import argparse
import os
import random
from collections import defaultdict
from dataclasses import fields

import numpy as np
import pandas as pd
import torch

from agent.quasimetric import (
	ContinualQuasimetricAgentConfig,
	ContinualQuasimetricSACAgent,
	QuasimetricConfig,
)
from agent.sac import SACAgent
from envs.metaworld_env import MetaWorldSingleEnvSequence
from replay_buffer_metric import ReplayBufferMetric


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def str2none(value):
	if value is None:
		return None
	if value.lower() in {"none", ""}:
		return None
	return value


def str_choice(value):
	return value.strip().strip("'\"‘’“”").lower()


def set_seed_everywhere(seed_value):
	seed_value = int(seed_value)
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	os.environ["PYTHONHASHSEED"] = str(seed_value)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True


def vector_observation_space(observation_space):
	spaces = getattr(observation_space, "spaces", None)
	if spaces is not None and "observation" in spaces:
		return spaces["observation"]
	return observation_space


def mean_std(values):
	values = np.asarray(values, dtype=np.float32)
	if values.size == 0:
		return np.nan, np.nan
	return float(np.mean(values)), float(np.std(values))


def summarize_eval_results(eval_results):
	return_mean, return_std = mean_std(eval_results["episodic_returns"])
	metaworld_success_mean, metaworld_success_std = mean_std(eval_results["successes"])
	gc_success_mean, gc_success_std = mean_std(eval_results.get("goal_successes", []))
	gc_final_distance_mean, gc_final_distance_std = mean_std(eval_results.get("final_goal_distances", []))
	gc_min_distance_mean, gc_min_distance_std = mean_std(eval_results.get("min_goal_distances", []))
	gc_mean_distance_mean, gc_mean_distance_std = mean_std(eval_results.get("mean_goal_distances", []))
	return {
		"return_mean": return_mean,
		"return_std": return_std,
		"metaworld_success_mean": metaworld_success_mean,
		"metaworld_success_std": metaworld_success_std,
		"gc_success_mean": gc_success_mean,
		"gc_success_std": gc_success_std,
		"gc_final_goal_distance_mean": gc_final_distance_mean,
		"gc_final_goal_distance_std": gc_final_distance_std,
		"gc_min_goal_distance_mean": gc_min_distance_mean,
		"gc_min_goal_distance_std": gc_min_distance_std,
		"gc_mean_goal_distance_mean": gc_mean_distance_mean,
		"gc_mean_goal_distance_std": gc_mean_distance_std,
	}


def append_eval_stats(stats, eval_metrics):
	stats["mean_return"].append(eval_metrics["return_mean"])
	stats["mean_success"].append(eval_metrics["metaworld_success_mean"])
	stats["metaworld_success"].append(eval_metrics["metaworld_success_mean"])
	stats["gc_success"].append(eval_metrics["gc_success_mean"])
	stats["gc_final_goal_distance"].append(eval_metrics["gc_final_goal_distance_mean"])
	stats["gc_min_goal_distance"].append(eval_metrics["gc_min_goal_distance_mean"])
	stats["gc_mean_goal_distance"].append(eval_metrics["gc_mean_goal_distance_mean"])


def resolve_existing_path(path):
	if path is None or os.path.isabs(path) or os.path.exists(path):
		return path

	script_relative_path = os.path.join(SCRIPT_DIR, path)
	if os.path.exists(script_relative_path):
		return script_relative_path
	return path


def resolve_eval_model_dir(args):
	model_dir = resolve_existing_path(args.model_dir)
	if args.eval_agent_type != "distilled":
		return model_dir
	if os.path.basename(os.path.normpath(model_dir)) == "model_distilledagent":
		return model_dir

	distilled_model_dir = os.path.join(os.path.dirname(model_dir), "model_distilledagent")
	if not os.path.isdir(distilled_model_dir):
		raise FileNotFoundError(
			f"Missing distilled checkpoint directory: {distilled_model_dir}. "
			"Pass --model_dir with either the training model directory or model_distilledagent."
		)
	return distilled_model_dir


def make_env_kwargs(args):
	env_name = args.env.lower()
	env_sequence = args.env_sequence
	base_task_name = args.base_task_name

	if env_name.startswith("metaworld_sequence_"):
		env_suffix = env_name[len("metaworld_sequence_"):]
		if env_suffix.startswith("set"):
			env_sequence = env_suffix
		else:
			base_task_name = f"{env_suffix}-v2"
			env_sequence = None

	return {
		"change_freq": args.change_freq,
		"base_task_name": base_task_name,
		"env_sequence": env_sequence,
		"goal_hidden": bool(args.goal_hidden),
		"goal_conditioned": bool(args.goal_conditioned),
		"gc_reward_type": args.gc_reward_type,
		"gc_success_threshold": args.gc_success_threshold,
		"gc_achieved_goal": args.gc_achieved_goal,
		"normalize_obs": args.normalize_obs,
		"normalize_avg_coef": args.normalize_avg_coef,
		"normalize_rewards": bool(args.normalize_rewards),
		"reset_obs_stats": bool(args.reset_obs_stats),
		"change_when_solved": bool(args.change_when_solved),
		"freeze_rand_vec": bool(args.freeze_rand_vec),
		"capture_video": False,
		"seed": args.seed + args.repeat_idx,
	}


def make_log_name(args):
	if args.log_name is not None:
		return args.log_name
	gc_log_suffix = f"_gc-{args.gc_reward_type}" if bool(args.goal_conditioned) else ""
	return f"sac_{args.env}_{args.seed}_{args.method}{gc_log_suffix}"


def task_model_name(log_name, task_idx, use_meta=False):
	model_name = f"{log_name}_{task_idx}"
	if use_meta:
		model_name += "_meta"
	return model_name


def checkpoint_path(model_dir, model_name, suffix):
	return os.path.join(model_dir, f"{model_name}_{suffix}.pt")


def load_state_dict_file(path, device):
	try:
		return torch.load(path, map_location=device, weights_only=True)
	except TypeError:
		return torch.load(path, map_location=device)
	except Exception as error:
		print(f"weights_only load failed for {path}: {error}")
		return torch.load(path, map_location=device, weights_only=False)


def load_optional_state(module, path, device):
	if not os.path.exists(path):
		return False
	module.load_state_dict(load_state_dict_file(path, device))
	return True


def load_optional_checkpoint_module(agent, name, model_dir, model_name, suffix, device):
	module = getattr(agent, name, None)
	if module is None:
		return False
	return load_optional_state(module, checkpoint_path(model_dir, model_name, suffix), device)


def load_agent_checkpoint(agent, model_dir, model_name, device):
	actor_path = checkpoint_path(model_dir, model_name, "actor")
	if not os.path.exists(actor_path):
		raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")

	agent.actor.load_state_dict(load_state_dict_file(actor_path, device))
	critic_loaded = load_optional_state(agent.critic, checkpoint_path(model_dir, model_name, "critic"), device)
	target_loaded = load_optional_state(
		agent.critic_target,
		checkpoint_path(model_dir, model_name, "critic_target"),
		device,
	)
	print(
		f"loaded agent checkpoint: {os.path.join(model_dir, model_name)} "
		f"actor=True critic={critic_loaded} critic_target={target_loaded}"
	)


def dataclass_from_saved(cls, values):
	if values is None:
		return cls()
	allowed = {field.name for field in fields(cls)}
	return cls(**{key: value for key, value in values.items() if key in allowed})


def load_meta_payload(model_dir, model_name, device):
	structure_path = checkpoint_path(model_dir, model_name, "quasimetric")
	if not os.path.exists(structure_path):
		raise FileNotFoundError(f"Missing meta quasimetric checkpoint: {structure_path}")
	return load_state_dict_file(structure_path, device)


def build_agent(obs_dim, action_dim, device, args):
	return SACAgent(
		obs_dim=obs_dim,
		action_dim=action_dim,
		action_range=[-1.0, 1.0],
		device=device,
		batch_size=args.batch_size,
		discount=args.discount,
		init_temperature=args.init_temperature,
		actor_lr=args.actor_lr,
		critic_lr=args.critic_lr,
		alpha_lr=args.alpha_lr,
		critic_tau=args.critic_tau,
		actor_update_frequency=args.actor_update_frequency,
		critic_target_update_frequency=args.critic_target_update_frequency,
	)


def build_meta_agent(obs_dim, action_dim, device, args, payload):
	quasimetric_cfg = dataclass_from_saved(QuasimetricConfig, payload.get("quasimetric_cfg"))
	continual_cfg = dataclass_from_saved(ContinualQuasimetricAgentConfig, payload.get("continual_cfg"))
	return ContinualQuasimetricSACAgent(
		obs_dim=obs_dim,
		action_dim=action_dim,
		action_range=[-1.0, 1.0],
		device=device,
		batch_size=args.batch_size,
		discount=args.discount,
		init_temperature=args.init_temperature,
		actor_lr=args.actor_lr,
		critic_lr=args.critic_lr,
		alpha_lr=args.alpha_lr,
		critic_tau=args.critic_tau,
		actor_update_frequency=args.actor_update_frequency,
		critic_target_update_frequency=args.critic_target_update_frequency,
		quasimetric_cfg=quasimetric_cfg,
		continual_cfg=continual_cfg,
	)


def load_meta_checkpoint(agent, payload, model_dir, model_name, device):
	actor_path = checkpoint_path(model_dir, model_name, "actor")
	if not os.path.exists(actor_path):
		raise FileNotFoundError(f"Missing meta actor checkpoint: {actor_path}")

	agent.actor.load_state_dict(load_state_dict_file(actor_path, device))
	critic_loaded = load_optional_checkpoint_module(agent, "critic", model_dir, model_name, "critic", device)
	target_loaded = load_optional_checkpoint_module(
		agent,
		"critic_target",
		model_dir,
		model_name,
		"critic_target",
		device,
	)
	goal_encoder_loaded = load_optional_checkpoint_module(
		agent,
		"goal_encoder",
		model_dir,
		model_name,
		"goal_encoder",
		device,
	)

	agent.quasimetric.load_checkpoint(payload["quasimetric"])
	if "structure_memory" in payload:
		agent.structure_memory.load_state_dict(payload["structure_memory"])

	print(
		f"loaded meta checkpoint: {os.path.join(model_dir, model_name)} "
		f"actor=True critic={critic_loaded} critic_target={target_loaded} "
		f"goal_encoder={goal_encoder_loaded} memory_tasks={agent.structure_memory.num_tasks}"
	)


def build_eval_agent(obs_dim, action_dim, model_dir, model_name, device, args):
	if args.eval_agent_type == "meta":
		payload = load_meta_payload(model_dir, model_name, device)
		agent = build_meta_agent(obs_dim, action_dim, device, args, payload)
		load_meta_checkpoint(agent, payload, model_dir, model_name, device)
		return agent

	agent = build_agent(obs_dim, action_dim, device, args)
	load_agent_checkpoint(agent, model_dir, model_name, device)
	return agent


def vectorize_goal_obs(obs):
	if isinstance(obs, dict):
		return np.asarray(obs["observation"], dtype=np.float32)
	return np.asarray(obs, dtype=np.float32)


def obs_vector(obs):
	if isinstance(obs, dict):
		return obs["observation"]
	return obs


def load_goal_array(args):
	if args.meta_goal_file is None:
		return None
	if not hasattr(args, "_meta_goal_array"):
		path = resolve_existing_path(args.meta_goal_file)
		if path.endswith(".npy"):
			args._meta_goal_array = np.load(path)
		elif path.endswith(".npz"):
			payload = np.load(path)
			key = "goals" if "goals" in payload else payload.files[0]
			args._meta_goal_array = payload[key]
		else:
			args._meta_goal_array = np.loadtxt(path, delimiter=args.meta_goal_file_delimiter)
	return args._meta_goal_array


def goal_from_file(args, agent_idx, task_eval_idx):
	goals = load_goal_array(args)
	if goals is None:
		return None
	goals = np.asarray(goals, dtype=np.float32)
	if goals.ndim == 1:
		return goals
	if goals.ndim == 2:
		if task_eval_idx >= goals.shape[0]:
			raise IndexError(f"goal file has {goals.shape[0]} rows, cannot read task {task_eval_idx}")
		return goals[task_eval_idx]
	if goals.ndim == 3:
		return goals[agent_idx, task_eval_idx]
	raise ValueError(f"Unsupported goal file shape: {goals.shape}")


def sample_memory_view_goal(view, args):
	if view is None or len(view) == 0:
		return None
	if args.meta_goal_memory_strategy == "first":
		goal_idx = 0
	elif args.meta_goal_memory_strategy == "last":
		goal_idx = len(view) - 1
	else:
		goal_idx = int(np.random.randint(0, len(view)))
	return np.array(view.next_obses[goal_idx], dtype=np.float32, copy=True)


def memory_task_view(agent, task_eval_idx):
	memory = getattr(agent, "structure_memory", None)
	tasks = getattr(memory, "_tasks", {})
	if not tasks:
		return None
	for task_id in (task_eval_idx + 1, task_eval_idx):
		if task_id in tasks:
			return tasks[task_id]
	views = list(tasks.values())
	if task_eval_idx < len(views):
		return views[task_eval_idx]
	return None


def goal_from_memory_any(agent, args):
	memory = getattr(agent, "structure_memory", None)
	views = list(getattr(memory, "_tasks", {}).values())
	views = [view for view in views if len(view) > 0]
	if not views:
		return None
	weights = np.asarray([len(view) for view in views], dtype=np.float64)
	weights /= weights.sum()
	view = views[int(np.random.choice(np.arange(len(views)), p=weights))]
	return sample_memory_view_goal(view, args)


def rollout_goal_success_only(agent, args):
	if args.rollout_goal_success_only < 0:
		return bool(getattr(agent, "behavior_goal_success_only", True))
	return bool(args.rollout_goal_success_only)


def sample_rollout_buffer_goal(rollout_goal_buffers, agent, agent_idx, task_eval_idx, args, source):
	if not rollout_goal_buffers:
		return None

	if source in {"rollout_any", "rollout_buffer"}:
		candidate_indices = [idx for idx in range(agent_idx + 1) if idx in rollout_goal_buffers]
		candidate_buffers = [rollout_goal_buffers[idx] for idx in candidate_indices if len(rollout_goal_buffers[idx]) > 0]
		if not candidate_buffers:
			return None
		weights = np.asarray([len(buffer) for buffer in candidate_buffers], dtype=np.float64)
		weights /= weights.sum()
		buffer = candidate_buffers[int(np.random.choice(np.arange(len(candidate_buffers)), p=weights))]
	else:
		buffer = rollout_goal_buffers.get(task_eval_idx)
		if buffer is None or len(buffer) == 0:
			return None

	return buffer.sample_behavior_goal(
		discount=getattr(agent, "goal_discount", 0.995),
		# success_only=rollout_goal_success_only(agent, args),
		success_only=1
	)


def rollout_agent_to_buffer(env, agent, replay_buffer, args):
	agent.eval()
	obs, _ = env.reset()
	obs = obs_vector(obs)
	episodes = 0
	steps = 0
	returns = []
	successes = []
	max_steps = args.rollout_goal_steps_per_task
	if max_steps is None:
		max_steps = args.rollout_goal_episodes * args.rollout_goal_max_episode_steps

	while episodes < args.rollout_goal_episodes and steps < max_steps:
		with torch.inference_mode():
			action = agent.act(obs, sample=bool(args.rollout_goal_sample_action))
		next_obs, reward, terminated, truncated, info = env.no_count_step(action)
		next_obs = obs_vector(next_obs)
		done = float(terminated) or float(truncated)
		done_no_max = 0.0 if truncated else done
		replay_buffer.add(obs, action, reward, info.get("success", False), next_obs, done, done_no_max)

		obs = next_obs
		steps += 1
		if terminated or truncated:
			episode_info = info.get("episode", {})
			returns.append(float(episode_info.get("r", np.nan)))
			env_successes = getattr(env.env, "successes", [])
			if len(env_successes) > 0:
				successes.append(float(env_successes[-1]))
			else:
				successes.append(float(info.get("success", False)))
			episodes += 1
			obs, _ = env.reset()
			obs = obs_vector(obs)

	agent.train()
	return {
		"episodes": episodes,
		"steps": steps,
		"return_mean": float(np.nanmean(returns)) if returns else np.nan,
		"success_mean": float(np.mean(successes)) if successes else np.nan,
	}


def evaluate_meta_with_student_episode_goals(
	env,
	meta_agent,
	student_agent,
	num_eval_episodes,
	obs_space,
	action_shape,
	device,
	args,
):
	test_env = env._wrap_env(env._make_base_env(), eval_mode=True)
	base_env = test_env.unwrapped
	original_freeze_rand_vec = bool(base_env._freeze_rand_vec)
	max_student_steps = max(1, int(args.rollout_goal_max_episode_steps))

	episodic_returns = []
	successes = []
	goal_successes = []
	final_goal_distances = []
	min_goal_distances = []
	mean_goal_distances = []
	goal_norms = []
	student_successes = []
	meta_agent.eval()
	student_agent.eval()

	try:
		for _ in range(num_eval_episodes):
			base_env._freeze_rand_vec = original_freeze_rand_vec
			student_obs, _ = test_env.reset()
			paired_rand_vec = np.array(base_env._last_rand_vec, copy=True)
			if env._uses_obs_normalization():
				student_obs = env._normalize_obs(student_obs)

			episode_buffer = ReplayBufferMetric(
				obs_space.shape,
				action_shape,
				max_student_steps,
				device,
			)
			student_info = {}
			for _ in range(max_student_steps):
				with torch.inference_mode():
					action = student_agent.act(
						obs_vector(student_obs),
						sample=bool(args.rollout_goal_sample_action),
					)
				next_student_obs, reward, terminated, truncated, student_info = test_env.step(action)
				if env._uses_obs_normalization():
					next_student_obs = env._normalize_obs(next_student_obs)
				done = float(terminated) or float(truncated)
				done_no_max = 0.0 if truncated else done
				episode_buffer.add(
					obs_vector(student_obs),
					action,
					reward,
					student_info.get("success", False),
					obs_vector(next_student_obs),
					done,
					done_no_max,
				)
				student_obs = next_student_obs
				if terminated or truncated:
					break

			student_episode_successes = test_env.pop_successes()
			student_successes.append(
				float(student_episode_successes[-1])
				if student_episode_successes
				else float(student_info.get("success", False))
			)
			goal = episode_buffer.sample_behavior_goal(
				discount=getattr(meta_agent, "goal_discount", 0.995),
				success_only=rollout_goal_success_only(meta_agent, args),
			)
			goal = validate_meta_goal(meta_agent, goal, "rollout_task_episode")
			if goal is None:
				raise RuntimeError("Student episode did not produce a meta behavior goal.")
			meta_agent.set_behavior_goal(goal)
			goal_norms.append(float(np.linalg.norm(goal)))

			base_env._last_rand_vec = paired_rand_vec.copy()
			base_env._freeze_rand_vec = True
			obs, _ = test_env.reset()
			if not np.array_equal(paired_rand_vec, base_env._last_rand_vec):
				raise RuntimeError("Meta episode did not reuse the student episode rand_vec.")
			if env._uses_obs_normalization():
				obs = env._normalize_obs(obs)

			current_goal_success = False
			current_goal_distances = []
			meta_info = {}
			while True:
				with torch.inference_mode():
					action = env._evaluate_action(meta_agent, obs)
				next_obs, _, terminated, truncated, meta_info = test_env.step(action)
				if "is_success" in meta_info:
					current_goal_success = current_goal_success or bool(meta_info["is_success"])
				if "goal_distance" in meta_info:
					current_goal_distances.append(float(meta_info["goal_distance"]))
				if env._uses_obs_normalization():
					next_obs = env._normalize_obs(next_obs)
				obs = next_obs
				if terminated or truncated:
					break

			episode_info = meta_info.get("episode", {})
			episodic_returns.append(float(episode_info.get("r", np.nan)))
			meta_episode_successes = test_env.pop_successes()
			successes.append(
				float(meta_episode_successes[-1])
				if meta_episode_successes
				else float(meta_info.get("success", False))
			)
			if current_goal_distances:
				goal_successes.append(current_goal_success)
				final_goal_distances.append(current_goal_distances[-1])
				min_goal_distances.append(float(np.min(current_goal_distances)))
				mean_goal_distances.append(float(np.mean(current_goal_distances)))
	finally:
		base_env._freeze_rand_vec = original_freeze_rand_vec
		student_agent.train()
		meta_agent.train()

	print(
		f"paired episode goals: episodes={num_eval_episodes} "
		f"student_success={round(float(np.mean(student_successes)), 3)} "
		f"goal_norm={round(float(np.mean(goal_norms)), 3)}"
	)
	return {
		"episodic_returns": episodic_returns,
		"successes": successes,
		"goal_successes": goal_successes,
		"final_goal_distances": final_goal_distances,
		"min_goal_distances": min_goal_distances,
		"mean_goal_distances": mean_goal_distances,
	}, {
		"source": "rollout_task_episode",
		"ready": 1.0,
		"norm": float(np.mean(goal_norms)),
	}


def build_rollout_goal_buffers(log_name, model_dir, obs_space, action_shape, device, env_kwargs, env_list, num_agents, args):
	rollout_goal_buffers = {}
	goal_env = MetaWorldSingleEnvSequence(**env_kwargs)
	capacity = args.rollout_goal_buffer_capacity
	if capacity is None:
		steps_per_task = args.rollout_goal_steps_per_task
		if steps_per_task is None:
			steps_per_task = args.rollout_goal_episodes * args.rollout_goal_max_episode_steps
		capacity = max(1, int(steps_per_task))

	print(
		f"building rollout goal buffers from {num_agents} trained task agents: "
		f"episodes_per_task={args.rollout_goal_episodes} capacity={capacity}"
	)
	for task_idx in range(num_agents):
		goal_env.set_task(env_list[task_idx])
		model_name = task_model_name(log_name, task_idx, use_meta=False)
		task_agent = build_agent(obs_space.shape[0], action_shape[0], device, args)
		load_agent_checkpoint(task_agent, model_dir, model_name, device)
		buffer = ReplayBufferMetric(obs_space.shape, action_shape, capacity, device)
		rollout_metrics = rollout_agent_to_buffer(goal_env, task_agent, buffer, args)
		rollout_goal_buffers[task_idx] = buffer
		print(
			f"rollout buffer task {task_idx + 1} {env_list[task_idx]}: "
			f"size={len(buffer)} episodes={rollout_metrics['episodes']} "
			f"return={round(rollout_metrics['return_mean'], 3)} "
			f"success={round(rollout_metrics['success_mean'], 3)}"
		)

	return rollout_goal_buffers


def validate_meta_goal(agent, goal, source):
	if goal is None:
		return None
	goal = np.asarray(goal, dtype=np.float32)
	expected_dim = getattr(agent, "obs_dim", None)
	if expected_dim is not None and goal.shape[-1] != expected_dim:
		raise ValueError(
			f"Meta goal from {source} has dim {goal.shape[-1]}, expected full observation dim {expected_dim}. "
			"Do not pass MetaWorld's 3D desired_goal directly unless the meta actor was trained for that dim."
		)
	return goal


def select_meta_goal(agent, env, agent_idx, task_eval_idx, args, rollout_goal_buffers=None):
	if args.eval_agent_type != "meta":
		return {"source": "", "ready": np.nan, "norm": np.nan}

	def get_goal(source):
		if source == "zero":
			return None
		if source == "goal_file":
			return goal_from_file(args, agent_idx, task_eval_idx)
		if source == "memory_task":
			return sample_memory_view_goal(memory_task_view(agent, task_eval_idx), args)
		if source == "memory_any":
			return goal_from_memory_any(agent, args)
		if source in {"rollout_task", "rollout_any", "rollout_task_buffer", "rollout_buffer"}:
			return sample_rollout_buffer_goal(rollout_goal_buffers, agent, agent_idx, task_eval_idx, args, source)
		if source == "env_reset":
			obs, _ = env.reset()
			return vectorize_goal_obs(obs)
		raise ValueError(f"Unsupported meta_goal_source: {source}")

	goal = get_goal(args.meta_goal_source)
	goal_source = args.meta_goal_source
	if goal is None and args.meta_goal_source != "zero":
		if args.meta_goal_fallback == "error":
			raise ValueError(
				f"Could not choose a meta behavior goal from {args.meta_goal_source}. "
				"Use --meta_goal_source rollout_task to build goals from task-agent rollouts, "
				"--meta_goal_fallback zero for a zero-goal debug eval, or provide --meta_goal_file."
			)
		goal_source = f"{args.meta_goal_source}->{args.meta_goal_fallback}"
		goal = get_goal(args.meta_goal_fallback)

	goal = validate_meta_goal(agent, goal, goal_source)
	agent.set_behavior_goal(goal)
	ready = goal is not None
	goal_norm = float(np.linalg.norm(goal)) if ready else 0.0
	print(f"meta goal: source={goal_source} ready={ready} norm={round(goal_norm, 3)}")
	return {"source": goal_source, "ready": float(ready), "norm": goal_norm}


def evaluate_loaded_agents(env, log_name, model_dir, device, env_kwargs, args):
	obs_space = vector_observation_space(env.env.observation_space)
	action_shape = env.env.action_space.shape
	action_dim = action_shape[0]
	num_agents = args.num_agents if args.num_agents is not None else len(env.env_list)

	if num_agents > len(env.env_list):
		raise ValueError(f"num_agents={num_agents} exceeds number of tasks={len(env.env_list)}")
	if args.agent_idx is not None and not 0 <= args.agent_idx < num_agents:
		raise ValueError(f"agent_idx={args.agent_idx} must be in [0, {num_agents - 1}]")
	if args.agent_idx is not None:
		agent_indices = [args.agent_idx]
	elif args.eval_agent_type == "distilled":
		agent_indices = [
			agent_idx
			for agent_idx in range(num_agents)
			if os.path.exists(
				checkpoint_path(model_dir, task_model_name(log_name, agent_idx), "actor")
			)
		]
		if not agent_indices:
			raise FileNotFoundError(f"No distilled actor checkpoints found in {model_dir}")
	else:
		agent_indices = range(num_agents)

	for agent_idx in agent_indices:
		model_name = task_model_name(log_name, agent_idx, use_meta=args.eval_agent_type == "meta")
		actor_path = checkpoint_path(model_dir, model_name, "actor")
		if not os.path.exists(actor_path):
			raise FileNotFoundError(
				f"Missing {args.eval_agent_type} actor checkpoint for agent_idx={agent_idx}: {actor_path}"
			)

	final_stats = defaultdict(list)
	print("---")
	print(
		f"evaluating {len(agent_indices)} loaded {args.eval_agent_type} agent(s) on previous tasks: "
		f"checkpoints={[agent_idx + 1 for agent_idx in agent_indices]}"
	)
	rollout_goal_buffers = None
	rollout_sources = {"rollout_task", "rollout_any", "rollout_task_buffer", "rollout_buffer"}
	paired_goal_sources = {"rollout_task", "rollout_task_buffer"}
	use_paired_episode_goals = (
		args.eval_agent_type == "meta" and args.meta_goal_source in paired_goal_sources
	)
	if not use_paired_episode_goals and args.eval_agent_type == "meta" and (
		args.meta_goal_source in rollout_sources or args.meta_goal_fallback in rollout_sources
	):
		rollout_goal_buffers = build_rollout_goal_buffers(
			log_name,
			model_dir,
			obs_space,
			action_shape,
			device,
			env_kwargs,
			env.env_list,
			num_agents,
			args,
		)

	for agent_idx in agent_indices:
		model_name = task_model_name(log_name, agent_idx, use_meta=args.eval_agent_type == "meta")
		agent = build_eval_agent(obs_space.shape[0], action_dim, model_dir, model_name, device, args)

		for task_eval_idx in range(agent_idx + 1):
			env.set_task(env.env_list[task_eval_idx])
			if use_paired_episode_goals:
				student_model_name = task_model_name(log_name, task_eval_idx, use_meta=False)
				student_agent = build_agent(obs_space.shape[0], action_dim, device, args)
				load_agent_checkpoint(student_agent, model_dir, student_model_name, device)
				eval_results, goal_info = evaluate_meta_with_student_episode_goals(
					env,
					agent,
					student_agent,
					args.num_eval_runs,
					obs_space,
					action_shape,
					device,
					args,
				)
			else:
				goal_info = select_meta_goal(
					agent,
					env,
					agent_idx,
					task_eval_idx,
					args,
					rollout_goal_buffers,
				)
				eval_results = env.evaluate_agent(
					agent,
					args.num_eval_runs,
					reseed_each_episode=False,
				)
			eval_metrics = summarize_eval_results(eval_results)

			print(
				f"Final task {env.env_list[task_eval_idx]} "
				f"success {round(eval_metrics['metaworld_success_mean'], 3)} "
				f"gc_success {round(eval_metrics['gc_success_mean'], 3)} "
				f"return {round(eval_metrics['return_mean'], 3)}"
			)

			append_eval_stats(final_stats, eval_metrics)
			final_stats["task"].append(env.base_task_name)
			final_stats["task_idx"].append(task_eval_idx + 1)
			final_stats["seed"].append(args.seed)
			final_stats["method"].append(args.method)
			final_stats["agent_idx"].append(agent_idx + 1)
			final_stats["eval_agent_type"].append(args.eval_agent_type)
			final_stats["checkpoint"].append(model_name)
			final_stats["meta_goal_source"].append(goal_info["source"])
			final_stats["meta_goal_ready"].append(goal_info["ready"])
			final_stats["meta_goal_norm"].append(goal_info["norm"])

	return final_stats


def output_csv_path(args, log_name):
	if args.output_csv is not None:
		return args.output_csv

	log_root = resolve_existing_path(args.log_dir)
	output_suffix = args.output_suffix
	if args.eval_agent_type == "meta" and output_suffix == "_final.csv":
		output_suffix = "_meta_final.csv"
	elif args.eval_agent_type == "distilled" and output_suffix == "_final.csv":
		output_suffix = "_distilled_final.csv"
	return os.path.join(log_root, args.env, f"{log_name}{output_suffix}")


def build_parser(description="Offline final evaluation for saved MetaWorld task agents"):
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument(
		"--eval_agent_type",
		type=str_choice,
		default="agent",
		choices=["agent", "meta", "distilled"],
		help="Evaluate ordinary task agents, goal-conditioned meta agents, or distilled task agents",
	)
	parser.add_argument("--method", type=str, default="buffer", help="Training method used in the saved log name")
	parser.add_argument("--repeat_idx", type=int, default=0, help="Index of the repeat")
	parser.add_argument("--env", type=str, default="metaworld_sequence_set6", help="Environment name used for training")
	parser.add_argument("--env_sequence", type=str2none, default=None, help="Override MetaWorld sequence, e.g. set6")
	parser.add_argument("--base_task_name", type=str2none, default=None, help="Single MetaWorld task name")
	parser.add_argument("--change_freq", type=int, default=900000, help="Task change frequency used by the env")
	parser.add_argument("--normalize_obs", type=str2none, default=None, help="Observation normalization mode")
	parser.add_argument("--normalize_avg_coef", type=float, default=0.0001, help="EMA coefficient for observation stats")
	parser.add_argument("--normalize_rewards", type=int, default=1, help="Whether to normalize non-GC rewards")
	parser.add_argument("--reset_obs_stats", type=int, default=0, help="Whether env resets obs stats on task switch")
	parser.add_argument("--change_when_solved", type=int, default=0, help="Whether env switches after solved evals")
	parser.add_argument(
		"--freeze_rand_vec",
		type=int,
		default=1,
		choices=[0, 1],
		help="Reuse one MetaWorld task instance across episode resets",
	)
	parser.add_argument("--goal_hidden", type=int, default=1, help="Use goal-hidden MetaWorld tasks")
	parser.add_argument("--goal_conditioned", type=int, default=0, help="Use goal-conditioned dict observations")
	parser.add_argument(
		"--gc_reward_type",
		type=str_choice,
		default="sparse",
		choices=["sparse", "dense", "success"],
		help="Goal-conditioned reward type",
	)
	parser.add_argument("--gc_success_threshold", type=float, default=0.05, help="Goal success distance threshold")
	parser.add_argument(
		"--gc_achieved_goal",
		type=str_choice,
		default="auto",
		choices=["auto", "object", "tcp"],
		help="Goal-conditioned achieved goal source",
	)
	parser.add_argument("--seed", type=int, default=0, help="Random seed used for training")
	parser.add_argument("--gpu", type=str, default="0", help="Comma separated GPU IDs")
	parser.add_argument("--batch_size", type=int, default=256, help="SAC batch size")
	parser.add_argument("--discount", type=float, default=0.99, help="SAC discount")
	parser.add_argument("--init_temperature", type=float, default=0.1, help="Initial SAC temperature")
	parser.add_argument("--actor_lr", type=float, default=1e-4, help="Actor learning rate")
	parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic learning rate")
	parser.add_argument("--alpha_lr", type=float, default=1e-4, help="Temperature learning rate")
	parser.add_argument("--critic_tau", type=float, default=0.005, help="Target critic tau")
	parser.add_argument("--actor_update_frequency", type=int, default=1, help="Actor update frequency")
	parser.add_argument("--critic_target_update_frequency", type=int, default=1, help="Critic target update frequency")
	parser.add_argument("--num_eval_runs", type=int, default=10, help="Evaluation episodes per task")
	parser.add_argument("--num_agents", type=int, default=None, help="Number of saved task agents to evaluate")
	parser.add_argument(
		"--agent_idx",
		type=int,
		default=None,
		help="Evaluate only this zero-based checkpoint index; by default evaluate every checkpoint",
	)
	parser.add_argument(
		"--meta_goal_source",
		type=str_choice,
		default="rollout_task",
		choices=[
			"zero",
			"memory_task",
			"memory_any",
			"goal_file",
			"env_reset",
			"rollout_task",
			"rollout_any",
			"rollout_task_buffer",
			"rollout_buffer",
		],
		help="How to choose behavior_goal when eval_agent_type=meta",
	)
	parser.add_argument(
		"--meta_goal_fallback",
		type=str_choice,
		default="error",
		choices=["error", "zero", "env_reset", "rollout_task", "rollout_any", "rollout_task_buffer", "rollout_buffer"],
		help="Fallback when meta_goal_source cannot produce a goal",
	)
	parser.add_argument(
		"--meta_goal_memory_strategy",
		type=str_choice,
		default="random",
		choices=["random", "first", "last"],
		help="Which transition to use when sampling a goal from checkpoint memory",
	)
	parser.add_argument("--meta_goal_file", type=str2none, default=None, help=".npy/.npz/.csv file containing full observation goals")
	parser.add_argument("--meta_goal_file_delimiter", type=str, default=",", help="Delimiter for text goal files")
	parser.add_argument(
		"--rollout_goal_episodes",
		type=int,
		default=5,
		help="Episodes per task for prebuilt goal buffers; paired rollout_task evaluation uses one student episode per meta episode",
	)
	parser.add_argument(
		"--rollout_goal_steps_per_task",
		type=int,
		default=None,
		help="Optional step cap per task-agent rollout goal buffer",
	)
	parser.add_argument(
		"--rollout_goal_max_episode_steps",
		type=int,
		default=200,
		help="Max episode length used to size rollout goal buffers when steps_per_task is unset",
	)
	parser.add_argument(
		"--rollout_goal_buffer_capacity",
		type=int,
		default=None,
		help="Optional explicit ReplayBufferMetric capacity per rollout task buffer",
	)
	parser.add_argument(
		"--rollout_goal_sample_action",
		type=int,
		default=0,
		help="Whether task agents sample stochastic actions while constructing rollout goal buffers",
	)
	parser.add_argument(
		"--rollout_goal_success_only",
		type=int,
		default=-1,
		help="Goal sampling from rollout buffer: -1 uses meta config, 0 allows any future state, 1 prefers successful states",
	)
	parser.add_argument(
		"--model_dir",
		type=str,
		default="model",
		help="Checkpoint directory; distilled evaluation also accepts model and selects sibling model_distilledagent",
	)
	parser.add_argument("--log_dir", type=str, default="log", help="Root directory for final CSV output")
	parser.add_argument("--log_name", type=str2none, default=None, help="Explicit checkpoint/log name prefix")
	parser.add_argument("--output_csv", type=str2none, default=None, help="Explicit CSV output path")
	parser.add_argument("--output_suffix", type=str, default="_final.csv", help="Output suffix when output_csv is unset")
	return parser


def parse_args():
	return build_parser().parse_args()


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	set_seed_everywhere(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	env_kwargs = make_env_kwargs(args)
	print("env_kwargs", env_kwargs)
	env = MetaWorldSingleEnvSequence(**env_kwargs)
	if not hasattr(env, "env_list"):
		env.env_list = [env.base_task_name]
	print("env_list:", env.env_list)

	model_dir = resolve_eval_model_dir(args)
	log_name = make_log_name(args)
	print("log_name:", log_name)
	print("model_dir:", model_dir)

	final_stats = evaluate_loaded_agents(env, log_name, model_dir, device, env_kwargs, args)
	final_stats = pd.DataFrame(final_stats)

	csv_path = output_csv_path(args, log_name)
	csv_dir = os.path.dirname(csv_path)
	if csv_dir:
		os.makedirs(csv_dir, exist_ok=True)
	final_stats.to_csv(csv_path, index=False)
	print("saved final eval:", csv_path)


if __name__ == "__main__":
	main()


# python finaleval.py --eval_agent_type meta \
#   --env metaworld_sequence_set12 \
#   --method buffer \
#   --model_dir model \
#   --meta_goal_source rollout_task
'''
python Metaworld/final_eval_main3.py \
  --eval_agent_type meta \
  --env metaworld_sequence_set12 \
  --method buffer \
  --seed 0 \
  --gpu 0 \
  --freeze_rand_vec 0 \
  --model_dir Metaworld/results/main3/set12/model \
	--agent_idx 9 \
  --num_eval_runs 10 \
  --meta_goal_source rollout_task \
  --rollout_goal_episodes 1 \
  --rollout_goal_success_only 0 \
  --output_csv Metaworld/results/main3/set12/main3_meta_final4.csv

python Metaworld/final_eval_main3.py \
	--eval_agent_type meta \
	--env metaworld_sequence_set12 \
	--method buffer \
	--seed 0 \
	--gpu 0 \
	--freeze_rand_vec 0 \
	--model_dir Metaworld/model \
	--agent_idx 9 \
	--num_eval_runs 10 \
	--output_csv Metaworld/results/main3/set12/main3_last_distilled_final.csv

	python Metaworld/final_eval_main3.py \
	--eval_agent_type meta \
	--env metaworld_sequence_set12 \
	--method buffer \
	--seed 0 \
	--gpu 0 \
	--freeze_rand_vec 0 \
	--model_dir Metaworld/results/model_3wd \
	--agent_idx 9 \
	--num_eval_runs 10 \
	--output_csv Metaworld/results/main3/set12/main3_wd.csv
	
	python Metaworld/final_eval_main3.py \
		--eval_agent_type meta \
		--env metaworld_sequence_set12 \
		--method buffer \
		--seed 0 \
		--gpu 0 \
		--freeze_rand_vec 0 \
		--model_dir Metaworld/results/main3-2/set12_trj20_contra_abl/model \
		--agent_idx 9 \
		--num_eval_runs 50 \
		--output_csv Metaworld/results/main3-2/set12/main3-2_set12_trj20_contra_abl.csv
	
		python Metaworld/final_eval_main3_episode.py \
				--eval_agent_type meta \
				--env metaworld_sequence_set6 \
				--method buffer \
				--seed 1 \
				--gpu 0 \
				--freeze_rand_vec 0 \
				--model_dir Metaworld/results/main3-2/set6_trj20_seed1/model \
				--agent_idx 9 \
				--num_eval_runs 10 \
				--output_csv Metaworld/results/main3-2/set12/main3-2_set6_trj20_seed1-2.csv



	--model_dir Metaworld/results/main3/set12/model \
	
	'''