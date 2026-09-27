"""Sequential finetuning baseline for the CQRL3 online QRL learner.

The same student and optimizer state are carried across the Fetch task stream.
At each task boundary only the task-local replay buffer is reset; no meta agent,
cross-task replay, policy selection, or distillation is used.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

try:
	from . import main as base
except ImportError:
	import main as base


DEFAULT_SAVE_PATH = os.path.join(base.DEFAULT_SAVE_PATH, "finetuning")
RESET_DEFAULT_SAVE_PATH = os.path.join(base.DEFAULT_SAVE_PATH, "reset")
OFFLINE_TO_ONLINE_SAVE_PATH = os.path.join(
	base.DEFAULT_SAVE_PATH,
	"finetuning_offline_to_online",
)


def load_resume_config(resume_run_dir):
	if resume_run_dir is None:
		return {}

	config_path = os.path.join(os.path.abspath(resume_run_dir), "run_config.json")
	if not os.path.isfile(config_path):
		raise FileNotFoundError(f"Resume config not found: {config_path}")
	with open(config_path, "r") as handle:
		config = json.load(handle)
	for key in (
		"start_task",
		"resume_run_dir",
		"resume_model_name",
		"resume_buffer_path",
	):
		config.pop(key, None)
	return config


def set_env_task_position(env, task_position, change_freq):
	env.task_counter = task_position
	env.timestep_counter = (task_position - 1) * int(change_freq)
	env.set_task(env.env_list[task_position - 1])


def resolve_resume_checkpoint(args):
	source_run_name = os.path.basename(os.path.normpath(args.resume_run_dir))
	model_dir = os.path.join(args.resume_run_dir, "model")
	if args.resume_model_name is not None:
		return model_dir, args.resume_model_name

	model_name = f"{source_run_name}_task{args.start_task - 1}"
	for candidate in (model_name, f"{model_name}_fast"):
		if os.path.isfile(
			os.path.join(model_dir, f"{candidate}_online_qrl.pt")
		):
			return model_dir, candidate
	return model_dir, model_name


def replay_buffer_checkpoint_path(run_dir, model_name):
	return os.path.join(run_dir, "replay_buffers", model_name)


def start_new_collection_episode(collector, replay_buffer):
	replay_buffer.current_episode_start = int(replay_buffer.idx)
	collector.initial_collect(0)


def capture_rng_state():
	state = {
		"python": random.getstate(),
		"numpy": np.random.get_state(),
		"torch": torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		state["torch_cuda"] = torch.cuda.get_rng_state_all()
	return state


def restore_rng_state(state):
	if not state:
		return
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])
	torch.set_rng_state(state["torch"])
	if "torch_cuda" in state and torch.cuda.is_available():
		torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_numpy_generator_state(owner):
	generator = getattr(owner, "np_random", None)
	if generator is None or not hasattr(generator, "bit_generator"):
		return None
	return copy.deepcopy(generator.bit_generator.state)


def restore_numpy_generator_state(owner, state):
	if state is None:
		return
	generator = getattr(owner, "np_random", None)
	if generator is not None and hasattr(generator, "bit_generator"):
		generator.bit_generator.state = copy.deepcopy(state)


def capture_env_state(env):
	return {
		"task_counter": env.task_counter,
		"timestep_counter": env.timestep_counter,
		"base_task_name": env.base_task_name,
		"obs_mean": None if env.obs_mean is None else np.array(env.obs_mean, copy=True),
		"obs_var": None if env.obs_var is None else np.array(env.obs_var, copy=True),
		"obs_count": env.obs_count,
		"task_env_rng": capture_numpy_generator_state(env.env.unwrapped),
		"action_space_rng": capture_numpy_generator_state(env.env.action_space),
		"observation_space_rng": capture_numpy_generator_state(
			env.env.observation_space
		),
		"pending_reset_seed": env._pending_reset_seed,
	}


def restore_env_state(env, state, expected_task):
	if not state:
		return
	if int(state["task_counter"]) != expected_task:
		raise ValueError(
			"Resume state targets task "
			f"{state['task_counter']}, but --start_task is {expected_task}."
		)
	if state["base_task_name"] != env.base_task_name:
		raise ValueError(
			"Resume task name does not match the configured sequence: "
			f"{state['base_task_name']} != {env.base_task_name}."
		)
	env.timestep_counter = int(state["timestep_counter"])
	env.obs_mean = None if state["obs_mean"] is None else np.array(state["obs_mean"], copy=True)
	env.obs_var = None if state["obs_var"] is None else np.array(state["obs_var"], copy=True)
	env.obs_count = float(state["obs_count"])
	restore_numpy_generator_state(env.env.unwrapped, state.get("task_env_rng"))
	restore_numpy_generator_state(env.env.action_space, state.get("action_space_rng"))
	restore_numpy_generator_state(
		env.env.observation_space,
		state.get("observation_space_rng"),
	)
	if "pending_reset_seed" in state:
		env._pending_reset_seed = state["pending_reset_seed"]


def save_resume_state(
	model_dir,
	model_name,
	env,
	completed_task_count,
	step,
	buffer_path=None,
):
	resume_path = os.path.join(model_dir, f"{model_name}_resume.pt")
	temporary_path = f"{resume_path}.tmp"
	payload = {
		"version": 3,
		"completed_task": completed_task_count,
		"next_task": env.task_counter,
		"step": step,
		"env": capture_env_state(env),
		"rng": capture_rng_state(),
	}
	if buffer_path is not None:
		payload["buffer_path"] = os.path.abspath(buffer_path)
	torch.save(payload, temporary_path)
	os.replace(temporary_path, resume_path)
	print("saved resume state:", resume_path)


def save_training_checkpoint(
	agent,
	replay_buffer,
	model_dir,
	run_dir,
	model_name,
	env,
	completed_task_count,
	step,
):
	buffer_path = replay_buffer_checkpoint_path(run_dir, model_name)
	replay_buffer.save_data(buffer_path)
	agent.save(model_dir, model_name)
	save_resume_state(
		model_dir,
		model_name,
		env,
		completed_task_count,
		step,
		buffer_path,
	)
	return buffer_path


def load_resume_training_state(args, env, agent=None, replay_buffer=None):
	model_dir, model_name = resolve_resume_checkpoint(args)
	model_path = os.path.join(model_dir, f"{model_name}_online_qrl.pt")
	if agent is not None and not os.path.isfile(model_path):
		raise FileNotFoundError(f"Finetuning checkpoint not found: {model_path}")

	resume_names = [model_name]
	if model_name.endswith("_fast"):
		resume_names.append(model_name[: -len("_fast")])
	resume_path = next(
		(
			path
			for resume_name in resume_names
			for path in [os.path.join(model_dir, f"{resume_name}_resume.pt")]
			if os.path.isfile(path)
		),
		os.path.join(model_dir, f"{model_name}_resume.pt"),
	)
	resume_state = {}
	if os.path.isfile(resume_path):
		resume_state = torch.load(
			resume_path,
			map_location="cpu",
			weights_only=False,
		)
		if int(resume_state["completed_task"]) != args.start_task - 1:
			raise ValueError(
				f"Resume checkpoint completed task {resume_state['completed_task']}, "
				f"but --start_task {args.start_task} requires task {args.start_task - 1}."
			)
		if int(resume_state["next_task"]) != args.start_task:
			raise ValueError(
				f"Resume checkpoint targets task {resume_state['next_task']}, "
				f"but --start_task is {args.start_task}."
			)
	else:
		print(
			"resume state marker not found; using an inferred step and freshly "
			"seeded environment state for this legacy checkpoint"
		)

	if agent is not None:
		agent.load(model_dir, model_name)
		print("loaded finetuning checkpoint:", model_path)
	if replay_buffer is not None:
		inferred_buffer_path = replay_buffer_checkpoint_path(
			args.resume_run_dir,
			model_name,
		)
		buffer_candidates = [
			getattr(args, "resume_buffer_path", None),
			resume_state.get("buffer_path"),
			inferred_buffer_path,
		]
		buffer_path = next(
			(
				os.path.abspath(path)
				for path in buffer_candidates
				if path is not None
				and os.path.isfile(
					os.path.join(os.path.abspath(path), "offline_data.npz")
				)
			),
			None,
		)
		if buffer_path is None:
			raise FileNotFoundError(
				"Unable to find the replay buffer paired with checkpoint "
				f"{model_name!r}. Checked: "
				+ ", ".join(
					str(path) for path in buffer_candidates if path is not None
				)
			)
		if not replay_buffer.load_data(buffer_path):
			raise ValueError(f"Unable to load replay buffer: {buffer_path}")
		print("loaded replay buffer:", buffer_path, "size:", len(replay_buffer))
	restore_env_state(env, resume_state.get("env"), args.start_task)
	restore_rng_state(resume_state.get("rng"))
	return int(resume_state.get("step", (args.start_task - 1) * args.change_freq))


def parse_args(carry_replay_buffer=False, reset_agent=False):
	if carry_replay_buffer and reset_agent:
		raise ValueError("reset_agent cannot be combined with carry_replay_buffer")

	bootstrap_parser = argparse.ArgumentParser(add_help=False)
	bootstrap_parser.add_argument("--resume_run_dir", type=base.str2none, default=None)
	bootstrap_args, _ = bootstrap_parser.parse_known_args()

	if carry_replay_buffer:
		description = "Resume shared-agent finetuning with the previous replay buffer"
	elif reset_agent:
		description = "Run an independent reset baseline on Fetch task streams"
	else:
		description = "Run a shared-agent sequential finetuning baseline on Fetch task streams"
	parser = argparse.ArgumentParser(
		description=description
	)
	base.add_common_args(parser)
	base.add_sac_args(parser)
	base.add_student_qrl_args(parser)
	base.add_quasimetric_args(parser)

	parser.add_argument(
		"--start_task",
		type=int,
		default=1,
		help="1-based task position in the full sequence at which training starts.",
	)
	parser.add_argument(
		"--resume_run_dir",
		type=base.str2none,
		default=None,
		help="Previous finetuning run directory containing task checkpoints.",
	)
	parser.add_argument(
		"--resume_model_name",
		type=base.str2none,
		default=None,
		help=(
			"Task checkpoint prefix; defaults to the checkpoint immediately "
			"before start_task."
		),
	)
	if carry_replay_buffer:
		parser.add_argument(
			"--resume_buffer_path",
			type=base.str2none,
			default=None,
			help=(
				"Replay buffer directory paired with the resume checkpoint; "
				"inferred from its resume metadata or run directory by default."
			),
		)

	method_action = next(
		action for action in parser._actions if action.dest == "method"
	)
	method_name = "reset" if reset_agent else "finetuning"
	method_action.choices = [method_name]
	method_action.default = method_name
	parser.set_defaults(
		save_path=(
			OFFLINE_TO_ONLINE_SAVE_PATH
			if carry_replay_buffer
			else RESET_DEFAULT_SAVE_PATH if reset_agent else DEFAULT_SAVE_PATH
		),
		wandb_project_name=(
			"finetuning-offline-to-online-fetch"
			if carry_replay_buffer
			else "reset-fetch" if reset_agent else "finetuning-fetch"
		),
	)
	parser.set_defaults(**load_resume_config(bootstrap_args.resume_run_dir))
	parser.set_defaults(method=method_name)
	if carry_replay_buffer:
		parser.set_defaults(
			method="finetuning",
			save_path=OFFLINE_TO_ONLINE_SAVE_PATH,
			wandb_project_name="finetuning-offline-to-online-fetch",
		)

	args = parser.parse_args()
	try:
		args.log_backends = base.normalize_log_backends(args.log_backends)
	except ValueError as error:
		parser.error(str(error))

	if args.random_steps <= 0:
		parser.error("--random_steps must be positive")
	if args.start_task < 1:
		parser.error("--start_task must be positive")
	if args.start_task > 1 and args.resume_run_dir is None:
		parser.error("--resume_run_dir is required when --start_task is greater than 1")
	if args.start_task == 1 and args.resume_run_dir is not None:
		parser.error("--start_task must be greater than 1 when --resume_run_dir is set")
	if args.resume_run_dir is None and args.resume_model_name is not None:
		parser.error("--resume_model_name requires --resume_run_dir")
	if (
		carry_replay_buffer
		and args.resume_run_dir is None
		and args.resume_buffer_path is not None
	):
		parser.error("--resume_buffer_path requires --resume_run_dir")
	if args.resume_run_dir is not None:
		args.resume_run_dir = os.path.abspath(args.resume_run_dir)
	return args


def main(carry_replay_buffer=False, reset_agent=False):
	args = parse_args(
		carry_replay_buffer=carry_replay_buffer,
		reset_agent=reset_agent,
	)
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	os.makedirs(args.save_path, exist_ok=True)
	base.set_seed_everywhere(args.seed)

	env_kwargs = base.make_env_kwargs(args)
	env = base.FetchGoalEnvSequence(**env_kwargs)
	print("env_list:", env.env_list)
	if not 1 <= args.start_task <= len(env.env_list):
		raise ValueError(
			f"start_task must be between 1 and {len(env.env_list)}, got {args.start_task}"
		)
	if args.start_task > 1:
		set_env_task_position(env, args.start_task, args.change_freq)
		print("starting task:", args.start_task, env.base_task_name)

	num_steps_per_run = len(env.env_list) * args.change_freq
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	start_time = time.perf_counter()

	order_name = (
		args.task_order
		or env_kwargs["env_sequence"]
		or env_kwargs["base_task_name"]
		or args.env
	)
	task_tag = base.safe_tag(str(order_name).replace(",", "-"))
	if carry_replay_buffer:
		method_tag = "finetuning_offline_to_online"
	elif reset_agent:
		method_tag = "reset"
	else:
		method_tag = "finetuning"
	log_name = f"fetch_{method_tag}_{task_tag}_seed{args.seed}_gc-{args.gc_reward_type}"
	if args.slide_goal_scale != 1.0:
		log_name += f"_slide-scale{base.safe_tag(args.slide_goal_scale)}"
	if args.start_task > 1:
		log_name += f"_from-task{args.start_task}"

	run_dir = os.path.join(args.save_path, log_name)
	model_dir = os.path.join(run_dir, "model")
	os.makedirs(model_dir, exist_ok=True)
	with open(os.path.join(run_dir, "run_config.json"), "w") as handle:
		json.dump(vars(args), handle, indent=2, sort_keys=True)
	writer, log_info = base.create_experiment_logger(args, log_name, run_dir)

	if log_info["active_backends"]:
		print("log_backends:", ", ".join(log_info["active_backends"]))
	else:
		print("log_backends: disabled")
	for key in ("tensorboard_path", "wandb_path", "wandb_url"):
		if log_info[key] is not None:
			print(f"{key}:", log_info[key])
	print("run_dir:", run_dir)

	obs_space = base.vector_observation_space(env.env.observation_space)
	action_space = env.env.action_space
	action_shape = action_space.shape
	replay_buffer_capacity = int(args.change_freq) + int(args.random_steps)
	replay_buffer_cls = (
		base.ReplayBufferMetric
		if args.replay_buffer_mode == "her"
		else base.ReplayBufferMetricNoHER
	)
	replay_buffer = replay_buffer_cls(
		obs_space.shape,
		action_shape,
		replay_buffer_capacity,
		device,
	)
	agent_optim_steps = args.change_freq if reset_agent else num_steps_per_run

	def build_agent():
		return base.build_student_agent(
			observation_space=obs_space,
			action_space=action_space,
			device=device,
			args=args,
			total_optim_steps=agent_optim_steps,
		)

	collector = base.Collector(env, replay_buffer)

	if args.start_task == 1:
		agent = build_agent()
		env.reset()
		step = 0
	elif reset_agent:
		step = load_resume_training_state(args, env)
		agent = build_agent()
	else:
		agent = build_agent()
		step = load_resume_training_state(
			args,
			env,
			agent,
			replay_buffer if carry_replay_buffer else None,
		)

	base.log_scalar(writer, "config/obs_dim", obs_space.shape[0], 0)
	base.log_scalar(writer, "config/action_dim", action_shape[0], 0)
	base.log_scalar(writer, "config/task_count", len(env.env_list), 0)
	base.log_scalar(writer, "config/finetuning", int(not reset_agent), 0)
	base.log_scalar(writer, "config/reset", int(reset_agent), 0)
	base.log_scalar(
		writer,
		"config/carry_replay_buffer",
		int(carry_replay_buffer),
		0,
	)

	intermediate_stats = defaultdict(list)
	task_counter = env.task_counter
	if args.start_task == 1 or not carry_replay_buffer:
		collector.initial_collect(args.random_steps)
	else:
		start_new_collection_episode(collector, replay_buffer)

	while step <= num_steps_per_run:
		if task_counter != env.task_counter:
			completed_task_count = task_counter
			task_counter = env.task_counter
			checkpoint_name = f"{log_name}_task{completed_task_count}"
			if carry_replay_buffer:
				save_training_checkpoint(
					agent,
					replay_buffer,
					model_dir,
					run_dir,
					checkpoint_name,
					env,
					completed_task_count,
					step,
				)
			else:
				agent.save(model_dir, checkpoint_name)
				save_resume_state(
					model_dir,
					checkpoint_name,
					env,
					completed_task_count,
					step,
				)

			if step == num_steps_per_run:
				break

			if carry_replay_buffer:
				start_new_collection_episode(collector, replay_buffer)
			else:
				replay_buffer.reset()
				if reset_agent:
					agent = build_agent()
				collector.initial_collect(args.random_steps)
			base.log_scalar(writer, "task/task_idx", task_counter, step)
			print(f"{args.method} on task:", task_counter, env.base_task_name)

		train_metrics = agent.update(replay_buffer, step)
		collector.run_one_step(step, agent)
		base.log_metric_dict(writer, "train", train_metrics, step)
		base.log_scalar(writer, "train/task_idx", task_counter, step)
		base.log_scalar(writer, "train/replay_buffer_size", len(replay_buffer), step)
		base.log_scalar(writer, "train/finetuning", int(not reset_agent), step)
		base.log_scalar(writer, "train/reset", int(reset_agent), step)

		if step % args.save_freq == 0:
			elapsed = time.perf_counter() - start_time
			sps = int(step / elapsed) if elapsed > 0 else 0
			print(
				"step:",
				step,
				"time:",
				round(elapsed / 60, 3),
				"SPS:",
				sps,
				"task:",
				(task_counter, env.base_task_name),
				"method:",
				args.method,
			)

			eval_metrics = base.evaluate_and_log(
				env,
				agent,
				writer,
				"eval",
				step,
				args.num_eval_runs,
			)
			base.append_eval_stats(intermediate_stats, eval_metrics)
			intermediate_stats["steps"].append(step)
			intermediate_stats["task"].append(env.base_task_name)
			intermediate_stats["seed"].append(args.seed)
			intermediate_stats["task_idx"].append(task_counter)
			intermediate_stats["method"].append(args.method)
			intermediate_stats["time"].append(round(elapsed / 3600, 3))

			base.log_scalar(writer, "charts/SPS", sps, step)
			base.log_scalar(writer, "eval/task_idx", task_counter, step)
			print(
				f"success {eval_metrics['success_mean']:.3f} +/- {eval_metrics['success_std']:.3f}, "
				f"gc_success {eval_metrics['gc_success_mean']:.3f} +/- {eval_metrics['gc_success_std']:.3f}, "
				f"eval return {eval_metrics['return_mean']:.3f} +/- {eval_metrics['return_std']:.3f}"
			)
			writer.flush()

		if args.save_model_freq > 0 and step > 0 and step % args.save_model_freq == 0:
			checkpoint_name = f"{log_name}_step{step}"
			if carry_replay_buffer:
				save_training_checkpoint(
					agent,
					replay_buffer,
					model_dir,
					run_dir,
					checkpoint_name,
					env,
					int(env.task_counter) - 1,
					step + 1,
				)
			else:
				agent.save(model_dir, checkpoint_name)

		step += 1

	base.write_stats_csv(
		os.path.join(run_dir, f"{log_name}.csv"),
		intermediate_stats,
	)
	agent.save(model_dir, f"{log_name}_final")
	writer.close()
	env.close()


if __name__ == "__main__":
	sys.exit(main())


'''
cd Fetch/quasimetric-rl

python -m online_continual.finetuning_baseline \
  --env fetch_sequence_custom \
  --env_sequence none \
  --task_order push,slide,pick-and-place \
  --slide_goal_scale 0.795 \
  --gpu 0 \
  --seed 1 \
  --save_path online_continual/results/finetuning


python -m online_continual.finetuning_baseline \
  --env fetch_sequence_custom \
  --env_sequence none \
  --task_order pick-and-place,slide,push \
  --slide_goal_scale 0.795 \
  --gpu 0 \
  --seed 0 \
  --save_path online_continual/results/finetuning_seed0_pickslidepush

'''





