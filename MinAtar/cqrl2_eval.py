import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from CL_envs import CL_envs_func_replacement
from cqrl2 import DiscreteQuasimetricLearner, QNetwork, observation_tensor, set_seed


CANONICAL_GAMES = ("breakout", "space_invaders", "freeway")


@dataclass(frozen=True)
class EvaluationSpec:
	target_task: int | None
	goal_task: int
	game: str | None = None


def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate a trained CQRL2 quasimetric meta policy."
	)
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument(
		"--stage-eval",
		action="store_true",
		help=(
			"Evaluate every sibling taskN checkpoint on tasks seen by that stage. "
			"Requires checkpoints produced by the updated cqrl2.py."
		),
	)
	parser.add_argument("--seq", type=int, default=0)
	parser.add_argument("--episodes", type=int, default=30)
	parser.add_argument("--max-steps", type=int, default=300)
	parser.add_argument("--seed", type=int, default=1000, help="Evaluation seed")
	parser.add_argument(
		"--policy",
		choices=("meta", "q", "both"),
		default="meta",
		help="Evaluate the quasimetric meta policy, final Q policy, or both.",
	)
	parser.add_argument(
		"--mode",
		choices=("transfer", "same-task", "all-games"),
		default="transfer",
		help=(
			"transfer: goal t-1 on task t; same-task: goal t on task t; "
			"all-games: every selected goal on all three games"
		),
	)
	parser.add_argument(
		"--task-ids",
		type=int,
		nargs="+",
		default=None,
		help="Target task IDs, or goal task IDs in all-games mode.",
	)
	parser.add_argument(
		"--meta-goal-source",
		choices=("rollout-task", "checkpoint"),
		default="rollout-task",
		help=(
			"Use Q rollout trajectory next observations as per-step goals, "
			"or reuse goals saved during training."
		),
	)
	parser.add_argument(
		"--rollout-goal-episodes",
		type=int,
		default=5,
		help="Number of Q rollout goal trajectories.",
	)
	parser.add_argument(
		"--rollout-goal-max-steps",
		type=int,
		default=300,
		help="Maximum number of next-observation goals in each Q trajectory.",
	)
	parser.add_argument(
		"--rollout-goal-success-only",
		type=int,
		choices=(0, 1),
		default=1,
		help="Prefer complete Q trajectories containing a positive reward.",
	)
	parser.add_argument(
		"--goal-discount",
		type=float,
		default=0.995,
		help="Deprecated compatibility option; trajectory goals are not discounted.",
	)
	parser.add_argument(
		"--qm-components",
		type=int,
		default=8,
		help="MRN component count used during training (not stored in old checkpoints).",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Torch device, for example cpu or cuda:0 (auto-detected by default).",
	)
	parser.add_argument("--output-json", type=Path, default=None)
	return parser.parse_args()


def load_checkpoint(path, device):
	if not path.is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {path}")
	try:
		checkpoint = torch.load(path, map_location=device, weights_only=True)
	except TypeError:
		checkpoint = torch.load(path, map_location=device)
	required = {"q_network", "quasimetric"}
	missing = required.difference(checkpoint)
	if missing:
		raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}")
	return checkpoint


def discover_stage_checkpoints(path):
	stage_match = re.fullmatch(r"(.+)_task(\d+)_checkpoint\.pt", path.name)
	if stage_match:
		prefix = stage_match.group(1)
	else:
		final_match = re.fullmatch(r"(.+)_checkpoint\.pt", path.name)
		if final_match is None:
			raise ValueError(
				"checkpoint filename must end with _checkpoint.pt or "
				"_taskN_checkpoint.pt for stage evaluation."
			)
		prefix = final_match.group(1)

	checkpoint_pattern = re.compile(
		rf"{re.escape(prefix)}_task(\d+)_checkpoint\.pt"
	)
	stage_paths = {}
	for candidate in path.parent.iterdir():
		candidate_match = checkpoint_pattern.fullmatch(candidate.name)
		if candidate_match:
			stage_paths[int(candidate_match.group(1))] = candidate
	if not stage_paths:
		raise FileNotFoundError(
			f"No stage checkpoints found beside {path}. Retrain with --save-model "
			"using the updated cqrl2.py to create _taskN_checkpoint.pt files."
		)

	expected_stages = set(range(max(stage_paths) + 1))
	missing_stages = sorted(expected_stages.difference(stage_paths))
	if missing_stages:
		raise FileNotFoundError(
			f"Stage checkpoint series is incomplete; missing tasks {missing_stages}."
		)
	return dict(sorted(stage_paths.items()))


def checkpoint_dimensions(checkpoint):
	quasimetric_state = checkpoint["quasimetric"]
	in_channels = quasimetric_state["state_encoder.conv.weight"].shape[1]
	latent_dim = quasimetric_state["state_encoder.output.weight"].shape[0]
	hidden_dim = quasimetric_state["transition_encoder.0.weight"].shape[0]
	num_actions = checkpoint["q_network"]["output.weight"].shape[0]
	transition_input = quasimetric_state["transition_encoder.0.weight"].shape[1]
	if transition_input == in_channels * 10 * 10 + num_actions:
		transition_input_mode = "state"
	elif transition_input == latent_dim + num_actions:
		transition_input_mode = "latent"
	else:
		raise ValueError(
			"Checkpoint dimensions are inconsistent with either T(s,a) or T(z,a)."
		)
	return in_channels, num_actions, latent_dim, hidden_dim, transition_input_mode


def build_q_network(checkpoint, device):
	in_channels, num_actions, _, _, _ = checkpoint_dimensions(checkpoint)
	q_network = QNetwork(in_channels, num_actions).to(device)
	q_network.load_state_dict(checkpoint["q_network"])
	q_network.eval()
	return q_network


def build_models(checkpoint, components, device, policy, meta_goal_source):
	(
		in_channels,
		num_actions,
		latent_dim,
		hidden_dim,
		transition_input_mode,
	) = checkpoint_dimensions(
		checkpoint
	)
	if components <= 0 or latent_dim % components != 0:
		raise ValueError(
			f"qm-components must divide latent dimension {latent_dim}; got {components}."
		)

	quasimetric = None
	if policy in ("meta", "both"):
		model_args = SimpleNamespace(
			qm_latent_dim=latent_dim,
			qm_components=components,
			qm_hidden_dim=hidden_dim,
			qm_transition_input=transition_input_mode,
			qm_lr=1e-4,
			qm_discount=0.995,
			qm_lambda=0.95,
			qm_next_state_sample=0.2,
			qm_backup_clip=5.0,
			qm_action_invariance_coef=1.0,
			qm_transition_consistency_coef=1.0,
			qm_contrastive_coef=0.05,
			qm_behavior_cloning_coef=1.0,
			qm_nce_mode="forward_nce",
			qm_target_tau=0.01,
			qm_current_batch_ratio=0.5,
		)
		quasimetric = DiscreteQuasimetricLearner(
			in_channels, num_actions, model_args, device
		)
		quasimetric.load_state_dict(checkpoint["quasimetric"])
		quasimetric.eval()

	q_network = None
	if policy in ("q", "both") or meta_goal_source == "rollout-task":
		q_network = build_q_network(checkpoint, device)
	return quasimetric, q_network, in_channels, num_actions


def checkpoint_goals(checkpoint):
	goals = checkpoint.get("behavior_goals") or {}
	goals = {int(task_id): goal for task_id, goal in goals.items()}
	if not goals and checkpoint.get("behavior_goal") is not None:
		goals[0] = checkpoint["behavior_goal"]
	if not goals:
		raise ValueError("Checkpoint contains no behavior goal for meta evaluation.")
	return goals


def evaluation_specs(mode, task_ids, goal_ids):
	available_goals = sorted(goal_ids)
	if mode == "transfer":
		target_tasks = task_ids or [task_id + 1 for task_id in available_goals[:-1]]
		specs = [EvaluationSpec(task_id, task_id - 1) for task_id in target_tasks]
	elif mode == "same-task":
		target_tasks = task_ids or available_goals
		specs = [EvaluationSpec(task_id, task_id) for task_id in target_tasks]
	else:
		selected_goals = task_ids or available_goals
		specs = [
			EvaluationSpec(None, goal_task, game)
			for goal_task in selected_goals
			for game in CANONICAL_GAMES
		]

	for spec in specs:
		if spec.goal_task not in goal_ids:
			raise ValueError(
				f"Goal for task {spec.goal_task} is unavailable; "
				f"available goals: {available_goals}."
			)
		if spec.target_task is not None and not 0 <= spec.target_task < 7:
			raise ValueError("Target task IDs must be between 0 and 6.")
	return specs


def make_environment(spec, sequence, seed):
	if spec.game is None:
		return CL_envs_func_replacement(sequence, spec.target_task, seed)
	game_id = CANONICAL_GAMES.index(spec.game)
	return CL_envs_func_replacement(
		sequence, game_id, seed, evaluation=True
	)


@torch.inference_mode()
def select_action(policy, observation, goal, quasimetric, q_network, device):
	observation = observation_tensor(observation, device).unsqueeze(0)
	if policy == "meta":
		distances = quasimetric.action_distances(
			observation, goal.unsqueeze(0)
		)
		return int(distances.argmin(dim=1).item())
	q_values = q_network(observation)
	return int(q_values.argmax(dim=1).item())


def generate_rollout_goal_trajectories(
	task_id,
	sequence,
	q_network,
	device,
	episodes,
	max_steps,
	seed,
	success_only,
	rollout_policy="final-student",
):
	spec = EvaluationSpec(target_task=task_id, goal_task=task_id)
	environment = make_environment(spec, sequence, seed)
	trajectories = []
	returns = []
	try:
		for episode in range(episodes):
			observation = environment.reset(seed=seed + 10000 + task_id * 1000 + episode)
			trajectory = []
			episode_return = 0.0
			for _ in range(max_steps):
				action = select_action(
					"q", observation, None, None, q_network, device
				)
				next_observation, reward, done, _ = environment.step(action)
				trajectory.append(
					(np.array(next_observation, dtype=np.float32, copy=True), float(reward))
				)
				episode_return += reward
				observation = next_observation
				if done:
					break
			trajectories.append(trajectory)
			returns.append(float(episode_return))
	finally:
		environment.close()

	if not all(trajectories):
		raise ValueError("Q rollout produced an empty goal trajectory.")
	successful_trajectories = [
		trajectory
		for trajectory in trajectories
		if any(reward > 0.0 for _, reward in trajectory)
	]
	selected_trajectories = trajectories
	selection = "all-q-trajectories"
	if success_only and successful_trajectories:
		selected_trajectories = successful_trajectories
		selection = "positive-reward-q-trajectories"
	goal_trajectories = [
		torch.stack(
			[
			observation_tensor(goal_observation, torch.device("cpu"))
				for goal_observation, _ in trajectory
			]
		)
		for trajectory in selected_trajectories
	]
	all_goals = torch.cat(goal_trajectories, dim=0)
	goal_norms = torch.linalg.vector_norm(
		all_goals.flatten(start_dim=1), dim=1
	).tolist()
	unique_goal_count = int(
		torch.unique(all_goals.flatten(start_dim=1), dim=0).shape[0]
	)
	trajectory_lengths = [len(trajectory) for trajectory in goal_trajectories]
	transition_count = sum(len(trajectory) for trajectory in trajectories)
	info = {
		"source": "rollout-task",
		"selection": selection,
		"goal_scope": "step",
		"goal_count": int(all_goals.shape[0]),
		"unique_goal_count": unique_goal_count,
		"goal_trajectory_count": len(goal_trajectories),
		"goal_trajectory_lengths": trajectory_lengths,
		"goal_trajectory_assignment": "round-robin-per-evaluation-episode",
		"goal_after_trajectory": "hold-final-next-observation",
		"rollout_policy": rollout_policy,
		"rollout_game": environment.game_name,
		"rollout_episodes": episodes,
		"rollout_transitions": transition_count,
		"rollout_successful_trajectories": len(successful_trajectories),
		"rollout_mean_return": float(np.mean(returns)),
		"goal_norm": float(np.mean(goal_norms)),
		"goal_norm_std": float(np.std(goal_norms)),
	}
	print(
		f"rollout goals: task={task_id} game={environment.game_name} "
		f"selection={selection} trajectories={len(goal_trajectories)} "
		f"lengths={trajectory_lengths} goals={info['goal_count']} "
		f"unique={unique_goal_count} return={info['rollout_mean_return']:.3f} "
		f"norm={info['goal_norm']:.3f}+/-{info['goal_norm_std']:.3f}"
	)
	return goal_trajectories, info


def resolve_meta_goals(
	checkpoint_goals, specs, args, q_network, device, task_q_networks=None
):
	goal_task_ids = sorted({spec.goal_task for spec in specs})
	if args.meta_goal_source == "checkpoint":
		goals = {}
		infos = {}
		for task_id in goal_task_ids:
			goal = checkpoint_goals[task_id]
			goal_norm = float(torch.linalg.vector_norm(goal.float()).item())
			goals[task_id] = [goal.unsqueeze(0)]
			infos[task_id] = {
				"source": "checkpoint",
				"selection": "saved-behavior-goal",
				"goal_scope": "task",
				"goal_count": 1,
				"unique_goal_count": 1,
				"goal_trajectory_count": 1,
				"goal_trajectory_lengths": [1],
				"goal_trajectory_assignment": "fixed",
				"goal_after_trajectory": "hold-final-next-observation",
				"goal_norm": goal_norm,
				"goal_norm_std": 0.0,
			}
		return goals, infos

	if task_q_networks is None:
		print(
			"rollout-task goals use the final q_network from the supplied "
			"checkpoint for every task."
		)
	else:
		print("rollout-task goals use each task's stage student checkpoint.")
	goals = {}
	infos = {}
	for task_id in goal_task_ids:
		goal_q_network = q_network
		rollout_policy = "final-student"
		if task_q_networks is not None:
			if task_id not in task_q_networks:
				raise FileNotFoundError(
					f"Missing task {task_id} student checkpoint for rollout goal."
				)
			goal_q_network = task_q_networks[task_id]
			rollout_policy = f"task{task_id}-stage-student"
		goals[task_id], infos[task_id] = generate_rollout_goal_trajectories(
			task_id=task_id,
			sequence=args.seq,
			q_network=goal_q_network,
			device=device,
			episodes=args.rollout_goal_episodes,
			max_steps=args.rollout_goal_max_steps,
			seed=args.seed,
			success_only=bool(args.rollout_goal_success_only),
			rollout_policy=rollout_policy,
		)
	return goals, infos


def evaluate(
	spec,
	policy,
	goals,
	quasimetric,
	q_network,
	device,
	sequence,
	episodes,
	max_steps,
	seed,
	in_channels,
	num_actions,
):
	environment = make_environment(spec, sequence, seed)
	if environment.observation_space.shape[2] != in_channels:
		raise ValueError("Environment and checkpoint observation dimensions differ.")
	if environment.action_space.n != num_actions:
		raise ValueError("Environment and checkpoint action dimensions differ.")
	if policy == "meta" and (
		goals is None
		or not goals
		or any(goal_trajectory.shape[0] == 0 for goal_trajectory in goals)
	):
		raise ValueError("Meta evaluation requires non-empty goal trajectories.")

	returns = []
	lengths = []
	truncated_episodes = 0
	try:
		for episode in range(episodes):
			observation = environment.reset(seed=seed + episode)
			goal_trajectory = (
				goals[episode % len(goals)] if policy == "meta" else None
			)
			episode_return = 0.0
			done = False
			for episode_step in range(1, max_steps + 1):
				goal = None
				if policy == "meta":
					goal_index = min(episode_step - 1, goal_trajectory.shape[0] - 1)
					goal = goal_trajectory[goal_index]
				action = select_action(
					policy,
					observation,
					goal,
					quasimetric,
					q_network,
					device,
				)
				observation, reward, done, _ = environment.step(action)
				episode_return += reward
				if done:
					break
			if not done:
				truncated_episodes += 1
			returns.append(float(episode_return))
			lengths.append(episode_step)
	finally:
		environment.close()

	return {
		"policy": policy,
		"target_task": spec.target_task,
		"goal_task": spec.goal_task if policy == "meta" else None,
		"game": environment.game_name,
		"episodes": episodes,
		"mean_return": float(np.mean(returns)),
		"std_return": float(np.std(returns)),
		"stderr_return": float(np.std(returns) / math.sqrt(episodes)),
		"median_return": float(np.median(returns)),
		"mean_length": float(np.mean(lengths)),
		"truncated_episodes": truncated_episodes,
		"returns": returns,
		"lengths": lengths,
	}


def print_results(results):
	print(
		f"{'stage':>5} {'policy':<7} {'target':>6} {'goal':>5} "
		f"{'source':<13} {'game':<16} "
		f"{'return (mean +/- std)':>23} {'length':>8} {'trunc':>6}"
	)
	for result in results:
		stage = "-" if result["checkpoint_stage"] is None else result["checkpoint_stage"]
		target = "-" if result["target_task"] is None else result["target_task"]
		goal = "-" if result["goal_task"] is None else result["goal_task"]
		source = result.get("goal_source", "-")
		print(
			f"{str(stage):>5} {result['policy']:<7} {str(target):>6} "
			f"{str(goal):>5} "
			f"{source:<13} "
			f"{result['game']:<16} "
			f"{result['mean_return']:>9.3f} +/- {result['std_return']:<7.3f} "
			f"{result['mean_length']:>8.1f} "
			f"{result['truncated_episodes']:>6}"
		)


def summarize_stages(results):
	groups = {}
	for result in results:
		if result["checkpoint_stage"] is None:
			continue
		key = (result["checkpoint_stage"], result["policy"])
		groups.setdefault(key, []).append(result)

	summaries = []
	for (stage, policy), stage_results in sorted(groups.items()):
		returns = np.asarray(
			[result["mean_return"] for result in stage_results], dtype=np.float32
		)
		games = sorted({result["game"] for result in stage_results})
		summaries.append(
			{
				"checkpoint_stage": stage,
				"policy": policy,
				"evaluated_tasks": len(stage_results),
				"average_task_return": float(np.mean(returns)),
				"per_game_mean_return": {
					game: float(
						np.mean(
							[
								result["mean_return"]
								for result in stage_results
								if result["game"] == game
							]
						)
					)
					for game in games
				},
			}
		)
	return summaries


def print_stage_summaries(summaries):
	if not summaries:
		return
	print("\nstage average performance (un-normalized across seen tasks)")
	print(f"{'stage':>5} {'policy':<7} {'tasks':>5} {'average return':>14}")
	for summary in summaries:
		print(
			f"{summary['checkpoint_stage']:>5} {summary['policy']:<7} "
			f"{summary['evaluated_tasks']:>5} "
			f"{summary['average_task_return']:>14.3f}"
		)


def stage_task_ids(task_ids, stage):
	if task_ids is None:
		return None
	return [task_id for task_id in task_ids if task_id <= stage]


def validate_stage_payload(checkpoint, stage, path):
	metadata = checkpoint.get("metadata") or {}
	saved_stage = metadata.get("task_id")
	if saved_stage is not None and int(saved_stage) != stage:
		raise ValueError(
			f"Checkpoint filename says task {stage}, but metadata says task "
			f"{saved_stage}: {path}"
		)


def main():
	args = parse_args()
	if args.episodes <= 0 or args.max_steps <= 0:
		raise ValueError("episodes and max-steps must be positive.")
	if args.rollout_goal_episodes <= 0 or args.rollout_goal_max_steps <= 0:
		raise ValueError("rollout goal episodes and max steps must be positive.")
	if not 0.0 < args.goal_discount <= 1.0:
		raise ValueError("goal-discount must be in (0, 1].")
	set_seed(args.seed)
	device = torch.device(
		args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
	)
	if args.stage_eval:
		checkpoint_paths = discover_stage_checkpoints(args.checkpoint)
		checkpoints = {
			stage: load_checkpoint(path, torch.device("cpu"))
			for stage, path in checkpoint_paths.items()
		}
		for stage, checkpoint in checkpoints.items():
			validate_stage_payload(checkpoint, stage, checkpoint_paths[stage])
	else:
		checkpoint_paths = {None: args.checkpoint}
		checkpoints = {None: load_checkpoint(args.checkpoint, device)}

	specs_by_stage = {}
	for stage, checkpoint in checkpoints.items():
		saved_goals = checkpoint_goals(checkpoint)
		requested_tasks = (
			stage_task_ids(args.task_ids, stage)
			if stage is not None
			else args.task_ids
		)
		if requested_tasks == []:
			continue
		specs = evaluation_specs(args.mode, requested_tasks, saved_goals)
		if specs:
			specs_by_stage[stage] = specs
	if not specs_by_stage:
		raise ValueError(
			"No evaluation tasks remain for the selected stages, mode, and task IDs."
		)

	policies = ("meta", "q") if args.policy == "both" else (args.policy,)
	shared_meta_goals = {}
	shared_goal_infos = {}
	if args.stage_eval and "meta" in policies and args.meta_goal_source == "rollout-task":
		all_specs = [spec for specs in specs_by_stage.values() for spec in specs]
		goal_task_ids = sorted({spec.goal_task for spec in all_specs})
		task_q_networks = {
			task_id: build_q_network(checkpoints[task_id], device)
			for task_id in goal_task_ids
		}
		latest_stage = max(specs_by_stage)
		shared_meta_goals, shared_goal_infos = resolve_meta_goals(
			checkpoint_goals(checkpoints[latest_stage]),
			all_specs,
			args,
			None,
			device,
			task_q_networks=task_q_networks,
		)
		del task_q_networks

	results = []
	for stage, specs in specs_by_stage.items():
		checkpoint = checkpoints[stage]
		quasimetric, q_network, in_channels, num_actions = build_models(
			checkpoint,
			args.qm_components,
			device,
			args.policy,
			"checkpoint" if args.stage_eval else args.meta_goal_source,
		)
		meta_goals = shared_meta_goals
		goal_infos = shared_goal_infos
		if "meta" in policies and not meta_goals:
			meta_goals, goal_infos = resolve_meta_goals(
				checkpoint_goals(checkpoint), specs, args, q_network, device
			)
		for spec in specs:
			for policy in policies:
				goals = None
				if policy == "meta":
					goals = [
						goal_trajectory.to(device=device, dtype=torch.float32)
						for goal_trajectory in meta_goals[spec.goal_task]
					]
				result = evaluate(
					spec,
					policy,
					goals,
					quasimetric,
					q_network,
					device,
					args.seq,
					args.episodes,
					args.max_steps,
					args.seed,
					in_channels,
					num_actions,
				)
				result["checkpoint_stage"] = stage
				result["checkpoint"] = str(checkpoint_paths[stage])
				if policy == "meta":
					goal_info = goal_infos[spec.goal_task]
					result.update(
						{
							"goal_source": goal_info["source"],
							"goal_selection": goal_info["selection"],
							"goal_scope": goal_info["goal_scope"],
							"goal_count": goal_info["goal_count"],
							"unique_goal_count": goal_info["unique_goal_count"],
							"goal_trajectory_count": goal_info[
								"goal_trajectory_count"
							],
							"goal_trajectory_lengths": goal_info[
								"goal_trajectory_lengths"
							],
							"goal_trajectory_assignment": goal_info[
								"goal_trajectory_assignment"
							],
							"goal_after_trajectory": goal_info[
								"goal_after_trajectory"
							],
							"goal_norm": goal_info["goal_norm"],
							"goal_norm_std": goal_info["goal_norm_std"],
							**{
								key: value
								for key, value in goal_info.items()
								if key.startswith("rollout_")
							},
						}
					)
				results.append(result)

	print(
		f"checkpoint: {args.checkpoint}"
		+ (f"; stages: {list(specs_by_stage)}" if args.stage_eval else "")
	)
	print(
		f"mode: {args.mode}; goal source: {args.meta_goal_source}; "
		f"device: {device}; episodes: {args.episodes}"
	)
	print_results(results)
	stage_summaries = summarize_stages(results)
	print_stage_summaries(stage_summaries)
	if args.output_json is not None:
		args.output_json.parent.mkdir(parents=True, exist_ok=True)
		with args.output_json.open("w", encoding="utf-8") as output_file:
			json.dump(
				{
					"checkpoint": str(args.checkpoint),
					"stage_eval": args.stage_eval,
					"mode": args.mode,
					"meta_goal_source": args.meta_goal_source,
					"sequence": args.seq,
					"seed": args.seed,
					"results": results,
					"stage_summaries": stage_summaries,
				},
				output_file,
				indent=2,
			)
		print(f"saved: {args.output_json}")


if __name__ == "__main__":
	main()
'''
conda run -n RLL3 python MinAtar/cqrl2_eval.py \
  --checkpoint MinAtar/results/CQRL_wo_trans_steps_3500000_switch_500000_seq_0_seed_0_checkpoint.pt \
	--stage-eval \
  --seq 0 \
	--policy both \
	--mode same-task \
	--meta-goal-source rollout-task \
	--rollout-goal-episodes 5 \
  --episodes 100 \
  --max-steps 300 \
  --device cuda:0 \
  --output-json MinAtar/results/cqrl2_seq0_eval.json


python MinAtar/cqrl2_eval.py \
  --checkpoint MinAtar/results/CQRL_wo_trans_steps_3500000_switch_500000_seq_0_seed_0_task0_checkpoint.pt \
  --stage-eval \
  --seq 0 \
  --policy both \
  --mode same-task \
  --meta-goal-source rollout-task \
  --rollout-goal-episodes 5 \
  --episodes 100 \
  --max-steps 300 \
  --device cuda:0 \
  --output-json MinAtar/results/cqrl2_seq2_seed1_stage_eval.json

  python MinAtar/cqrl2_eval.py \
  --checkpoint MinAtar/results/cqrl_newstateinput/CQRL_wo_trans_steps_3500000_switch_500000_seq_0_seed_0_task0_checkpoint.pt \
  --stage-eval \
  --seq 0 \
  --policy q \
  --mode same-task \
  --meta-goal-source rollout-task \
  --rollout-goal-episodes 50 \
  --episodes 50 \
  --max-steps 300 \
  --device cuda:0 \
  --output-json MinAtar/results/cqrl_newstateinput/cqrl2_seq0_see0_stage_eval.json


  mode same-task   transfer  all-games
'''

