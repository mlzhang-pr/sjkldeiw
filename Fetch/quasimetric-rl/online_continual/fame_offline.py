"""Train FAME's meta policy offline from cumulative CQRL replay snapshots."""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

try:
	from . import fame
except ImportError:
	import fame


DEFAULT_BUFFER_ROOT = os.path.join(
	fame.ONLINE_CONTINUAL_DIR,
	"results",
	"cqrl_wd",
	"quasimetric_buffers",
	"fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795",
)
DEFAULT_SAVE_PATH = os.path.join(fame.ONLINE_CONTINUAL_DIR, "results", "fame_offline")
TASK_DIRECTORY_PATTERN = re.compile(r"^task_(\d+)_(.+)$")
SLIDE_SCALE_PATTERN = re.compile(r"(?:^|_)slide-scale([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
BUFFER_FIELDS = (
	"obses",
	"next_obses",
	"actions",
	"rewards",
	"successes",
	"not_dones",
	"not_dones_no_max",
	"task_ids",
)


@dataclass(frozen=True)
class TaskSnapshot:
	index: int
	task_name: str
	directory: str
	archive_path: str
	size: int
	obs_shape: tuple
	action_shape: tuple


def inspect_snapshot(directory, index, task_name):
	archive_path = os.path.join(directory, "offline_data.npz")
	if not os.path.isfile(archive_path):
		raise FileNotFoundError(f"Missing replay archive: {archive_path}")

	with np.load(archive_path) as data:
		missing = sorted(set(BUFFER_FIELDS + ("others",)) - set(data.files))
		if missing:
			raise ValueError(f"Replay archive {archive_path} is missing fields: {missing}")

		size = len(data["obses"])
		for field in BUFFER_FIELDS:
			if len(data[field]) != size:
				raise ValueError(
					f"Replay archive {archive_path} has {len(data[field])} {field} rows, expected {size}."
				)

		metadata = np.asarray(data["others"]).reshape(-1)
		if metadata.size < 4:
			raise ValueError(f"Replay archive {archive_path} has invalid others metadata: {metadata}")
		if bool(metadata[3]):
			raise ValueError(
				f"Replay archive {archive_path} is a wrapped full buffer; cumulative suffix extraction is ambiguous."
			)
		if int(metadata[1]) != size:
			raise ValueError(
				f"Replay archive {archive_path} reports idx={int(metadata[1])}, but stores {size} rows."
			)

		return TaskSnapshot(
			index=index,
			task_name=task_name,
			directory=directory,
			archive_path=archive_path,
			size=size,
			obs_shape=tuple(data["obses"].shape[1:]),
			action_shape=tuple(data["actions"].shape[1:]),
		)


def discover_task_snapshots(buffer_root):
	buffer_root = os.path.abspath(buffer_root)
	if not os.path.isdir(buffer_root):
		raise FileNotFoundError(f"Offline buffer root does not exist: {buffer_root}")

	snapshots = []
	for entry in os.scandir(buffer_root):
		match = TASK_DIRECTORY_PATTERN.match(entry.name)
		if not entry.is_dir() or match is None:
			continue
		snapshots.append(
			inspect_snapshot(
				entry.path,
				index=int(match.group(1)),
				task_name=match.group(2),
			)
		)

	snapshots.sort(key=lambda snapshot: snapshot.index)
	if not snapshots:
		raise FileNotFoundError(f"No task_XX_<name>/offline_data.npz snapshots found under {buffer_root}")

	expected_indices = list(range(1, len(snapshots) + 1))
	actual_indices = [snapshot.index for snapshot in snapshots]
	if actual_indices != expected_indices:
		raise ValueError(f"Task snapshot indices must be contiguous from 1; found {actual_indices}.")

	reference_obs_shape = snapshots[0].obs_shape
	reference_action_shape = snapshots[0].action_shape
	previous_size = 0
	for snapshot in snapshots:
		if snapshot.obs_shape != reference_obs_shape or snapshot.action_shape != reference_action_shape:
			raise ValueError(
				f"Task {snapshot.index} has shapes obs={snapshot.obs_shape}, action={snapshot.action_shape}; "
				f"expected obs={reference_obs_shape}, action={reference_action_shape}."
			)
		if snapshot.size <= previous_size:
			raise ValueError(
				f"Cumulative snapshot sizes must increase; task {snapshot.index} has {snapshot.size} rows "
				f"after {previous_size}."
			)
		previous_size = snapshot.size
	return snapshots


def validate_cumulative_prefixes(snapshots):
	for previous, current in zip(snapshots, snapshots[1:]):
		with np.load(previous.archive_path) as previous_data, np.load(current.archive_path) as current_data:
			for field in BUFFER_FIELDS:
				if not np.array_equal(previous_data[field], current_data[field][: previous.size]):
					raise ValueError(
						f"{current.archive_path} is not cumulative: its {field} prefix differs from task "
						f"{previous.index}."
					)
		print(f"validated cumulative prefix: task {previous.index} -> task {current.index}")


def infer_slide_goal_scale(buffer_root):
	match = SLIDE_SCALE_PATTERN.search(os.path.basename(os.path.normpath(buffer_root)))
	return float(match.group(1)) if match is not None else 1.0


def configure_data_dependent_args(args, snapshots):
	snapshot_order = [snapshot.task_name for snapshot in snapshots]
	if args.task_order is None:
		args.task_order = ",".join(snapshot_order)
	else:
		requested_order = [fame.normalize_key(task) for task in args.task_order.split(",")]
		normalized_snapshot_order = [fame.normalize_key(task) for task in snapshot_order]
		if requested_order != normalized_snapshot_order:
			raise ValueError(
				f"--task_order {requested_order} does not match snapshot order {normalized_snapshot_order}."
			)

	if args.slide_goal_scale is None:
		args.slide_goal_scale = infer_slide_goal_scale(args.buffer_root)
	args.buffer_root = os.path.abspath(args.buffer_root)


def trim_loaded_buffer_to_suffix(replay_buffer, start_index, expected_size):
	loaded_size = len(replay_buffer)
	if loaded_size != expected_size:
		raise ValueError(f"Loaded {loaded_size} replay rows, expected {expected_size}.")
	if not 0 <= start_index < loaded_size:
		raise ValueError(f"Invalid cumulative suffix start {start_index} for replay size {loaded_size}.")
	if start_index > 0 and replay_buffer.not_dones[start_index - 1, 0] >= 0.5:
		raise ValueError(f"Task boundary at replay row {start_index} splits an unfinished episode.")

	suffix_size = loaded_size - start_index
	for field in BUFFER_FIELDS[:-1]:
		values = getattr(replay_buffer, field)
		values[:suffix_size] = values[start_index:loaded_size].copy()
	replay_buffer.task_ids[:suffix_size] = replay_buffer.task_ids[start_index:loaded_size].copy()
	replay_buffer.idx = suffix_size
	replay_buffer.full = False
	replay_buffer.last_save = 0
	replay_buffer._rebuild_episode_ends()
	return suffix_size


def load_task_delta(replay_buffer, snapshot, previous_snapshot_size):
	if not replay_buffer.load_data(snapshot.directory):
		raise RuntimeError(f"Unable to load replay snapshot: {snapshot.directory}")
	return trim_loaded_buffer_to_suffix(replay_buffer, previous_snapshot_size, snapshot.size)


def count_successful_episodes(replay_buffer):
	successful_episodes = 0
	episode_succeeded = False
	for index in fame.chronological_indices(replay_buffer):
		episode_succeeded = episode_succeeded or bool(replay_buffer.successes[index, 0] > 0.5)
		if replay_buffer.not_dones[index, 0] < 0.5:
			successful_episodes += int(episode_succeeded)
			episode_succeeded = False
	return successful_episodes


def select_fast_transitions(args, task_buffer, fast_buffer):
	fast_buffer.reset()
	if args.use_all_transitions:
		copied_transitions = fame.copy_replay_buffer(task_buffer, fast_buffer)
		return copied_transitions, count_successful_episodes(task_buffer)
	return fame.copy_recent_trajectories(task_buffer, fast_buffer, args.store_traj_num)


def evaluate_seen_tasks(
	args,
	eval_env,
	meta_agent,
	snapshots,
	stage_idx,
	update_step,
	writer,
	eval_history,
	eval_csv_path,
):
	python_rng_state = fame.random.getstate()
	numpy_rng_state = np.random.get_state()
	torch_rng_state = torch.get_rng_state()
	cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
	was_training = meta_agent.training
	meta_agent.eval()
	task_metrics = []
	try:
		for snapshot in snapshots[:stage_idx]:
			eval_env.set_task(snapshot.task_name)
			prefix = f"offline_eval/task_{snapshot.index:02d}_{fame.safe_tag(snapshot.task_name)}"
			metrics = fame.evaluate_and_log(
				eval_env,
				meta_agent,
				writer,
				prefix,
				update_step,
				args.num_eval_runs,
			)
			task_metrics.append(metrics)
			for key, value in (
				("update_step", update_step),
				("stage_idx", stage_idx),
				("task_idx", snapshot.index),
				("task", snapshot.task_name),
				("mean_return", metrics["return_mean"]),
				("mean_success", metrics["success_mean"]),
				("success_std", metrics["success_std"]),
				("gc_success", metrics["gc_success_mean"]),
				("gc_success_std", metrics["gc_success_std"]),
			):
				eval_history[key].append(value)
	finally:
		meta_agent.train(was_training)
		fame.random.setstate(python_rng_state)
		np.random.set_state(numpy_rng_state)
		torch.set_rng_state(torch_rng_state)
		if cuda_rng_state is not None:
			torch.cuda.set_rng_state_all(cuda_rng_state)

	mean_success = float(np.mean([metrics["success_mean"] for metrics in task_metrics]))
	mean_gc_success = float(np.mean([metrics["gc_success_mean"] for metrics in task_metrics]))
	mean_return = float(np.mean([metrics["return_mean"] for metrics in task_metrics]))
	fame.log_scalar(writer, "offline_eval/seen_mean_success", mean_success, update_step)
	fame.log_scalar(writer, "offline_eval/seen_mean_gc_success", mean_gc_success, update_step)
	fame.log_scalar(writer, "offline_eval/seen_mean_return", mean_return, update_step)
	fame.log_scalar(writer, "offline_eval/stage_idx", stage_idx, update_step)
	fame.write_stats_csv(eval_csv_path, eval_history)
	writer.flush()
	print(
		f"offline eval step={update_step} stage={stage_idx}: "
		f"seen_success={mean_success:.3f} gc_success={mean_gc_success:.3f} "
		f"return={mean_return:.3f}"
	)


def train_meta_stage(
	args,
	meta_agent,
	meta_buffer,
	fast_buffer,
	snapshot,
	eval_env,
	snapshots,
	writer,
	integration_step,
	eval_history,
	eval_csv_path,
):
	if snapshot.index <= 1:
		print("FAME integration: first task only populates meta memory.")
		return {}, integration_step
	if len(meta_buffer) == 0 or len(fast_buffer) == 0 or args.meta_update_steps == 0:
		return {}, integration_step

	old_task_count = snapshot.index - 1
	last_metrics = {}
	print(
		"FAME integration:",
		f"old_memory={len(meta_buffer)}",
		f"new_memory={len(fast_buffer)}",
		f"updates={args.meta_update_steps}",
	)
	for update_idx in range(args.meta_update_steps):
		replay_buffers = [meta_buffer]
		if update_idx % old_task_count == 0:
			replay_buffers.append(fast_buffer)
		last_metrics = meta_agent.behavior_clone_step(replay_buffers)
		completed_updates = update_idx + 1
		update_step = integration_step + completed_updates
		should_evaluate = args.eval_interval > 0 and (
			completed_updates % args.eval_interval == 0
			or completed_updates == args.meta_update_steps
		)
		if should_evaluate:
			fame.log_metric_dict(writer, "meta_integration", last_metrics, update_step)
			evaluate_seen_tasks(
				args,
				eval_env,
				meta_agent,
				snapshots,
				snapshot.index,
				update_step,
				writer,
				eval_history,
				eval_csv_path,
			)
	return last_metrics, integration_step + args.meta_update_steps


def build_parser():
	parser = argparse.ArgumentParser(
		description="Train FAME's meta policy offline from cumulative CQRL quasimetric buffers"
	)
	fame.add_common_args(parser)
	fame.add_fame_args(parser)
	fame.add_student_qrl_args(parser)
	parser.add_argument(
		"--buffer_root",
		type=str,
		default=DEFAULT_BUFFER_ROOT,
		help="Directory containing cumulative task_XX_<name>/offline_data.npz snapshots.",
	)
	parser.add_argument(
		"--use_all_transitions",
		action="store_true",
		help="Use every new task transition instead of FAME's most recent --store_traj_num trajectories.",
	)
	parser.add_argument(
		"--skip_prefix_validation",
		action="store_true",
		help="Skip the exact cumulative-prefix comparison between adjacent snapshots.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Validate and load all task deltas without constructing or training the meta agent.",
	)
	parser.add_argument(
		"--eval_interval",
		type=int,
		default=1000,
		help="Evaluate success on all seen tasks every N meta updates; use 0 to disable.",
	)
	parser.set_defaults(
		env="fetch_sequence_custom",
		env_sequence=None,
		task_order=None,
		slide_goal_scale=None,
		save_path=DEFAULT_SAVE_PATH,
		log_backends=["none"],
		wandb_project_name="fame-offline-fetch",
	)
	return parser


def parse_args(argv=None):
	parser = build_parser()
	args = parser.parse_args(argv)
	try:
		args.log_backends = fame.normalize_log_backends(args.log_backends)
	except ValueError as error:
		parser.error(str(error))
	if args.batch_size <= 0:
		parser.error("--batch_size must be positive")
	if args.store_traj_num <= 0:
		parser.error("--store_traj_num must be positive")
	if args.meta_update_steps < 0:
		parser.error("--meta_update_steps cannot be negative")
	if args.meta_buffer_capacity <= 0:
		parser.error("--meta_buffer_capacity must be positive")
	if args.eval_interval < 0:
		parser.error("--eval_interval cannot be negative")
	if args.num_eval_runs <= 0:
		parser.error("--num_eval_runs must be positive")
	return args


def make_run_name(args, snapshots):
	task_tag = fame.safe_tag("-".join(snapshot.task_name for snapshot in snapshots))
	log_name = f"fetch_fame_offline_{task_tag}_seed{args.seed}_gc-{args.gc_reward_type}"
	if args.slide_goal_scale != 1.0:
		log_name += f"_slide-scale{fame.safe_tag(args.slide_goal_scale)}"
	return log_name


def validate_environment_shapes(env, snapshots):
	expected_tasks = [fame.normalize_key(snapshot.task_name) for snapshot in snapshots]
	actual_tasks = [fame.normalize_key(task_name) for task_name in env.env_list]
	if actual_tasks != expected_tasks:
		raise ValueError(f"Environment tasks {actual_tasks} do not match replay tasks {expected_tasks}.")

	observation_space = fame.vector_observation_space(env.env.observation_space)
	action_space = env.env.action_space
	if tuple(observation_space.shape) != snapshots[0].obs_shape:
		raise ValueError(
			f"Environment observation shape {observation_space.shape} does not match replay shape "
			f"{snapshots[0].obs_shape}."
		)
	if tuple(action_space.shape) != snapshots[0].action_shape:
		raise ValueError(
			f"Environment action shape {action_space.shape} does not match replay shape {snapshots[0].action_shape}."
		)
	return observation_space, action_space


def main(argv=None):
	args = parse_args(argv)
	snapshots = discover_task_snapshots(args.buffer_root)
	configure_data_dependent_args(args, snapshots)
	if not args.skip_prefix_validation:
		validate_cumulative_prefixes(snapshots)

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	fame.set_seed_everywhere(args.seed)
	env = fame.FetchGoalEnvSequence(**fame.make_env_kwargs(args))
	eval_env = fame.FetchGoalEnvSequence(**fame.make_env_kwargs(args))

	writer = None
	try:
		observation_space, action_space = validate_environment_shapes(env, snapshots)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		replay_buffer_cls = (
			fame.ReplayBufferMetric if args.replay_buffer_mode == "her" else fame.ReplayBufferMetricNoHER
		)
		max_snapshot_size = snapshots[-1].size
		max_task_size = max(
			snapshot.size - (snapshots[index - 1].size if index > 0 else 0)
			for index, snapshot in enumerate(snapshots)
		)
		task_buffer = replay_buffer_cls(
			observation_space.shape,
			action_space.shape,
			max(max_snapshot_size, args.batch_size),
			device,
		)
		fast_buffer = replay_buffer_cls(
			observation_space.shape,
			action_space.shape,
			max(max_task_size, args.batch_size),
			device,
		)

		if args.dry_run:
			previous_snapshot_size = 0
			for snapshot in snapshots:
				delta_size = load_task_delta(task_buffer, snapshot, previous_snapshot_size)
				copied_transitions, successful_episodes = select_fast_transitions(
					args, task_buffer, fast_buffer
				)
				print(
					f"task {snapshot.index} ({snapshot.task_name}): snapshot={snapshot.size} "
					f"delta={delta_size} selected={copied_transitions} "
					f"successful_episodes={successful_episodes}"
				)
				previous_snapshot_size = snapshot.size
			print("dry run complete: offline buffers and environment are compatible")
			return 0

		os.makedirs(args.save_path, exist_ok=True)
		log_name = make_run_name(args, snapshots)
		run_dir = os.path.join(args.save_path, log_name)
		model_dir = os.path.join(run_dir, "model")
		os.makedirs(model_dir, exist_ok=True)
		with open(os.path.join(run_dir, "run_config.json"), "w") as handle:
			json.dump(vars(args), handle, indent=2, sort_keys=True)

		writer, log_info = fame.create_experiment_logger(args, log_name, run_dir)
		print("env_list:", env.env_list)
		print("device:", device)
		print("buffer_root:", args.buffer_root)
		print("run_dir:", run_dir)
		if log_info["active_backends"]:
			print("log_backends:", ", ".join(log_info["active_backends"]))
		else:
			print("log_backends: disabled")

		total_optim_steps = max(1, args.meta_update_steps * max(1, len(snapshots) - 1))
		meta_agent = fame.build_fame_agent(
			observation_space,
			action_space,
			device,
			args,
			total_optim_steps,
		)
		meta_buffer = replay_buffer_cls(
			observation_space.shape,
			action_space.shape,
			max(args.meta_buffer_capacity, args.batch_size),
			device,
		)

		fame.log_scalar(writer, "config/obs_dim", observation_space.shape[0], 0)
		fame.log_scalar(writer, "config/action_dim", action_space.shape[0], 0)
		fame.log_scalar(writer, "config/task_count", len(snapshots), 0)
		fame.log_scalar(writer, "config/meta_update_steps", args.meta_update_steps, 0)
		fame.log_scalar(writer, "config/eval_interval", args.eval_interval, 0)
		fame.log_scalar(writer, "config/num_eval_runs", args.num_eval_runs, 0)

		start_time = time.perf_counter()
		previous_snapshot_size = 0
		integration_step = 0
		training_summary = []
		eval_history = defaultdict(list)
		eval_csv_path = os.path.join(run_dir, "offline_eval.csv")
		for snapshot in snapshots:
			delta_size = load_task_delta(task_buffer, snapshot, previous_snapshot_size)
			copied_transitions, successful_episodes = select_fast_transitions(
				args, task_buffer, fast_buffer
			)
			print(
				f"offline FAME task {snapshot.index} ({snapshot.task_name}): "
				f"delta={delta_size} selected={copied_transitions} successes={successful_episodes}"
			)

			metrics, stage_end_step = train_meta_stage(
				args,
				meta_agent,
				meta_buffer,
				fast_buffer,
				snapshot,
				eval_env,
				snapshots,
				writer,
				integration_step,
				eval_history,
				eval_csv_path,
			)
			integrated_transitions = fame.copy_replay_buffer(fast_buffer, meta_buffer)
			if snapshot.index == 1 and args.eval_interval > 0:
				evaluate_seen_tasks(
					args,
					eval_env,
					meta_agent,
					snapshots,
					snapshot.index,
					stage_end_step,
					writer,
					eval_history,
					eval_csv_path,
				)
			fame.log_scalar(writer, "offline/task_idx", snapshot.index, stage_end_step)
			fame.log_scalar(writer, "offline/task_delta_size", delta_size, stage_end_step)
			fame.log_scalar(writer, "offline/selected_transitions", copied_transitions, stage_end_step)
			fame.log_scalar(writer, "offline/integrated_transitions", integrated_transitions, stage_end_step)
			fame.log_scalar(writer, "offline/meta_buffer_size", len(meta_buffer), stage_end_step)
			fame.log_scalar(writer, "offline/successful_episodes", successful_episodes, stage_end_step)

			checkpoint_name = f"{log_name}_task{snapshot.index}_meta"
			meta_agent.save(model_dir, checkpoint_name)
			training_summary.append(
				{
					"task_idx": snapshot.index,
					"task": snapshot.task_name,
					"snapshot_size": snapshot.size,
					"task_delta_size": delta_size,
					"selected_transitions": copied_transitions,
					"successful_episodes": successful_episodes,
					"meta_buffer_size": len(meta_buffer),
					"integration_metrics": metrics,
					"checkpoint": f"{checkpoint_name}_online_qrl.pt",
				}
			)
			writer.flush()
			integration_step = stage_end_step
			previous_snapshot_size = snapshot.size

		final_checkpoint_name = f"{log_name}_final_meta"
		meta_agent.save(model_dir, final_checkpoint_name)
		elapsed_seconds = time.perf_counter() - start_time
		summary_path = os.path.join(run_dir, "offline_training_summary.json")
		with open(summary_path, "w") as handle:
			json.dump(
				{
					"buffer_root": args.buffer_root,
					"elapsed_seconds": elapsed_seconds,
					"final_checkpoint": f"{final_checkpoint_name}_online_qrl.pt",
					"eval_csv": os.path.basename(eval_csv_path),
					"tasks": training_summary,
				},
				handle,
				indent=2,
			)
		print(f"offline FAME training complete in {elapsed_seconds / 60.0:.2f} minutes")
		print("final checkpoint:", os.path.join(model_dir, f"{final_checkpoint_name}_online_qrl.pt"))
		print("summary:", summary_path)
		return 0
	finally:
		if writer is not None:
			writer.close()
		env.close()
		eval_env.close()


if __name__ == "__main__":
	raise SystemExit(main())
'''
conda activate RLL3
cd Fetch/quasimetric-rl

python -m online_continual.fame_offline \
  --buffer_root online_continual/results/cqrl_wd/quasimetric_buffers/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --use_all_transitions \
  --meta_update_steps 100000 \
  --eval_interval 5000 \
  --num_eval_runs 10 \
  --gpu 1 \
  --log_backends wandb \
  --wandb_project_name fame-offline-fetch \
  --wandb_group fame_push-slide-pick-and-place \
  --wandb_mode online

'''