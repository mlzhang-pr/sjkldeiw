"""Train only the CQRL4 meta agent from saved cumulative buffers.

Each task snapshot is expected to extend the previous snapshot. The script
isolates the newly appended task segment, keeps its last complete trajectories,
and replays CQRL4's task-boundary meta update schedule without creating or
updating a student agent.
"""

import argparse
import hashlib
import json
import os
import re
from argparse import Namespace

import numpy as np
import torch

try:
	from . import CQRL4
except ImportError:
	import CQRL4


base = CQRL4.base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUFFER_ROOT = os.path.join(
	SCRIPT_DIR,
	"results",
	"cqrl_wd",
	"quasimetric_buffers",
)
TRANSITION_FIELDS = (
	"obses",
	"next_obses",
	"actions",
	"rewards",
	"successes",
	"not_dones",
	"not_dones_no_max",
)
TASK_DIRECTORY_PATTERN = re.compile(r"task_(\d+)_(.+)")


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Train a fresh CQRL4 meta agent using only the last complete "
			"trajectories from each task in cumulative quasimetric buffers."
		)
	)
	parser.add_argument(
		"--buffer_dir",
		default=DEFAULT_BUFFER_ROOT,
		help=(
			"A run-specific directory containing task_XX_* snapshots, or a "
			"parent directory containing exactly one such run directory."
		),
	)
	parser.add_argument(
		"--run_dir",
		default=None,
		help=(
			"Source run containing run_config.json. By default it is inferred "
			"as the sibling of quasimetric_buffers with the same run name."
		),
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		help="Output directory; defaults to <run_dir>/meta_only_last<N>.",
	)
	parser.add_argument(
		"--trajectories_per_task",
		type=int,
		default=200,
		help="Number of most recent complete trajectories retained per task.",
	)
	parser.add_argument(
		"--meta_update_steps",
		type=int,
		default=None,
		help="Meta updates per task stage; defaults to the source CQRL4 config.",
	)
	parser.add_argument(
		"--qm_diag_backup",
		type=float,
		default=None,
		help=(
			"Diagonal backup weight; defaults to the source CQRL4 config. "
			"Use 1 for paired samples or 0 for all batch pairs."
		),
	)
	parser.add_argument("--gpu", default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--num_eval_runs", type=int, default=None)
	parser.add_argument(
		"--log_backends",
		nargs="+",
		choices=["none", "tensorboard", "wandb"],
		default=None,
		help="Logging override; defaults to the source CQRL4 config.",
	)
	parser.add_argument(
		"--wandb_mode",
		choices=["online", "offline", "disabled"],
		default=None,
	)
	parser.add_argument(
		"--no_eval",
		action="store_true",
		help="Skip final meta-agent evaluation.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help=(
			"Validate and extract all task slices without running meta updates "
			"or saving model checkpoints."
		),
	)
	args = parser.parse_args()

	if args.trajectories_per_task <= 0:
		parser.error("--trajectories_per_task must be positive")
	if args.meta_update_steps is not None and args.meta_update_steps <= 0:
		parser.error("--meta_update_steps must be positive")
	if args.qm_diag_backup is not None and not 0.0 <= args.qm_diag_backup <= 1.0:
		parser.error("--qm_diag_backup must be between 0 and 1")
	if args.num_eval_runs is not None and args.num_eval_runs <= 0:
		parser.error("--num_eval_runs must be positive")
	return args


def contains_task_snapshots(path):
	if not os.path.isdir(path):
		return False
	return any(
		entry.is_dir()
		and TASK_DIRECTORY_PATTERN.fullmatch(entry.name)
		and os.path.isfile(os.path.join(entry.path, "offline_data.npz"))
		for entry in os.scandir(path)
	)


def resolve_buffer_run_dir(path):
	path = os.path.abspath(path)
	if contains_task_snapshots(path):
		return path
	if not os.path.isdir(path):
		raise FileNotFoundError(f"Buffer directory not found: {path}")

	candidates = sorted(
		entry.path
		for entry in os.scandir(path)
		if entry.is_dir() and contains_task_snapshots(entry.path)
	)
	if len(candidates) != 1:
		raise ValueError(
			f"Expected exactly one buffer run below {path}, found {len(candidates)}. "
			"Pass the run-specific directory with --buffer_dir."
		)
	return candidates[0]


def infer_source_run_dir(buffer_run_dir):
	buffer_root = os.path.dirname(buffer_run_dir)
	if os.path.basename(buffer_root) != "quasimetric_buffers":
		raise ValueError(
			"Cannot infer --run_dir because the buffer run is not directly below "
			f"a quasimetric_buffers directory: {buffer_run_dir}"
		)
	return os.path.join(
		os.path.dirname(buffer_root),
		os.path.basename(buffer_run_dir),
	)


def load_source_config(run_dir):
	config_path = os.path.join(run_dir, "run_config.json")
	if not os.path.isfile(config_path):
		raise FileNotFoundError(f"CQRL4 run config not found: {config_path}")
	with open(config_path, "r") as handle:
		return json.load(handle)


def current_cqrl_defaults():
	parser = argparse.ArgumentParser(add_help=False)
	base.add_common_args(parser)
	base.add_sac_args(parser)
	base.add_student_qrl_args(parser)
	base.add_quasimetric_args(parser)
	return vars(parser.parse_args([]))


def build_training_args(source_config, cli_args):
	config = current_cqrl_defaults()
	config.update(source_config)
	config["method"] = "cqrl"
	if cli_args.gpu is not None:
		config["gpu"] = cli_args.gpu
	if cli_args.seed is not None:
		config["seed"] = cli_args.seed
	if cli_args.meta_update_steps is not None:
		config["meta_update_steps"] = cli_args.meta_update_steps
	if cli_args.qm_diag_backup is not None:
		config["qm_diag_backup"] = cli_args.qm_diag_backup
	if cli_args.num_eval_runs is not None:
		config["num_eval_runs"] = cli_args.num_eval_runs
	if cli_args.log_backends is not None:
		config["log_backends"] = base.normalize_log_backends(cli_args.log_backends)
	elif cli_args.dry_run:
		config["log_backends"] = []
	else:
		config["log_backends"] = base.normalize_log_backends(
			config.get("log_backends", [])
		)
	if cli_args.wandb_mode is not None:
		config["wandb_mode"] = cli_args.wandb_mode
	config.setdefault("wandb_project_name", "cqrl-fetch")
	config.setdefault("wandb_entity", None)
	config.setdefault("wandb_group", None)
	config.setdefault("wandb_mode", "online")
	return Namespace(**config)


def discover_task_snapshot_paths(buffer_run_dir):
	snapshots = []
	for entry in os.scandir(buffer_run_dir):
		match = TASK_DIRECTORY_PATTERN.fullmatch(entry.name)
		if not entry.is_dir() or match is None:
			continue
		data_path = os.path.join(entry.path, "offline_data.npz")
		if os.path.isfile(data_path):
			snapshots.append(
				{
					"task_idx": int(match.group(1)),
					"task": match.group(2),
					"directory": entry.path,
					"data_path": data_path,
				}
			)
	snapshots.sort(key=lambda record: record["task_idx"])
	if not snapshots:
		raise FileNotFoundError(
			f"No task_XX_*/offline_data.npz snapshots found in {buffer_run_dir}"
		)
	expected_indices = list(range(1, len(snapshots) + 1))
	actual_indices = [record["task_idx"] for record in snapshots]
	if actual_indices != expected_indices:
		raise ValueError(
			f"Task snapshots must be contiguous from 1; found {actual_indices}."
		)
	return snapshots


def array_digest(values):
	values = np.ascontiguousarray(values)
	digest = hashlib.sha256()
	digest.update(str(values.dtype).encode("ascii"))
	digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
	digest.update(values.view(np.uint8))
	return digest.digest()


def inspect_task_slices(buffer_run_dir, trajectories_per_task):
	snapshots = discover_task_snapshot_paths(buffer_run_dir)
	previous_size = 0
	previous_digests = {}

	for snapshot in snapshots:
		with np.load(snapshot["data_path"]) as data:
			missing_fields = [
				name for name in TRANSITION_FIELDS if name not in data.files
			]
			if missing_fields:
				raise ValueError(
					f"{snapshot['data_path']} is missing fields: {missing_fields}"
				)
			snapshot_size = len(data["obses"])
			if any(len(data[name]) != snapshot_size for name in TRANSITION_FIELDS):
				raise ValueError(
					f"Transition arrays have inconsistent lengths in {snapshot['data_path']}"
				)
			if snapshot_size <= previous_size:
				raise ValueError(
					f"Snapshot {snapshot['data_path']} has {snapshot_size} transitions; "
					f"expected more than the previous cumulative size {previous_size}."
				)

			if "others" in data.files:
				metadata = data["others"]
				if len(metadata) < 4:
					raise ValueError(f"Invalid buffer metadata in {snapshot['data_path']}")
				if bool(metadata[3]):
					raise ValueError(
						"Wrapped/full replay-buffer snapshots are not supported because "
						"their raw storage is not cumulative chronological data: "
						f"{snapshot['data_path']}"
					)
				if int(metadata[1]) != snapshot_size:
					raise ValueError(
						f"Saved index {int(metadata[1])} does not match array size "
						f"{snapshot_size} in {snapshot['data_path']}"
					)

			current_digests = {}
			for name in TRANSITION_FIELDS:
				values = np.asarray(data[name])
				if previous_size and array_digest(values[:previous_size]) != previous_digests[name]:
					raise ValueError(
						f"{snapshot['data_path']} is not a cumulative extension of the "
						f"previous snapshot; prefix mismatch in {name}."
					)
				current_digests[name] = array_digest(values)

			not_dones = np.asarray(data["not_dones"]).reshape(snapshot_size, -1)[:, 0]
			if previous_size and not_dones[previous_size - 1] >= 0.5:
				raise ValueError(
					f"Task {snapshot['task_idx']} starts after an incomplete trajectory."
				)
			new_not_dones = not_dones[previous_size:]
			done_positions = np.flatnonzero(new_not_dones < 0.5)
			if done_positions.size == 0 or done_positions[-1] != len(new_not_dones) - 1:
				raise ValueError(
					f"Task {snapshot['task_idx']} snapshot does not end on a complete trajectory."
				)
			if done_positions.size < trajectories_per_task:
				raise ValueError(
					f"Task {snapshot['task_idx']} has only {done_positions.size} complete "
					f"trajectories; {trajectories_per_task} requested."
				)
			if done_positions.size == trajectories_per_task:
				selected_start = previous_size
			else:
				selected_start = previous_size + int(
					done_positions[-trajectories_per_task - 1] + 1
				)

		snapshot.update(
			{
				"snapshot_size": snapshot_size,
				"new_start": previous_size,
				"new_transitions": snapshot_size - previous_size,
				"available_trajectories": int(done_positions.size),
				"selected_start": selected_start,
				"selected_end": snapshot_size,
				"selected_transitions": snapshot_size - selected_start,
				"selected_trajectories": trajectories_per_task,
			}
		)
		previous_size = snapshot_size
		previous_digests = current_digests
	return snapshots


def append_selected_trajectories(snapshot, target_buffers):
	with np.load(snapshot["data_path"]) as data:
		arrays = {name: np.asarray(data[name]) for name in TRANSITION_FIELDS}

	episode_count = 0
	successful_episodes = 0
	episode_success = False
	for source_idx in range(snapshot["selected_start"], snapshot["selected_end"]):
		episode_success = episode_success or bool(
			arrays["successes"][source_idx, 0] > 0.5
		)
		transition = (
			arrays["obses"][source_idx],
			arrays["actions"][source_idx],
			arrays["rewards"][source_idx],
			arrays["successes"][source_idx],
			arrays["next_obses"][source_idx],
			not bool(arrays["not_dones"][source_idx, 0]),
			not bool(arrays["not_dones_no_max"][source_idx, 0]),
		)
		for target_buffer in target_buffers:
			target_buffer.add(*transition)
		if arrays["not_dones"][source_idx, 0] < 0.5:
			episode_count += 1
			successful_episodes += int(episode_success)
			episode_success = False

	if episode_count != snapshot["selected_trajectories"]:
		raise RuntimeError(
			f"Expected {snapshot['selected_trajectories']} trajectories for task "
			f"{snapshot['task_idx']}, copied {episode_count}."
		)
	return successful_episodes


def json_metrics(metrics):
	result = {}
	for key, value in metrics.items():
		if torch.is_tensor(value):
			if value.numel() != 1:
				continue
			value = value.detach().cpu().item()
		elif isinstance(value, (np.integer, np.floating)):
			value = value.item()
		if isinstance(value, (bool, int, float, str)) or value is None:
			result[key] = value
	return result


def evaluate_meta_agent(env, meta_agent, snapshots, num_eval_runs, writer, step):
	records = []
	meta_agent.eval()
	for snapshot in snapshots:
		task_idx = snapshot["task_idx"]
		task_name = env.env_list[task_idx - 1]
		env.set_task(task_name)
		eval_results = env.evaluate_agent(
			meta_agent,
			num_eval_runs,
			reseed_each_episode=True,
		)
		metrics = json_metrics(base.summarize_eval_results(eval_results))
		base.log_metric_dict(
			writer,
			f"meta_only_eval/{task_idx}_{base.safe_tag(task_name)}",
			metrics,
			step + task_idx,
		)
		record = {
			"task_idx": task_idx,
			"task": task_name,
			"num_eval_runs": num_eval_runs,
			**metrics,
		}
		records.append(record)
		print(
			f"meta evaluation task {task_idx}: {task_name}",
			f"success={metrics['success_mean']:.3f} +/- {metrics['success_std']:.3f}",
			f"gc_success={metrics['gc_success_mean']:.3f} +/- "
			f"{metrics['gc_success_std']:.3f}",
			f"return={metrics['return_mean']:.3f} +/- {metrics['return_std']:.3f}",
		)
	meta_agent.train(True)
	return records


def validate_snapshot_tasks(snapshots, env):
	if len(snapshots) > len(env.env_list):
		raise ValueError(
			f"Found {len(snapshots)} task snapshots for an environment with "
			f"{len(env.env_list)} tasks."
		)
	for snapshot in snapshots:
		expected_name = str(env.env_list[snapshot["task_idx"] - 1]).replace("/", "_")
		if snapshot["task"] != expected_name:
			raise ValueError(
				f"Task {snapshot['task_idx']} snapshot is named {snapshot['task']!r}, "
				f"but run_config.json defines {expected_name!r}."
			)


def write_json(path, payload):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)


def main():
	cli_args = parse_args()
	buffer_run_dir = resolve_buffer_run_dir(cli_args.buffer_dir)
	run_dir = os.path.abspath(
		cli_args.run_dir or infer_source_run_dir(buffer_run_dir)
	)
	source_config = load_source_config(run_dir)
	args = build_training_args(source_config, cli_args)
	update_steps = int(args.meta_update_steps)
	if update_steps <= 0:
		raise ValueError("meta_update_steps must be positive")

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	base.set_seed_everywhere(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	snapshots = inspect_task_slices(
		buffer_run_dir,
		cli_args.trajectories_per_task,
	)

	run_name = os.path.basename(os.path.normpath(run_dir))
	output_dir = os.path.abspath(
		cli_args.output_dir
		or os.path.join(
			run_dir,
			f"meta_only_last{cli_args.trajectories_per_task}",
		)
	)
	model_dir = os.path.join(output_dir, "model")
	filtered_data_dir = os.path.join(output_dir, "filtered_buffers")
	os.makedirs(model_dir, exist_ok=True)

	logging_config = vars(args).copy()
	logging_config.update(
		{
			"meta_only": True,
			"source_run_dir": run_dir,
			"source_buffer_dir": buffer_run_dir,
			"trajectories_per_task": cli_args.trajectories_per_task,
			"meta_update_steps_per_stage": update_steps,
			"output_dir": output_dir,
		}
	)
	write_json(os.path.join(output_dir, "run_config.json"), logging_config)
	log_name = f"{run_name}_meta_only_last{cli_args.trajectories_per_task}"
	writer, log_info = base.create_experiment_logger(
		Namespace(**logging_config),
		log_name,
		output_dir,
	)

	env = None
	try:
		env = base.FetchGoalEnvSequence(**base.make_env_kwargs(args))
		validate_snapshot_tasks(snapshots, env)
		obs_space = base.vector_observation_space(env.env.observation_space)
		action_space = env.env.action_space
		replay_buffer_cls = (
			base.ReplayBufferMetric
			if args.replay_buffer_mode == "her"
			else base.ReplayBufferMetricNoHER
		)
		total_selected_transitions = sum(
			snapshot["selected_transitions"] for snapshot in snapshots
		)
		largest_task_slice = max(
			snapshot["selected_transitions"] for snapshot in snapshots
		)
		minimum_capacity = max(
			int(args.batch_size),
			int(args.qm_min_buffer_size),
		)
		cumulative_buffer = replay_buffer_cls(
			obs_space.shape,
			action_space.shape,
			max(total_selected_transitions + 1, minimum_capacity),
			device,
		)
		recent_buffer = replay_buffer_cls(
			obs_space.shape,
			action_space.shape,
			max(largest_task_slice + 1, minimum_capacity),
			device,
		)
		meta_agent = base.build_meta_agent(
			obs_space.shape[0],
			action_space.shape[0],
			device,
			args,
		)
		meta_agent.train(True)

		print("source run:", run_dir)
		print("source buffers:", buffer_run_dir)
		print("tasks:", env.env_list)
		print("device:", device)
		print("trajectories per task:", cli_args.trajectories_per_task)
		print("meta updates per stage:", update_steps)

		stage_records = []
		global_update_count = 0
		last_structure_metrics = {}
		last_actor_metrics = {}
		for stage_idx, snapshot in enumerate(snapshots, start=1):
			recent_buffer.reset()
			cumulative_start = len(cumulative_buffer)
			successful_trajectories = append_selected_trajectories(
				snapshot,
				(cumulative_buffer, recent_buffer),
			)
			if len(recent_buffer) != snapshot["selected_transitions"]:
				raise RuntimeError(
					f"Task {snapshot['task_idx']} expected "
					f"{snapshot['selected_transitions']} recent transitions, got "
					f"{len(recent_buffer)}."
				)
			stage_data_dir = os.path.join(
				filtered_data_dir,
				f"stage_{stage_idx:02d}_{snapshot['task']}",
			)
			cumulative_buffer.save_data(stage_data_dir)
			print(
				f"meta stage {stage_idx}/{len(snapshots)}: {snapshot['task']}",
				f"available={snapshot['available_trajectories']}",
				f"selected={snapshot['selected_trajectories']}",
				f"recent_transitions={len(recent_buffer)}",
				f"cumulative_transitions={len(cumulative_buffer)}",
			)

			structure_metrics = {}
			actor_metrics = {}
			recent_updates = 0
			if not cli_args.dry_run:
				structure_metrics, actor_metrics, recent_updates = (
					CQRL4.update_meta_from_recent(
						args,
						meta_agent,
						cumulative_buffer,
						recent_buffer,
						stage_idx,
						writer,
						stage_idx * int(args.change_freq),
					)
				)
				if not structure_metrics and not actor_metrics:
					raise RuntimeError(
						f"Meta stage {stage_idx} produced no metrics. Check "
						"qm_min_buffer_size and meta_update_steps."
					)
				last_structure_metrics = json_metrics(structure_metrics)
				last_actor_metrics = json_metrics(actor_metrics)
				stage_checkpoint = (
					f"{run_name}_meta_only_last{cli_args.trajectories_per_task}_"
					f"stage{stage_idx}_{base.safe_tag(snapshot['task'])}"
				)
				meta_agent.save(model_dir, stage_checkpoint)
			else:
				stage_checkpoint = None

			global_update_count += 0 if cli_args.dry_run else update_steps
			stage_records.append(
				{
					**{
						key: value
						for key, value in snapshot.items()
						if key not in {"directory", "data_path"}
					},
					"source_data": snapshot["data_path"],
					"successful_trajectories": successful_trajectories,
					"cumulative_start": cumulative_start,
					"cumulative_end": len(cumulative_buffer),
					"filtered_buffer": stage_data_dir,
					"meta_updates": 0 if cli_args.dry_run else update_steps,
					"recent_updates": recent_updates,
					"meta_checkpoint": (
						None
						if stage_checkpoint is None
						else os.path.join(model_dir, stage_checkpoint)
					),
					"last_structure_metrics": json_metrics(structure_metrics),
					"last_actor_metrics": json_metrics(actor_metrics),
				}
			)

		final_buffer_dir = os.path.join(filtered_data_dir, "final")
		cumulative_buffer.save_data(final_buffer_dir)
		evaluation_records = []
		final_checkpoint = None
		if not cli_args.dry_run:
			final_checkpoint = (
				f"{run_name}_meta_only_last{cli_args.trajectories_per_task}_final"
			)
			meta_agent.save(model_dir, final_checkpoint)
			if not cli_args.no_eval:
				evaluation_records = evaluate_meta_agent(
					env,
					meta_agent,
					snapshots,
					int(args.num_eval_runs),
					writer,
					global_update_count,
				)

		report = {
			"source_run_dir": run_dir,
			"source_buffer_dir": buffer_run_dir,
			"output_dir": output_dir,
			"device": str(device),
			"seed": int(args.seed),
			"dry_run": cli_args.dry_run,
			"trajectories_per_task": cli_args.trajectories_per_task,
			"task_count": len(snapshots),
			"total_selected_trajectories": (
				len(snapshots) * cli_args.trajectories_per_task
			),
			"total_selected_transitions": len(cumulative_buffer),
			"meta_update_steps_per_stage": update_steps,
			"total_meta_update_steps": global_update_count,
			"filtered_buffer": final_buffer_dir,
			"final_meta_checkpoint": (
				None
				if final_checkpoint is None
				else os.path.join(model_dir, final_checkpoint)
			),
			"logging": log_info,
			"stages": stage_records,
			"last_structure_metrics": last_structure_metrics,
			"last_actor_metrics": last_actor_metrics,
			"meta_evaluation": evaluation_records,
		}
		report_path = os.path.join(output_dir, "meta_only_report.json")
		write_json(report_path, report)
		print("filtered buffer:", final_buffer_dir)
		if final_checkpoint is not None:
			print("saved meta checkpoint:", report["final_meta_checkpoint"])
		print("saved report:", report_path)
		return report
	finally:
		writer.close()
		if env is not None:
			env.close()


if __name__ == "__main__":
	main()


'''
conda run -n RLL3 python -m online_continual.cqrl4_only_update_meta \
  --trajectories_per_task 1500 \
	--qm_diag_backup 0.5 \
  --gpu 1 \
  --output_dir  online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/meta_only_last1500_diag_backup

  
  --qm_diag_backup 0：完整跨样本配对。
--qm_diag_backup 1：维持现状。
--qm_diag_backup 0.5：两者等权混合。
  
  '''