"""Incrementally train a CQRL meta agent from task-specific student rollouts."""

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch

try:
	from . import CQRL as cqrl
except ImportError:
	import CQRL as cqrl


base = cqrl.base


def add_meta_learner_override_args(parser):
	group = parser.add_argument_group(
		"meta learner overrides",
		"Override meta learner hyperparameters from run_config.json.",
	)
	base.add_sac_args(group)
	base.add_quasimetric_args(group)
	argument_names = []
	for action in group._group_actions:
		action.default = argparse.SUPPRESS
		argument_names.append(action.dest)
	return tuple(argument_names)


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Collect offline data from trained CQRL student checkpoints and "
			"incrementally train one meta agent after each task."
		)
	)
	parser.add_argument(
		"--run_dir",
		required=True,
		help="CQRL run directory containing run_config.json and model/.",
	)
	parser.add_argument(
		"--store_num",
		type=int,
		default=10000,
		help=(
			"Number of complete trajectories rolled out from each task-specific "
			"student. Only successful trajectories are retained for training."
		),
	)
	parser.add_argument(
		"--tasks",
		nargs=2,
		default=None,
		metavar=("TASK_A", "TASK_B"),
		help=(
			"Train and evaluate on exactly two tasks, specified by task name or "
			"1-based position in the source sequence. Defaults to all tasks."
		),
	)
	parser.add_argument(
		"--meta_update_steps",
		type=int,
		default=None,
		help=(
			"Meta updates after each collected task; defaults to meta_update_steps "
			"in run_config.json."
		),
	)
	parser.add_argument(
		"--gpu",
		default=None,
		help="CUDA device override; defaults to the source run configuration.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Collection/training seed override; defaults to the source run seed.",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		help=(
			"Output directory; defaults to "
			"<run_dir>/offline_meta_incremental_store<store_num>."
		),
	)
	parser.add_argument(
		"--deterministic",
		action="store_true",
		help="Use each student's mean action instead of its training-time sampled action.",
	)
	parser.add_argument(
		"--log_interval",
		type=int,
		default=1000,
		help="Print meta-training metrics every this many updates.",
	)
	parser.add_argument(
		"--num_eval_runs",
		type=int,
		default=None,
		help="Evaluation episodes per task; defaults to num_eval_runs in run_config.json.",
	)
	parser.add_argument(
		"--train_eval_interval",
		type=int,
		default=10000,
		help="Run evaluation during offline training every this many meta updates.",
	)
	parser.add_argument(
		"--train_eval_runs",
		type=int,
		default=None,
		help="Episodes per task during training evaluation; defaults to --num_eval_runs.",
	)
	parser.add_argument(
		"--distill_loss_weight",
		type=float,
		default=1.0,
		help=(
			"Weight for current student-to-meta actor distillation during each "
			"task-boundary meta update; use 0 to disable."
		),
	)
	meta_learner_arg_names = add_meta_learner_override_args(parser)
	args = parser.parse_args()
	args.meta_learner_overrides = {
		name: getattr(args, name)
		for name in meta_learner_arg_names
		if hasattr(args, name)
	}
	args.meta_learner_arg_names = meta_learner_arg_names

	if args.store_num <= 0:
		parser.error("--store_num must be positive")
	if args.meta_update_steps is not None and args.meta_update_steps <= 0:
		parser.error("--meta_update_steps must be positive")
	if args.log_interval <= 0:
		parser.error("--log_interval must be positive")
	if args.num_eval_runs is not None and args.num_eval_runs <= 0:
		parser.error("--num_eval_runs must be positive")
	if args.train_eval_interval <= 0:
		parser.error("--train_eval_interval must be positive")
	if args.train_eval_runs is not None and args.train_eval_runs <= 0:
		parser.error("--train_eval_runs must be positive")
	if args.distill_loss_weight < 0.0:
		parser.error("--distill_loss_weight must be non-negative")
	return args


def load_source_config(run_dir):
	config_path = os.path.join(run_dir, "run_config.json")
	if not os.path.isfile(config_path):
		raise FileNotFoundError(f"CQRL run config not found: {config_path}")
	with open(config_path, "r") as handle:
		return json.load(handle)


def task_checkpoint_name(run_name, task_idx):
	return f"{run_name}_task{task_idx}_fast"


def select_tasks(task_names, selectors):
	all_tasks = list(enumerate(task_names, start=1))
	if selectors is None:
		return all_tasks

	selected_tasks = []
	for selector in selectors:
		if selector.isdigit():
			task_idx = int(selector)
			if not 1 <= task_idx <= len(task_names):
				raise ValueError(
					f"Task index {task_idx} is outside 1..{len(task_names)}."
				)
			task_name = task_names[task_idx - 1]
		else:
			if selector not in task_names:
				raise ValueError(
					f"Unknown task {selector!r}. Available tasks: "
					f"{', '.join(task_names)}"
				)
			task_name = selector
			task_idx = task_names.index(task_name) + 1
		selected_tasks.append((task_idx, task_name))

	if len({task_idx for task_idx, _ in selected_tasks}) != len(selected_tasks):
		raise ValueError("--tasks must select two different tasks.")
	return selected_tasks


def make_logging_args(source_args, cli_args, run_dir, output_dir, selected_tasks):
	logging_config = vars(source_args).copy()
	log_backends = logging_config.get("log_backends", [])
	if isinstance(log_backends, str):
		log_backends = [log_backends]
	log_backends = [backend for backend in log_backends if backend != "none"]
	if "wandb" not in log_backends:
		log_backends.append("wandb")
	logging_config.update(
		{
			"log_backends": base.normalize_log_backends(log_backends),
			"offline_incremental": True,
			"offline_source_run_dir": run_dir,
			"offline_output_dir": output_dir,
			"offline_store_num": cli_args.store_num,
			"offline_tasks": [task_name for _, task_name in selected_tasks],
			"offline_deterministic": cli_args.deterministic,
		}
	)
	logging_config.setdefault("wandb_project_name", "cqrl-fetch")
	logging_config.setdefault("wandb_entity", None)
	logging_config.setdefault("wandb_group", None)
	logging_config.setdefault("wandb_mode", "online")
	return Namespace(**logging_config)


def collect_task_data(
	env,
	student,
	replay_buffer,
	store_num,
	deterministic,
	writer=None,
	log_prefix="offline_collection",
	step_offset=0,
):
	collector = base.Collector(env, replay_buffer)
	collector.obs, _ = collector._reset_env()
	completed_episodes = 0
	successful_episodes = 0
	episode_success = False
	episode_transitions = []
	attempted_transitions = 0
	collected_transitions = 0

	while completed_episodes < store_num:
		with torch.inference_mode():
			action = collector._act(student, sample=not deterministic)
		next_obs, reward, terminated, truncated, info = env.no_count_step(action)
		next_obs, next_goal_obs = collector._split_obs(next_obs)

		actual_done = bool(terminated or truncated)
		done_no_max = False if truncated else actual_done
		episode_success = episode_success or bool(info.get("success", False))
		episode_transitions.append(
			(
				np.array(collector.obs, copy=True),
				np.array(action, copy=True),
				reward,
				info.get("success", False),
				np.array(next_obs, copy=True),
				actual_done,
				done_no_max,
			)
		)
		attempted_transitions += 1

		collector.obs = next_obs
		collector.goal_obs = next_goal_obs
		if actual_done:
			completed_episodes += 1
			episode_success_value = int(episode_success)
			if episode_success:
				if collected_transitions + len(episode_transitions) > replay_buffer.capacity:
					raise RuntimeError(
						f"Replay buffer capacity {replay_buffer.capacity} cannot hold "
						f"all successful trajectories."
					)
				for transition in episode_transitions:
					replay_buffer.add(*transition)
				successful_episodes += 1
				collected_transitions += len(episode_transitions)
			process_step = step_offset + completed_episodes
			base.log_scalar(
				writer,
				f"{log_prefix}/episode_success",
				episode_success_value,
				process_step,
			)
			base.log_scalar(
				writer,
				f"{log_prefix}/success_rate",
				successful_episodes / completed_episodes,
				process_step,
			)
			base.log_scalar(
				writer,
				f"{log_prefix}/successful_trajectories",
				successful_episodes,
				process_step,
			)
			base.log_scalar(
				writer,
				f"{log_prefix}/attempted_transitions",
				attempted_transitions,
				process_step,
			)
			base.log_scalar(
				writer,
				f"{log_prefix}/retained_transitions",
				collected_transitions,
				process_step,
			)
			episode_success = False
			episode_transitions = []
			collector.obs, _ = collector._reset_env()

	return {
		"completed_episodes": completed_episodes,
		"successful_episodes": successful_episodes,
		"trajectory_success_rate": successful_episodes / completed_episodes,
		"attempted_transitions": attempted_transitions,
		"collected_transitions": collected_transitions,
		"discarded_transitions": attempted_transitions - collected_transitions,
	}


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


def evaluate_training_progress(
	env,
	meta_agent,
	selected_tasks,
	num_eval_runs,
	stage_idx,
	update_idx,
	global_update_idx,
	writer,
	process_step,
):
	evaluation_records = []
	for task_idx, task_name in selected_tasks:
		env.set_task(task_name)
		eval_results = env.evaluate_agent(
			meta_agent,
			num_eval_runs,
			reseed_each_episode=False,
		)
		metrics = json_metrics(base.summarize_eval_results(eval_results))
		task_prefix = (
			f"offline_train_eval/stage_{stage_idx:02d}/"
			f"{task_idx}_{base.safe_tag(task_name)}"
		)
		base.log_metric_dict(writer, task_prefix, metrics, process_step)
		evaluation_records.append(
			{
				"stage_idx": stage_idx,
				"update": update_idx,
				"global_update": global_update_idx,
				"task_idx": task_idx,
				"task": task_name,
				"num_eval_runs": num_eval_runs,
				**metrics,
			}
		)
		print(
			f"training evaluation stage {stage_idx} update {update_idx}: "
			f"{task_name}",
			f"success={metrics['success_mean']:.3f} +/- {metrics['success_std']:.3f}",
			f"gc_success={metrics['gc_success_mean']:.3f} +/- "
			f"{metrics['gc_success_std']:.3f}",
			f"return={metrics['return_mean']:.3f} +/- {metrics['return_std']:.3f}",
		)

	mean_success = float(
		np.mean([record["success_mean"] for record in evaluation_records])
	)
	mean_gc_success = float(
		np.mean([record["gc_success_mean"] for record in evaluation_records])
	)
	base.log_scalar(
		writer,
		"offline_train_eval/mean_success_across_tasks",
		mean_success,
		process_step,
	)
	base.log_scalar(
		writer,
		"offline_train_eval/mean_gc_success_across_tasks",
		mean_gc_success,
		process_step,
	)
	base.log_scalar(
		writer,
		"offline_train_eval/stage_update",
		update_idx,
		process_step,
	)
	base.log_scalar(
		writer,
		"offline_train_eval/global_update",
		global_update_idx,
		process_step,
	)
	if writer is not None:
		writer.flush()
	return evaluation_records


def train_meta_agent(
	meta_agent,
	offline_buffer,
	student,
	student_buffer,
	distill_loss_weight,
	update_steps,
	log_interval,
	eval_env,
	selected_tasks,
	eval_interval,
	num_eval_runs,
	stage_idx,
	writer=None,
	step_offset=0,
	update_offset=0,
):
	last_structure_metrics = {}
	last_actor_metrics = {}
	evaluation_history = []
	meta_agent.train(True)
	for update_idx in range(1, update_steps + 1):
		structure_metrics, actor_metrics = meta_agent.update_structure_and_actor(
			offline_buffer,
			teacher_agent=student,
			distill_buffer=student_buffer,
			distill_loss_weight=distill_loss_weight,
		)
		if not structure_metrics and not actor_metrics:
			raise RuntimeError(
				"Meta update produced no metrics. The offline buffer is likely smaller "
				"than qm_min_buffer_size."
			)
		last_structure_metrics = json_metrics(structure_metrics)
		last_actor_metrics = json_metrics(actor_metrics)
		process_step = step_offset + update_idx
		global_update_idx = update_offset + update_idx
		stage_prefix = f"offline_train/stage_{stage_idx:02d}"
		base.log_metric_dict(
			writer,
			f"{stage_prefix}/structure",
			last_structure_metrics,
			process_step,
		)
		base.log_metric_dict(
			writer,
			f"{stage_prefix}/actor",
			last_actor_metrics,
			process_step,
		)
		base.log_scalar(
			writer,
			"offline_train/stage_update",
			update_idx,
			process_step,
		)
		base.log_scalar(
			writer,
			"offline_train/global_update",
			global_update_idx,
			process_step,
		)
		base.log_scalar(writer, "offline_train/stage_idx", stage_idx, process_step)
		if (
			update_idx == 1
			or update_idx % eval_interval == 0
			or update_idx == update_steps
		):
			evaluation_history.extend(
				evaluate_training_progress(
					eval_env,
					meta_agent,
					selected_tasks,
					num_eval_runs,
					stage_idx,
					update_idx,
					global_update_idx,
					writer,
					process_step,
				)
			)
		if update_idx == 1 or update_idx % log_interval == 0 or update_idx == update_steps:
			print(
				f"meta stage {stage_idx} update {update_idx}/{update_steps}",
				{"structure": last_structure_metrics, "actor": last_actor_metrics},
			)
			if writer is not None:
				writer.flush()
	return last_structure_metrics, last_actor_metrics, evaluation_history


def evaluate_meta_agent(
	env,
	meta_agent,
	selected_tasks,
	num_eval_runs,
	writer=None,
	step_offset=0,
):
	evaluation_records = []
	for eval_idx, (task_idx, task_name) in enumerate(selected_tasks, start=1):
		env.set_task(task_name)
		eval_results = env.evaluate_agent(
			meta_agent,
			num_eval_runs,
			reseed_each_episode=False,
		)
		metrics = json_metrics(base.summarize_eval_results(eval_results))
		process_step = step_offset + eval_idx
		task_prefix = f"offline_eval/{task_idx}_{base.safe_tag(task_name)}"
		base.log_metric_dict(writer, task_prefix, metrics, process_step)
		base.log_scalar(writer, "offline_eval/task_idx", task_idx, process_step)
		base.log_scalar(
			writer,
			"offline_eval/success_mean",
			metrics["success_mean"],
			process_step,
		)
		base.log_scalar(
			writer,
			"offline_eval/gc_success_mean",
			metrics["gc_success_mean"],
			process_step,
		)
		record = {
			"task_idx": task_idx,
			"task": task_name,
			"num_eval_runs": num_eval_runs,
			**metrics,
		}
		evaluation_records.append(record)
		print(
			f"meta evaluation task {eval_idx}/{len(selected_tasks)}: {task_name}",
			f"success={metrics['success_mean']:.3f} +/- {metrics['success_std']:.3f}",
			f"gc_success={metrics['gc_success_mean']:.3f} +/- "
			f"{metrics['gc_success_std']:.3f}",
			f"return={metrics['return_mean']:.3f} +/- {metrics['return_std']:.3f}",
		)
	if evaluation_records:
		final_step = step_offset + len(evaluation_records)
		base.log_scalar(
			writer,
			"offline_eval/mean_success_across_tasks",
			float(np.mean([record["success_mean"] for record in evaluation_records])),
			final_step,
		)
		base.log_scalar(
			writer,
			"offline_eval/mean_gc_success_across_tasks",
			float(
				np.mean(
					[record["gc_success_mean"] for record in evaluation_records]
				)
			),
			final_step,
		)
		if writer is not None:
			writer.flush()
	return evaluation_records


def main():
	cli_args = parse_args()
	run_dir = os.path.abspath(cli_args.run_dir)
	source_config = load_source_config(run_dir)
	source_args = Namespace(**source_config)
	if not hasattr(source_args, "qm_backup_coef"):
		source_args.qm_backup_coef = 1.0
	if not hasattr(source_args, "qm_q_loss_coef"):
		source_args.qm_q_loss_coef = 1.0

	if cli_args.gpu is not None:
		source_args.gpu = cli_args.gpu
	if cli_args.seed is not None:
		source_args.seed = cli_args.seed
	for name, value in cli_args.meta_learner_overrides.items():
		setattr(source_args, name, value)
	meta_learner_overrides = dict(cli_args.meta_learner_overrides)
	if cli_args.meta_update_steps is not None:
		meta_learner_overrides["meta_update_steps"] = cli_args.meta_update_steps
	update_steps = (
		cli_args.meta_update_steps
		if cli_args.meta_update_steps is not None
		else int(source_args.meta_update_steps)
	)
	num_eval_runs = (
		cli_args.num_eval_runs
		if cli_args.num_eval_runs is not None
		else int(source_args.num_eval_runs)
	)
	train_eval_runs = (
		cli_args.train_eval_runs
		if cli_args.train_eval_runs is not None
		else num_eval_runs
	)
	if update_steps <= 0:
		raise ValueError("meta_update_steps must be positive")
	if num_eval_runs <= 0:
		raise ValueError("num_eval_runs must be positive")

	os.environ["CUDA_VISIBLE_DEVICES"] = str(source_args.gpu)
	base.set_seed_everywhere(source_args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	env = base.FetchGoalEnvSequence(**base.make_env_kwargs(source_args))
	all_task_names = list(env.env_list)
	selected_tasks = select_tasks(all_task_names, cli_args.tasks)
	task_names = [task_name for _, task_name in selected_tasks]
	task_count = len(selected_tasks)
	max_episode_steps = int(source_args.max_episode_steps)
	task_buffer_capacity = cli_args.store_num * max_episode_steps
	offline_buffer_capacity = task_count * task_buffer_capacity
	minimum_buffer_size = int(source_args.qm_min_buffer_size)
	if task_buffer_capacity < minimum_buffer_size:
		env.close()
		raise ValueError(
			f"The first incremental stage can hold at most {task_buffer_capacity} "
			f"transitions, but qm_min_buffer_size is {minimum_buffer_size}. "
			"Increase --store_num."
		)

	run_name = os.path.basename(os.path.normpath(run_dir))
	model_dir = os.path.join(run_dir, "model")
	task_tag = "-".join(base.safe_tag(task_name) for task_name in task_names)
	default_output_name = f"offline_meta_incremental_store{cli_args.store_num}"
	if cli_args.tasks is not None:
		default_output_name = (
			f"offline_meta_incremental_{task_tag}_store{cli_args.store_num}_contra{cli_args.qm_contrastive_coef}_test_bc"
		)
	output_dir = os.path.abspath(
		cli_args.output_dir
		or os.path.join(run_dir, default_output_name)
	)
	output_model_dir = os.path.join(output_dir, "model")
	offline_data_dir = os.path.join(output_dir, "offline_data")
	os.makedirs(output_model_dir, exist_ok=True)
	logging_args = make_logging_args(
		source_args,
		cli_args,
		run_dir,
		output_dir,
		selected_tasks,
	)
	logging_args.offline_meta_update_steps = update_steps
	logging_args.offline_meta_update_steps_per_stage = update_steps
	logging_args.offline_incremental_stage_count = task_count
	logging_args.offline_num_eval_runs = num_eval_runs
	logging_args.offline_train_eval_interval = cli_args.train_eval_interval
	logging_args.offline_train_eval_runs = train_eval_runs
	logging_args.offline_distill_loss_weight = cli_args.distill_loss_weight
	log_name = f"{run_name}_{os.path.basename(output_dir)}"
	writer, log_info = base.create_experiment_logger(
		logging_args,
		log_name,
		output_dir,
	)
	if log_info["active_backends"]:
		print("log_backends:", ", ".join(log_info["active_backends"]))
	else:
		print("log_backends: disabled")
	for key in ("wandb_path", "wandb_url"):
		if log_info[key] is not None:
			print(f"{key}:", log_info[key])

	obs_space = base.vector_observation_space(env.env.observation_space)
	action_space = env.env.action_space
	replay_buffer_cls = (
		base.ReplayBufferMetric
		if source_args.replay_buffer_mode == "her"
		else base.ReplayBufferMetricNoHER
	)
	offline_buffer = replay_buffer_cls(
		obs_space.shape,
		action_space.shape,
		offline_buffer_capacity,
		device,
	)
	meta_agent = base.build_meta_agent(
		obs_space.shape[0],
		action_space.shape[0],
		device,
		source_args,
	)
	total_optim_steps = len(all_task_names) * int(source_args.change_freq)
	task_records = []
	stage_records = []
	training_evaluation_records = []
	structure_metrics = {}
	actor_metrics = {}
	process_step = 0
	global_update_count = 0

	print("source run:", run_dir)
	print("source tasks:", all_task_names)
	print("selected tasks:", selected_tasks)
	print("device:", device)
	for selected_idx, (task_idx, task_name) in enumerate(selected_tasks, start=1):
		checkpoint_name = task_checkpoint_name(run_name, task_idx)
		checkpoint_path = os.path.join(
			model_dir,
			f"{checkpoint_name}_online_qrl.pt",
		)
		if not os.path.isfile(checkpoint_path):
			env.close()
			raise FileNotFoundError(
				f"Student checkpoint for task {task_idx} ({task_name}) not found: "
				f"{checkpoint_path}"
			)

		env.set_task(task_name)
		task_buffer = replay_buffer_cls(
			obs_space.shape,
			action_space.shape,
			task_buffer_capacity,
			device,
		)
		student = cqrl.build_fast_agent(
			obs_space,
			action_space,
			device,
			source_args,
			total_optim_steps,
		)
		student.load(model_dir, checkpoint_name)
		student.eval()
		print(
			f"collecting task {selected_idx}/{task_count}: {task_name}",
			f"source_task_idx={task_idx}",
			f"checkpoint={checkpoint_name}",
			f"trajectories={cli_args.store_num}",
		)
		task_stats = collect_task_data(
			env,
			student,
			task_buffer,
			cli_args.store_num,
			cli_args.deterministic,
			writer=writer,
			log_prefix=(
				f"offline_collection/{task_idx}_{base.safe_tag(task_name)}"
			),
			step_offset=process_step,
		)
		process_step += cli_args.store_num
		start_index = len(offline_buffer)
		copied = cqrl.copy_replay_buffer(task_buffer, offline_buffer)
		offline_buffer.set_task_id_range(
			task_idx,
			start_index,
			start_index + copied,
		)
		print(
			f"collected task {selected_idx}/{task_count}: {task_name}",
			f"successful_trajectories={task_stats['successful_episodes']}/"
			f"{task_stats['completed_episodes']}",
			f"success_rate={task_stats['trajectory_success_rate']:.3f}",
			f"transitions={copied}",
		)
		task_record = {
			"task_idx": task_idx,
			"task": task_name,
			"checkpoint": checkpoint_name,
			"start_index": start_index,
			"end_index": start_index + copied,
			"transitions": copied,
			**task_stats,
		}
		task_records.append(task_record)
		expected_transitions = sum(
			record["transitions"] for record in task_records
		)
		if len(offline_buffer) != expected_transitions:
			env.close()
			raise RuntimeError(
				f"Expected {expected_transitions} offline transitions after stage "
				f"{selected_idx}, got {len(offline_buffer)}"
			)
		if len(offline_buffer) < minimum_buffer_size:
			env.close()
			raise ValueError(
				f"Incremental stage {selected_idx} collected {len(offline_buffer)} "
				f"transitions, but qm_min_buffer_size is {minimum_buffer_size}. "
				"Increase --store_num."
			)

		stage_data_dir = os.path.join(
			offline_data_dir,
			f"stage_{selected_idx:02d}_{base.safe_tag(task_name)}",
		)
		offline_buffer.save_data(stage_data_dir)
		base.log_scalar(
			writer,
			"offline_incremental/stage_idx",
			selected_idx,
			process_step,
		)
		base.log_scalar(
			writer,
			"offline_incremental/cumulative_buffer_size",
			len(offline_buffer),
			process_step,
		)
		print(
			f"training meta stage {selected_idx}/{task_count}: {task_name}",
			f"seen_tasks={selected_idx}",
			f"cumulative_transitions={len(offline_buffer)}",
			f"updates={update_steps}",
		)
		stage_update_start = global_update_count
		structure_metrics, actor_metrics, stage_evaluations = train_meta_agent(
			meta_agent,
			offline_buffer,
			student,
			task_buffer,
			cli_args.distill_loss_weight,
			update_steps,
			cli_args.log_interval,
			env,
			selected_tasks[:selected_idx],
			cli_args.train_eval_interval,
			train_eval_runs,
			selected_idx,
			writer=writer,
			step_offset=process_step,
			update_offset=global_update_count,
		)
		del student
		del task_buffer
		training_evaluation_records.extend(stage_evaluations)
		process_step += update_steps
		global_update_count += update_steps
		stage_checkpoint = (
			f"{run_name}_offline_meta_stage{selected_idx}_"
			f"{base.safe_tag(task_name)}"
		)
		meta_agent.save(output_model_dir, stage_checkpoint)
		stage_record = {
			"stage_idx": selected_idx,
			"task_idx": task_idx,
			"task": task_name,
			"seen_tasks": [
				{"task_idx": seen_task_idx, "task": seen_task_name}
				for seen_task_idx, seen_task_name in selected_tasks[:selected_idx]
			],
			"new_transitions": copied,
			"cumulative_transitions": len(offline_buffer),
			"offline_data_dir": stage_data_dir,
			"update_start": stage_update_start,
			"update_end": global_update_count,
			"update_steps": update_steps,
			"meta_checkpoint": os.path.join(
				output_model_dir,
				stage_checkpoint,
			),
			"last_structure_metrics": structure_metrics,
			"last_actor_metrics": actor_metrics,
			"training_evaluation": stage_evaluations,
		}
		stage_records.append(stage_record)
		task_record["incremental_stage_idx"] = selected_idx
		task_record["meta_checkpoint"] = stage_record["meta_checkpoint"]
		print("saved incremental meta checkpoint:", stage_record["meta_checkpoint"])

	offline_buffer.save_data(offline_data_dir)
	print("offline buffer:", offline_data_dir, "transitions:", len(offline_buffer))
	total_trajectories = sum(record["completed_episodes"] for record in task_records)
	total_successful_trajectories = sum(
		record["successful_episodes"] for record in task_records
	)
	overall_trajectory_success_rate = (
		total_successful_trajectories / total_trajectories
	)
	base.log_scalar(
		writer,
		"offline_collection/overall_success_rate",
		overall_trajectory_success_rate,
		process_step,
	)
	base.log_scalar(
		writer,
		"offline_collection/total_successful_trajectories",
		total_successful_trajectories,
		process_step,
	)
	base.log_scalar(
		writer,
		"offline_collection/retained_transitions",
		len(offline_buffer),
		process_step,
	)
	writer.flush()
	print(
		"collection trajectory success:",
		f"{total_successful_trajectories}/{total_trajectories}",
		f"rate={overall_trajectory_success_rate:.3f}",
	)

	meta_checkpoint = f"{run_name}_offline_meta"
	meta_agent.save(output_model_dir, meta_checkpoint)
	evaluation_records = evaluate_meta_agent(
		env,
		meta_agent,
		selected_tasks,
		num_eval_runs,
		writer=writer,
		step_offset=process_step,
	)
	evaluation_path = os.path.join(output_dir, "meta_evaluation.json")
	with open(evaluation_path, "w") as handle:
		json.dump(
			{
				"num_eval_runs_per_task": num_eval_runs,
				"tasks": evaluation_records,
			},
			handle,
			indent=2,
			sort_keys=True,
		)
	evaluation_stats = {
		key: [record[key] for record in evaluation_records]
		for key in evaluation_records[0]
	}
	base.write_stats_csv(
		os.path.join(output_dir, "meta_evaluation.csv"),
		evaluation_stats,
	)

	report = {
		"source_run_dir": run_dir,
		"source_run_name": run_name,
		"source_tasks": all_task_names,
		"selected_tasks": [
			{"task_idx": task_idx, "task": task_name}
			for task_idx, task_name in selected_tasks
		],
		"seed": source_args.seed,
		"device": str(device),
		"deterministic_collection": cli_args.deterministic,
		"store_trajectories_per_task": cli_args.store_num,
		"total_trajectories": total_trajectories,
		"total_successful_trajectories": total_successful_trajectories,
		"overall_trajectory_success_rate": overall_trajectory_success_rate,
		"total_transitions": len(offline_buffer),
		"offline_data_dir": offline_data_dir,
		"logging": log_info,
		"incremental_training": True,
		"meta_update_steps_per_stage": update_steps,
		"total_meta_update_steps": global_update_count,
		"distill_loss_weight": cli_args.distill_loss_weight,
		"train_eval_interval": cli_args.train_eval_interval,
		"train_eval_runs_per_task": train_eval_runs,
		"training_evaluation": training_evaluation_records,
		"meta_learner_overrides": meta_learner_overrides,
		"meta_learner_config": {
			"meta_update_steps_per_stage": update_steps,
			**{
				name: getattr(source_args, name)
				for name in cli_args.meta_learner_arg_names
			},
		},
		"meta_checkpoint": os.path.join(output_model_dir, meta_checkpoint),
		"num_eval_runs_per_task": num_eval_runs,
		"meta_evaluation": evaluation_records,
		"tasks": task_records,
		"incremental_stages": stage_records,
		"last_structure_metrics": structure_metrics,
		"last_actor_metrics": actor_metrics,
	}
	with open(os.path.join(output_dir, "offline_report.json"), "w") as handle:
		json.dump(report, handle, indent=2, sort_keys=True)
	with open(os.path.join(output_dir, "source_run_config.json"), "w") as handle:
		json.dump(source_config, handle, indent=2, sort_keys=True)

	writer.close()
	env.close()
	print("saved meta checkpoint:", report["meta_checkpoint"])
	print("saved meta evaluation:", evaluation_path)
	print("saved report:", os.path.join(output_dir, "offline_report.json"))


if __name__ == "__main__":
	sys.exit(main())

'''
	python -m online_continual.CQRL_offline_finetuning \
  --run_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --store_num 2000 \
  --meta_update_steps 10000 \
  --qm_contrastive_coef 0.1 \
  --num_eval_runs 100 \
	--bc_alpha 0.1 \
	--tasks push pick-and-place \
	--qm_backup_coef 1.0 \
	--qm_q_loss_coef 1.0 \
	--qm_action_invariance_coef 0.0 \
	--train_eval_interval 1000 \
--train_eval_runs 10 \
  --gpu 1 \
  --qm_ranking_coef 0.1 \
	--qm_ranking_margin 0.05 \
	--qm_current_batch_ratio 0.5 \
	--distill_loss_weight 0.5

python -m online_continual.CQRL_offline_finetuning \
  --run_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --store_num 1500 \
  --meta_update_steps 30000 \
  --qm_contrastive_coef 0.1 \
  --qm_nce_mode backward_nce \
  --qm_backup_coef 0.5 \
	--qm_q_loss_coef 1.0 \
  --num_eval_runs 100 \
	--bc_alpha 0.0 \
	--tasks push pick-and-place \
	--qm_action_invariance_coef 0.0 \
	--train_eval_interval 1000 \
--train_eval_runs 10 \
  --gpu 1 \
  --qm_ranking_coef 0.1 \
	--qm_ranking_margin 0.05 \
	--qm_current_batch_ratio 0.5 \
	--distill_loss_weight 1.0

	-----stable version
python -m online_continual.CQRL_offline_finetuning \
  --run_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --store_num 1500 \
  --meta_update_steps 120000 \
  --qm_contrastive_coef 1.0 \
  --qm_nce_mode backward_nce \
  --qm_backup_coef 1.0 \
	--qm_q_loss_coef 1.0 \
  --num_eval_runs 100 \
	--bc_alpha 1.0 \
	--tasks push pick-and-place \
	--qm_action_invariance_coef 1.0 \
	--qm_transition_consistency_coef 0.0 \
	--train_eval_interval 1000 \
--train_eval_runs 10 \
  --gpu 1 \
  --qm_ranking_coef 0.0 \
	--qm_ranking_margin 0.05 \
	--qm_current_batch_ratio 0.5 \
	--distill_loss_weight 0.0

python -m online_continual.CQRL_offline_finetuning \
  --run_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --store_num 1500 \
  --meta_update_steps 60000 \
  --qm_contrastive_coef 0.0 \
  --qm_nce_mode backward_nce \
  --qm_backup_coef 1.0 \
	--qm_q_loss_coef 1.0 \
  --num_eval_runs 100 \
	--bc_alpha 1.0 \
	--tasks push pick-and-place \
	--qm_action_invariance_coef 1.0 \
	--qm_transition_consistency_coef 0.0 \
	--train_eval_interval 1000 \
--train_eval_runs 10 \
  --gpu 0 \
  --qm_ranking_coef 0.0 \
	--qm_ranking_margin 0.05 \
	--qm_current_batch_ratio 0.5 \
	--distill_loss_weight 0.0

	python -m online_continual.CQRL_offline_finetuning \
  --run_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --store_num 1500 \
  --meta_update_steps 120000 \
  --qm_contrastive_coef 0.0 \
  --qm_nce_mode backward_nce \
  --qm_backup_coef 0.0 \
	--qm_q_loss_coef 0.0 \
  --num_eval_runs 100 \
	--bc_alpha 1.0 \
	--tasks push pick-and-place \
	--qm_action_invariance_coef 0.0 \
	--qm_transition_consistency_coef 0.0 \
	--train_eval_interval 1000 \
--train_eval_runs 10 \
  --gpu 1 \
  --qm_ranking_coef 0.0 \
	--qm_ranking_margin 0.05 \
	--qm_current_batch_ratio 0.5 \
	--distill_loss_weight 0.0

	
python -m online_continual.CQRL_offline_finetuning \
  --run_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --store_num 1500 \
  --meta_update_steps 120000 \
  --qm_contrastive_coef 1.0 \
  --qm_nce_mode backward_nce \
  --qm_backup_coef 1.0 \
	--qm_q_loss_coef 1.0 \
  --num_eval_runs 100 \
	--bc_alpha 0.1 \
	--qm_action_invariance_coef 1.0 \
	--qm_transition_consistency_coef 0.0 \
	--train_eval_interval 1000 \
	--train_eval_runs 10 \
  	--gpu 0 \
  	--qm_ranking_coef 0.0 \
	--qm_ranking_margin 0.05 \
	--qm_current_batch_ratio 0.5 \
	--distill_loss_weight 0.00



'''
