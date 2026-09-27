"""Distill an offline CQRL meta agent into a fast student and test zero-shot."""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

try:
	from . import CQRL as cqrl
	from .CQRL_offline_ft_zeroshot import (
		build_agent_args,
		evaluate as evaluate_zero_shot,
		load_json,
		make_env_kwargs,
		output_paths,
		report_task_names,
		resolve_checkpoint,
		validate_targets,
	)
except ImportError:
	import CQRL as cqrl
	from CQRL_offline_ft_zeroshot import (
		build_agent_args,
		evaluate as evaluate_zero_shot,
		load_json,
		make_env_kwargs,
		output_paths,
		report_task_names,
		resolve_checkpoint,
		validate_targets,
	)

from fetch_env import normalize_fetch_task_name


base = cqrl.base
STUDENT_CHECKPOINT_SUFFIX = "_online_qrl.pt"


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Load an offline-finetuned CQRL meta agent, distill it into a fast "
			"student with CQRL's regularized update, and zero-shot evaluate the "
			"student without target-task updates."
		)
	)
	parser.add_argument(
		"--offline_dir",
		required=True,
		help="Offline finetuning output directory containing offline_report.json.",
	)
	parser.add_argument(
		"--target_tasks",
		nargs="+",
		required=True,
		metavar="TASK",
		help="One or more Fetch tasks for zero-shot evaluation.",
	)
	parser.add_argument(
		"--checkpoint",
		default="final",
		help=(
			"Meta checkpoint selector: final, stageN, a checkpoint basename, or "
			"a path to a checkpoint stem/component file."
		),
	)
	parser.add_argument(
		"--offline_data",
		default=None,
		help=(
			"Replay-buffer directory used for distillation. Defaults to the "
			"selected stage data or the final offline_data directory."
		),
	)
	parser.add_argument(
		"--student_checkpoint",
		default="random",
		help=(
			"Student initialization: random, final, taskN, a checkpoint basename, "
			"or a path to an _online_qrl.pt file/stem. Random is zero-shot clean."
		),
	)
	parser.add_argument(
		"--student_output",
		default=None,
		help=(
			"Distilled student checkpoint stem. Defaults to offline_dir/model/ "
			"with a generated name."
		),
	)
	parser.add_argument(
		"--no_save_student",
		action="store_true",
		help="Do not save the distilled student checkpoint.",
	)
	parser.add_argument(
		"--distill_steps",
		type=int,
		default=None,
		help="Distillation updates; defaults to warmup_steps in the source config.",
	)
	parser.add_argument(
		"--lambda_reg",
		"--distill_loss_weight",
		dest="lambda_reg",
		type=float,
		default=None,
		help="CQRL KL regularization weight; defaults to source lambda_reg.",
	)
	parser.add_argument(
		"--distill_seed",
		type=int,
		default=None,
		help="Student initialization/distillation seed; defaults to source seed.",
	)
	parser.add_argument(
		"--log_interval",
		type=int,
		default=1000,
		help="Print distillation metrics every this many updates.",
	)
	parser.add_argument(
		"--config",
		default=None,
		help="Source run config override; defaults to source_run_config.json.",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="CSV output path; a JSON file is written beside it.",
	)
	eval_seed_group = parser.add_mutually_exclusive_group()
	eval_seed_group.add_argument("--eval_seed", type=int, default=None)
	eval_seed_group.add_argument("--eval_seeds", type=int, nargs="+", default=None)
	parser.add_argument(
		"--num_eval_runs",
		type=int,
		default=None,
		help="Evaluation episodes per task; defaults to the source configuration.",
	)
	parser.add_argument(
		"--gpu",
		default=None,
		help="CUDA device override; defaults to the source configuration.",
	)
	parser.add_argument(
		"--eval_slide_goal_scale",
		type=float,
		default=None,
		help="Evaluation-only FetchSlide goal-distance scale.",
	)
	parser.add_argument(
		"--eval_goal_outer_range",
		type=float,
		default=None,
		help="Evaluation-only Push/Pick x-y goal half-range greater than 0.15.",
	)
	parser.add_argument(
		"--eval_initial_state_outer_range",
		type=float,
		default=None,
		help=(
			"Evaluation-only Push/Pick initial-object x-y half-range greater "
			"than 0.15."
		),
	)
	parser.add_argument(
		"--allow_seen_tasks",
		action="store_true",
		help="Allow evaluation targets used to train the selected meta checkpoint.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Load the config, replay buffer, teacher, and student without training.",
	)
	args = parser.parse_args()

	args.offline_dir = os.path.abspath(args.offline_dir)
	args.config = os.path.abspath(
		args.config or os.path.join(args.offline_dir, "source_run_config.json")
	)
	args.target_tasks = list(
		dict.fromkeys(normalize_fetch_task_name(task) for task in args.target_tasks)
	)
	if args.distill_steps is not None and args.distill_steps <= 0:
		parser.error("--distill_steps must be positive")
	if args.lambda_reg is not None and args.lambda_reg <= 0.0:
		parser.error("--lambda_reg must be positive")
	if args.log_interval <= 0:
		parser.error("--log_interval must be positive")
	if args.num_eval_runs is not None and args.num_eval_runs <= 0:
		parser.error("--num_eval_runs must be positive")
	if args.eval_slide_goal_scale is not None and not (
		0.0 < args.eval_slide_goal_scale <= 1.0
	):
		parser.error("--eval_slide_goal_scale must be in (0, 1]")
	validate_outer_range_args(parser, args)
	return args


def validate_outer_range_args(parser, args):
	for option, value in (
		("--eval_goal_outer_range", args.eval_goal_outer_range),
		(
			"--eval_initial_state_outer_range",
			args.eval_initial_state_outer_range,
		),
	):
		if value is None:
			continue
		if value <= 0.15:
			parser.error(f"{option} must be greater than 0.15")
		unsupported_tasks = [
			task
			for task in args.target_tasks
			if task not in {"push", "pick-and-place"}
		]
		if unsupported_tasks:
			parser.error(f"{option} only supports push and pick-and-place")


def stage_entry(report, selector):
	match = re.fullmatch(r"stage(\d+)", selector)
	if match is None:
		return None
	stage_idx = int(match.group(1))
	return next(
		(
			entry
			for entry in report.get("incremental_stages", [])
			if int(entry["stage_idx"]) == stage_idx
		),
		None,
	)


def resolve_offline_data(cli_args, report):
	if cli_args.offline_data is not None:
		data_dir = cli_args.offline_data
	else:
		stage = stage_entry(report, cli_args.checkpoint)
		data_dir = (
			stage.get("offline_data_dir")
			if stage is not None
			else report.get("offline_data_dir")
		)
		if data_dir is None:
			data_dir = os.path.join(cli_args.offline_dir, "offline_data")
	if not os.path.isabs(data_dir):
		data_dir = os.path.join(cli_args.offline_dir, data_dir)
	data_dir = os.path.abspath(data_dir)
	if os.path.basename(data_dir) == "offline_data.npz":
		data_dir = os.path.dirname(data_dir)
	data_path = os.path.join(data_dir, "offline_data.npz")
	if not os.path.isfile(data_path):
		raise FileNotFoundError(f"Offline replay data not found: {data_path}")
	with np.load(data_path) as data:
		buffer_size = len(data["obses"])
		obs_shape = tuple(data["obses"].shape[1:])
		action_shape = tuple(data["actions"].shape[1:])
	if buffer_size == 0:
		raise ValueError(f"Offline replay data is empty: {data_path}")
	return data_dir, buffer_size, obs_shape, action_shape


def strip_student_suffix(path):
	if path.endswith(STUDENT_CHECKPOINT_SUFFIX):
		return path[: -len(STUDENT_CHECKPOINT_SUFFIX)]
	return path


def resolve_student_checkpoint(selector, report, offline_dir):
	if selector == "random":
		return None

	source_run_name = report.get("source_run_name")
	source_run_dir = report.get("source_run_dir")
	if source_run_dir is None:
		source_run_dir = os.path.dirname(offline_dir)
	source_model_dir = os.path.join(os.path.abspath(source_run_dir), "model")
	if selector == "final":
		if source_run_name is None:
			raise ValueError("offline_report.json does not define source_run_name.")
		checkpoint = os.path.join(
			source_model_dir,
			f"{source_run_name}_final_fast",
		)
	else:
		task_match = re.fullmatch(r"task(\d+)", selector)
		if task_match:
			if source_run_name is None:
				raise ValueError(
					"offline_report.json does not define source_run_name."
				)
			checkpoint = os.path.join(
				source_model_dir,
				f"{source_run_name}_task{int(task_match.group(1))}_fast",
			)
		elif os.path.isabs(selector) or os.path.dirname(selector):
			checkpoint = os.path.abspath(selector)
		else:
			candidates = (
				os.path.join(source_model_dir, selector),
				os.path.join(offline_dir, "model", selector),
			)
			checkpoint = next(
				(
					candidate
					for candidate in candidates
					if os.path.isfile(
						strip_student_suffix(candidate)
						+ STUDENT_CHECKPOINT_SUFFIX
					)
				),
				candidates[0],
			)

	checkpoint = strip_student_suffix(checkpoint)
	checkpoint_path = checkpoint + STUDENT_CHECKPOINT_SUFFIX
	if not os.path.isfile(checkpoint_path):
		raise FileNotFoundError(
			f"Student initialization checkpoint not found: {checkpoint_path}"
		)
	return os.path.dirname(checkpoint), os.path.basename(checkpoint), checkpoint


def distilled_student_output(cli_args, teacher_name, distill_steps, lambda_reg):
	if cli_args.student_output is None:
		student_name = (
			f"{teacher_name}_distilled_student_steps{distill_steps}_"
			f"lambda{base.safe_tag(lambda_reg)}"
		)
		checkpoint = os.path.join(cli_args.offline_dir, "model", student_name)
	else:
		checkpoint = cli_args.student_output
		if not os.path.isabs(checkpoint):
			checkpoint = os.path.abspath(checkpoint)
		checkpoint = strip_student_suffix(checkpoint)
		student_name = os.path.basename(checkpoint)
	return os.path.dirname(checkpoint), student_name, checkpoint


def scalar_metrics(metrics):
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


def distill_student(
	student,
	teacher,
	replay_buffer,
	distill_steps,
	lambda_reg,
	log_interval,
):
	student.train(True)
	teacher.eval()
	history = []
	last_metrics = {}
	for update_idx in range(1, distill_steps + 1):
		metrics = student.update(
			replay_buffer,
			update_idx,
			teacher_agent=teacher,
			regularization_weight=lambda_reg,
		)
		if (
			update_idx == 1
			or update_idx % log_interval == 0
			or update_idx == distill_steps
		):
			last_metrics = scalar_metrics(metrics)
			history.append({"update": update_idx, **last_metrics})
			print(
				f"student distillation {update_idx}/{distill_steps}",
				{
					key: last_metrics[key]
					for key in ("loss", "cqrl_kl", "cqrl_regularized")
					if key in last_metrics
				},
			)
	return last_metrics, history


def main():
	cli_args = parse_args()
	report_path = os.path.join(cli_args.offline_dir, "offline_report.json")
	report = load_json(report_path, "Offline report")
	agent_args = build_agent_args(cli_args, report)
	agent_args.training_slide_goal_scale = float(agent_args.slide_goal_scale)
	if cli_args.eval_slide_goal_scale is not None:
		agent_args.slide_goal_scale = cli_args.eval_slide_goal_scale

	distill_steps = (
		cli_args.distill_steps
		if cli_args.distill_steps is not None
		else int(agent_args.warmup_steps)
	)
	lambda_reg = (
		cli_args.lambda_reg
		if cli_args.lambda_reg is not None
		else float(agent_args.lambda_reg)
	)
	if distill_steps <= 0:
		raise ValueError("The resolved distillation step count must be positive.")
	if lambda_reg <= 0.0:
		raise ValueError("The resolved CQRL lambda_reg must be positive.")
	distill_seed = (
		cli_args.distill_seed
		if cli_args.distill_seed is not None
		else int(agent_args.seed)
	)

	model_tasks = report_task_names(report, agent_args)
	teacher_dir, teacher_name, trained_tasks = resolve_checkpoint(
		cli_args.offline_dir,
		report,
		cli_args.checkpoint,
	)
	model_observation_dim = validate_targets(
		model_tasks,
		trained_tasks,
		cli_args.target_tasks,
		cli_args.allow_seen_tasks,
	)
	data_dir, buffer_size, data_obs_shape, data_action_shape = (
		resolve_offline_data(cli_args, report)
	)
	student_initialization = resolve_student_checkpoint(
		cli_args.student_checkpoint,
		report,
		cli_args.offline_dir,
	)

	os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_args.gpu)
	base.set_seed_everywhere(distill_seed)
	evaluation_tasks = list(dict.fromkeys(model_tasks + cli_args.target_tasks))
	agent_args.task_order = ",".join(evaluation_tasks)
	agent_args.base_task_name = None
	agent_args.eval_goal_outer_range = cli_args.eval_goal_outer_range
	agent_args.eval_initial_state_outer_range = (
		cli_args.eval_initial_state_outer_range
	)

	env = base.FetchGoalEnvSequence(**make_env_kwargs(agent_args))
	try:
		obs_space = base.vector_observation_space(env.env.observation_space)
		action_space = env.env.action_space
		if obs_space.shape[0] != model_observation_dim:
			raise RuntimeError(
				f"Evaluation environment observation dim {obs_space.shape[0]} does "
				f"not match checkpoint dim {model_observation_dim}."
			)
		if tuple(obs_space.shape) != data_obs_shape:
			raise ValueError(
				f"Offline observation shape {data_obs_shape} does not match agent "
				f"shape {tuple(obs_space.shape)}."
			)
		if tuple(action_space.shape) != data_action_shape:
			raise ValueError(
				f"Offline action shape {data_action_shape} does not match agent "
				f"shape {tuple(action_space.shape)}."
			)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		teacher = base.build_meta_agent(
			obs_space.shape[0],
			action_space.shape[0],
			device,
			agent_args,
		)
		teacher.load(teacher_dir, teacher_name)
		teacher.eval()
		total_optim_steps = len(model_tasks) * int(agent_args.change_freq)
		student = cqrl.build_fast_agent(
			obs_space,
			action_space,
			device,
			agent_args,
			total_optim_steps,
		)
		if student_initialization is not None:
			student.load(
				student_initialization[0],
				student_initialization[1],
			)

		replay_buffer_cls = (
			base.ReplayBufferMetric
			if agent_args.replay_buffer_mode == "her"
			else base.ReplayBufferMetricNoHER
		)
		replay_buffer = replay_buffer_cls(
			obs_space.shape,
			action_space.shape,
			buffer_size,
			device,
		)
		if not replay_buffer.load_data(data_dir):
			raise RuntimeError(f"Failed to load offline replay data: {data_dir}")
	finally:
		env.close()

	student_output_dir, student_name, student_checkpoint = (
		distilled_student_output(
			cli_args,
			teacher_name,
			distill_steps,
			lambda_reg,
		)
	)
	print("offline_dir:", cli_args.offline_dir)
	print("teacher_checkpoint:", os.path.join(teacher_dir, teacher_name))
	print("teacher_trained_tasks:", trained_tasks)
	print(
		"student_initialization:",
		"random" if student_initialization is None else student_initialization[2],
	)
	print("offline_data:", data_dir, "transitions:", len(replay_buffer))
	print("distill_steps:", distill_steps)
	print("lambda_reg:", lambda_reg)
	print("distill_seed:", distill_seed)
	print("target_tasks:", cli_args.target_tasks)
	print("eval_seeds:", agent_args.eval_seeds)
	print("device:", device)
	if cli_args.dry_run:
		print("dry run complete: teacher, student, and replay buffer loaded")
		return 0

	last_metrics, distillation_history = distill_student(
		student,
		teacher,
		replay_buffer,
		distill_steps,
		lambda_reg,
		cli_args.log_interval,
	)
	student.eval()
	if not cli_args.no_save_student:
		student.save(student_output_dir, student_name)
		print("distilled student checkpoint:", student_checkpoint)

	records, stats = evaluate_zero_shot(
		agent_args,
		student,
		student_name,
		trained_tasks,
		cli_args.target_tasks,
	)
	csv_path, json_path = output_paths(cli_args, student_name)
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	base.write_stats_csv(csv_path, stats)
	with open(json_path, "w") as handle:
		json.dump(
			{
				"offline_dir": cli_args.offline_dir,
				"teacher_checkpoint": os.path.join(
					teacher_dir,
					teacher_name,
				),
				"teacher_trained_tasks": trained_tasks,
				"student_initialization": (
					"random"
					if student_initialization is None
					else student_initialization[2]
				),
				"student_checkpoint": (
					None if cli_args.no_save_student else student_checkpoint
				),
				"offline_data_dir": data_dir,
				"offline_transitions": len(replay_buffer),
				"distill_steps": distill_steps,
				"lambda_reg": lambda_reg,
				"distill_seed": distill_seed,
				"last_distillation_metrics": last_metrics,
				"distillation_history": distillation_history,
				"model_tasks": model_tasks,
				"target_tasks": cli_args.target_tasks,
				"zero_shot": True,
				"training_slide_goal_scale": (
					agent_args.training_slide_goal_scale
				),
				"eval_slide_goal_scale": agent_args.slide_goal_scale,
				"eval_goal_outer_range": agent_args.eval_goal_outer_range,
				"eval_initial_state_outer_range": (
					agent_args.eval_initial_state_outer_range
				),
				"results": records,
			},
			handle,
			indent=2,
			sort_keys=True,
		)
	print("results csv:", csv_path)
	print("results json:", json_path)
	return 0


if __name__ == "__main__":
	sys.exit(main())
