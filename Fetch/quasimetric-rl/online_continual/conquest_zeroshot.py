"""Evaluate a trained Fetch agent at fixed or randomly sampled OOD positions."""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

if __package__:
	from .conquest_eval import (
		build_eval_agent,
		checkpoint_plan,
		find_training_config,
		load_training_config,
		parse_checkpoint,
		resolve_agent_kind,
		validate_checkpoints,
	)
	from .main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		evaluate_and_log,
		make_env_kwargs,
		safe_tag,
		set_seed_everywhere,
		write_stats_csv,
	)
else:
	from conquest_eval import (
		build_eval_agent,
		checkpoint_plan,
		find_training_config,
		load_training_config,
		parse_checkpoint,
		resolve_agent_kind,
		validate_checkpoints,
	)
	from main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		evaluate_and_log,
		make_env_kwargs,
		safe_tag,
		set_seed_everywhere,
		write_stats_csv,
	)

from fetch_env import FETCH_OBSERVATION_DIMS, normalize_fetch_task_name


def parse_args():
	bootstrap_parser = argparse.ArgumentParser(add_help=False)
	bootstrap_parser.add_argument("--run_dir", type=str, default=None)
	bootstrap_parser.add_argument("--config", type=str, default=None)
	bootstrap_args, _ = bootstrap_parser.parse_known_args()

	run_dir = os.path.abspath(bootstrap_args.run_dir) if bootstrap_args.run_dir else None
	config_path = find_training_config(run_dir, bootstrap_args.config) if run_dir else None
	training_config = load_training_config(config_path)

	parser = argparse.ArgumentParser(
		description=(
			"Load a trained continual Fetch agent and evaluate fixed positions or "
			"randomly sampled positions outside the native training ranges."
		)
	)
	parser.add_argument("--run_dir", type=str, required=True)
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--model_dir", type=str, default=None)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument(
		"--checkpoint",
		type=parse_checkpoint,
		default="final",
		help="Choose all, final, latest, taskN, or stepN.",
	)
	parser.add_argument(
		"--agent_kind",
		choices=["auto", "meta", "student", "fast"],
		default="meta",
	)
	target_group = parser.add_mutually_exclusive_group(required=True)
	target_group.add_argument(
		"--target_task",
		type=str,
		help="Object task to test: push, pick-and-place, or slide.",
	)
	target_group.add_argument(
		"--target_tasks",
		type=str,
		nargs="+",
		help="One or more object tasks to test.",
	)
	parser.add_argument(
		"--initial_position",
		type=float,
		nargs=3,
		default=None,
		metavar=("X", "Y", "Z"),
		help="Absolute MuJoCo world coordinates for the initial object position.",
	)
	parser.add_argument(
		"--goal_position",
		type=float,
		nargs=3,
		default=None,
		metavar=("X", "Y", "Z"),
		help="Absolute MuJoCo world coordinates for the desired object position.",
	)
	parser.add_argument(
		"--eval_goal_outer_range",
		type=float,
		default=None,
		help=(
			"Push/Pick evaluation x-y half-range. Each goal is randomly sampled "
			"outside the native +/-0.15 training range."
		),
	)
	parser.add_argument(
		"--eval_initial_state_outer_range",
		type=float,
		default=None,
		help=(
			"Push/Pick evaluation x-y half-range. Each initial object state is "
			"randomly sampled outside the native +/-0.15 training range."
		),
	)
	parser.add_argument(
		"--eval_slide_goal_scale",
		type=float,
		default=None,
		help=(
			"FetchSlide goal-distance scale used only for evaluation; for example, "
			"use 1.0 for a model trained with 0.795."
		),
	)
	parser.add_argument(
		"--allow_in_distribution",
		action="store_true",
		help=(
			"Allow x-y coordinates inside the native training ranges. By default "
			"both coordinates must be outside their respective ranges."
		),
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Validate coordinates and checkpoint loading without running episodes.",
	)
	eval_seed_group = parser.add_mutually_exclusive_group()
	eval_seed_group.add_argument("--eval_seed", type=int, default=None)
	eval_seed_group.add_argument("--eval_seeds", type=int, nargs="+", default=None)
	add_common_args(parser)
	add_sac_args(parser)
	add_student_qrl_args(parser)
	add_quasimetric_args(parser)
	parser.set_defaults(**training_config)
	args = parser.parse_args()

	args.run_dir = os.path.abspath(args.run_dir)
	args.config = config_path
	args.model_dir = os.path.abspath(
		args.model_dir or os.path.join(args.run_dir, "model")
	)
	target_tasks = args.target_tasks or [args.target_task]
	args.target_tasks = list(
		dict.fromkeys(normalize_fetch_task_name(task) for task in target_tasks)
	)
	args.training_slide_goal_scale = float(args.slide_goal_scale)
	fixed_requested = args.initial_position is not None or args.goal_position is not None
	range_requested = (
		args.eval_initial_state_outer_range is not None
		or args.eval_goal_outer_range is not None
	)
	slide_scale_requested = args.eval_slide_goal_scale is not None
	if fixed_requested and (range_requested or slide_scale_requested):
		parser.error(
			"Fixed positions and distribution-shift sampling are mutually exclusive"
		)
	if range_requested and slide_scale_requested:
		parser.error(
			"Slide goal-scale and Push/Pick outer-range tests must use separate commands"
		)
	if not fixed_requested and not range_requested and not slide_scale_requested:
		parser.error(
			"Specify fixed positions, outer ranges, or --eval_slide_goal_scale"
		)
	if fixed_requested:
		if len(args.target_tasks) != 1:
			parser.error("Fixed positions support exactly one target task")
		if args.initial_position is None or args.goal_position is None:
			parser.error("--initial_position and --goal_position must be used together")
		if args.target_tasks[0] == "reach":
			parser.error(
				"--initial_position requires an object task; FetchReach has no object"
			)
		args.initial_position = np.asarray(args.initial_position, dtype=np.float64)
		args.goal_position = np.asarray(args.goal_position, dtype=np.float64)
		if (
			np.linalg.norm(args.initial_position - args.goal_position)
			<= args.gc_success_threshold
		):
			parser.error(
				"Initial and goal positions are already within the success threshold"
			)
		args.evaluation_mode = "fixed"
	elif range_requested:
		unsupported_tasks = [
			task
			for task in args.target_tasks
			if task not in {"push", "pick-and-place"}
		]
		if unsupported_tasks:
			parser.error(
				"Outer-range random sampling only supports push and pick-and-place; "
				+ "unsupported: "
				+ ", ".join(unsupported_tasks)
			)
		for name in ("eval_initial_state_outer_range", "eval_goal_outer_range"):
			value = getattr(args, name)
			if value is not None and value <= 0.15:
				parser.error(f"--{name} must be greater than 0.15")
		args.evaluation_mode = "random_outer_range"
	else:
		if args.target_tasks != ["slide"]:
			parser.error(
				"--eval_slide_goal_scale requires --target_task slide"
			)
		if not 0.0 < args.eval_slide_goal_scale <= 1.0:
			parser.error("--eval_slide_goal_scale must be in (0, 1]")
		args.slide_goal_scale = args.eval_slide_goal_scale
		args.evaluation_mode = "slide_goal_scale"
	if args.num_eval_runs <= 0:
		parser.error("--num_eval_runs must be positive")
	if args.eval_seeds is None:
		args.eval_seeds = [args.eval_seed if args.eval_seed is not None else args.seed]

	if args.output is None:
		run_name = os.path.basename(os.path.normpath(args.run_dir))
		task_tag = "-".join(args.target_tasks)
		if args.evaluation_mode == "fixed":
			start_tag = safe_tag(
				"-".join(f"{value:.4g}" for value in args.initial_position)
			)
			goal_tag = safe_tag(
				"-".join(f"{value:.4g}" for value in args.goal_position)
			)
			filename = (
				f"{run_name}_fixed-ood_{task_tag}_{args.checkpoint}_"
				f"start-{start_tag}_goal-{goal_tag}.csv"
			)
		elif args.evaluation_mode == "random_outer_range":
			range_tags = []
			if args.eval_initial_state_outer_range is not None:
				range_tags.append(
					"initial-" + safe_tag(args.eval_initial_state_outer_range)
				)
			if args.eval_goal_outer_range is not None:
				range_tags.append("goal-" + safe_tag(args.eval_goal_outer_range))
			filename = (
				f"{run_name}_random-ood_{task_tag}_{args.checkpoint}_"
				+ "_".join(range_tags)
				+ ".csv"
			)
		else:
			filename = (
				f"{run_name}_slide-ood_{args.checkpoint}_scale-"
				f"{safe_tag(args.eval_slide_goal_scale)}.csv"
			)
		args.output = os.path.join(args.run_dir, filename)
	args.output = os.path.abspath(args.output)
	return args


def custom_env_kwargs(args):
	kwargs = make_env_kwargs(args)
	kwargs["eval_initial_position"] = args.initial_position
	kwargs["eval_goal_position"] = args.goal_position
	kwargs["eval_initial_state_outer_range"] = args.eval_initial_state_outer_range
	kwargs["eval_goal_outer_range"] = args.eval_goal_outer_range
	return kwargs


def coordinate_profile(env, args, target_task):
	env.set_task(target_task)
	base_env = env.env.unwrapped
	initial_center = np.asarray(base_env.initial_gripper_xpos[:2], dtype=np.float64)
	target_offset = np.asarray(base_env.target_offset, dtype=np.float64)
	if target_offset.ndim == 0:
		target_offset = np.full(3, target_offset, dtype=np.float64)
	goal_center = (
		np.asarray(base_env.initial_gripper_xpos[:3], dtype=np.float64)
		+ target_offset
	)[:2]
	initial_range = float(base_env.obj_range)
	goal_range = float(base_env.target_range)
	profile = {
		"evaluation_mode": args.evaluation_mode,
		"training_slide_goal_scale": args.training_slide_goal_scale,
		"eval_slide_goal_scale": args.slide_goal_scale,
		"initial_center_x": float(initial_center[0]),
		"initial_center_y": float(initial_center[1]),
		"goal_center_x": float(goal_center[0]),
		"goal_center_y": float(goal_center[1]),
		"training_initial_xy_range": initial_range,
		"training_goal_xy_range": goal_range,
		"eval_initial_state_outer_range": args.eval_initial_state_outer_range,
		"eval_goal_outer_range": args.eval_goal_outer_range,
	}
	if args.evaluation_mode in {"random_outer_range", "slide_goal_scale"}:
		profile["initial_outside_training_range"] = (
			args.eval_initial_state_outer_range is not None
		)
		profile["goal_outside_training_range"] = (
			args.eval_goal_outer_range is not None
		)
		profile["goal_distribution_shifted"] = bool(
			args.evaluation_mode == "slide_goal_scale"
			and args.slide_goal_scale != args.training_slide_goal_scale
		)
		return profile

	initial_offset = float(
		np.max(np.abs(args.initial_position[:2] - initial_center))
	)
	goal_offset = float(np.max(np.abs(args.goal_position[:2] - goal_center)))
	profile.update(
		{
			"initial_xy_offset": initial_offset,
			"goal_xy_offset": goal_offset,
			"initial_outside_training_range": initial_offset > initial_range,
			"goal_outside_training_range": goal_offset > goal_range,
		}
	)
	if not args.allow_in_distribution:
		inside = []
		if not profile["initial_outside_training_range"]:
			inside.append(
				f"initial_position x-y offset {initial_offset:.6f} <= {initial_range:.6f}"
			)
		if not profile["goal_outside_training_range"]:
			inside.append(
				f"goal_position x-y offset {goal_offset:.6f} <= {goal_range:.6f}"
			)
		if inside:
			raise ValueError(
				"Coordinates are not outside the native training support: "
				+ "; ".join(inside)
				+ ". Pass --allow_in_distribution to test them intentionally."
			)
	return profile


def validate_evaluation_sample(env, args, target_task, profile):
	test_env = env._wrap_env(target_task, eval_mode=True)
	try:
		_, info = test_env.reset(seed=args.eval_seeds[0])
		actual_initial = info.get("fetch_achieved_goal", info.get("achieved_goal"))
		actual_goal = info.get("fetch_desired_goal", info.get("desired_goal"))
		actual_initial = np.asarray(actual_initial, dtype=np.float64)[:3]
		actual_goal = np.asarray(actual_goal, dtype=np.float64)[:3]
		if args.evaluation_mode == "fixed":
			np.testing.assert_allclose(
				actual_initial,
				args.initial_position,
				atol=1e-6,
			)
			np.testing.assert_allclose(actual_goal, args.goal_position, atol=1e-6)
		else:
			if args.eval_initial_state_outer_range is not None:
				initial_offset = np.max(
					np.abs(
						actual_initial[:2]
						- np.array(
							[profile["initial_center_x"], profile["initial_center_y"]]
						)
					)
				)
				assert profile["training_initial_xy_range"] < initial_offset
				assert initial_offset <= args.eval_initial_state_outer_range + 1e-6
			if args.eval_goal_outer_range is not None:
				goal_offset = np.max(
					np.abs(
						actual_goal[:2]
						- np.array(
							[profile["goal_center_x"], profile["goal_center_y"]]
						)
					)
				)
				assert profile["training_goal_xy_range"] < goal_offset
				assert goal_offset <= args.eval_goal_outer_range + 1e-6
		return actual_initial, actual_goal
	finally:
		test_env.close()


def append_record(stats, record):
	for key, value in record.items():
		stats[key].append(value)


def evaluate_seed(args, agent, plan, agent_kind, eval_seed, profiles, stats):
	args.seed = eval_seed
	set_seed_everywhere(eval_seed)
	env = FetchGoalEnvSequence(**custom_env_kwargs(args))
	try:
		for stage_idx, model_name, _ in plan:
			print(f"Loading checkpoint: {model_name}")
			agent.load(args.model_dir, model_name)
			for target_task in args.target_tasks:
				env.set_task(target_task)
				profile = profiles[target_task]
				eval_metrics = evaluate_and_log(
					env,
					agent,
					None,
					args.evaluation_mode,
					stage_idx,
					args.num_eval_runs,
					reseed_each_episode=False,
				)
				record = {
					"target_task": target_task,
					"seed": eval_seed,
					"num_eval_runs": args.num_eval_runs,
					"checkpoint": model_name,
					"checkpoint_stage": stage_idx,
					"method": args.method,
					"agent_kind": agent_kind,
					**profile,
					**eval_metrics,
				}
				if args.evaluation_mode == "fixed":
					record.update(
						{
							"initial_x": float(args.initial_position[0]),
							"initial_y": float(args.initial_position[1]),
							"initial_z": float(args.initial_position[2]),
							"goal_x": float(args.goal_position[0]),
							"goal_y": float(args.goal_position[1]),
							"goal_z": float(args.goal_position[2]),
							"start_goal_distance": float(
								np.linalg.norm(
									args.initial_position - args.goal_position
								)
							),
						}
					)
				append_record(stats, record)
				print(
					f"{args.evaluation_mode} {target_task}, seed {eval_seed}: "
					f"success {eval_metrics['success_mean']:.3f} +/- "
					f"{eval_metrics['success_std']:.3f}, gc_success "
					f"{eval_metrics['gc_success_mean']:.3f} +/- "
					f"{eval_metrics['gc_success_std']:.3f}, return "
					f"{eval_metrics['return_mean']:.3f} +/- "
					f"{eval_metrics['return_std']:.3f}"
				)
	finally:
		env.close()


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	args.seed = args.eval_seeds[0]
	set_seed_everywhere(args.seed)

	env = FetchGoalEnvSequence(**custom_env_kwargs(args))
	try:
		training_tasks = list(env.env_list)
		for target_task in args.target_tasks:
			if FETCH_OBSERVATION_DIMS[target_task] > env.fetch_observation_dim:
				raise ValueError(
					f"Target task {target_task!r} requires observation dim "
					f"{FETCH_OBSERVATION_DIMS[target_task]}, but the checkpoint "
					f"was built for dim {env.fetch_observation_dim}."
				)
		agent_kind = resolve_agent_kind(args)
		plan = checkpoint_plan(args, len(training_tasks), agent_kind)
		validate_checkpoints(args.model_dir, plan, agent_kind, args.method)
		profiles = {}
		validation_samples = {}
		for target_task in args.target_tasks:
			profile = coordinate_profile(env, args, target_task)
			profiles[target_task] = profile
			validation_samples[target_task] = validate_evaluation_sample(
				env,
				args,
				target_task,
				profile,
			)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		agent = build_eval_agent(args, env, device, agent_kind)
	finally:
		env.close()

	print("run_dir:", args.run_dir)
	print("model_dir:", args.model_dir)
	print("training_tasks:", training_tasks)
	print("target_tasks:", args.target_tasks)
	print("agent_kind:", agent_kind)
	for target_task in args.target_tasks:
		actual_initial, actual_goal = validation_samples[target_task]
		print(f"{target_task}_validation_initial_position:", actual_initial.tolist())
		print(f"{target_task}_validation_goal_position:", actual_goal.tolist())
		print(f"{target_task}_coordinate_profile:", profiles[target_task])

	if args.dry_run:
		for _, model_name, _ in plan:
			print(f"Loading checkpoint: {model_name}")
			agent.load(args.model_dir, model_name)
		print("Dry run completed; no rollout or CSV was produced.")
		return 0

	stats = defaultdict(list)
	for eval_seed in args.eval_seeds:
		evaluate_seed(args, agent, plan, agent_kind, eval_seed, profiles, stats)
	write_stats_csv(args.output, stats)
	print("results:", args.output)
	return 0


if __name__ == "__main__":
	sys.exit(main())

