"""Zero-shot evaluation for a CQRL offline-finetuned meta agent."""

import argparse
import json
import os
import re
import sys
from argparse import Namespace
from collections import defaultdict

import torch

try:
	from . import CQRL as cqrl
except ImportError:
	import CQRL as cqrl

from fetch_env import (
	FETCH_OBSERVATION_DIMS,
	normalize_fetch_task_name,
	parse_fetch_task_order,
)


base = cqrl.base
CHECKPOINT_SUFFIXES = (
	"_critic_target.pt",
	"_quasimetric.pt",
	"_critic.pt",
	"_actor.pt",
)


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Load a CQRL offline-finetuned meta agent and evaluate it on unseen "
			"Fetch tasks or shifted goal ranges without parameter updates."
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
		help="One or more unseen Fetch tasks to evaluate.",
	)
	parser.add_argument(
		"--checkpoint",
		default="final",
		help=(
			"Checkpoint selector: final, stageN, a checkpoint basename, or a path "
			"to a checkpoint stem/component file."
		),
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
		help=(
			"FetchSlide goal-distance scale used only for evaluation; for example, "
			"evaluate a model trained with 0.795 at the native 1.0 scale."
		),
	)
	parser.add_argument(
		"--eval_goal_outer_range",
		type=float,
		default=None,
		help=(
			"Evaluation-only Push/Pick x-y half-range. Goals are sampled outside "
			"the environment's native range; for example, 0.20 tests the square "
			"ring outside the native +/-0.15 range."
		),
	)
	parser.add_argument(
		"--eval_initial_state_outer_range",
		type=float,
		default=None,
		help=(
			"Evaluation-only Push/Pick initial object x-y half-range. Initial "
			"states are sampled outside the environment's native +/-0.15 range."
		),
	)
	parser.add_argument(
		"--allow_seen_tasks",
		action="store_true",
		help="Allow targets that appear in the offline training task list.",
	)
	parser.add_argument(
		"--dry_run",
		action="store_true",
		help="Build the environment and agent and load the checkpoint without rollout.",
	)
	args = parser.parse_args()

	args.offline_dir = os.path.abspath(args.offline_dir)
	args.config = os.path.abspath(
		args.config or os.path.join(args.offline_dir, "source_run_config.json")
	)
	args.target_tasks = list(
		dict.fromkeys(normalize_fetch_task_name(task) for task in args.target_tasks)
	)
	if args.num_eval_runs is not None and args.num_eval_runs <= 0:
		parser.error("--num_eval_runs must be positive")
	if args.eval_slide_goal_scale is not None and not (
		0.0 < args.eval_slide_goal_scale <= 1.0
	):
		parser.error("--eval_slide_goal_scale must be in (0, 1]")
	if args.eval_goal_outer_range is not None:
		if args.eval_goal_outer_range <= 0.15:
			parser.error("--eval_goal_outer_range must be greater than 0.15")
		unsupported_tasks = [
			task
			for task in args.target_tasks
			if task not in {"push", "pick-and-place"}
		]
		if unsupported_tasks:
			parser.error(
				"--eval_goal_outer_range only supports push and pick-and-place"
			)
	if args.eval_initial_state_outer_range is not None:
		if args.eval_initial_state_outer_range <= 0.15:
			parser.error(
				"--eval_initial_state_outer_range must be greater than 0.15"
			)
		unsupported_tasks = [
			task
			for task in args.target_tasks
			if task not in {"push", "pick-and-place"}
		]
		if unsupported_tasks:
			parser.error(
				"--eval_initial_state_outer_range only supports push and "
				"pick-and-place"
			)
	return args


def make_env_kwargs(args):
	kwargs = base.make_env_kwargs(args)
	kwargs["eval_goal_outer_range"] = args.eval_goal_outer_range
	kwargs["eval_initial_state_outer_range"] = (
		args.eval_initial_state_outer_range
	)
	return kwargs


def load_json(path, description):
	if not os.path.isfile(path):
		raise FileNotFoundError(f"{description} not found: {path}")
	with open(path, "r") as handle:
		return json.load(handle)


def default_agent_config():
	parser = argparse.ArgumentParser(add_help=False)
	base.add_common_args(parser)
	base.add_sac_args(parser)
	base.add_student_qrl_args(parser)
	base.add_quasimetric_args(parser)
	return vars(parser.parse_args([]))


def build_agent_args(cli_args, report):
	config = default_agent_config()
	config.update(load_json(cli_args.config, "Source run config"))
	config.update(report.get("meta_learner_config", {}))
	if cli_args.gpu is not None:
		config["gpu"] = cli_args.gpu
	if cli_args.num_eval_runs is not None:
		config["num_eval_runs"] = cli_args.num_eval_runs
	if cli_args.eval_seeds is not None:
		eval_seeds = cli_args.eval_seeds
	elif cli_args.eval_seed is not None:
		eval_seeds = [cli_args.eval_seed]
	else:
		eval_seeds = [int(config.get("seed", report.get("seed", 0)))]
	config["eval_seeds"] = list(dict.fromkeys(eval_seeds))
	return Namespace(**config)


def report_task_names(report, agent_args):
	task_names = report.get("source_tasks")
	if task_names:
		return [normalize_fetch_task_name(task) for task in task_names]
	if agent_args.task_order is not None:
		return parse_fetch_task_order(agent_args.task_order)
	raise ValueError(
		"Cannot determine the task sequence used to build the meta agent. "
		"Expected source_tasks in offline_report.json or task_order in the config."
	)


def selected_task_names(report):
	selected_tasks = report.get("selected_tasks", [])
	return [
		normalize_fetch_task_name(
			task["task"] if isinstance(task, dict) else task
		)
		for task in selected_tasks
	]


def strip_checkpoint_suffix(path):
	for suffix in CHECKPOINT_SUFFIXES:
		if path.endswith(suffix):
			return path[: -len(suffix)]
	return path


def resolve_checkpoint(offline_dir, report, selector):
	model_dir = os.path.join(offline_dir, "model")
	trained_tasks = selected_task_names(report)
	if selector == "final":
		checkpoint = report.get("meta_checkpoint")
		if checkpoint is None:
			run_name = report.get("source_run_name")
			if run_name is None:
				raise ValueError("offline_report.json does not define meta_checkpoint.")
			checkpoint = os.path.join(model_dir, f"{run_name}_offline_meta")
	else:
		stage_match = re.fullmatch(r"stage(\d+)", selector)
		if stage_match:
			stage_idx = int(stage_match.group(1))
			stage = next(
				(
					entry
					for entry in report.get("incremental_stages", [])
					if int(entry["stage_idx"]) == stage_idx
				),
				None,
			)
			if stage is None:
				raise ValueError(f"Offline report does not contain stage {stage_idx}.")
			checkpoint = stage["meta_checkpoint"]
			trained_tasks = [
				normalize_fetch_task_name(
					task["task"] if isinstance(task, dict) else task
				)
				for task in stage.get("seen_tasks", [])
			]
		elif os.path.isabs(selector) or os.path.dirname(selector):
			checkpoint = os.path.abspath(selector)
		else:
			checkpoint = os.path.join(model_dir, selector)

	checkpoint = strip_checkpoint_suffix(checkpoint)
	if not os.path.dirname(checkpoint):
		checkpoint = os.path.join(model_dir, checkpoint)
	if not all(os.path.isfile(f"{checkpoint}{suffix}") for suffix in CHECKPOINT_SUFFIXES):
		missing = [
			f"{checkpoint}{suffix}"
			for suffix in CHECKPOINT_SUFFIXES
			if not os.path.isfile(f"{checkpoint}{suffix}")
		]
		raise FileNotFoundError(
			"Incomplete meta checkpoint; missing: " + ", ".join(missing)
		)
	if not trained_tasks:
		raise ValueError(
			"Cannot determine which tasks trained this checkpoint from "
			"offline_report.json."
		)
	return os.path.dirname(checkpoint), os.path.basename(checkpoint), trained_tasks


def validate_targets(model_tasks, trained_tasks, target_tasks, allow_seen_tasks):
	seen_tasks = set(trained_tasks)
	if not allow_seen_tasks:
		seen_targets = [task for task in target_tasks if task in seen_tasks]
		if seen_targets:
			raise ValueError(
				"Zero-shot targets were present in the source task sequence: "
				+ ", ".join(seen_targets)
				+ ". Pass --allow_seen_tasks to evaluate them intentionally."
			)
	model_observation_dim = max(FETCH_OBSERVATION_DIMS[task] for task in model_tasks)
	for task in target_tasks:
		target_observation_dim = FETCH_OBSERVATION_DIMS[task]
		if target_observation_dim > model_observation_dim:
			raise ValueError(
				f"Target task {task!r} requires observation dim "
				f"{target_observation_dim}, but the checkpoint was built for "
				f"dim {model_observation_dim}."
			)
	return model_observation_dim


def output_paths(cli_args, checkpoint_name):
	if cli_args.output is not None:
		csv_path = os.path.abspath(cli_args.output)
	else:
		target_tag = "-".join(base.safe_tag(task) for task in cli_args.target_tasks)
		if cli_args.eval_slide_goal_scale is not None:
			target_tag += (
				"_eval-slide-scale"
				+ base.safe_tag(cli_args.eval_slide_goal_scale)
			)
		if cli_args.eval_goal_outer_range is not None:
			target_tag += (
				"_eval-outer-range"
				+ base.safe_tag(cli_args.eval_goal_outer_range)
			)
		if cli_args.eval_initial_state_outer_range is not None:
			target_tag += (
				"_eval-initial-outer-range"
				+ base.safe_tag(cli_args.eval_initial_state_outer_range)
			)
		checkpoint_tag = base.safe_tag(checkpoint_name)
		csv_path = os.path.join(
			cli_args.offline_dir,
			f"zeroshot_{checkpoint_tag}_to-{target_tag}.csv",
		)
	root, extension = os.path.splitext(csv_path)
	if extension.lower() != ".csv":
		csv_path = f"{csv_path}.csv"
		root = csv_path[:-4]
	return csv_path, f"{root}.json"


def evaluate(args, agent, checkpoint_name, trained_tasks, target_tasks):
	records = []
	stats = defaultdict(list)
	for eval_seed in args.eval_seeds:
		args.seed = eval_seed
		base.set_seed_everywhere(eval_seed)
		env = base.FetchGoalEnvSequence(**make_env_kwargs(args))
		try:
			for target_task in target_tasks:
				env.set_task(target_task)
				agent.eval()
				with torch.inference_mode():
					eval_results = env.evaluate_agent(
						agent,
						args.num_eval_runs,
						reseed_each_episode=False,
					)
				metrics = base.summarize_eval_results(eval_results)
				record = {
					"target_task": target_task,
					"seed": eval_seed,
					"num_eval_runs": args.num_eval_runs,
					"checkpoint": checkpoint_name,
					"trained_tasks": ",".join(trained_tasks),
					"zero_shot": True,
					"training_slide_goal_scale": args.training_slide_goal_scale,
					"eval_slide_goal_scale": args.slide_goal_scale,
					"eval_goal_outer_range": args.eval_goal_outer_range,
					"eval_initial_state_outer_range": (
						args.eval_initial_state_outer_range
					),
					**metrics,
				}
				records.append(record)
				for key, value in record.items():
					stats[key].append(value)
				print(
					f"zero-shot task {target_task}, seed {eval_seed}: "
					f"success={metrics['success_mean']:.3f} +/- "
					f"{metrics['success_std']:.3f}, "
					f"gc_success={metrics['gc_success_mean']:.3f} +/- "
					f"{metrics['gc_success_std']:.3f}, "
					f"return={metrics['return_mean']:.3f} +/- "
					f"{metrics['return_std']:.3f}"
				)
		finally:
			env.close()
	return records, stats


def main():
	cli_args = parse_args()
	report_path = os.path.join(cli_args.offline_dir, "offline_report.json")
	report = load_json(report_path, "Offline report")
	agent_args = build_agent_args(cli_args, report)
	agent_args.training_slide_goal_scale = float(agent_args.slide_goal_scale)
	if cli_args.eval_slide_goal_scale is not None:
		agent_args.slide_goal_scale = cli_args.eval_slide_goal_scale
	model_tasks = report_task_names(report, agent_args)
	checkpoint_dir, checkpoint_name, trained_tasks = resolve_checkpoint(
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

	os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_args.gpu)
	agent_args.seed = agent_args.eval_seeds[0]
	base.set_seed_everywhere(agent_args.seed)
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
		if obs_space.shape[0] != model_observation_dim:
			raise RuntimeError(
				f"Evaluation environment observation dim {obs_space.shape[0]} does not "
				f"match checkpoint dim {model_observation_dim}."
			)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		agent = base.build_meta_agent(
			obs_space.shape[0],
			env.env.action_space.shape[0],
			device,
			agent_args,
		)
		agent.load(checkpoint_dir, checkpoint_name)
		agent.eval()
	finally:
		env.close()

	print("offline_dir:", cli_args.offline_dir)
	print("model_tasks:", model_tasks)
	print("trained_tasks:", trained_tasks)
	print("target_tasks:", cli_args.target_tasks)
	print("checkpoint:", os.path.join(checkpoint_dir, checkpoint_name))
	print("device:", device)
	print("eval_seeds:", agent_args.eval_seeds)
	print("training_slide_goal_scale:", agent_args.training_slide_goal_scale)
	print("eval_slide_goal_scale:", agent_args.slide_goal_scale)
	print("eval_goal_outer_range:", agent_args.eval_goal_outer_range)
	print(
		"eval_initial_state_outer_range:",
		agent_args.eval_initial_state_outer_range,
	)
	if cli_args.dry_run:
		print("dry run complete: checkpoint loaded; no rollout performed")
		return 0

	records, stats = evaluate(
		agent_args,
		agent,
		checkpoint_name,
		trained_tasks,
		cli_args.target_tasks,
	)
	csv_path, json_path = output_paths(cli_args, checkpoint_name)
	base.write_stats_csv(csv_path, stats)
	os.makedirs(os.path.dirname(json_path), exist_ok=True)
	with open(json_path, "w") as handle:
		json.dump(
			{
				"offline_dir": cli_args.offline_dir,
				"checkpoint": os.path.join(checkpoint_dir, checkpoint_name),
				"model_tasks": model_tasks,
				"trained_tasks": trained_tasks,
				"target_tasks": cli_args.target_tasks,
				"zero_shot": True,
				"training_slide_goal_scale": agent_args.training_slide_goal_scale,
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
'''
conda activate RLL3

python -m online_continual.CQRL_offline_ft_zeroshot \
  --offline_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/offline_meta_incremental_push-pick-and-place_store1500 \
  --checkpoint final \
	--target_tasks slide \
	--eval_slide_goal_scale 1.0 \
	--allow_seen_tasks \
  --eval_seeds 0 1 \
  --num_eval_runs 50 \
  --gpu 0
python -m online_continual.CQRL_offline_ft_zeroshot \
  --offline_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/offline_meta_incremental_push-pick-and-place_store1500_contra0.0_onlyclone \
  --checkpoint final \
	--target_tasks slide \
	--eval_slide_goal_scale 1.0 \
	--allow_seen_tasks \
  --eval_seeds 0 1 \
  --num_eval_runs 50 \
  --gpu 0

  python -m online_continual.CQRL_offline_ft_zeroshot \
  --offline_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/offline_meta_incremental_push-pick-and-place_store1500_contra0.0_onlyclone \
  --checkpoint final \
  --target_tasks push pick-and-place \
  --eval_goal_outer_range 0.30 \
  --allow_seen_tasks \
  --eval_seeds 0 \
  --num_eval_runs 50 \
  --gpu 0

  python -m online_continual.CQRL_offline_ft_zeroshot \
  --offline_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/offline_meta_incremental_push-pick-and-place_store1500 \
  --checkpoint final \
  --target_tasks push pick-and-place \
  --eval_goal_outer_range 0.30 \
  --allow_seen_tasks \
  --eval_seeds 0 \
  --num_eval_runs 50 \
  --gpu 0
---------------------
	python -m online_continual.CQRL_offline_ft_zeroshot \
	--offline_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/offline_meta_incremental_push-pick-and-place_store1500 \
	--checkpoint final \
	--target_tasks push pick-and-place \
	--eval_initial_state_outer_range 0.20 \
	--eval_goal_outer_range 0.30 \
	--allow_seen_tasks \
	--eval_seeds 0 1 \
	--num_eval_runs 50 \
	--gpu 0
python -m online_continual.CQRL_offline_ft_zeroshot \
    --offline_dir online_continual/results/cqrl/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795/offline_meta_incremental_push-pick-and-place_store1500_contra0.0_onlyclone \
    --checkpoint final \
    --target_tasks push pick-and-place \
    --eval_initial_state_outer_range 0.20 \
    --eval_goal_outer_range 0.30 \
    --allow_seen_tasks \
    --eval_seeds 0 1 \
    --num_eval_runs 50 \
    --gpu 0
	
	success  0.465 , 0.41


python -m online_continual.CQRL_offline_ft_zeroshot \
    --offline_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
    --checkpoint final \
    --target_tasks push pick-and-place \
    --eval_initial_state_outer_range 0.20 \
    --eval_goal_outer_range 0.30 \
    --allow_seen_tasks \
    --eval_seeds 0 1 \
    --num_eval_runs 25 \
    --gpu 0

online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795
'''