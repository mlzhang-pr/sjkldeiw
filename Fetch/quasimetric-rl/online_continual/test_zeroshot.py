import argparse
import os
import sys
from collections import defaultdict

import torch

if __package__:
	from .final_eval import (
		build_eval_agent,
		checkpoint_exists,
		find_training_config,
		load_training_config,
		resolve_agent_kind,
	)
	from .main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		append_eval_stats,
		evaluate_and_log,
		make_env_kwargs,
		set_seed_everywhere,
		write_stats_csv,
	)
else:
	from final_eval import (
		build_eval_agent,
		checkpoint_exists,
		find_training_config,
		load_training_config,
		resolve_agent_kind,
	)
	from main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		append_eval_stats,
		evaluate_and_log,
		make_env_kwargs,
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
		description="Load continual Fetch checkpoints and zero-shot test a target task"
	)
	parser.add_argument("--run_dir", type=str, required=True)
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--model_dir", type=str, default=None)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument(
		"--source_stages",
		type=int,
		nargs="+",
		default=None,
		help=(
			"1-based completed stages to load. Each is evaluated on the following task "
			"unless --target_task is provided."
		),
	)
	parser.add_argument(
		"--target_task",
		type=str,
		default=None,
		help=(
			"Fetch task to evaluate regardless of source stage: reach, push, "
			"pick-and-place, or slide."
		),
	)
	parser.add_argument(
		"--agent_kind",
		choices=["auto", "meta", "student"],
		default="auto",
		help="Auto uses the meta agent for buffer methods and the student otherwise.",
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
	args.model_dir = os.path.abspath(args.model_dir or os.path.join(args.run_dir, "model"))
	if args.target_task is not None:
		args.target_task = normalize_fetch_task_name(args.target_task)
	run_name = os.path.basename(os.path.normpath(args.run_dir))
	output_name = f"{run_name}_zeroshot"
	if args.target_task is not None:
		output_name += f"_to-{args.target_task}"
	args.output = os.path.abspath(
		args.output or os.path.join(args.run_dir, f"{output_name}.csv")
	)
	if args.eval_seeds is None:
		args.eval_seeds = [args.eval_seed if args.eval_seed is not None else args.seed]
	return args


def build_zero_shot_plan(args, task_names, agent_kind):
	if args.target_task is None and len(task_names) < 2:
		raise ValueError("Zero-shot next-task evaluation requires at least two tasks.")

	max_source_stage = len(task_names) if args.target_task is not None else len(task_names) - 1
	source_stages = args.source_stages or list(range(1, max_source_stage + 1))
	source_stages = list(dict.fromkeys(source_stages))
	invalid_stages = [stage for stage in source_stages if stage < 1 or stage > max_source_stage]
	if invalid_stages:
		raise ValueError(
			f"source_stages must be between 1 and {max_source_stage}; "
			f"got {invalid_stages}."
		)
	if args.target_task is not None:
		model_observation_dim = max(FETCH_OBSERVATION_DIMS[task] for task in task_names)
		target_observation_dim = FETCH_OBSERVATION_DIMS[args.target_task]
		if target_observation_dim > model_observation_dim:
			raise ValueError(
				f"Target task '{args.target_task}' requires observation dim "
				f"{target_observation_dim}, but the checkpoint was built for dim "
				f"{model_observation_dim}."
			)

	run_name = os.path.basename(os.path.normpath(args.run_dir))
	kind_suffix = "_meta" if agent_kind == "meta" else ""
	plan = []
	missing = []
	for source_stage in source_stages:
		model_name = f"{run_name}_task{source_stage}{kind_suffix}"
		if not checkpoint_exists(args.model_dir, model_name, agent_kind):
			missing.append(model_name)
		if args.target_task is None:
			target_task = task_names[source_stage]
			target_task_idx = source_stage + 1
		else:
			target_task = args.target_task
			target_task_idx = (
				task_names.index(target_task) + 1 if target_task in task_names else ""
			)
		plan.append(
			(
				source_stage,
				model_name,
				task_names[source_stage - 1],
				target_task,
				target_task_idx,
			)
		)

	if missing:
		expected = "*_actor.pt" if agent_kind == "meta" else "*_online_qrl.pt"
		raise FileNotFoundError(
			f"Missing {agent_kind} checkpoints in {args.model_dir}: {', '.join(missing)}. "
			f"Expected files matching {expected}."
		)
	return plan


def evaluate_seed(args, agent, plan, agent_kind, eval_seed, stats):
	args.seed = eval_seed
	set_seed_everywhere(eval_seed)
	env = FetchGoalEnvSequence(**make_env_kwargs(args))

	try:
		for source_stage, model_name, source_task, target_task, target_task_idx in plan:
			print(f"Loading checkpoint: {model_name}")
			agent.load(args.model_dir, model_name)
			env.set_task(target_task)
			eval_metrics = evaluate_and_log(
				env,
				agent,
				None,
				"zeroshot",
				source_stage,
				args.num_eval_runs,
				reseed_each_episode=False,
			)
			print(
				f"Task {source_task} -> {target_task}: "
				f"success {eval_metrics['success_mean']:.3f} +/- "
				f"{eval_metrics['success_std']:.3f}, "
				f"gc_success {eval_metrics['gc_success_mean']:.3f} +/- "
				f"{eval_metrics['gc_success_std']:.3f}, "
				f"return {eval_metrics['return_mean']:.3f} +/- "
				f"{eval_metrics['return_std']:.3f}"
			)

			append_eval_stats(stats, eval_metrics)
			stats["source_task"].append(source_task)
			stats["source_task_idx"].append(source_stage)
			stats["target_task"].append(target_task)
			stats["target_task_idx"].append(target_task_idx)
			stats["seed"].append(eval_seed)
			stats["method"].append(args.method)
			stats["agent_kind"].append(agent_kind)
			stats["checkpoint"].append(model_name)
	finally:
		env.close()


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	args.seed = args.eval_seeds[0]
	set_seed_everywhere(args.seed)

	env = FetchGoalEnvSequence(**make_env_kwargs(args))
	try:
		agent_kind = resolve_agent_kind(args)
		task_names = list(env.env_list)
		plan = build_zero_shot_plan(args, task_names, agent_kind)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		agent = build_eval_agent(args, env, device, agent_kind)
	finally:
		env.close()

	print("run_dir:", args.run_dir)
	print("config:", args.config or "CLI defaults")
	print("model_dir:", args.model_dir)
	print("agent_kind:", agent_kind)
	print("eval_seeds:", args.eval_seeds)
	print("zero-shot pairs:", [(entry[2], entry[3]) for entry in plan])

	stats = defaultdict(list)
	for eval_seed in args.eval_seeds:
		evaluate_seed(args, agent, plan, agent_kind, eval_seed, stats)

	write_stats_csv(args.output, stats)
	print("results:", args.output)


if __name__ == "__main__":
	sys.exit(main())

'''
cd Fetch/quasimetric-rl
conda activate RLL3

python -m online_continual.test_zeroshot \
  --run_dir online_continual/results/fetch_continual_push-pick-and-place_seed0_buffer_gc-sparse \
  --source_stages 1 \
	--target_task slide \
  --agent_kind meta \
  --eval_seeds 0 1 2 \
  --num_eval_runs 50 \
  --gpu 0

  python -m online_continual.test_zeroshot \
  --run_dir online_continual/results/fetch_continual_push-pick-and-place_seed0_buffer_gc-sparse \
  --source_stages 2 \
  --target_task slide \
  --agent_kind meta \
  --eval_seeds 0 \
  --num_eval_runs 50 \
  --gpu 0
  
   python -m online_continual.test_zeroshot \
    --run_dir online_continual/results/fetch_continual_push-slide_seed0_buffer_gc-sparse_slide-scale0.8 \
    --source_stages 2 \
    --target_task pick-and-place \
    --agent_kind student \
    --eval_seeds 0 \
    --num_eval_runs 50 \
    --gpu 0
  
'''


