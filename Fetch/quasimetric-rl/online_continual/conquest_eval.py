import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
import yaml

if __package__:
	from .conquest import build_fast_agent as build_conquest_fast_agent
	from .main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		append_eval_stats,
		build_meta_agent,
		build_student_agent,
		evaluate_and_log,
		make_env_kwargs,
		set_seed_everywhere,
		vector_observation_space,
		write_stats_csv,
	)
else:
	from conquest import build_fast_agent as build_conquest_fast_agent
	from main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		append_eval_stats,
		build_meta_agent,
		build_student_agent,
		evaluate_and_log,
		make_env_kwargs,
		set_seed_everywhere,
		vector_observation_space,
		write_stats_csv,
	)


def find_training_config(run_dir, config_path=None):
	if config_path is not None:
		return os.path.abspath(config_path)

	candidates = [
		os.path.join(run_dir, "run_config.json"),
		os.path.join(run_dir, "wandb", "latest-run", "files", "config.yaml"),
	]
	return next((path for path in candidates if os.path.isfile(path)), None)


def load_training_config(config_path):
	if config_path is None:
		return {}
	with open(config_path, "r") as handle:
		if config_path.endswith(".json"):
			return json.load(handle)
		raw_config = yaml.safe_load(handle) or {}
	return {
		key: entry["value"]
		for key, entry in raw_config.items()
		if key != "_wandb" and isinstance(entry, dict) and "value" in entry
	}


def parse_checkpoint(value):
	if value in {"all", "final", "latest"} or re.fullmatch(r"(?:task|step)\d+", value):
		return value
	raise argparse.ArgumentTypeError(
		"checkpoint must be all, final, latest, taskN, or stepN"
	)


def parse_args():
	bootstrap_parser = argparse.ArgumentParser(add_help=False)
	bootstrap_parser.add_argument("--run_dir", type=str, default=None)
	bootstrap_parser.add_argument("--config", type=str, default=None)
	bootstrap_args, _ = bootstrap_parser.parse_known_args()

	run_dir = os.path.abspath(bootstrap_args.run_dir) if bootstrap_args.run_dir else None
	config_path = find_training_config(run_dir, bootstrap_args.config) if run_dir else None
	training_config = load_training_config(config_path)

	parser = argparse.ArgumentParser(
		description="Load trained continual Fetch agents and run final evaluation"
	)
	parser.add_argument("--run_dir", type=str, required=True)
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--model_dir", type=str, default=None)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument(
		"--checkpoint",
		type=parse_checkpoint,
		default="all",
		help=(
			"Choose all, final, latest, taskN, or stepN. Use latest to evaluate "
			"an unfinished run."
		),
	)
	parser.add_argument(
		"--agent_kind",
		choices=["auto", "meta", "student", "fast"],
		default="meta",
		help=(
			"Choose the CONQUEST fast or meta policy; student aliases fast."
		),
	)
	eval_seed_group = parser.add_mutually_exclusive_group()
	eval_seed_group.add_argument(
		"--eval_seed",
		type=int,
		default=None,
		help="Override the training seed used to initialize evaluation environments.",
	)
	eval_seed_group.add_argument(
		"--eval_seeds",
		type=int,
		nargs="+",
		default=None,
		help="Evaluate each checkpoint with multiple environment seeds.",
	)
	add_common_args(parser)
	add_sac_args(parser)
	add_student_qrl_args(parser)
	add_quasimetric_args(parser)
	method_action = next(action for action in parser._actions if action.dest == "method")
	method_action.choices = ["cqrl"]
	method_action.default = "cqrl"
	parser.set_defaults(**training_config)
	args = parser.parse_args()

	args.run_dir = os.path.abspath(args.run_dir)
	args.config = config_path
	args.model_dir = os.path.abspath(args.model_dir or os.path.join(run_dir, "model"))
	args.output = os.path.abspath(
		args.output
		or os.path.join(
			run_dir,
			f"{os.path.basename(run_dir)}_loaded_{args.checkpoint}.csv",
		)
	)
	if args.eval_seeds is None:
		args.eval_seeds = [args.eval_seed if args.eval_seed is not None else args.seed]
	return args


def resolve_agent_kind(args):
	if args.agent_kind in {"auto", "student"}:
		return "fast"
	return args.agent_kind


def checkpoint_file_suffix(agent_kind, method=None):
	if agent_kind in {"fast", "student"}:
		return "_online_qrl.pt"
	return "_actor.pt"


def checkpoint_kind_suffix(agent_kind, method=None):
	return f"_{agent_kind}"


def checkpoint_exists(model_dir, model_name, agent_kind, method=None):
	suffix = checkpoint_file_suffix(agent_kind, method)
	return os.path.isfile(os.path.join(model_dir, f"{model_name}{suffix}"))


def latest_checkpoint_name(model_dir, run_name, agent_kind, method=None):
	kind_suffix = checkpoint_kind_suffix(agent_kind, method)
	file_suffix = checkpoint_file_suffix(agent_kind, method)
	pattern = re.compile(
		rf"^{re.escape(run_name)}_step(\d+){re.escape(kind_suffix + file_suffix)}$"
	)
	candidates = []
	for filename in os.listdir(model_dir):
		match = pattern.fullmatch(filename)
		if match:
			model_name = filename[: -len(file_suffix)]
			candidates.append((int(match.group(1)), model_name))
	if not candidates:
		raise FileNotFoundError(
			f"No step checkpoints found for {agent_kind} in {model_dir}."
		)
	return max(candidates, key=lambda entry: entry[0])[1]


def checkpoint_plan(args, task_count, agent_kind):
	run_name = os.path.basename(os.path.normpath(args.run_dir))
	kind_suffix = checkpoint_kind_suffix(agent_kind, args.method)
	if getattr(args, "meta_only", False):
		if agent_kind != "meta":
			raise ValueError("Meta-only runs can only be evaluated with --agent_kind meta.")
		source_run_dir = getattr(args, "source_run_dir", None)
		if not source_run_dir:
			raise ValueError("Meta-only run config is missing source_run_dir.")
		trajectories_per_task = getattr(args, "trajectories_per_task", None)
		if trajectories_per_task is None:
			raise ValueError("Meta-only run config is missing trajectories_per_task.")
		source_run_name = os.path.basename(os.path.normpath(source_run_dir))
		run_name = f"{source_run_name}_meta_only_last{trajectories_per_task}"
		kind_suffix = ""

	if args.checkpoint == "final":
		model_name = f"{run_name}_final{kind_suffix}"
		return [(task_count, model_name, list(range(task_count)))]
	if args.checkpoint == "latest":
		model_name = latest_checkpoint_name(
			args.model_dir,
			run_name,
			agent_kind,
			args.method,
		)
		return [(task_count, model_name, list(range(task_count)))]
	task_match = re.fullmatch(r"task(\d+)", args.checkpoint)
	if task_match:
		stage_idx = int(task_match.group(1))
		if stage_idx < 1 or stage_idx > task_count:
			raise ValueError(
				f"Checkpoint task index must be between 1 and {task_count}; "
				f"got {stage_idx}."
			)
		model_name = f"{run_name}_task{stage_idx}{kind_suffix}"
		return [(stage_idx, model_name, list(range(stage_idx)))]
	if re.fullmatch(r"step\d+", args.checkpoint):
		model_name = f"{run_name}_{args.checkpoint}{kind_suffix}"
		return [(task_count, model_name, list(range(task_count)))]

	plan = []
	for stage_idx in range(1, task_count + 1):
		model_name = f"{run_name}_task{stage_idx}{kind_suffix}"
		if stage_idx == task_count and not checkpoint_exists(
			args.model_dir,
			model_name,
			agent_kind,
			args.method,
		):
			model_name = f"{run_name}_final{kind_suffix}"
		plan.append((stage_idx, model_name, list(range(stage_idx))))
	return plan


def validate_checkpoints(model_dir, plan, agent_kind, method=None):
	missing = [
		model_name
		for _, model_name, _ in plan
		if not checkpoint_exists(model_dir, model_name, agent_kind, method)
	]
	if missing:
		expected = f"*{checkpoint_file_suffix(agent_kind, method)}"
		raise FileNotFoundError(
			f"Missing {agent_kind} checkpoints in {model_dir}: {', '.join(missing)}. "
			f"Expected files matching {expected}."
		)


def build_eval_agent(args, env, device, agent_kind):
	obs_space = vector_observation_space(env.env.observation_space)
	action_space = env.env.action_space
	if agent_kind == "fast":
		total_optim_steps = len(env.env_list) * args.change_freq
		return build_conquest_fast_agent(
			observation_space=obs_space,
			action_space=action_space,
			device=device,
			args=args,
			total_optim_steps=total_optim_steps,
		)
	if agent_kind == "meta":
		return build_meta_agent(
			obs_dim=obs_space.shape[0],
			action_dim=action_space.shape[0],
			device=device,
			args=args,
		)
	total_optim_steps = len(env.env_list) * args.change_freq
	return build_student_agent(
		observation_space=obs_space,
		action_space=action_space,
		device=device,
		args=args,
		total_optim_steps=total_optim_steps,
	)


def evaluate_seed(args, agent, plan, agent_kind, eval_seed, final_stats):
	args.seed = eval_seed
	set_seed_everywhere(eval_seed)
	env = FetchGoalEnvSequence(**make_env_kwargs(args))
	print("eval_seed:", eval_seed)

	try:
		for stage_idx, model_name, task_indices in plan:
			print(f"Loading checkpoint: {model_name}")
			agent.load(args.model_dir, model_name)

			for task_idx in task_indices:
				task_name = env.env_list[task_idx]
				env.set_task(task_name)
				eval_metrics = evaluate_and_log(
					env,
					agent,
					None,
					"final",
					stage_idx,
					args.num_eval_runs,
					reseed_each_episode=False,
				)
				print(
					f"Stage {stage_idx} task {task_name}: "
					f"success {eval_metrics['success_mean']:.3f} +/- {eval_metrics['success_std']:.3f}, "
					f"gc_success {eval_metrics['gc_success_mean']:.3f} +/- "
					f"{eval_metrics['gc_success_std']:.3f}, "
					f"return {eval_metrics['return_mean']:.3f} +/- "
					f"{eval_metrics['return_std']:.3f}"
				)

				append_eval_stats(final_stats, eval_metrics)
				final_stats["task"].append(task_name)
				final_stats["task_idx"].append(task_idx + 1)
				final_stats["seed"].append(eval_seed)
				final_stats["method"].append(args.method)
				final_stats["agent_kind"].append(agent_kind)
				final_stats["agent_idx"].append(stage_idx)
				final_stats["checkpoint"].append(model_name)
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
		plan = checkpoint_plan(args, len(env.env_list), agent_kind)
		validate_checkpoints(args.model_dir, plan, agent_kind, args.method)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		agent = build_eval_agent(args, env, device, agent_kind)
		task_names = list(env.env_list)
	finally:
		env.close()

	final_stats = defaultdict(list)
	print("run_dir:", args.run_dir)
	print("config:", args.config or "CLI defaults")
	print("model_dir:", args.model_dir)
	print("agent_kind:", agent_kind)
	print("eval_seeds:", args.eval_seeds)
	print("tasks:", task_names)

	for eval_seed in args.eval_seeds:
		evaluate_seed(args, agent, plan, agent_kind, eval_seed, final_stats)

	write_stats_csv(args.output, final_stats)
	print("results:", args.output)


if __name__ == "__main__":
	sys.exit(main())