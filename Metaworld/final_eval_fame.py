"""Evaluate the latest FAME meta policy on every task and seed."""

import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import yaml

from agent.sac import SACAgent
from envs.metaworld_env import MetaWorldSingleEnvSequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_existing_path(path):
	path = os.path.expanduser(path)
	if os.path.isabs(path):
		return path

	cwd_path = os.path.abspath(path)
	if os.path.exists(cwd_path):
		return cwd_path
	return os.path.abspath(os.path.join(SCRIPT_DIR, path))


def load_training_config(config_path):
	with open(config_path, "r", encoding="utf-8") as handle:
		if config_path.endswith(".json"):
			return json.load(handle)
		raw_config = yaml.safe_load(handle) or {}
	return {
		key: entry["value"]
		for key, entry in raw_config.items()
		if key != "_wandb" and isinstance(entry, dict) and "value" in entry
	}


def find_training_config(run_dir, config_path=None):
	if config_path is not None:
		path = resolve_existing_path(config_path)
		if not os.path.isfile(path):
			raise FileNotFoundError(f"Training config does not exist: {path}")
		return path

	candidates = [
		os.path.join(run_dir, "run_config.json"),
		os.path.join(run_dir, "wandb", "latest-run", "files", "config.yaml"),
	]
	return next((path for path in candidates if os.path.isfile(path)), None)


def resolve_model_dir(run_dir, model_dir=None):
	if model_dir is not None:
		resolved = resolve_existing_path(model_dir)
	elif os.path.isdir(os.path.join(run_dir, "model")):
		resolved = os.path.join(run_dir, "model")
	else:
		resolved = run_dir
	if not os.path.isdir(resolved):
		raise FileNotFoundError(f"Model directory does not exist: {resolved}")
	return resolved


def checkpoint_path(model_dir, model_name, suffix):
	return os.path.join(model_dir, f"{model_name}_{suffix}.pt")


def latest_meta_checkpoint(model_dir, requested_name=None):
	if requested_name is not None:
		model_name = requested_name
		match = re.search(r"_(\d+)_meta$", model_name)
		checkpoint_idx = int(match.group(1)) if match else -1
	else:
		pattern = re.compile(r"^(.+)_(\d+)_meta_actor\.pt$")
		candidates = []
		for filename in os.listdir(model_dir):
			match = pattern.fullmatch(filename)
			if match:
				candidates.append(
					(int(match.group(2)), filename[: -len("_actor.pt")])
				)
		if not candidates:
			raise FileNotFoundError(
				f"No FAME meta actor matching *_N_meta_actor.pt in {model_dir}."
			)
		checkpoint_idx, model_name = max(candidates, key=lambda item: item[0])

	actor_path = checkpoint_path(model_dir, model_name, "actor")
	if not os.path.isfile(actor_path):
		raise FileNotFoundError(f"Missing FAME meta actor checkpoint: {actor_path}")
	return model_name, checkpoint_idx


def infer_training_config(model_name):
	match = re.fullmatch(r"sac_(.+)_(-?\d+)_([^_]+)_\d+_meta", model_name)
	if match is None:
		raise ValueError(
			"Cannot infer environment, training seed, and method from checkpoint "
			f"name {model_name!r}. Pass --config or the explicit CLI overrides."
		)
	return {
		"env": match.group(1),
		"seed": int(match.group(2)),
		"method": match.group(3),
	}


def merge_training_config(saved_config, inferred_config, args):
	config = {
		"change_freq": 1_000_000,
		"normalize_obs": None,
		"normalize_avg_coef": 0.0001,
		"normalize_rewards": True,
		"reset_obs_stats": False,
		"change_when_solved": False,
		"freeze_rand_vec": False,
		"goal_hidden": True,
		"goal_conditioned": False,
		"gc_reward_type": "sparse",
		"gc_success_threshold": 0.05,
		"gc_achieved_goal": "auto",
		"batch_size": 256,
		"discount": 0.99,
		"init_temperature": 0.1,
		"actor_lr": 1e-4,
		"critic_lr": 1e-4,
		"alpha_lr": 1e-4,
		"critic_tau": 0.005,
		"actor_update_frequency": 1,
		"critic_target_update_frequency": 1,
	}
	config.update(inferred_config)
	config.update(saved_config)
	if args.env is not None:
		config["env"] = args.env
	if args.training_seed is not None:
		config["seed"] = args.training_seed
	if args.method is not None:
		config["method"] = args.method
	if args.freeze_rand_vec is not None:
		config["freeze_rand_vec"] = bool(args.freeze_rand_vec)
	return config


def set_seed_everywhere(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def make_env_kwargs(training_config, eval_seed):
	env_name = str(training_config["env"]).lower()
	env_sequence = training_config.get("env_sequence")
	base_task_name = training_config.get("base_task_name")
	if env_name.startswith("metaworld_sequence_"):
		env_suffix = env_name[len("metaworld_sequence_"):]
		if env_suffix.startswith("set"):
			env_sequence = env_suffix
		else:
			base_task_name = f"{env_suffix}-v2"
			env_sequence = None

	return {
		"change_freq": training_config.get("change_freq", 1_000_000),
		"base_task_name": base_task_name,
		"env_sequence": env_sequence,
		"goal_hidden": bool(training_config.get("goal_hidden", True)),
		"goal_conditioned": bool(training_config.get("goal_conditioned", False)),
		"gc_reward_type": training_config.get("gc_reward_type", "sparse"),
		"gc_success_threshold": training_config.get("gc_success_threshold", 0.05),
		"gc_achieved_goal": training_config.get("gc_achieved_goal", "auto"),
		"normalize_obs": training_config.get("normalize_obs"),
		"normalize_avg_coef": training_config.get("normalize_avg_coef", 0.0001),
		"normalize_rewards": bool(training_config.get("normalize_rewards", True)),
		"reset_obs_stats": bool(training_config.get("reset_obs_stats", False)),
		"change_when_solved": bool(training_config.get("change_when_solved", False)),
		"freeze_rand_vec": bool(training_config.get("freeze_rand_vec", False)),
		"capture_video": False,
		"seed": eval_seed,
	}


def vector_observation_space(observation_space):
	spaces = getattr(observation_space, "spaces", None)
	if spaces is not None and "observation" in spaces:
		return spaces["observation"]
	return observation_space


def load_state_dict(path, device):
	try:
		return torch.load(path, map_location=device, weights_only=True)
	except TypeError:
		return torch.load(path, map_location=device)


def build_and_load_meta_agent(
	obs_dim,
	action_dim,
	device,
	training_config,
	model_dir,
	model_name,
):
	agent = SACAgent(
		obs_dim=obs_dim,
		action_dim=action_dim,
		action_range=[-1.0, 1.0],
		device=device,
		batch_size=int(training_config.get("batch_size", 256)),
		discount=float(training_config.get("discount", 0.99)),
		init_temperature=float(training_config.get("init_temperature", 0.1)),
		actor_lr=float(training_config.get("actor_lr", 1e-4)),
		critic_lr=float(training_config.get("critic_lr", 1e-4)),
		alpha_lr=float(training_config.get("alpha_lr", 1e-4)),
		critic_tau=float(training_config.get("critic_tau", 0.005)),
		actor_update_frequency=int(
			training_config.get("actor_update_frequency", 1)
		),
		critic_target_update_frequency=int(
			training_config.get("critic_target_update_frequency", 1)
		),
	)
	agent.actor.load_state_dict(
		load_state_dict(checkpoint_path(model_dir, model_name, "actor"), device)
	)

	critic_path = checkpoint_path(model_dir, model_name, "critic")
	critic_target_path = checkpoint_path(model_dir, model_name, "critic_target")
	critic_loaded = os.path.isfile(critic_path)
	target_loaded = os.path.isfile(critic_target_path)
	if critic_loaded:
		agent.critic.load_state_dict(load_state_dict(critic_path, device))
	if target_loaded:
		agent.critic_target.load_state_dict(
			load_state_dict(critic_target_path, device)
		)
	agent.eval()
	print(
		f"loaded FAME meta checkpoint: {os.path.join(model_dir, model_name)} "
		f"actor=True critic={critic_loaded} critic_target={target_loaded}"
	)
	return agent


def mean_std(values):
	values = np.asarray(values, dtype=np.float32)
	if values.size == 0:
		return np.nan, np.nan
	return float(np.mean(values)), float(np.std(values))


def summarize_eval_results(eval_results):
	metrics = {}
	for output_name, source_name in (
		("return", "episodic_returns"),
		("metaworld_success", "successes"),
		("gc_success", "goal_successes"),
		("gc_final_goal_distance", "final_goal_distances"),
		("gc_min_goal_distance", "min_goal_distances"),
		("gc_mean_goal_distance", "mean_goal_distances"),
	):
		mean, std = mean_std(eval_results.get(source_name, []))
		metrics[f"{output_name}_mean"] = mean
		metrics[f"{output_name}_std"] = std
	return metrics


def evaluate_seed(
	eval_seed,
	training_config,
	agent,
	model_name,
	checkpoint_idx,
	num_eval_runs,
	reseed_each_episode,
):
	set_seed_everywhere(eval_seed)
	env = MetaWorldSingleEnvSequence(
		**make_env_kwargs(training_config, eval_seed)
	)
	rows = []
	try:
		env_list = list(env.env_list)
		print(f"eval_seed={eval_seed} tasks={len(env_list)}")
		if checkpoint_idx >= 0 and checkpoint_idx + 1 < len(env_list):
			print(
				f"checkpoint contains {checkpoint_idx + 1} completed task(s); "
				f"also evaluating {len(env_list) - checkpoint_idx - 1} unseen task(s)."
			)
		for task_idx, task_name in enumerate(env_list):
			env.set_task(task_name)
			eval_results = env.evaluate_agent(
				agent,
				num_eval_runs,
				reseed_each_episode=bool(reseed_each_episode),
			)
			metrics = summarize_eval_results(eval_results)
			print(
				f"seed {eval_seed} task {task_idx + 1}/{len(env_list)} {task_name}: "
				f"success {metrics['metaworld_success_mean']:.3f} +/- "
				f"{metrics['metaworld_success_std']:.3f}, return "
				f"{metrics['return_mean']:.3f} +/- {metrics['return_std']:.3f}"
			)
			rows.append(
				{
					"eval_seed": eval_seed,
					"task_idx": task_idx + 1,
					"task": str(task_name),
					"num_eval_runs": num_eval_runs,
					"checkpoint_idx": checkpoint_idx,
					"checkpoint": model_name,
					**metrics,
				}
			)
	finally:
		close = getattr(env, "close", None)
		if callable(close):
			close()
	return rows


def output_with_suffix(output_path, suffix):
	stem, extension = os.path.splitext(output_path)
	return f"{stem}{suffix}{extension or '.csv'}"


def build_task_summary(results):
	return (
		results.groupby(["task_idx", "task"], as_index=False)
		.agg(
			eval_seed_count=("eval_seed", "nunique"),
			success_mean=("metaworld_success_mean", "mean"),
			success_std_across_seeds=("metaworld_success_mean", "std"),
			return_mean=("return_mean", "mean"),
			return_std_across_seeds=("return_mean", "std"),
		)
	)


def build_average_accuracy_table(results):
	per_seed = (
		results.groupby("eval_seed", as_index=False)
		.agg(
			task_count=("task_idx", "nunique"),
			average_accuracy=("metaworld_success_mean", "mean"),
			average_return=("return_mean", "mean"),
		)
	)
	per_seed["eval_seed"] = per_seed["eval_seed"].astype(str)
	per_seed["eval_seed_count"] = 1
	per_seed["average_accuracy_percent"] = per_seed["average_accuracy"] * 100.0
	per_seed["accuracy_std_across_seeds"] = np.nan
	per_seed["return_std_across_seeds"] = np.nan

	overall_accuracy = float(per_seed["average_accuracy"].mean())
	overall_return = float(per_seed["average_return"].mean())
	overall = pd.DataFrame(
		[
			{
				"eval_seed": "all",
				"task_count": int(results["task_idx"].nunique()),
				"average_accuracy": overall_accuracy,
				"average_return": overall_return,
				"eval_seed_count": int(results["eval_seed"].nunique()),
				"average_accuracy_percent": overall_accuracy * 100.0,
				"accuracy_std_across_seeds": float(
					per_seed["average_accuracy"].std(ddof=0)
				),
				"return_std_across_seeds": float(
					per_seed["average_return"].std(ddof=0)
				),
			}
		]
	)
	return pd.concat([per_seed, overall], ignore_index=True)


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Load the latest FAME meta policy and evaluate it on every task in "
			"the training sequence for multiple environment seeds."
		)
	)
	parser.add_argument(
		"--run_dir",
		required=True,
		help="FAME model directory, or a run directory containing model/",
	)
	parser.add_argument("--config", default=None, help="Optional run_config.json or W&B config.yaml")
	parser.add_argument("--model_dir", default=None, help="Explicit checkpoint directory")
	parser.add_argument("--model_name", default=None, help="Explicit checkpoint name without _actor.pt")
	parser.add_argument("--env", default=None, help="Override inferred training environment")
	parser.add_argument("--training_seed", type=int, default=None, help="Override inferred training seed")
	parser.add_argument("--method", default=None, help="Override inferred training method")
	parser.add_argument("--freeze_rand_vec", type=int, choices=[0, 1], default=None)
	parser.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2])
	parser.add_argument("--num_eval_runs", type=int, default=10)
	parser.add_argument("--reseed_each_episode", type=int, choices=[0, 1], default=0)
	parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
	parser.add_argument("--output", default=None, help="Per-seed CSV output path")
	return parser.parse_args()


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	run_dir = resolve_existing_path(args.run_dir)
	if not os.path.isdir(run_dir):
		raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
	model_dir = resolve_model_dir(run_dir, args.model_dir)
	model_name, checkpoint_idx = latest_meta_checkpoint(model_dir, args.model_name)
	inferred_config = infer_training_config(model_name)
	config_path = find_training_config(run_dir, args.config)
	saved_config = load_training_config(config_path) if config_path else {}
	training_config = merge_training_config(saved_config, inferred_config, args)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	set_seed_everywhere(args.eval_seeds[0])
	probe_env = MetaWorldSingleEnvSequence(
		**make_env_kwargs(training_config, args.eval_seeds[0])
	)
	try:
		obs_space = vector_observation_space(probe_env.env.observation_space)
		action_dim = probe_env.env.action_space.shape[0]
		task_names = list(probe_env.env_list)
	finally:
		close = getattr(probe_env, "close", None)
		if callable(close):
			close()

	agent = build_and_load_meta_agent(
		obs_space.shape[0],
		action_dim,
		device,
		training_config,
		model_dir,
		model_name,
	)
	print("config:", config_path or "inferred from checkpoint name and FAME defaults")
	print("model:", os.path.join(model_dir, model_name))
	print("device:", device)
	print("training env/seed/method:", training_config["env"], training_config["seed"], training_config["method"])
	print("eval_seeds:", args.eval_seeds)
	print("task_sequence:", task_names)

	rows = []
	for eval_seed in args.eval_seeds:
		rows.extend(
			evaluate_seed(
				eval_seed,
				training_config,
				agent,
				model_name,
				checkpoint_idx,
				args.num_eval_runs,
				args.reseed_each_episode,
			)
		)

	output_path = (
		resolve_existing_path(args.output)
		if args.output
		else os.path.join(model_dir, f"{model_name}_multi_seed_eval.csv")
	)
	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	results = pd.DataFrame(rows)
	results.to_csv(output_path, index=False)
	task_summary = build_task_summary(results)
	task_summary_path = output_with_suffix(output_path, "_summary")
	task_summary.to_csv(task_summary_path, index=False)
	accuracy_summary = build_average_accuracy_table(results)
	accuracy_summary_path = output_with_suffix(output_path, "_average_accuracy")
	accuracy_summary.to_csv(accuracy_summary_path, index=False)

	overall = accuracy_summary.iloc[-1]
	print("results:", output_path)
	print("summary:", task_summary_path)
	print(
		f"average accuracy: {overall['average_accuracy']:.4f} "
		f"({overall['average_accuracy_percent']:.2f}%) +/- "
		f"{overall['accuracy_std_across_seeds']:.4f} across seeds"
	)
	print("average accuracy summary:", accuracy_summary_path)


if __name__ == "__main__":
	main()
