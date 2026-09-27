"""Evaluate a CONQUEST meta policy on completed tasks and seeds."""

import argparse
import json
import os
import random
import re
from dataclasses import fields

import numpy as np
import pandas as pd
import torch
import yaml

from agent.quasimetric import QuasimetricConfig
from agent.quasimetric.awr_agent import ContinualQuasimetricAWRAgent
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
	config_path = next((path for path in candidates if os.path.isfile(path)), None)
	if config_path is None:
		raise FileNotFoundError(
			f"No training config found in {run_dir}. Pass --config explicitly."
		)
	return config_path


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
	env_keys = {
		"env",
		"base_task_name",
		"goal_hidden",
		"normalize_obs",
		"normalize_rewards",
		"change_freq",
		"env_sequence",
		"obs_drift_mean",
		"obs_drift_std",
		"obs_scale_drift",
		"obs_noise_std",
		"normalize_avg_coef",
		"reset_obs_stats",
		"change_when_solved",
		"freeze_rand_vec",
		"goal_conditioned",
		"gc_reward_type",
		"gc_success_threshold",
		"gc_achieved_goal",
	}
	env_kwargs = {
		key: value
		for key, value in training_config.items()
		if key in env_keys
	}

	env_name = str(training_config.get("env", "metaworld_sequence_set6")).lower()
	if env_name.startswith("metaworld_sequence_"):
		env_suffix = env_name[len("metaworld_sequence_"):]
		if env_suffix.startswith("set"):
			env_kwargs["env_sequence"] = env_suffix
		else:
			env_kwargs["base_task_name"] = f"{env_suffix}-v2"
			env_kwargs["env_sequence"] = None

	env_kwargs.update(
		capture_video=False,
		env_type="rl",
		seed=eval_seed,
	)
	return env_kwargs


def checkpoint_path(model_dir, model_name, suffix):
	return os.path.join(model_dir, f"{model_name}_{suffix}.pt")


def expected_log_name(training_config):
	env_name = training_config.get("env", "metaworld_sequence_set6")
	training_seed = training_config.get("seed", 0)
	method = training_config.get("method", "buffer")
	gc_suffix = ""
	if bool(training_config.get("goal_conditioned", 0)):
		gc_suffix = f"_gc-{training_config.get('gc_reward_type', 'sparse')}"
	return f"sac_{env_name}_{training_seed}_{method}{gc_suffix}"


def latest_meta_checkpoint(model_dir, training_config, requested_name=None):
	if requested_name is not None:
		model_name = requested_name
		match = re.search(r"_(\d+)_meta$", model_name)
		checkpoint_idx = int(match.group(1)) if match else -1
	else:
		prefix = expected_log_name(training_config)
		pattern = re.compile(
			rf"^{re.escape(prefix)}_(\d+)_meta_actor\.pt$"
		)
		candidates = []
		for filename in os.listdir(model_dir):
			match = pattern.fullmatch(filename)
			if match:
				checkpoint_idx = int(match.group(1))
				model_name = filename[: -len("_actor.pt")]
				candidates.append((checkpoint_idx, model_name))
		if not candidates:
			raise FileNotFoundError(
				f"No AWR meta actor matching {prefix}_N_meta_actor.pt in {model_dir}. "
				"Pass --model_name if the checkpoint prefix differs."
			)
		checkpoint_idx, model_name = max(candidates, key=lambda item: item[0])

	missing = [
		checkpoint_path(model_dir, model_name, suffix)
		for suffix in ("actor", "awr", "quasimetric")
		if not os.path.isfile(checkpoint_path(model_dir, model_name, suffix))
	]
	if missing:
		raise FileNotFoundError(
			"Incomplete AWR meta checkpoint; missing: " + ", ".join(missing)
		)
	return model_name, checkpoint_idx


def load_torch(path, device, weights_only=False):
	try:
		return torch.load(path, map_location=device, weights_only=weights_only)
	except TypeError:
		return torch.load(path, map_location=device)


def quasimetric_config_from_payload(payload, obs_dim, action_dim):
	saved_config = payload.get("quasimetric_cfg") or {}
	allowed = {field.name for field in fields(QuasimetricConfig)}
	config = QuasimetricConfig(
		**{key: value for key, value in saved_config.items() if key in allowed}
	)
	if "transition_input" in saved_config:
		return config

	model_state = payload.get("quasimetric", {}).get("model_state", {})
	transition_weight = model_state.get("latent_transition_encoder.trunk.0.weight")
	if transition_weight is None:
		return config

	checkpoint_state_dim = transition_weight.shape[1] - action_dim
	if checkpoint_state_dim == obs_dim:
		config.transition_input = "state"
	elif checkpoint_state_dim == config.latent_dim:
		config.transition_input = "latent"
	else:
		raise ValueError(
			"Cannot infer transition_input from checkpoint: "
			f"input_dim={transition_weight.shape[1]}, obs_dim={obs_dim}, "
			f"action_dim={action_dim}, latent_dim={config.latent_dim}."
		)
	return config


def build_and_load_meta_agent(
	obs_dim,
	action_dim,
	device,
	training_config,
	model_dir,
	model_name,
):
	awr_path = checkpoint_path(model_dir, model_name, "awr")
	quasimetric_path = checkpoint_path(model_dir, model_name, "quasimetric")
	awr_payload = load_torch(awr_path, device, weights_only=False)
	quasimetric_payload = load_torch(quasimetric_path, device, weights_only=False)
	quasimetric_config = quasimetric_config_from_payload(
		quasimetric_payload,
		obs_dim,
		action_dim,
	)

	agent = ContinualQuasimetricAWRAgent(
		obs_dim=obs_dim,
		action_dim=action_dim,
		action_range=[-1.0, 1.0],
		device=device,
		batch_size=int(training_config.get("batch_size", 256)),
		actor_lr=float(training_config.get("actor_lr", 1e-4)),
		awr_beta=float(awr_payload.get("awr_beta", training_config.get("awr_beta", 1.0))),
		awr_weight_clip=float(
			awr_payload.get("awr_weight_clip", training_config.get("awr_weight_clip", 20.0))
		),
		awr_num_goals=int(
			awr_payload.get("awr_num_goals", training_config.get("awr_num_goals", 4))
		),
		quasimetric_cfg=quasimetric_config,
	)
	actor_state = load_torch(
		checkpoint_path(model_dir, model_name, "actor"),
		device,
		weights_only=True,
	)
	agent.actor.load_state_dict(actor_state)
	agent.quasimetric.load_checkpoint(quasimetric_payload["quasimetric"])
	agent.eval()
	return agent


def vector_observation_space(observation_space):
	spaces = getattr(observation_space, "spaces", None)
	if spaces is not None and "observation" in spaces:
		return spaces["observation"]
	return observation_space


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


def select_eval_tasks(task_names, checkpoint_idx, eval_all_tasks=False):
	task_names = list(task_names)
	if eval_all_tasks or checkpoint_idx < 0:
		return task_names
	return task_names[: checkpoint_idx + 1]


def evaluate_seed(
	eval_seed,
	training_config,
	agent,
	model_name,
	checkpoint_idx,
	num_eval_runs,
	reseed_each_episode,
	eval_all_tasks,
):
	set_seed_everywhere(eval_seed)
	env_kwargs = make_env_kwargs(training_config, eval_seed)
	env = MetaWorldSingleEnvSequence(**env_kwargs)
	rows = []
	try:
		env_list = select_eval_tasks(
			env.env_list,
			checkpoint_idx,
			eval_all_tasks,
		)
		print(f"eval_seed={eval_seed} tasks={len(env_list)}")
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


def summary_path(output_path):
	stem, extension = os.path.splitext(output_path)
	return f"{stem}_summary{extension or '.csv'}"


def average_accuracy_path(output_path):
	stem, extension = os.path.splitext(output_path)
	return f"{stem}_average_accuracy{extension or '.csv'}"


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
			"Load a CONQUEST meta policy and evaluate its completed tasks "
			"for multiple environment seeds."
		)
	)
	parser.add_argument("--run_dir", required=True, help="Training output directory")
	parser.add_argument("--config", default=None, help="Optional run_config.json or W&B config.yaml")
	parser.add_argument("--model_dir", default=None, help="Defaults to RUN_DIR/model")
	parser.add_argument("--model_name", default=None, help="Explicit checkpoint name without _actor.pt")
	parser.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2])
	parser.add_argument("--num_eval_runs", type=int, default=10)
	parser.add_argument("--reseed_each_episode", type=int, choices=[0, 1], default=0)
	parser.add_argument(
		"--eval_all_tasks",
		action="store_true",
		help="Evaluate unseen tasks after the selected checkpoint as well",
	)
	parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
	parser.add_argument("--output", default=None, help="Per-seed CSV output path")
	return parser.parse_args()


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	run_dir = resolve_existing_path(args.run_dir)
	if not os.path.isdir(run_dir):
		raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
	config_path = find_training_config(run_dir, args.config)
	training_config = load_training_config(config_path)
	model_dir = resolve_existing_path(args.model_dir) if args.model_dir else os.path.join(run_dir, "model")
	if not os.path.isdir(model_dir):
		raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

	model_name, checkpoint_idx = latest_meta_checkpoint(
		model_dir,
		training_config,
		args.model_name,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	set_seed_everywhere(args.eval_seeds[0])
	probe_env = MetaWorldSingleEnvSequence(
		**make_env_kwargs(training_config, args.eval_seeds[0])
	)
	try:
		obs_space = vector_observation_space(probe_env.env.observation_space)
		action_dim = probe_env.env.action_space.shape[0]
		task_names = select_eval_tasks(
			probe_env.env_list,
			checkpoint_idx,
			args.eval_all_tasks,
		)
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
	print("config:", config_path)
	print("model:", os.path.join(model_dir, model_name))
	print("device:", device)
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
				args.eval_all_tasks,
			)
		)

	output_path = (
		resolve_existing_path(args.output)
		if args.output
		else os.path.join(run_dir, f"{model_name}_multi_seed_eval.csv")
	)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	results = pd.DataFrame(rows)
	results.to_csv(output_path, index=False)

	aggregate = (
		results.groupby(["task_idx", "task"], as_index=False)
		.agg(
			eval_seed_count=("eval_seed", "nunique"),
			success_mean=("metaworld_success_mean", "mean"),
			success_std_across_seeds=("metaworld_success_mean", "std"),
			return_mean=("return_mean", "mean"),
			return_std_across_seeds=("return_mean", "std"),
		)
	)
	aggregate_output = summary_path(output_path)
	aggregate.to_csv(aggregate_output, index=False)
	average_accuracy = build_average_accuracy_table(results)
	average_accuracy_output = average_accuracy_path(output_path)
	average_accuracy.to_csv(average_accuracy_output, index=False)
	overall = average_accuracy.iloc[-1]
	print("results:", output_path)
	print("summary:", aggregate_output)
	print(
		f"average accuracy: {overall['average_accuracy']:.4f} "
		f"({overall['average_accuracy_percent']:.2f}%) +/- "
		f"{overall['accuracy_std_across_seeds']:.4f} across seeds"
	)
	print("average accuracy summary:", average_accuracy_output)


if __name__ == "__main__":
	main()
