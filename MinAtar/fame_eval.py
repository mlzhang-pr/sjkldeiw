import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from CL_envs import CL_envs_func_replacement
from model import CNN


GAME_NAMES = ("breakout", "space_invaders", "freeway")
DQN_GAME_BENCHMARKS = {
	"breakout": 12.7,
	"space_invaders": 30.9,
	"freeway": 3.04,
}
DEFAULT_RESET_TEMPLATE = (
	"DQN_env_name_all_gamma_0.99_steps_3500000_switch_500000_batch_64_"
	"lr1_1e-05_seq_{seq}_reset_1_seed_{seed}_returns.pkl"
)
DEFAULT_BENCHMARK_TEMPLATES = (
	(
		"Finetune",
		"DQN_env_name_all_gamma_0.99_steps_3500000_switch_500000_batch_64_"
		"lr1_1e-05_seq_{seq}_reset_0_seed_{seed}_returns.pkl",
	),
	(
		"PT-DQN",
		"PT_DQN_0.5x_env_name_all_gamma_0.99_steps_3500000_switch_500000_"
		"update_50000_decay_0.75_lr1_1e-08_lr2_0.0001_batch_64_seq_{seq}_"
		"CNNhalf_1_boundary0_reset_1_seed_{seed}_returns.pkl",
	),
	(
		"MultiHead",
		"DQN_multi_task_env_name_all_gamma_0.99_steps_3500000_switch_500000_"
		"batch_64_lr1_1e-05_seq_{seq}_reset_1_clearbuffer_1_seed_{seed}_"
		"returns.pkl",
	),
	(
		"LargeBuffer",
		"DQN_large_buffer_env_name_all_gamma_0.99_steps_3500000_switch_"
		"500000_batch_64_lr1_0.0001_seq_{seq}_reset_1_clearbuffer_1_"
		"seed_{seed}_returns.pkl",
	),
)
FAME_RUN_PATTERN = re.compile(
	r"(?P<prefix>.+)_seq_(?P<sequence>\d+)"
	r"(?P<middle>_.+)_seed_(?P<training_seed>\d+)_Meta(?P<stage>\d+)\.pt"
)


def moving_average(values, window):
	values = np.asarray(values, dtype=np.float64)
	if values.ndim != 1:
		raise ValueError("Return curves must be one-dimensional.")
	if not 0 < window <= len(values):
		raise ValueError("Smoothing window must fit within the return curve.")
	cumulative = np.cumsum(np.insert(values, 0, 0.0))
	smoothed = (cumulative[window:] - cumulative[:-window]) / window
	return np.concatenate((values[: window - 1] / window, smoothed))


def compute_forgetting(
	training_returns, task_games, final_game_returns, switch_steps, smoothing_window
):
	training_returns = moving_average(training_returns, smoothing_window)
	expected_steps = len(task_games) * switch_steps
	if len(training_returns) != expected_steps:
		raise ValueError(
			f"Expected {expected_steps} training returns for {len(task_games)} tasks, "
			f"got {len(training_returns)}."
		)

	per_game_samples = {game: [] for game in GAME_NAMES}
	for task_id, game in enumerate(task_games):
		if game not in final_game_returns:
			raise ValueError(f"Missing final evaluation for game {game!r}.")
		stage_end = training_returns[(task_id + 1) * switch_steps - 1]
		per_game_samples[game].append(stage_end - final_game_returns[game])

	per_game = {
		game: float(np.mean(samples)) if samples else 0.0
		for game, samples in per_game_samples.items()
	}
	return {
		"mean": float(np.mean(list(per_game.values()))),
		"per_game": per_game,
		"task_samples": [
			float(sample)
			for game in GAME_NAMES
			for sample in per_game_samples[game]
		],
	}


def normalize_forgetting(forgetting):
	per_game = {
		game: forgetting["per_game"][game] / DQN_GAME_BENCHMARKS[game]
		for game in GAME_NAMES
	}
	return {"mean": float(np.mean(list(per_game.values()))), "per_game": per_game}


def compute_forward_transfer(
	method_returns, reset_returns, benchmark_returns, switch_steps
):
	method_returns = np.asarray(method_returns, dtype=np.float64)
	reset_returns = np.asarray(reset_returns, dtype=np.float64)
	benchmark_returns = [
		np.asarray(values, dtype=np.float64) for values in benchmark_returns
	]
	if method_returns.ndim != 1 or reset_returns.shape != method_returns.shape:
		raise ValueError("Method and reset return curves must be matching vectors.")
	if not benchmark_returns or any(
		values.shape != method_returns.shape for values in benchmark_returns
	):
		raise ValueError("Benchmark return curves must match the method curve.")
	if switch_steps <= 0 or len(method_returns) % switch_steps:
		raise ValueError("Return curve length must be divisible by switch-steps.")

	scores = []
	for start in range(0, len(method_returns), switch_steps):
		stop = start + switch_steps
		method_mean = method_returns[start:stop].mean()
		reset_mean = reset_returns[start:stop].mean()
		benchmark = max(values[start:stop].max() for values in benchmark_returns)
		denominator = benchmark - reset_mean
		scores.append(
			float((method_mean - reset_mean) / denominator)
			if denominator != 0
			else None
		)
	return scores


def summarize_samples(values):
	values = np.asarray(values, dtype=np.float64)
	values = values[np.isfinite(values)]
	if not len(values):
		return {"mean": None, "std": None, "stderr": None, "count": 0}
	return {
		"mean": float(values.mean()),
		"std": float(values.std()),
		"stderr": float(values.std() / math.sqrt(len(values))),
		"count": int(len(values)),
	}


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate trained FAME meta policies.")
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument(
		"--aggregate",
		action="store_true",
		help=(
			"Evaluate the final checkpoint for every matching sequence/seed and "
			"compute the paper's performance, FT, and forgetting metrics."
		),
	)
	parser.add_argument(
		"--stage-eval",
		action="store_true",
		help="Evaluate every sibling MetaN checkpoint on tasks seen by that stage.",
	)
	parser.add_argument("--seq", type=int, default=0)
	parser.add_argument("--task-ids", type=int, nargs="+", default=None)
	parser.add_argument("--episodes", type=int, default=100)
	parser.add_argument("--max-steps", type=int, default=300)
	parser.add_argument(
		"--evaluation-steps",
		type=int,
		default=6_000,
		help="Environment-step budget per game in aggregate paper-metric mode.",
	)
	parser.add_argument("--seed", type=int, default=1000)
	parser.add_argument("--torch-threads", type=int, default=1)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Torch device, for example cpu or cuda:0 (auto-detected by default).",
	)
	parser.add_argument(
		"--returns-dir",
		type=Path,
		default=None,
		help="Directory containing training-return pickle files (default: beside models).",
	)
	parser.add_argument("--switch-steps", type=int, default=500_000)
	parser.add_argument("--smoothing-window", type=int, default=10_000)
	parser.add_argument("--reset-template", default=DEFAULT_RESET_TEMPLATE)
	parser.add_argument(
		"--benchmark-template",
		action="append",
		default=None,
		metavar="LABEL=TEMPLATE",
		help=(
			"Additional FT benchmark curve template using {seq} and {seed}; "
			"repeat for multiple methods. Defaults to Finetune and PT-DQN."
		),
	)
	parser.add_argument("--output-json", type=Path, default=None)
	return parser.parse_args()


def checkpoint_stage(path):
	match = re.fullmatch(r"(.+)_Meta(\d+)\.pt", path.name)
	if match is None:
		raise ValueError("FAME checkpoint filename must end with _MetaN.pt.")
	return int(match.group(2))


def discover_stage_checkpoints(path):
	match = re.fullmatch(r"(.+)_Meta(\d+)\.pt", path.name)
	if match is None:
		raise ValueError("FAME checkpoint filename must end with _MetaN.pt.")
	prefix = match.group(1)
	pattern = re.compile(rf"{re.escape(prefix)}_Meta(\d+)\.pt")
	stage_paths = {}
	for candidate in path.parent.iterdir():
		candidate_match = pattern.fullmatch(candidate.name)
		if candidate_match:
			stage_paths[int(candidate_match.group(1))] = candidate
	if not stage_paths:
		raise FileNotFoundError(f"No FAME stage checkpoints found beside {path}.")
	missing = sorted(set(range(max(stage_paths) + 1)).difference(stage_paths))
	if missing:
		raise FileNotFoundError(f"FAME stage series is missing stages {missing}.")
	return dict(sorted(stage_paths.items()))


def parse_fame_run(path):
	match = FAME_RUN_PATTERN.fullmatch(path.name)
	if match is None:
		raise ValueError(
			"Aggregate metrics require a checkpoint filename containing "
			"_seq_N_..._seed_N_MetaN.pt."
		)
	return {
		"path": path,
		"prefix": match.group("prefix"),
		"middle": match.group("middle"),
		"sequence": int(match.group("sequence")),
		"training_seed": int(match.group("training_seed")),
		"stage": int(match.group("stage")),
		"run_name": path.name[: path.name.rfind("_Meta")],
	}


def discover_metric_runs(path):
	stage_paths = discover_stage_checkpoints(path)
	final_stage = max(stage_paths)
	reference = parse_fame_run(stage_paths[final_stage])
	family = (reference["prefix"], reference["middle"], reference["stage"])
	runs = []
	seen = set()
	for candidate in path.parent.glob(f"*_Meta{final_stage}.pt"):
		try:
			run = parse_fame_run(candidate)
		except ValueError:
			continue
		if (run["prefix"], run["middle"], run["stage"]) != family:
			continue
		key = (run["sequence"], run["training_seed"])
		if key in seen:
			raise ValueError(f"Duplicate FAME checkpoint for seq/seed {key}.")
		seen.add(key)
		runs.append(run)
	if not runs:
		raise FileNotFoundError(f"No matching final FAME checkpoints beside {path}.")
	return sorted(runs, key=lambda run: (run["sequence"], run["training_seed"]))


def load_return_curve(path):
	if not path.is_file():
		raise FileNotFoundError(f"Return curve does not exist: {path}")
	with path.open("rb") as input_file:
		values = np.asarray(pickle.load(input_file), dtype=np.float64)
	if values.ndim != 1 or not np.all(np.isfinite(values)):
		raise ValueError(f"Return curve must be a finite vector: {path}")
	return values


def parse_benchmark_templates(values):
	templates = list(DEFAULT_BENCHMARK_TEMPLATES)
	labels = {label for label, _ in templates}
	for value in values or []:
		label, separator, template = value.partition("=")
		if not separator or not label or not template:
			raise ValueError(
				"benchmark-template must have the form LABEL=TEMPLATE."
			)
		if label in labels:
			raise ValueError(f"Duplicate FT benchmark label: {label}")
		templates.append((label, template))
		labels.add(label)
	return templates


def resolve_return_path(returns_dir, template, sequence, training_seed):
	path = Path(template.format(seq=sequence, seed=training_seed))
	return path if path.is_absolute() else returns_dir / path


def select_global_benchmark_templates(
	runs, returns_dir, reset_template, benchmark_templates
):
	ft_runs = [
		run
		for run in runs
		if resolve_return_path(
			returns_dir,
			reset_template,
			run["sequence"],
			run["training_seed"],
		).is_file()
	]
	selected = []
	excluded = []
	for label, template in benchmark_templates:
		missing_runs = [
			{
				"sequence": run["sequence"],
				"training_seed": run["training_seed"],
			}
			for run in ft_runs
			if not resolve_return_path(
				returns_dir,
				template,
				run["sequence"],
				run["training_seed"],
			).is_file()
		]
		if missing_runs:
			excluded.append({"label": label, "missing_runs": missing_runs})
		else:
			selected.append((label, template))
	return selected, excluded


def load_model(path, device):
	if not path.is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {path}")
	try:
		state_dict = torch.load(path, map_location=device, weights_only=True)
	except TypeError:
		state_dict = torch.load(path, map_location=device)
	in_channels = state_dict["conv.weight"].shape[1]
	num_actions = state_dict["output.weight"].shape[0]
	model = CNN(in_channels, num_actions).to(device)
	model.load_state_dict(state_dict)
	model.eval()
	return model, in_channels, num_actions


@torch.inference_mode()
def select_action(model, observation, device):
	observation = np.moveaxis(np.asarray(observation, dtype=np.float32), 2, 0)
	observation = torch.as_tensor(observation, device=device).unsqueeze(0)
	return int(model(observation).argmax(dim=1).item())


def evaluate(
	model,
	task_id,
	args,
	device,
	in_channels,
	num_actions,
	sequence=None,
	evaluation_seed=None,
	canonical_game=False,
):
	sequence = args.seq if sequence is None else sequence
	evaluation_seed = args.seed if evaluation_seed is None else evaluation_seed
	environment = CL_envs_func_replacement(
		sequence, task_id, evaluation_seed, evaluation=canonical_game
	)
	if environment.observation_space.shape[2] != in_channels:
		raise ValueError("Environment and checkpoint observation dimensions differ.")
	if environment.action_space.n != num_actions:
		raise ValueError("Environment and checkpoint action dimensions differ.")

	returns = []
	lengths = []
	truncated_episodes = 0
	try:
		for episode in range(args.episodes):
			observation = environment.reset(seed=evaluation_seed + episode)
			episode_return = 0.0
			done = False
			for episode_step in range(1, args.max_steps + 1):
				action = select_action(model, observation, device)
				observation, reward, done, _ = environment.step(action)
				episode_return += reward
				if done:
					break
			if not done:
				truncated_episodes += 1
			returns.append(float(episode_return))
			lengths.append(episode_step)
	finally:
		environment.close()

	return {
		"task": task_id,
		"game": environment.game_name,
		"episodes": args.episodes,
		"mean_return": float(np.mean(returns)),
		"std_return": float(np.std(returns)),
		"stderr_return": float(np.std(returns) / math.sqrt(args.episodes)),
		"median_return": float(np.median(returns)),
		"mean_length": float(np.mean(lengths)),
		"truncated_episodes": truncated_episodes,
		"returns": returns,
		"lengths": lengths,
	}


def evaluate_for_steps(
	model,
	game_id,
	args,
	device,
	in_channels,
	num_actions,
	sequence,
	evaluation_seed,
):
	environment = CL_envs_func_replacement(
		sequence, game_id, evaluation_seed, evaluation=True
	)
	if environment.observation_space.shape[2] != in_channels:
		raise ValueError("Environment and checkpoint observation dimensions differ.")
	if environment.action_space.n != num_actions:
		raise ValueError("Environment and checkpoint action dimensions differ.")

	returns = []
	lengths = []
	truncated_episodes = 0
	episode_return = 0.0
	episode_length = 0
	try:
		observation = environment.reset()
		for _ in range(args.evaluation_steps):
			action = select_action(model, observation, device)
			observation, reward, done, _ = environment.step(action)
			episode_return += reward
			episode_length += 1
			if done or episode_length >= args.max_steps:
				if not done:
					truncated_episodes += 1
				returns.append(float(episode_return))
				lengths.append(episode_length)
				observation = environment.reset()
				episode_return = 0.0
				episode_length = 0
	finally:
		environment.close()

	if not returns:
		raise RuntimeError(
			f"No episode completed within {args.evaluation_steps} evaluation steps."
		)
	return {
		"task": game_id,
		"game": environment.game_name,
		"evaluation_steps": args.evaluation_steps,
		"episodes": len(returns),
		"mean_return": float(np.mean(returns)),
		"std_return": float(np.std(returns)),
		"stderr_return": float(np.std(returns) / math.sqrt(len(returns))),
		"median_return": float(np.median(returns)),
		"mean_length": float(np.mean(lengths)),
		"truncated_episodes": truncated_episodes,
		"partial_episode_steps": episode_length,
		"returns": returns,
		"lengths": lengths,
	}


def task_games(sequence, task_count, seed):
	games = []
	for task_id in range(task_count):
		environment = CL_envs_func_replacement(sequence, task_id, seed)
		try:
			games.append(environment.game_name)
		finally:
			environment.close()
	return games


def aggregate_forward_transfer(run_records, switch_steps):
	by_sequence = defaultdict(list)
	excluded_runs = []
	for record in run_records:
		if "Reset" not in record["benchmark_returns"]:
			excluded_runs.append(
				{
					"sequence": record["sequence"],
					"training_seed": record["training_seed"],
					"reason": "missing Reset return curve",
				}
			)
			continue
		by_sequence[record["sequence"]].append(record)

	sequence_scores = {}
	all_scores = []
	benchmark_methods = None
	for sequence, records in sorted(by_sequence.items()):
		method_returns = np.mean(
			[record["training_returns"] for record in records], axis=0
		)
		reset_returns = np.mean(
			[record["benchmark_returns"]["Reset"] for record in records], axis=0
		)
		methods = set.intersection(
			*(set(record["benchmark_returns"]) for record in records)
		)
		if benchmark_methods is None:
			benchmark_methods = methods
		else:
			benchmark_methods.intersection_update(methods)
		benchmark_returns = [
			np.mean(
				[record["benchmark_returns"][method] for record in records],
				axis=0,
			)
			for method in sorted(methods)
		]
		scores = compute_forward_transfer(
			method_returns, reset_returns, benchmark_returns, switch_steps
		)
		sequence_scores[str(sequence)] = scores
		all_scores.extend(scores)

	return {
		**summarize_samples(all_scores),
		"sequence_stage_scores": sequence_scores,
		"benchmark_methods": sorted(benchmark_methods or []),
		"seed_aggregation": "curves averaged within each sequence before FT",
		"excluded_runs": excluded_runs,
	}


def print_aggregate_metrics(metrics):
	print("\nFAME metrics (mean +/- standard error)")
	print(f"{'metric':<20} {'mean':>10} {'stderr':>10} {'n':>6}")
	for game in GAME_NAMES:
		result = metrics["average_performance"][game]
		print(
			f"{game:<20} {result['mean']:>10.3f} "
			f"{result['stderr']:>10.3f} {result['count']:>6}"
		)
	for name in ("forward_transfer", "forgetting"):
		result = metrics[name]
		print(
			f"{name:<20} {result['mean']:>10.3f} "
			f"{result['stderr']:>10.3f} {result['count']:>6}"
		)
	print(
		"FT benchmark methods: "
		+ ", ".join(metrics["forward_transfer"]["benchmark_methods"])
	)


def run_aggregate(args, device):
	runs = discover_metric_runs(args.checkpoint)
	returns_dir = args.returns_dir
	if returns_dir is None:
		returns_dir = args.checkpoint.parent.parent / "results"
	benchmark_templates = parse_benchmark_templates(args.benchmark_template)
	benchmark_templates, excluded_benchmark_methods = (
		select_global_benchmark_templates(
			runs, returns_dir, args.reset_template, benchmark_templates
		)
	)
	aggregate_records = []
	serialized_runs = []
	sequence_ft_results = []

	for _, sequence_runs in groupby(runs, key=lambda run: run["sequence"]):
		sequence_records = []
		for run in sequence_runs:
			sequence = run["sequence"]
			training_seed = run["training_seed"]
			model, in_channels, num_actions = load_model(run["path"], device)
			game_evaluations = {}
			for game_id, game in enumerate(GAME_NAMES):
				game_evaluations[game] = evaluate_for_steps(
					model,
					game_id,
					args,
					device,
					in_channels,
					num_actions,
					sequence,
					training_seed,
				)

			training_path = returns_dir / f"{run['run_name']}_returns.pkl"
			training_returns = load_return_curve(training_path)
			reset_path = resolve_return_path(
				returns_dir, args.reset_template, sequence, training_seed
			)
			benchmark_returns = {
				"FAME": moving_average(training_returns, args.smoothing_window)
			}
			benchmark_paths = {"FAME": training_path}
			missing_benchmarks = []
			if reset_path.is_file():
				benchmark_returns["Reset"] = moving_average(
					load_return_curve(reset_path), args.smoothing_window
				)
				benchmark_paths["Reset"] = reset_path
			else:
				missing_benchmarks.append(
					{"label": "Reset", "path": str(reset_path)}
				)
			for label, template in benchmark_templates:
				if label in benchmark_returns:
					raise ValueError(f"Duplicate FT benchmark label: {label}")
				benchmark_path = resolve_return_path(
					returns_dir, template, sequence, training_seed
				)
				if benchmark_path.is_file():
					benchmark_returns[label] = moving_average(
						load_return_curve(benchmark_path), args.smoothing_window
					)
					benchmark_paths[label] = benchmark_path
				else:
					missing_benchmarks.append(
						{"label": label, "path": str(benchmark_path)}
					)

			curve_lengths = {len(values) for values in benchmark_returns.values()}
			if curve_lengths != {len(training_returns)}:
				raise ValueError(
					f"FT curves have different lengths for seq={sequence}, "
					f"seed={training_seed}."
				)
			games = task_games(sequence, run["stage"] + 1, training_seed)
			final_game_returns = {
				game: evaluation["mean_return"]
				for game, evaluation in game_evaluations.items()
			}
			forgetting = compute_forgetting(
				training_returns,
				games,
				final_game_returns,
				args.switch_steps,
				args.smoothing_window,
			)
			normalized_forgetting = normalize_forgetting(forgetting)
			sequence_records.append(
				{
					"sequence": sequence,
					"training_seed": training_seed,
					"training_returns": benchmark_returns["FAME"],
					"benchmark_returns": benchmark_returns,
				}
			)
			aggregate_records.append(
				{
					"final_game_returns": final_game_returns,
					"forgetting": forgetting,
					"normalized_forgetting": normalized_forgetting,
				}
			)
			serialized_runs.append(
				{
					"sequence": sequence,
					"training_seed": training_seed,
					"checkpoint": str(run["path"]),
					"training_returns": str(training_path),
					"benchmark_returns": {
						label: str(path) for label, path in benchmark_paths.items()
					},
					"missing_benchmarks": missing_benchmarks,
					"task_games": games,
					"game_evaluations": game_evaluations,
					"forgetting": forgetting,
					"normalized_forgetting": normalized_forgetting,
				}
			)
			print(
				f"evaluated seq={sequence}, seed={training_seed}: "
				f"{final_game_returns}"
			)
		sequence_ft_results.append(
			aggregate_forward_transfer(sequence_records, args.switch_steps)
		)

	all_ft_scores = [
		score
		for result in sequence_ft_results
		for scores in result["sequence_stage_scores"].values()
		for score in scores
	]
	valid_method_sets = [
		set(result["benchmark_methods"])
		for result in sequence_ft_results
		if result["count"]
	]
	forward_transfer = {
		**summarize_samples(all_ft_scores),
		"sequence_stage_scores": {
			sequence: scores
			for result in sequence_ft_results
			for sequence, scores in result["sequence_stage_scores"].items()
		},
		"benchmark_methods": sorted(
			set.intersection(*valid_method_sets) if valid_method_sets else set()
		),
		"seed_aggregation": "curves averaged within each sequence before FT",
		"excluded_runs": [
			run
			for result in sequence_ft_results
			for run in result["excluded_runs"]
		],
		"excluded_benchmark_methods": excluded_benchmark_methods,
		"paper_benchmark_complete": not excluded_benchmark_methods,
	}

	metrics = {
		"average_performance": {
			game: summarize_samples(
				[record["final_game_returns"][game] for record in aggregate_records]
			)
			for game in GAME_NAMES
		},
		"forward_transfer": forward_transfer,
		"forgetting": summarize_samples(
			[
				record["normalized_forgetting"]["mean"]
				for record in aggregate_records
			]
		),
		"forgetting_raw": summarize_samples(
			[record["forgetting"]["mean"] for record in aggregate_records]
		),
		"forgetting_per_game": {
			game: summarize_samples(
				[
					record["forgetting"]["per_game"][game]
					for record in aggregate_records
				]
			)
			for game in GAME_NAMES
		},
		"forgetting_per_game_normalized": {
			game: summarize_samples(
				[
					record["normalized_forgetting"]["per_game"][game]
					for record in aggregate_records
				]
			)
			for game in GAME_NAMES
		},
	}
	print_aggregate_metrics(metrics)
	return {
		"checkpoint_family": str(args.checkpoint),
		"returns_dir": str(returns_dir),
		"evaluation_steps": args.evaluation_steps,
		"max_steps": args.max_steps,
		"switch_steps": args.switch_steps,
		"smoothing_window": args.smoothing_window,
		"forgetting_normalizers": DQN_GAME_BENCHMARKS,
		"missing_game_policy": "zero forgetting for games absent from a sequence",
		"runs": serialized_runs,
		"metrics": metrics,
	}


def summarize(results):
	summaries = []
	for stage in sorted({result["checkpoint_stage"] for result in results}):
		stage_results = [
			result for result in results if result["checkpoint_stage"] == stage
		]
		games = sorted({result["game"] for result in stage_results})
		summaries.append(
			{
				"checkpoint_stage": stage,
				"evaluated_tasks": len(stage_results),
				"average_task_return": float(
					np.mean([result["mean_return"] for result in stage_results])
				),
				"per_game_mean_return": {
					game: float(
						np.mean(
							[
								result["mean_return"]
								for result in stage_results
								if result["game"] == game
							]
						)
					)
					for game in games
				},
			}
		)
	return summaries


def print_results(results, summaries):
	print(
		f"{'stage':>5} {'task':>5} {'game':<16} "
		f"{'return (mean +/- std)':>23} {'length':>8} {'trunc':>6}"
	)
	for result in results:
		print(
			f"{result['checkpoint_stage']:>5} {result['task']:>5} "
			f"{result['game']:<16} "
			f"{result['mean_return']:>9.3f} +/- {result['std_return']:<7.3f} "
			f"{result['mean_length']:>8.1f} {result['truncated_episodes']:>6}"
		)
	print("\nstage average performance (un-normalized across seen tasks)")
	print(f"{'stage':>5} {'tasks':>5} {'average return':>14}")
	for summary in summaries:
		print(
			f"{summary['checkpoint_stage']:>5} {summary['evaluated_tasks']:>5} "
			f"{summary['average_task_return']:>14.3f}"
		)


def main():
	args = parse_args()
	if args.episodes <= 0 or args.max_steps <= 0:
		raise ValueError("episodes and max-steps must be positive.")
	if args.evaluation_steps <= 0:
		raise ValueError("evaluation-steps must be positive.")
	if args.torch_threads <= 0:
		raise ValueError("torch-threads must be positive.")
	if args.aggregate and args.stage_eval:
		raise ValueError("aggregate and stage-eval cannot be combined.")
	if args.aggregate and args.task_ids is not None:
		raise ValueError("task-ids cannot be used with aggregate metrics.")
	if args.switch_steps <= 0 or args.smoothing_window <= 0:
		raise ValueError("switch-steps and smoothing-window must be positive.")
	torch.set_num_threads(args.torch_threads)
	device = torch.device(
		args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
	)
	if args.aggregate:
		payload = run_aggregate(args, device)
		if args.output_json is not None:
			args.output_json.parent.mkdir(parents=True, exist_ok=True)
			with args.output_json.open("w", encoding="utf-8") as output_file:
				json.dump(payload, output_file, indent=2, allow_nan=False)
			print(f"saved: {args.output_json}")
		return
	if args.stage_eval:
		checkpoint_paths = discover_stage_checkpoints(args.checkpoint)
	else:
		stage = checkpoint_stage(args.checkpoint)
		checkpoint_paths = {stage: args.checkpoint}

	results = []
	for stage, path in checkpoint_paths.items():
		model, in_channels, num_actions = load_model(path, device)
		task_ids = args.task_ids or list(range(stage + 1))
		task_ids = [task_id for task_id in task_ids if task_id <= stage]
		for task_id in task_ids:
			if not 0 <= task_id < 7:
				raise ValueError("Task IDs must be between 0 and 6.")
			result = evaluate(
				model, task_id, args, device, in_channels, num_actions
			)
			result["checkpoint_stage"] = stage
			result["checkpoint"] = str(path)
			results.append(result)

	if not results:
		raise ValueError("No evaluation tasks remain for the selected stages.")
	summaries = summarize(results)
	print(f"checkpoint: {args.checkpoint}; stages: {list(checkpoint_paths)}")
	print(f"sequence: {args.seq}; device: {device}; episodes: {args.episodes}")
	print_results(results, summaries)
	if args.output_json is not None:
		args.output_json.parent.mkdir(parents=True, exist_ok=True)
		with args.output_json.open("w", encoding="utf-8") as output_file:
			json.dump(
				{
					"checkpoint": str(args.checkpoint),
					"stage_eval": args.stage_eval,
					"sequence": args.seq,
					"seed": args.seed,
					"results": results,
					"stage_summaries": summaries,
				},
				output_file,
				indent=2,
				allow_nan=False,
			)
		print(f"saved: {args.output_json}")


if __name__ == "__main__":
	main()

'''
cd MinAtar

conda run -n RLL3 python fame_eval.py \
  --checkpoint models/FAME_steps_3500000_switch_500000_update_50000_lr1_0.001_lr2_1e-05_size_fast2meta_12000_detection_step_600_seq_0_epoch_meta_200_warmstep_50000_lambda_reg_1.0_seed_0_Meta6.pt \
  --aggregate \
  --device cpu \
  --evaluation-steps 6000 \
  --max-steps 300 \
  --output-json results/fame_metrics.json
'''