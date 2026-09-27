import argparse
import json
import pickle
import re
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from conquest import QNetwork
from fame_eval import (
	DEFAULT_BENCHMARK_TEMPLATES,
	DEFAULT_RESET_TEMPLATE,
	DQN_GAME_BENCHMARKS,
	GAME_NAMES,
	aggregate_forward_transfer,
	compute_forgetting,
	evaluate_for_steps,
	moving_average,
	normalize_forgetting,
	resolve_return_path,
	select_global_benchmark_templates,
	summarize_samples,
)


METHOD_NAME = "CONQUEST"
FAME_RETURN_TEMPLATE = (
	"FAME_steps_3500000_switch_500000_update_50000_lr1_0.001_lr2_1e-05_"
	"size_fast2meta_12000_detection_step_600_seq_{seq}_epoch_meta_200_"
	"warmstep_50000_lambda_reg_1.0_seed_{seed}_returns.pkl"
)
DEFAULT_FT_BENCHMARK_TEMPLATES = (
	("FAME", FAME_RETURN_TEMPLATE),
	*DEFAULT_BENCHMARK_TEMPLATES,
)
RUN_PATTERN = re.compile(
	r"(?P<prefix>.+)_seq_(?P<sequence>\d+)_seed_"
	r"(?P<training_seed>\d+)_checkpoint\.pt"
)
IGNORED_CONFIG_KEYS = {
	"evaluation_episodes",
	"evaluation_max_steps",
	"evaluation_seed",
	"gpu",
	"log_interval",
	"output_dir",
	"save",
	"save_model",
	"seed",
	"seq",
}


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate CONQUEST's extracted shared policy and compute the MinAtar "
			"average-performance, forward-transfer, and forgetting metrics."
		)
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		required=True,
		help="Any final CONQUEST *_checkpoint.pt in the run family.",
	)
	parser.add_argument(
		"--evaluation-steps",
		type=int,
		default=6_000,
		help="Environment-step budget for each canonical game.",
	)
	parser.add_argument("--max-steps", type=int, default=300)
	parser.add_argument("--smoothing-window", type=int, default=10_000)
	parser.add_argument(
		"--switch-steps",
		type=int,
		default=None,
		help="Task duration; inferred from checkpoint metadata by default.",
	)
	parser.add_argument("--torch-threads", type=int, default=1)
	parser.add_argument(
		"--device",
		default=None,
		help="Torch device such as cpu or cuda:0 (auto-detected by default).",
	)
	parser.add_argument(
		"--baseline-returns-dir",
		type=Path,
		default=None,
		help="Directory containing baseline return curves (default: parent results dir).",
	)
	parser.add_argument("--reset-template", default=DEFAULT_RESET_TEMPLATE)
	parser.add_argument(
		"--benchmark-template",
		action="append",
		default=None,
		metavar="LABEL=TEMPLATE",
		help=(
			"Additional FT benchmark template using {seq} and {seed}; may be "
			"repeated."
		),
	)
	parser.add_argument("--output-json", type=Path, default=None)
	return parser.parse_args()


def parse_run(path):
	match = RUN_PATTERN.fullmatch(path.name)
	if match is None:
		raise ValueError(
			"Checkpoint filename must end with "
			"_seq_N_seed_N_checkpoint.pt."
		)
	return {
		"path": path,
		"prefix": match.group("prefix"),
		"sequence": int(match.group("sequence")),
		"training_seed": int(match.group("training_seed")),
		"run_name": path.name[: -len("_checkpoint.pt")],
	}


def discover_runs(path):
	if not path.is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {path}")
	reference = parse_run(path)
	runs = []
	seen = set()
	for candidate in path.parent.glob("*_checkpoint.pt"):
		try:
			run = parse_run(candidate)
		except ValueError:
			continue
		if run["prefix"] != reference["prefix"]:
			continue
		key = (run["sequence"], run["training_seed"])
		if key in seen:
			raise ValueError(f"Duplicate final checkpoint for seq/seed {key}.")
		seen.add(key)
		runs.append(run)
	if not runs:
		raise FileNotFoundError(f"No matching final checkpoints beside {path}.")
	return sorted(runs, key=lambda run: (run["sequence"], run["training_seed"]))


def parse_benchmark_templates(values):
	templates = list(DEFAULT_FT_BENCHMARK_TEMPLATES)
	labels = {label for label, _ in templates}
	for value in values or []:
		label, separator, template = value.partition("=")
		if not separator or not label or not template:
			raise ValueError(
				"benchmark-template must have the form LABEL=TEMPLATE."
			)
		if label in labels or label in {METHOD_NAME, "Reset"}:
			raise ValueError(f"Duplicate FT benchmark label: {label}")
		templates.append((label, template))
		labels.add(label)
	return templates


def infer_baseline_returns_dir(checkpoint):
	candidates = [checkpoint.parent, checkpoint.parent.parent]
	for candidate in candidates:
		if next(
			candidate.glob(
				"DQN_env_name_all_*_reset_1_seed_*_returns.pkl"
			),
			None,
		) is not None:
			return candidate
	raise FileNotFoundError(
		"Could not infer the baseline return directory. Pass "
		"--baseline-returns-dir explicitly. Checked: "
		+ ", ".join(str(candidate) for candidate in candidates)
	)


def load_checkpoint(path, device):
	try:
		checkpoint = torch.load(path, map_location="cpu", weights_only=True)
	except TypeError:
		checkpoint = torch.load(path, map_location="cpu")
	required = {"extracted_policy", "metadata"}
	missing = required.difference(checkpoint)
	if missing:
		raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}")
	return checkpoint


def training_config(metadata):
	config = metadata.get("config") or {}
	return {
		key: value
		for key, value in config.items()
		if key not in IGNORED_CONFIG_KEYS and not key.startswith("wandb_")
	}


def validate_training_config(reference, current, path):
	if reference == current:
		return
	differing_keys = sorted(
		key
		for key in set(reference).union(current)
		if reference.get(key) != current.get(key)
	)
	raise ValueError(
		f"Checkpoint {path} has a different training configuration for: "
		+ ", ".join(differing_keys)
	)


def build_extracted_policy(checkpoint, device):
	state_dict = checkpoint["extracted_policy"]
	try:
		in_channels = state_dict["encoder.conv.weight"].shape[1]
		num_actions = state_dict["output.weight"].shape[0]
	except KeyError as error:
		raise KeyError(
			f"Invalid extracted-policy state dict; missing {error.args[0]!r}."
		) from error
	policy = QNetwork(in_channels, num_actions).to(device)
	policy.load_state_dict(state_dict)
	policy.eval()
	return policy, in_channels, num_actions


def load_return_curve(path):
	if not path.is_file():
		raise FileNotFoundError(f"Return curve does not exist: {path}")
	with path.open("rb") as input_file:
		values = np.asarray(pickle.load(input_file), dtype=np.float64)
	if values.ndim != 1 or not np.all(np.isfinite(values)):
		raise ValueError(f"Return curve must be a finite vector: {path}")
	return values


def validate_run_metadata(run, metadata, return_steps, requested_switch_steps):
	if metadata.get("sequence") != run["sequence"]:
		raise ValueError(f"Sequence metadata does not match {run['path']}.")
	if metadata.get("seed") != run["training_seed"]:
		raise ValueError(f"Seed metadata does not match {run['path']}.")
	config = metadata.get("config") or {}
	switch_steps = requested_switch_steps or config.get("switch")
	if not isinstance(switch_steps, int) or switch_steps <= 0:
		raise ValueError(
			"A positive switch interval is required in checkpoint metadata or "
			"--switch-steps."
		)
	if return_steps % switch_steps:
		raise ValueError(
			f"Return curve length {return_steps} is not divisible by "
			f"switch interval {switch_steps}."
		)
	configured_steps = config.get("t_steps")
	if configured_steps is not None and configured_steps != return_steps:
		raise ValueError(
			f"Checkpoint declares {configured_steps} steps, but return curve has "
			f"{return_steps}."
		)
	games = metadata.get("games")
	if not isinstance(games, list) or len(games) != return_steps // switch_steps:
		raise ValueError("Checkpoint metadata has an invalid task game sequence.")
	return switch_steps, games


def combine_forward_transfer(sequence_results, excluded_benchmark_methods):
	all_scores = [
		score
		for result in sequence_results
		for scores in result["sequence_stage_scores"].values()
		for score in scores
	]
	valid_method_sets = [
		set(result["benchmark_methods"])
		for result in sequence_results
		if result["count"]
	]
	excluded_runs = [
		run
		for result in sequence_results
		for run in result["excluded_runs"]
	]
	return {
		**summarize_samples(all_scores),
		"sequence_stage_scores": {
			sequence: scores
			for result in sequence_results
			for sequence, scores in result["sequence_stage_scores"].items()
		},
		"benchmark_methods": sorted(
			set.intersection(*valid_method_sets) if valid_method_sets else set()
		),
		"seed_aggregation": "curves averaged within each sequence before FT",
		"sequence_seed_counts": {
			str(result["sequence"]): result["seed_count"]
			if result["count"]
			else 0
			for result in sequence_results
		},
		"excluded_runs": excluded_runs,
		"excluded_benchmark_methods": excluded_benchmark_methods,
		"paper_benchmark_complete": not excluded_runs
		and not excluded_benchmark_methods,
	}


def aggregate_sequence_forward_transfer(records, switch_steps):
	sequence = records[0]["sequence"]
	if all("Reset" in record["benchmark_returns"] for record in records):
		result = aggregate_forward_transfer(records, switch_steps)
		result["sequence"] = sequence
		result["seed_count"] = len(records)
		return result
	return {
		**summarize_samples([]),
		"sequence": sequence,
		"seed_count": 0,
		"sequence_stage_scores": {},
		"benchmark_methods": [],
		"seed_aggregation": "sequence excluded because Reset coverage is incomplete",
		"excluded_runs": [
			{
				"sequence": sequence,
				"training_seed": record["training_seed"],
				"reason": "incomplete Reset coverage for sequence",
			}
			for record in records
		],
	}


def print_metrics(metrics):
	print("\nCONQUEST metrics (mean +/- standard error)")
	print(f"{'metric':<20} {'mean':>10} {'stderr':>10} {'n':>6}")
	for game in GAME_NAMES:
		result = metrics["average_performance"][game]
		print(
			f"{game:<20} {result['mean']:>10.3f} "
			f"{result['stderr']:>10.3f} {result['count']:>6}"
		)
	for name in ("forward_transfer", "forgetting"):
		result = metrics[name]
		mean = "nan" if result["mean"] is None else f"{result['mean']:.3f}"
		stderr = (
			"nan" if result["stderr"] is None else f"{result['stderr']:.3f}"
		)
		print(f"{name:<20} {mean:>10} {stderr:>10} {result['count']:>6}")
	print(
		"FT benchmark methods: "
		+ ", ".join(metrics["forward_transfer"]["benchmark_methods"])
	)
	forward_transfer = metrics["forward_transfer"]
	if not forward_transfer["paper_benchmark_complete"]:
		print("WARNING: FT is partial because benchmark data are incomplete.")
		if forward_transfer["excluded_runs"]:
			excluded = ", ".join(
				f"seq={run['sequence']}/seed={run['training_seed']}"
				for run in forward_transfer["excluded_runs"]
			)
			print(f"Excluded FT runs: {excluded}")
		if forward_transfer["excluded_benchmark_methods"]:
			labels = ", ".join(
				item["label"]
				for item in forward_transfer["excluded_benchmark_methods"]
			)
			print(f"Excluded FT benchmark methods: {labels}")


def run(args):
	runs = discover_runs(args.checkpoint)
	baseline_returns_dir = (
		args.baseline_returns_dir
		if args.baseline_returns_dir is not None
		else infer_baseline_returns_dir(args.checkpoint)
	)
	benchmark_templates = parse_benchmark_templates(args.benchmark_template)
	benchmark_templates, excluded_benchmark_methods = (
		select_global_benchmark_templates(
			runs,
			baseline_returns_dir,
			args.reset_template,
			benchmark_templates,
		)
	)
	device = torch.device(
		args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
	)
	aggregate_records = []
	serialized_runs = []
	sequence_ft_results = []
	resolved_switch_steps = None
	reference_training_config = None

	for _, sequence_runs in groupby(runs, key=lambda run: run["sequence"]):
		sequence_records = []
		for run_spec in sequence_runs:
			sequence = run_spec["sequence"]
			training_seed = run_spec["training_seed"]
			checkpoint = load_checkpoint(run_spec["path"], device)
			current_training_config = training_config(checkpoint["metadata"])
			if reference_training_config is None:
				reference_training_config = current_training_config
			else:
				validate_training_config(
					reference_training_config,
					current_training_config,
					run_spec["path"],
				)
			policy, in_channels, num_actions = build_extracted_policy(
				checkpoint, device
			)
			game_evaluations = {
				game: evaluate_for_steps(
					policy,
					game_id,
					args,
					device,
					in_channels,
					num_actions,
					sequence,
					training_seed,
				)
				for game_id, game in enumerate(GAME_NAMES)
			}

			training_path = run_spec["path"].with_name(
				f"{run_spec['run_name']}_returns.pkl"
			)
			training_returns = load_return_curve(training_path)
			switch_steps, games = validate_run_metadata(
				run_spec,
				checkpoint["metadata"],
				len(training_returns),
				args.switch_steps,
			)
			if resolved_switch_steps is None:
				resolved_switch_steps = switch_steps
			elif switch_steps != resolved_switch_steps:
				raise ValueError("Runs use different task switch intervals.")

			reset_path = resolve_return_path(
				baseline_returns_dir,
				args.reset_template,
				sequence,
				training_seed,
			)
			benchmark_returns = {
				METHOD_NAME: moving_average(
					training_returns, args.smoothing_window
				)
			}
			benchmark_paths = {METHOD_NAME: training_path}
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
			if "Reset" in benchmark_returns:
				for label, template in benchmark_templates:
					benchmark_path = resolve_return_path(
						baseline_returns_dir,
						template,
						sequence,
						training_seed,
					)
					if not benchmark_path.is_file():
						raise FileNotFoundError(
							f"Globally selected FT benchmark is missing: "
							f"{benchmark_path}"
						)
					benchmark_returns[label] = moving_average(
						load_return_curve(benchmark_path), args.smoothing_window
					)
					benchmark_paths[label] = benchmark_path

			curve_lengths = {len(values) for values in benchmark_returns.values()}
			if curve_lengths != {len(training_returns)}:
				raise ValueError(
					f"FT curves have different lengths for seq={sequence}, "
					f"seed={training_seed}."
				)
			final_game_returns = {
				game: evaluation["mean_return"]
				for game, evaluation in game_evaluations.items()
			}
			forgetting = compute_forgetting(
				training_returns,
				games,
				final_game_returns,
				switch_steps,
				args.smoothing_window,
			)
			normalized_forgetting = normalize_forgetting(forgetting)
			sequence_records.append(
				{
					"sequence": sequence,
					"training_seed": training_seed,
					"training_returns": benchmark_returns[METHOD_NAME],
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
					"checkpoint": str(run_spec["path"]),
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
			aggregate_sequence_forward_transfer(
				sequence_records, resolved_switch_steps
			)
		)

	forward_transfer = combine_forward_transfer(
		sequence_ft_results, excluded_benchmark_methods
	)
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
	print_metrics(metrics)
	return {
		"method": METHOD_NAME,
		"checkpoint_family": str(args.checkpoint),
		"baseline_returns_dir": str(baseline_returns_dir),
		"evaluation_steps": args.evaluation_steps,
		"max_steps": args.max_steps,
		"switch_steps": resolved_switch_steps,
		"smoothing_window": args.smoothing_window,
		"evaluation_seed": "training seed",
		"forgetting_normalizers": DQN_GAME_BENCHMARKS,
		"missing_game_policy": "zero forgetting for games absent from a sequence",
		"training_config": reference_training_config,
		"runs": serialized_runs,
		"metrics": metrics,
	}


def main():
	args = parse_args()
	if args.evaluation_steps <= 0 or args.max_steps <= 0:
		raise ValueError("evaluation-steps and max-steps must be positive.")
	if args.smoothing_window <= 0:
		raise ValueError("smoothing-window must be positive.")
	if args.switch_steps is not None and args.switch_steps <= 0:
		raise ValueError("switch-steps must be positive.")
	if args.torch_threads <= 0:
		raise ValueError("torch-threads must be positive.")
	torch.set_num_threads(args.torch_threads)
	payload = run(args)
	if args.output_json is not None:
		args.output_json.parent.mkdir(parents=True, exist_ok=True)
		with args.output_json.open("w", encoding="utf-8") as output_file:
			json.dump(payload, output_file, indent=2, allow_nan=False)
		print(f"saved: {args.output_json}")


if __name__ == "__main__":
	main()
'''
cd MinAtar
conda run -n RLL3 python conquest_eval.py \
	--checkpoint results/conquest/CONQUEST_steps_3500000_switch_500000_seq_0_seed_0_checkpoint.pt \
  --device cpu
'''
