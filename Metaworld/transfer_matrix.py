"""Evaluate and plot MetaWorld continual-learning transfer matrices."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATORS = {
	"awr": SCRIPT_DIR / "conquest_eval.py",
	"fame": SCRIPT_DIR / "final_eval_fame.py",
}
CHECKPOINT_PATTERN = re.compile(r"^(?P<name>.+_(?P<stage>\d+)_meta)_actor\.pt$")


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Build R[i,j], the performance on task j after training through task i, "
			"its stage-to-stage gain, or a hybrid meta/forward-transfer matrix."
		)
	)
	parser.add_argument("--run-dir", required=True, help="An AWR or FAME run directory")
	parser.add_argument(
		"--agent-kind",
		choices=("awr", "fame"),
		default="awr",
		help="Checkpoint format and evaluator used for meta-agent performance",
	)
	parser.add_argument(
		"--model-dir",
		default=None,
		help="Checkpoint directory (default: RUN_DIR/model when present, else RUN_DIR)",
	)
	parser.add_argument(
		"--evaluate",
		action="store_true",
		help="Evaluate all missing intermediate checkpoints before plotting",
	)
	parser.add_argument("--config", default=None, help="Optional training config path")
	parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
	parser.add_argument("--num-eval-runs", type=int, default=25)
	parser.add_argument("--reseed-each-episode", type=int, choices=(0, 1), default=0)
	parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
	parser.add_argument(
		"--matrix",
		choices=("performance", "gain", "hybrid"),
		default="performance",
		help=(
			"performance: R[i,j]; gain: R[i,j] - R[i-1,j], the marginal "
			"effect of learning task i; hybrid: meta success for j<=i and "
			"student-vs-reset normalized AUC forward transfer for j>i"
		),
	)
	parser.add_argument(
		"--region",
		choices=("all", "forward", "backward"),
		default="all",
		help="Optionally retain only future-task or past-task cells",
	)
	parser.add_argument(
		"--metric",
		choices=("metaworld_success_mean", "return_mean"),
		default="metaworld_success_mean",
	)
	parser.add_argument(
		"--student-curves",
		nargs="+",
		default=None,
		help=(
			"Student learning-curve CSVs for hybrid mode. Sequential logs use "
			"task_idx; full pairwise logs must also contain source_task_idx."
		),
	)
	parser.add_argument(
		"--reset-curves",
		nargs="+",
		default=None,
		help="Reset/scratch learning-curve CSVs for hybrid mode",
	)
	parser.add_argument(
		"--curve-metric",
		default="mean_success",
		help="Success column in student/reset curve CSVs",
	)
	parser.add_argument(
		"--task-horizon",
		type=float,
		default=None,
		help="AUC horizon in local environment steps (default: infer from logging grid)",
	)
	parser.add_argument(
		"--allow-unpaired-seeds",
		action="store_true",
		help="Compare mean AUCs when student and reset training seeds do not match",
	)
	parser.add_argument(
		"--allow-incomplete-forward",
		action="store_true",
		help="Leave unavailable pairwise forward-transfer cells blank",
	)
	parser.add_argument(
		"--eval-dir",
		default=None,
		help="Per-checkpoint CSV directory (default: RUN_DIR/transfer_matrix_evals)",
	)
	parser.add_argument("--output", default=None, help="Output PDF or PNG path")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument(
		"--allow-incomplete",
		action="store_true",
		help="Plot available checkpoints even if some stages are missing",
	)
	args = parser.parse_args()
	if args.num_eval_runs <= 0:
		parser.error("--num-eval-runs must be positive")
	if not args.eval_seeds:
		parser.error("--eval-seeds must not be empty")
	if args.task_horizon is not None and args.task_horizon <= 0:
		parser.error("--task-horizon must be positive")
	if args.matrix == "hybrid":
		if not args.student_curves or not args.reset_curves:
			parser.error("--matrix hybrid requires --student-curves and --reset-curves")
		if args.metric != "metaworld_success_mean":
			parser.error("--matrix hybrid requires --metric metaworld_success_mean")
		if args.region != "all":
			parser.error("--region applies only to performance/gain matrices")
	return args


def resolve_path(path):
	path = Path(path).expanduser()
	if path.is_absolute():
		return path
	cwd_path = (Path.cwd() / path).resolve()
	if cwd_path.exists():
		return cwd_path
	return (SCRIPT_DIR / path).resolve()


def resolve_output_path(path):
	path = Path(path).expanduser()
	if path.is_absolute():
		return path
	return (Path.cwd() / path).resolve()


def resolve_model_dir(run_dir, requested_model_dir):
	if requested_model_dir:
		model_dir = resolve_path(requested_model_dir)
	elif (run_dir / "model").is_dir():
		model_dir = run_dir / "model"
	else:
		model_dir = run_dir
	if not model_dir.is_dir():
		raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
	return model_dir


def discover_checkpoints(model_dir):

	checkpoints = {}
	for actor_path in model_dir.glob("*_meta_actor.pt"):
		match = CHECKPOINT_PATTERN.fullmatch(actor_path.name)
		if match:
			checkpoints[int(match.group("stage"))] = match.group("name")
	if not checkpoints:
		raise FileNotFoundError(f"No *_N_meta_actor.pt checkpoints found in {model_dir}")
	return dict(sorted(checkpoints.items()))


def checkpoint_stage(csv_path):
	try:
		header = pd.read_csv(csv_path, usecols=["checkpoint_idx"], nrows=1)
	except (OSError, ValueError, pd.errors.EmptyDataError):
		return None
	if header.empty or pd.isna(header.iloc[0]["checkpoint_idx"]):
		return None
	return int(header.iloc[0]["checkpoint_idx"])


def discover_evaluations(run_dir, eval_dir):
	files = {}
	for directory in (run_dir, eval_dir):
		if not directory.is_dir():
			continue
		for csv_path in sorted(directory.glob("*_meta_multi_seed_eval.csv")):
			stage = checkpoint_stage(csv_path)
			if stage is not None:
				files[stage] = csv_path
	return files


def evaluation_matches(csv_path, eval_seeds, num_eval_runs):
	try:
		data = pd.read_csv(csv_path, usecols=["eval_seed", "num_eval_runs"])
	except (OSError, ValueError, pd.errors.EmptyDataError):
		return False
	return (
		set(data["eval_seed"].astype(int).unique()) == set(eval_seeds)
		and set(data["num_eval_runs"].astype(int).unique()) == {num_eval_runs}
	)


def evaluate_checkpoints(args, run_dir, model_dir, eval_dir, checkpoints):
	eval_dir.mkdir(parents=True, exist_ok=True)
	available = discover_evaluations(run_dir, eval_dir)
	evaluator = EVALUATORS[args.agent_kind]
	for stage, model_name in checkpoints.items():
		existing = available.get(stage)
		if (
			existing is not None
			and not args.overwrite
			and evaluation_matches(existing, args.eval_seeds, args.num_eval_runs)
		):
			print(f"stage {stage}: using {existing}")
			continue

		output = eval_dir / f"{model_name}_multi_seed_eval.csv"
		command = [
			sys.executable,
			str(evaluator),
			"--run_dir",
			str(run_dir),
			"--model_dir",
			str(model_dir),
			"--model_name",
			model_name,
			"--eval_seeds",
			*(str(seed) for seed in args.eval_seeds),
			"--num_eval_runs",
			str(args.num_eval_runs),
			"--reseed_each_episode",
			str(args.reseed_each_episode),
			"--gpu",
			args.gpu,
			"--output",
			str(output),
		]
		if args.config:
			command.extend(("--config", str(resolve_path(args.config))))
		print(f"stage {stage}: evaluating {model_name}", flush=True)
		subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def load_results(evaluation_files, metric):
	frames = []
	required = {"checkpoint_idx", "task_idx", "task", "eval_seed", metric}
	for stage, csv_path in sorted(evaluation_files.items()):
		frame = pd.read_csv(csv_path)
		missing_columns = required.difference(frame.columns)
		if missing_columns:
			raise ValueError(
				f"{csv_path} is missing columns: {sorted(missing_columns)}"
			)
		stages = set(frame["checkpoint_idx"].dropna().astype(int).unique())
		if stages != {stage}:
			raise ValueError(
				f"{csv_path} contains checkpoint_idx={sorted(stages)}, expected {stage}"
			)
		frames.append(frame)
	return pd.concat(frames, ignore_index=True)


def build_performance_matrix(results, metric):
	task_names = results.groupby("task_idx")["task"].nunique()
	if (task_names != 1).any():
		conflicts = task_names[task_names != 1].index.tolist()
		raise ValueError(f"Task names disagree across checkpoints at task_idx={conflicts}")

	task_lookup = (
		results[["task_idx", "task"]]
		.drop_duplicates()
		.sort_values("task_idx")
		.set_index("task_idx")["task"]
	)
	matrix = results.pivot_table(
		index="checkpoint_idx",
		columns="task_idx",
		values=metric,
		aggfunc="mean",
	)
	matrix = matrix.sort_index().reindex(columns=task_lookup.index)
	return matrix, task_lookup


def load_curve_files(paths, metric, role):
	frames = []
	for raw_path in paths:
		path = resolve_path(raw_path)
		if not path.is_file():
			raise FileNotFoundError(f"{role} curve CSV does not exist: {path}")
		frame = pd.read_csv(path)
		required = {"steps", "seed", "task", metric}
		missing = required.difference(frame.columns)
		if missing:
			raise ValueError(f"{path} is missing columns: {sorted(missing)}")

		if "target_task_idx" in frame.columns:
			frame["_target_task_idx"] = frame["target_task_idx"]
		elif "task_idx" in frame.columns:
			frame["_target_task_idx"] = frame["task_idx"]
		else:
			raise ValueError(f"{path} needs task_idx or target_task_idx")

		if role == "student":
			if "source_task_idx" in frame.columns:
				frame["_source_task_idx"] = frame["source_task_idx"]
			elif "source_stage" in frame.columns:
				frame["_source_task_idx"] = frame["source_stage"] + 1
			else:
				# A sequential log observes only cumulative history -> next task.
				frame["_source_task_idx"] = frame["_target_task_idx"] - 1

		frame["_curve_file"] = str(path)
		frames.append(frame)

	curves = pd.concat(frames, ignore_index=True)
	for column in ("steps", "seed", "_target_task_idx", metric):
		curves[column] = pd.to_numeric(curves[column], errors="raise")
	curves["_target_task_idx"] = curves["_target_task_idx"].astype(int)
	if role == "student":
		curves["_source_task_idx"] = pd.to_numeric(
			curves["_source_task_idx"], errors="raise"
		).astype(int)
		curves = curves[curves["_source_task_idx"] >= 1].copy()
		invalid = curves["_source_task_idx"] >= curves["_target_task_idx"]
		if invalid.any():
			pairs = curves.loc[
				invalid, ["_source_task_idx", "_target_task_idx"]
			].drop_duplicates()
			raise ValueError(
				f"Student curves must describe forward pairs source < target; found {pairs.values.tolist()}"
			)

	group_columns = ["_curve_file", "seed", "_target_task_idx"]
	if role == "student":
		group_columns.append("_source_task_idx")
	curves["_local_step"] = curves["steps"] - curves.groupby(group_columns)[
		"steps"
	].transform("min")
	return curves


def curve_auc(curve, metric, requested_horizon):
	points = (
		curve.groupby("_local_step", as_index=False)[metric]
		.mean()
		.sort_values("_local_step")
	)
	x = points["_local_step"].to_numpy(dtype=float)
	y = points[metric].to_numpy(dtype=float)
	if len(x) < 2:
		raise ValueError("Each learning curve needs at least two evaluation points")
	if not np.isfinite(x).all() or not np.isfinite(y).all():
		raise ValueError("Learning curves cannot contain NaN or infinite values")
	if np.any((y < 0.0) | (y > 1.0)):
		raise ValueError("Forward-transfer success curves must stay in [0, 1]")

	steps = np.diff(x)
	if np.any(steps <= 0):
		raise ValueError("Learning-curve steps must be strictly increasing")
	inferred_horizon = float(x[-1] + np.median(steps))
	horizon = float(requested_horizon or inferred_horizon)
	if x[-1] > horizon:
		raise ValueError(
			f"Curve reaches local step {x[-1]:g}, beyond task horizon {horizon:g}"
		)
	if x[0] > 0:
		x = np.insert(x, 0, 0.0)
		y = np.insert(y, 0, y[0])
	if x[-1] < horizon:
		x = np.append(x, horizon)
		y = np.append(y, y[-1])
	return float(np.trapz(y, x) / horizon), horizon


def summarize_curve_aucs(curves, metric, requested_horizon, role):
	group_columns = ["_target_task_idx", "seed"]
	if role == "student":
		group_columns.insert(0, "_source_task_idx")
	rows = []
	for keys, curve in curves.groupby(group_columns, sort=True):
		if not isinstance(keys, tuple):
			keys = (keys,)
		auc, horizon = curve_auc(curve, metric, requested_horizon)
		row = dict(zip(group_columns, keys))
		row.update(
			{
				"task": str(curve["task"].iloc[0]),
				f"{role}_auc": auc,
				f"{role}_horizon": horizon,
			}
		)
		rows.append(row)
	return pd.DataFrame(rows)


def compute_forward_transfer(student_curves, reset_curves, metric, args):
	student_aucs = summarize_curve_aucs(
		student_curves, metric, args.task_horizon, "student"
	)
	reset_aucs = summarize_curve_aucs(
		reset_curves, metric, args.task_horizon, "reset"
	)

	matched = student_aucs.merge(
		reset_aucs.drop(columns="task"),
		on=["_target_task_idx", "seed"],
		how="left",
	)
	missing_seed_matches = matched["reset_auc"].isna()
	if missing_seed_matches.any() and not args.allow_unpaired_seeds:
		missing = matched.loc[
			missing_seed_matches,
			["_source_task_idx", "_target_task_idx", "seed"],
		].values.tolist()
		raise ValueError(
			"Reset curves are missing matching training seeds for "
			f"{missing}. Add matching reset runs or pass --allow-unpaired-seeds."
		)

	if missing_seed_matches.any():
		print(
			"warning: student/reset training seeds do not match; comparing "
			"AUCs averaged independently across available seeds"
		)
		student_summary = student_aucs.groupby(
			["_source_task_idx", "_target_task_idx", "task"], as_index=False
		).agg(
			student_auc=("student_auc", "mean"),
			student_horizon=("student_horizon", "mean"),
			student_seed_count=("seed", "nunique"),
		)
		reset_summary = reset_aucs.groupby("_target_task_idx", as_index=False).agg(
			reset_auc=("reset_auc", "mean"),
			reset_horizon=("reset_horizon", "mean"),
			reset_seed_count=("seed", "nunique"),
		)
		components = student_summary.merge(reset_summary, on="_target_task_idx")
		components["pairing"] = "unpaired-seed means"
	else:
		components = matched.groupby(
			["_source_task_idx", "_target_task_idx", "task"], as_index=False
		).agg(
			student_auc=("student_auc", "mean"),
			reset_auc=("reset_auc", "mean"),
			student_horizon=("student_horizon", "mean"),
			reset_horizon=("reset_horizon", "mean"),
			student_seed_count=("seed", "nunique"),
			reset_seed_count=("seed", "nunique"),
		)
		components["pairing"] = "matched seeds"

	horizon_mismatch = ~np.isclose(
		components["student_horizon"], components["reset_horizon"]
	)
	if horizon_mismatch.any():
		pairs = components.loc[
			horizon_mismatch, ["_source_task_idx", "_target_task_idx"]
		].values.tolist()
		raise ValueError(f"Student/reset AUC horizons disagree for pairs {pairs}")

	denominator = 1.0 - components["reset_auc"]
	if np.isclose(denominator, 0.0).any():
		pairs = components.loc[
			np.isclose(denominator, 0.0),
			["_source_task_idx", "_target_task_idx"],
		].values.tolist()
		raise ValueError(f"Forward transfer is undefined when reset AUC is 1: {pairs}")
	components["forward_transfer"] = (
		components["student_auc"] - components["reset_auc"]
	) / denominator
	return components


def validate_curve_tasks(curves, task_lookup, role):
	for target_task_idx, task in curves[
		["_target_task_idx", "task"]
	].drop_duplicates().itertuples(index=False, name=None):
		target_idx = int(target_task_idx)
		if target_idx not in task_lookup.index:
			raise ValueError(f"Unknown target_task_idx={target_idx} in {role} curves")
		expected = clean_task_name(task_lookup.loc[target_idx])
		actual = clean_task_name(task)
		if actual != expected:
			raise ValueError(
				f"Task {target_idx} is {actual!r} in {role} curves, expected {expected!r}"
			)


def build_hybrid_matrix(performance, components):
	hybrid = pd.DataFrame(np.nan, index=performance.index, columns=performance.columns)
	meta_values = hybrid.copy()
	forward_values = hybrid.copy()
	for stage in performance.index:
		source_task_idx = int(stage) + 1
		for target_task_idx in performance.columns:
			if int(target_task_idx) <= source_task_idx:
				value = performance.loc[stage, target_task_idx]
				hybrid.loc[stage, target_task_idx] = value
				meta_values.loc[stage, target_task_idx] = value

	for source_task_idx, target_task_idx, forward_transfer in components[
		["_source_task_idx", "_target_task_idx", "forward_transfer"]
	].itertuples(index=False, name=None):
		stage = int(source_task_idx) - 1
		target_task_idx = int(target_task_idx)
		if stage not in hybrid.index:
			raise ValueError(f"No meta checkpoint row for source_task_idx={stage + 1}")
		if target_task_idx not in hybrid.columns:
			raise ValueError(f"No matrix column for target_task_idx={target_task_idx}")
		value = float(forward_transfer)
		hybrid.loc[stage, target_task_idx] = value
		forward_values.loc[stage, target_task_idx] = value
	return hybrid, meta_values, forward_values


def missing_forward_pairs(performance, components):
	available = {
		(int(source), int(target))
		for source, target in components[
			["_source_task_idx", "_target_task_idx"]
		].itertuples(index=False, name=None)
	}
	expected = {
		(int(stage) + 1, int(target_task_idx))
		for stage in performance.index
		for target_task_idx in performance.columns
		if int(stage) + 1 < int(target_task_idx)
	}
	return sorted(expected.difference(available))


def clean_task_name(task_name):
	return re.sub(r"-v\d+$", "", str(task_name))


def select_matrix(performance, matrix_type, region):
	matrix = performance.copy()
	if matrix_type == "gain":
		stages = matrix.index.to_numpy(dtype=int)
		if len(stages) > 1 and not np.all(np.diff(stages) == 1):
			raise ValueError("Gain matrix requires consecutive checkpoint stages")
		matrix = matrix.diff().iloc[1:]

	if region != "all":
		for stage in matrix.index:
			learned_task_idx = int(stage) + 1
			for task_idx in matrix.columns:
				keep = (
					int(task_idx) > learned_task_idx
					if region == "forward"
					else int(task_idx) < learned_task_idx
				)
				if not keep:
					matrix.loc[stage, task_idx] = np.nan
	return matrix


def label_matrix(matrix, task_lookup):
	column_labels = [
		f"{int(task_idx)}. {clean_task_name(task_lookup.loc[task_idx])}"
		for task_idx in matrix.columns
	]
	row_labels = []
	for stage in matrix.index:
		task_idx = int(stage) + 1
		task_name = task_lookup.get(task_idx, f"task-{task_idx}")
		row_labels.append(f"{task_idx}. {clean_task_name(task_name)}")
	labeled = matrix.copy()
	labeled.index = row_labels
	labeled.columns = column_labels
	return labeled


def plot_matrix(matrix, args, output_path):
	if matrix.empty or not matrix.notna().any().any():
		raise ValueError("The selected matrix region has no values to plot")

	figure_width = max(9.0, 0.9 * matrix.shape[1] + 4.0)
	figure_height = max(4.5, 0.6 * matrix.shape[0] + 2.5)
	figure, axis = plt.subplots(figsize=(figure_width, figure_height))

	if args.matrix == "gain":
		finite_values = np.abs(matrix.to_numpy(dtype=float))
		limit = float(np.nanmax(finite_values))
		limit = max(limit, 0.05)
		cmap = "RdYlGn"
		vmin, vmax, center = -limit, limit, 0.0
		colorbar_label = "Change in score"
	else:
		cmap = "viridis"
		if args.metric == "metaworld_success_mean":
			vmin, vmax = 0.0, 1.0
		else:
			vmin, vmax = None, None
		center = None
		colorbar_label = "Mean success" if vmin == 0.0 else "Mean return"

	sns.heatmap(
		matrix,
		mask=matrix.isna(),
		annot=True,
		fmt=".2f",
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		center=center,
		linewidths=0.5,
		linecolor="white",
		cbar_kws={"label": colorbar_label},
		ax=axis,
	)
	axis.set_xlabel("Evaluation task")
	axis.set_ylabel(
		"Newly learned task" if args.matrix == "gain" else "Training completed through"
	)
	axis.set_title(
		"Marginal task transfer" if args.matrix == "gain" else "Continual-learning performance"
	)
	axis.tick_params(axis="x", rotation=45)
	axis.tick_params(axis="y", rotation=0)
	figure.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)


def plot_hybrid_matrix(meta_values, forward_values, output_path):
	if meta_values.empty or not meta_values.notna().any().any():
		raise ValueError("The hybrid matrix has no meta-agent values")

	figure_width = max(10.0, 0.9 * meta_values.shape[1] + 5.0)
	figure_height = max(5.0, 0.6 * meta_values.shape[0] + 2.5)
	figure, axis = plt.subplots(figsize=(figure_width, figure_height))
	axis.set_facecolor("#e6e6e6")

	sns.heatmap(
		meta_values,
		mask=meta_values.isna(),
		annot=True,
		fmt=".2f",
		cmap="Blues",
		vmin=0.0,
		vmax=1.0,
		linewidths=0.5,
		linecolor="white",
		cbar_kws={"label": "Meta-agent success", "shrink": 0.8, "pad": 0.02},
		ax=axis,
	)

	if forward_values.notna().any().any():
		limit = max(
			1.0,
			float(np.nanmax(np.abs(forward_values.to_numpy(dtype=float)))),
		)
		sns.heatmap(
			forward_values,
			mask=forward_values.isna(),
			annot=True,
			fmt=".2f",
			cmap="RdYlGn",
			vmin=-limit,
			vmax=limit,
			center=0.0,
			linewidths=0.5,
			linecolor="white",
			cbar_kws={"label": "Normalized student FT", "shrink": 0.8, "pad": 0.10},
			ax=axis,
		)

	axis.set_xlabel("Evaluation / target task")
	axis.set_ylabel("Meta agent after task")
	axis.set_title("Meta retention (lower) and student forward transfer (upper)")
	axis.tick_params(axis="x", rotation=45)
	axis.tick_params(axis="y", rotation=0)
	figure.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)


def main():
	args = parse_args()
	run_dir = resolve_path(args.run_dir)
	if not run_dir.is_dir():
		raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
	model_dir = resolve_model_dir(run_dir, args.model_dir)
	eval_dir = (
		resolve_output_path(args.eval_dir)
		if args.eval_dir
		else run_dir / "transfer_matrix_evals"
	)
	checkpoints = discover_checkpoints(model_dir)

	if args.evaluate:
		evaluate_checkpoints(args, run_dir, model_dir, eval_dir, checkpoints)

	evaluation_files = discover_evaluations(run_dir, eval_dir)
	missing_stages = sorted(set(checkpoints).difference(evaluation_files))
	if missing_stages and not args.allow_incomplete:
		raise RuntimeError(
			"Missing per-task evaluations for checkpoint stages "
			f"{missing_stages}. Re-run with --evaluate, or use --allow-incomplete "
			"only for a diagnostic plot."
		)
	if not evaluation_files:
		raise FileNotFoundError(
			f"No *_meta_multi_seed_eval.csv files found in {run_dir} or {eval_dir}"
		)

	results = load_results(evaluation_files, args.metric)
	performance, task_lookup = build_performance_matrix(results, args.metric)
	components = None
	meta_values = None
	forward_values = None
	if args.matrix == "hybrid":
		student_curves = load_curve_files(
			args.student_curves, args.curve_metric, "student"
		)
		reset_curves = load_curve_files(args.reset_curves, args.curve_metric, "reset")
		validate_curve_tasks(student_curves, task_lookup, "student")
		validate_curve_tasks(reset_curves, task_lookup, "reset")
		components = compute_forward_transfer(
			student_curves, reset_curves, args.curve_metric, args
		)
		matrix, meta_values, forward_values = build_hybrid_matrix(
			performance, components
		)
		missing_pairs = missing_forward_pairs(performance, components)
		if missing_pairs and not args.allow_incomplete_forward:
			preview = ", ".join(f"{source}->{target}" for source, target in missing_pairs[:10])
			raise RuntimeError(
				f"Missing {len(missing_pairs)} pairwise student curves ({preview}). "
				"A sequential run supplies only (j-1)->j. Run the remaining "
				"adaptation pairs or pass --allow-incomplete-forward to leave them blank."
			)
		default_name = "transfer_matrix_hybrid.pdf"
	else:
		matrix = select_matrix(performance, args.matrix, args.region)
		missing_pairs = []
		default_name = f"transfer_matrix_{args.matrix}_{args.region}.pdf"

	labeled_matrix = label_matrix(matrix, task_lookup)
	output_path = (
		resolve_output_path(args.output) if args.output else run_dir / default_name
	)
	if output_path.suffix.lower() not in {".pdf", ".png"}:
		raise ValueError("--output must end in .pdf or .png")
	matrix_path = output_path.with_suffix(".csv")
	matrix_path.parent.mkdir(parents=True, exist_ok=True)
	labeled_matrix.to_csv(matrix_path, index_label="training_stage")
	if args.matrix == "hybrid":
		labeled_meta = label_matrix(meta_values, task_lookup)
		labeled_forward = label_matrix(forward_values, task_lookup)
		plot_hybrid_matrix(labeled_meta, labeled_forward, output_path)
		components_path = output_path.with_name(
			f"{output_path.stem}_ft_components.csv"
		)
		export_components = components.rename(
			columns={
				"_source_task_idx": "source_task_idx",
				"_target_task_idx": "target_task_idx",
			}
		)
		export_components.to_csv(components_path, index=False)
		print(f"FT components: {components_path}")
	else:
		plot_matrix(labeled_matrix, args, output_path)

	print(f"evaluation files: {len(evaluation_files)}")
	print(f"matrix: {matrix_path}")
	print(f"figure: {output_path}")
	if missing_stages:
		print(f"warning: omitted checkpoint stages {missing_stages}")
	if missing_pairs:
		print(f"warning: left {len(missing_pairs)} unavailable forward pairs blank")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

'''
python Metaworld/transfer_matrix.py \
  --run-dir Metaworld/results/main3-2-awr/set12_warmup_seed1 \
  --evaluate \
  --eval-seeds 0 \
  --num-eval-runs 15 \
  --gpu 0 \
  --matrix performance

python Metaworld/transfer_matrix.py \
	--run-dir Metaworld/results/main3-2-awr/set12_warmup_seed1 \
	--matrix hybrid \
	--student-curves log-awr/metaworld_sequence_set12/sac_metaworld_sequence_set12_1_buffer.csv \
	--reset-curves Metaworld/log/metaworld_sequence_set12/sac_metaworld_sequence_set12_0_independent.csv \
	--allow-unpaired-seeds \
	--allow-incomplete-forward

'''
