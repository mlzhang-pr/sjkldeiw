"""Diagnose cross-task state compatibility on held-out rollouts.

The test asks whether a representation assigns smaller distances to physically
compatible states from another task. It compares each source state's nearest
cross-task state in raw Fetch coordinates against all random cross-task
pairings. This is a necessary state-overlap check, not proof that a policy can
execute the stitched path; that requires a separate two-leg rollout
intervention.

The input can be a learned ``*_state_tsne.npz`` artifact or a handcrafted
``*_robot_state_tsne_3d.npz`` artifact. The diagnostic uses the saved
high-dimensional ``representations`` and never uses the t-SNE coordinates.
"""

import argparse
import json
import os
import sys
from itertools import permutations

import numpy as np


DEFAULT_MATCH_DIMS = (0, 1, 2, 3, 4, 5)


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Test whether held-out states from different Fetch tasks can be "
			"matched in a state representation."
		)
	)
	parser.add_argument(
		"--data",
		type=str,
		default=None,
		help="Path to a *_state_tsne.npz artifact.",
	)
	parser.add_argument(
		"--run_dir",
		type=str,
		default=None,
		help="Run directory used to infer the final state-representation artifact.",
	)
	parser.add_argument("--source_task", type=str, default=None)
	parser.add_argument("--target_task", type=str, default=None)
	parser.add_argument(
		"--match_dims",
		type=int,
		nargs="+",
		default=list(DEFAULT_MATCH_DIMS),
		help=(
			"Raw observation dimensions defining physical compatibility. The default "
			"uses Fetch gripper and object positions."
		),
	)
	parser.add_argument(
		"--distance_mode",
		choices=["auto", "mrn", "euclidean"],
		default="auto",
		help=(
			"auto uses standardized Euclidean distance for robot-state artifacts "
			"and MRN distance for learned representations."
		),
	)
	parser.add_argument("--components", type=int, default=8)
	parser.add_argument(
		"--max_samples_per_task",
		type=int,
		default=1000,
		help="Maximum held-out states per task; <= 0 keeps every state.",
	)
	parser.add_argument(
		"--top_k",
		type=int,
		default=10,
		help="Raw-neighbor set size used by cross-task retrieval recall.",
	)
	parser.add_argument("--bootstrap_samples", type=int, default=2000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument(
		"--min_win_rate",
		type=float,
		default=0.6,
		help="Minimum matched-vs-random pairwise win rate for evidence_supported.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Output JSON path. Defaults beside the input artifact.",
	)
	args = parser.parse_args()

	if (args.source_task is None) != (args.target_task is None):
		parser.error("--source_task and --target_task must be provided together")
	if args.source_task == args.target_task and args.source_task is not None:
		parser.error("source and target tasks must differ")
	if args.components <= 0:
		parser.error("--components must be positive")
	if args.top_k <= 0:
		parser.error("--top_k must be positive")
	if args.bootstrap_samples <= 0:
		parser.error("--bootstrap_samples must be positive")
	if not 0.0 <= args.min_win_rate <= 1.0:
		parser.error("--min_win_rate must be between 0 and 1")

	args.data = resolve_data_path(args.data, args.run_dir)
	args.output = os.path.abspath(
		args.output or f"{os.path.splitext(args.data)[0]}_crosstask_stitching.json"
	)
	return args


def resolve_data_path(data_path, run_dir):
	if data_path is not None:
		path = os.path.abspath(data_path)
		if not os.path.isfile(path):
			raise FileNotFoundError(f"State-representation artifact not found: {path}")
		return path
	if run_dir is None:
		raise ValueError("Provide --data or --run_dir.")

	run_dir = os.path.abspath(run_dir)
	run_name = os.path.basename(os.path.normpath(run_dir))
	path = os.path.join(run_dir, f"{run_name}_final_state_tsne.npz")
	if not os.path.isfile(path):
		raise FileNotFoundError(
			f"State-representation artifact not found: {path}. Generate it with "
			"python -m online_continual.tsne.tsne --run_dir <run_dir>."
		)
	return path


def load_rollout_data(path):
	required = {
		"representations",
		"state_observations",
		"episode_indices",
		"step_indices",
		"task_names",
		"checkpoints",
	}
	with np.load(path) as archive:
		missing = required.difference(archive.files)
		if missing:
			raise ValueError(f"Missing arrays in {path}: {sorted(missing)}")
		data = {key: np.array(archive[key], copy=True) for key in required}
		for optional_key in ("episode_successes", "robot_feature_names"):
			if optional_key in archive.files:
				data[optional_key] = np.array(archive[optional_key], copy=True)

	row_count = len(data["representations"])
	if any(len(data[key]) != row_count for key in required):
		raise ValueError("All state-representation arrays must have the same row count.")
	if data["representations"].ndim != 2 or data["state_observations"].ndim != 2:
		raise ValueError("representations and state_observations must be rank-2 arrays.")
	return data


def ordered_unique(values):
	return list(dict.fromkeys(str(value) for value in values))


def select_task_rows(data, task_name, max_samples, random_state):
	indices = np.flatnonzero(data["task_names"].astype(str) == task_name)
	if len(indices) == 0:
		raise ValueError(f"Task '{task_name}' is absent from the artifact.")
	if max_samples > 0 and len(indices) > max_samples:
		indices = np.sort(random_state.choice(indices, size=max_samples, replace=False))
	return indices


def standardized_coordinates(source_states, target_states, match_dims):
	dimensions = np.asarray(match_dims, dtype=np.int64)
	observation_dim = source_states.shape[1]
	if np.any(dimensions < 0) or np.any(dimensions >= observation_dim):
		raise ValueError(
			f"match_dims must be in [0, {observation_dim - 1}], got {dimensions.tolist()}."
		)
	combined = np.concatenate(
		[source_states[:, dimensions], target_states[:, dimensions]],
		axis=0,
	).astype(np.float64)
	scale = combined.std(axis=0)
	scale[scale < 1e-6] = 1.0
	return (
		source_states[:, dimensions] / scale,
		target_states[:, dimensions] / scale,
	)


def euclidean_distance_matrix(source, target):
	source_sq = np.sum(np.square(source), axis=1, keepdims=True)
	target_sq = np.sum(np.square(target), axis=1, keepdims=True).T
	squared = np.maximum(source_sq + target_sq - 2.0 * source @ target.T, 0.0)
	return np.sqrt(squared)


def standardized_pair(source, target):
	combined = np.concatenate([source, target], axis=0).astype(np.float64)
	scale = combined.std(axis=0)
	scale[scale < 1e-6] = 1.0
	return source / scale, target / scale, scale


def mrn_distance(x, y, components):
	x, y = np.broadcast_arrays(x, y)
	latent_dim = x.shape[-1]
	if latent_dim % components != 0:
		raise ValueError(
			f"Representation dim {latent_dim} is not divisible by {components} components."
		)
	component_dim = latent_dim // components
	difference = (x - y).reshape(*x.shape[:-1], components, component_dim)
	asymmetric_dim = component_dim // 2
	if asymmetric_dim:
		max_component = np.maximum(
			difference[..., :asymmetric_dim].max(axis=-1),
			0.0,
		)
	else:
		max_component = np.zeros(difference.shape[:-1], dtype=difference.dtype)
	l2_part = difference[..., asymmetric_dim:]
	if l2_part.shape[-1]:
		l2_component = np.linalg.norm(l2_part + 1e-8, axis=-1)
	else:
		l2_component = np.zeros_like(max_component)
	return (max_component + l2_component).mean(axis=-1) / np.sqrt(float(latent_dim))


def mrn_distance_matrix(source, target, components, chunk_size=64):
	distances = np.empty((len(source), len(target)), dtype=np.float32)
	for start in range(0, len(source), chunk_size):
		stop = min(start + chunk_size, len(source))
		distances[start:stop] = mrn_distance(
			source[start:stop, None, :],
			target[None, :, :],
			components,
		)
	return distances


def representation_distance_matrix(source, target, distance_mode, components):
	if distance_mode == "euclidean":
		source, target, scale = standardized_pair(source, target)
		return euclidean_distance_matrix(source, target), scale
	return mrn_distance_matrix(source, target, components), None


def same_task_next_distances(
	data,
	task_name,
	distance_mode,
	components,
	representation_scale,
):
	mask = data["task_names"].astype(str) == task_name
	indices = np.flatnonzero(mask)
	if len(indices) < 2:
		return np.empty((0,), dtype=np.float32)

	current = indices[:-1]
	following = indices[1:]
	consecutive = (
		(data["episode_indices"][following] == data["episode_indices"][current])
		& (data["step_indices"][following] == data["step_indices"][current] + 1)
	)
	current = current[consecutive]
	following = following[consecutive]
	current_representations = data["representations"][current]
	following_representations = data["representations"][following]
	if distance_mode == "euclidean":
		return np.linalg.norm(
			(current_representations - following_representations)
			/ representation_scale,
			axis=1,
		).astype(np.float32)
	return mrn_distance(
		current_representations,
		following_representations,
		components,
	).astype(np.float32)


def bootstrap_interval(values, samples, random_state):
	values = np.asarray(values, dtype=np.float64)
	indices = random_state.randint(0, len(values), size=(samples, len(values)))
	means = values[indices].mean(axis=1)
	return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def episode_means(values, episode_indices):
	return np.asarray(
		[
			values[episode_indices == episode_idx].mean()
			for episode_idx in np.unique(episode_indices)
		],
		dtype=np.float64,
	)


def sign_permutation_pvalue(values, samples, random_state):
	values = np.asarray(values, dtype=np.float64)
	observed = float(values.mean())
	signs = random_state.choice((-1.0, 1.0), size=(samples, len(values)))
	null_means = (signs * values).mean(axis=1)
	return float((np.count_nonzero(null_means >= observed) + 1) / (samples + 1))


def summarize_direction(
	data,
	source_task,
	target_task,
	args,
	random_state,
):
	source_indices = select_task_rows(
		data,
		source_task,
		args.max_samples_per_task,
		random_state,
	)
	target_indices = select_task_rows(
		data,
		target_task,
		args.max_samples_per_task,
		random_state,
	)
	source_states = data["state_observations"][source_indices]
	target_states = data["state_observations"][target_indices]
	source_representations = data["representations"][source_indices]
	target_representations = data["representations"][target_indices]

	source_coordinates, target_coordinates = standardized_coordinates(
		source_states,
		target_states,
		args.match_dims,
	)
	raw_distances = euclidean_distance_matrix(source_coordinates, target_coordinates)
	representation_distances, representation_scale = representation_distance_matrix(
		source_representations,
		target_representations,
		args.distance_mode,
		args.components,
	)

	matched_indices = raw_distances.argmin(axis=1)
	row_indices = np.arange(len(source_indices))
	matched_raw = raw_distances[row_indices, matched_indices]
	matched_representation = representation_distances[row_indices, matched_indices]
	random_raw = raw_distances.mean(axis=1)
	random_representation = representation_distances.mean(axis=1)
	representation_advantage = random_representation - matched_representation
	source_episode_indices = data["episode_indices"][source_indices]
	episode_representation_advantage = episode_means(
		representation_advantage,
		source_episode_indices,
	)

	representation_selected_indices = representation_distances.argmin(axis=1)
	representation_selected_raw = raw_distances[
		row_indices,
		representation_selected_indices,
	]
	representation_selected_raw_ranks = (
		(raw_distances < representation_selected_raw[:, None]).sum(axis=1) + 1
	)
	matched_representation_ranks = (
		representation_distances < matched_representation[:, None]
	).sum(axis=1) + 1
	top_k = min(args.top_k, len(target_indices))
	within_next = same_task_next_distances(
		data,
		source_task,
		args.distance_mode,
		args.components,
		representation_scale,
	)
	within_next_mean = float(within_next.mean()) if len(within_next) else None
	random_representation_mean = float(random_representation.mean())
	matched_representation_mean = float(matched_representation.mean())
	denominator = (
		None
		if within_next_mean is None
		else random_representation_mean - within_next_mean
	)
	normalized_score = (
		None
		if denominator is None or abs(denominator) < 1e-12
		else float(
			(random_representation_mean - matched_representation_mean)
			/ denominator
		)
	)
	confidence_interval = bootstrap_interval(
		episode_representation_advantage,
		args.bootstrap_samples,
		random_state,
	)
	p_value = sign_permutation_pvalue(
		episode_representation_advantage,
		args.bootstrap_samples,
		random_state,
	)
	pairwise_win_rate_by_state = (
		matched_representation[:, None] < representation_distances
	).mean(axis=1)
	pairwise_win_rate = float(
		episode_means(pairwise_win_rate_by_state, source_episode_indices).mean()
	)

	return {
		"source_task": source_task,
		"target_task": target_task,
		"source_samples": int(len(source_indices)),
		"target_samples": int(len(target_indices)),
		"source_episodes": int(len(episode_representation_advantage)),
		"physical_matched_distance_mean": float(matched_raw.mean()),
		"physical_random_distance_mean": float(random_raw.mean()),
		"representation_same_task_next_mean": within_next_mean,
		"representation_matched_mean": matched_representation_mean,
		"representation_random_mean": random_representation_mean,
		"representation_advantage_episode_mean": float(
			episode_representation_advantage.mean()
		),
		"representation_advantage_episode_95ci": confidence_interval,
		"representation_advantage_episode_permutation_p": p_value,
		"matched_vs_random_pairwise_win_rate": pairwise_win_rate,
		"physical_top_k_recall_at_representation_top1": float(
			(representation_selected_raw_ranks <= top_k).mean()
		),
		"physical_rank_at_representation_top1_median": float(
			np.median(representation_selected_raw_ranks)
		),
		"representation_rank_of_physical_top1_median": float(
			np.median(matched_representation_ranks)
		),
		"normalized_stitching_score": normalized_score,
		"evidence_supported": bool(
			confidence_interval[0] > 0.0
			and p_value < 0.05
			and pairwise_win_rate >= args.min_win_rate
		),
	}


def resolve_distance_mode(data, requested_mode):
	if requested_mode != "auto":
		return requested_mode
	return "euclidean" if "robot_feature_names" in data else "mrn"


def validate_artifact(data, match_dims, distance_mode, components):
	latent_dim = data["representations"].shape[1]
	observation_dim = data["state_observations"].shape[1]
	if distance_mode == "mrn" and latent_dim % components != 0:
		raise ValueError(
			f"Representation dim {latent_dim} is not divisible by {components} components."
		)
	if len(set(match_dims)) != len(match_dims):
		raise ValueError("match_dims must not contain duplicates.")
	if not match_dims:
		raise ValueError("match_dims must not be empty.")
	if min(match_dims) < 0 or max(match_dims) >= observation_dim:
		raise ValueError(
			f"match_dims must be in [0, {observation_dim - 1}], got {match_dims}."
		)
	checkpoints = ordered_unique(data["checkpoints"])
	if len(checkpoints) != 1:
		raise ValueError(
			"The artifact mixes representations from multiple checkpoints. Run the "
			"state collector with --checkpoint final before comparing tasks."
		)
	return checkpoints[0]


def main():
	args = parse_args()
	data = load_rollout_data(args.data)
	args.distance_mode = resolve_distance_mode(data, args.distance_mode)
	checkpoint = validate_artifact(
		data,
		args.match_dims,
		args.distance_mode,
		args.components,
	)
	task_names = ordered_unique(data["task_names"])
	if len(task_names) < 2:
		raise ValueError("Cross-task stitching requires at least two tasks.")

	if args.source_task is not None:
		pairs = [(args.source_task, args.target_task)]
	else:
		pairs = list(permutations(task_names, 2))

	random_state = np.random.RandomState(args.seed)
	directions = [
		summarize_direction(data, source, target, args, random_state)
		for source, target in pairs
	]
	report = {
		"data": args.data,
		"checkpoint": checkpoint,
		"match_dims": args.match_dims,
		"distance_mode": args.distance_mode,
		"components": args.components,
		"interpretation": (
			"For robot-state artifacts this measures handcrafted state-manifold "
			"overlap, not learned quasimetric quality. evidence_supported is a "
			"necessary representation-level result only; executable stitching "
			"requires two-leg policy rollouts with an intervened bridge goal."
		),
		"directions": directions,
	}

	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	with open(args.output, "w") as handle:
		json.dump(report, handle, indent=2)
		handle.write("\n")

	print("data:", args.data)
	print("checkpoint:", checkpoint)
	print("distance_mode:", args.distance_mode)
	for result in directions:
		print(
			f"{result['source_task']} -> {result['target_task']}: "
			f"matched={result['representation_matched_mean']:.6f}, "
			f"random={result['representation_random_mean']:.6f}, "
			f"win_rate={result['matched_vs_random_pairwise_win_rate']:.3f}, "
			f"episode_p={result['representation_advantage_episode_permutation_p']:.6f}, "
			f"supported={result['evidence_supported']}"
		)
	print("results:", args.output)
	return 0


if __name__ == "__main__":
	sys.exit(main())
