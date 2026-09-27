"""Measure latent waypoint coverage and its association with forward transfer."""

import argparse
import importlib.util
import json
import re
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import rankdata

from agent.quasimetric.config import QuasimetricConfig
from agent.quasimetric.distance import mrn_distance
from agent.quasimetric.structure import MultistepQuasimetricLearner


CHECKPOINT_PATTERN = re.compile(r"^(?P<name>.+)_(?P<stage>\d+)_meta_quasimetric\.pt$")
SCRIPT_DIR = Path(__file__).resolve().parent
AWR_EVALUATOR_PATH = SCRIPT_DIR / "final_eval_main3-2awr.py"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def unit_interval(value):
    value = float(value)
    if not 0.0 <= value < 0.5:
        raise argparse.ArgumentTypeError("value must be in [0, 0.5)")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect or load balanced task trajectories, measure directed latent "
            "waypoint coverage, and correlate it with forward transfer."
        )
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--buffer",
        help="Final cumulative offline_data.npz or the directory containing it",
    )
    data_group.add_argument(
        "--rollout-run-dir",
        help="AWR run directory whose final meta agent will collect fresh trajectories",
    )
    encoder_group = parser.add_mutually_exclusive_group()
    encoder_group.add_argument(
        "--model-dir",
        help=(
            "Directory containing *_STAGE_meta_quasimetric.pt; the highest stage is "
            "used as the final meta checkpoint (default: ROLLOUT_RUN_DIR/model)"
        ),
    )
    encoder_group.add_argument(
        "--checkpoint",
        help="One frozen quasimetric checkpoint used for buffer-based analysis",
    )
    parser.add_argument("--config", default=None, help="Optional rollout training config")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional rollout checkpoint name without _actor.pt",
    )
    parser.add_argument(
        "--ft-components",
        default=None,
        help="CSV containing source_task_idx, target_task_idx, and forward_transfer",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output path prefix, without a filename extension",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=positive_int,
        default=None,
        help="Episodes used per task (default: all rollout episodes, otherwise 10)",
    )
    parser.add_argument(
        "--episode-selection",
        choices=("all", "successful"),
        default=None,
        help="Episode subset used as waypoints (default: all for rollout, successful for buffer)",
    )
    parser.add_argument("--rollout-episodes", type=positive_int, default=20)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument(
        "--rollout-reseed-each-episode",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--rollout-sample-action",
        action="store_true",
        help="Sample meta-policy actions instead of using its deterministic mean",
    )
    parser.add_argument("--waypoints-per-task", type=positive_int, default=100)
    parser.add_argument(
        "--trim-fraction",
        type=unit_interval,
        default=0.05,
        help="Fraction removed from each end of every selected trajectory",
    )
    parser.add_argument("--knn-k", type=positive_int, default=5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Coverage threshold in units of each target waypoint's local kNN radius",
    )
    parser.add_argument(
        "--allow-failed-fallback",
        action="store_true",
        help="For tasks with no successful episode, use the highest-return episodes",
    )
    parser.add_argument("--permutations", type=nonnegative_int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-batch-size", type=positive_int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.buffer and not (args.model_dir or args.checkpoint):
        parser.error("--buffer requires --model-dir or --checkpoint")
    if args.rollout_run_dir and args.checkpoint:
        parser.error("--checkpoint cannot load the actor required by --rollout-run-dir")
    if args.episodes_per_task is None:
        args.episodes_per_task = args.rollout_episodes if args.rollout_run_dir else 10
    if args.episode_selection is None:
        args.episode_selection = "all" if args.rollout_run_dir else "successful"
    return args


def resolve_buffer_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "offline_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Replay snapshot does not exist: {path}")
    return path


def load_replay_snapshot(path):
    required = {"next_obses", "actions", "rewards", "successes", "task_ids", "episode_ids"}
    with np.load(path) as replay:
        missing = required.difference(replay.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        arrays = {key: np.asarray(replay[key]) for key in required}

    size = arrays["next_obses"].shape[0]
    if size == 0:
        raise ValueError("Replay snapshot is empty")
    for key, value in arrays.items():
        if value.shape[0] != size:
            raise ValueError(f"{key} has {value.shape[0]} rows, expected {size}")
    if arrays["next_obses"].ndim != 2 or arrays["actions"].ndim != 2:
        raise ValueError("next_obses and actions must be rank-2 arrays")

    arrays["task_ids"] = arrays["task_ids"].reshape(-1).astype(int)
    arrays["episode_ids"] = arrays["episode_ids"].reshape(-1).astype(int)
    arrays["rewards"] = arrays["rewards"].reshape(-1).astype(float)
    arrays["successes"] = arrays["successes"].reshape(-1).astype(float)
    return arrays


def load_awr_evaluator():
    spec = importlib.util.spec_from_file_location(
        "final_eval_main3_2awr", AWR_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AWR evaluator: {AWR_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation_vector(observation):
    if isinstance(observation, dict):
        return np.asarray(observation["observation"], dtype=np.float32)
    return np.asarray(observation, dtype=np.float32)


def collect_meta_rollouts(args, device, output_prefix):
    evaluator = load_awr_evaluator()
    run_dir = Path(evaluator.resolve_existing_path(args.rollout_run_dir))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Rollout run directory does not exist: {run_dir}")
    config_path = evaluator.find_training_config(str(run_dir), args.config)
    training_config = evaluator.load_training_config(config_path)
    model_dir = (
        Path(evaluator.resolve_existing_path(args.model_dir))
        if args.model_dir
        else run_dir / "model"
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    model_name, checkpoint_stage = evaluator.latest_meta_checkpoint(
        str(model_dir), training_config, args.model_name
    )
    encoder_checkpoint = model_dir / f"{model_name}_quasimetric.pt"

    evaluator.set_seed_everywhere(args.rollout_seed)
    env = evaluator.MetaWorldSingleEnvSequence(
        **evaluator.make_env_kwargs(training_config, args.rollout_seed)
    )
    try:
        obs_space = evaluator.vector_observation_space(env.env.observation_space)
        action_dim = int(env.env.action_space.shape[0])
        task_sequence = list(env.env_list)
        agent = evaluator.build_and_load_meta_agent(
            obs_space.shape[0],
            action_dim,
            device,
            training_config,
            str(model_dir),
            model_name,
        )

        observations = []
        next_observations = []
        actions = []
        rewards = []
        successes = []
        task_ids = []
        episode_ids = []
        episode_rows = []
        global_episode_id = 0
        for task_id, task_name in enumerate(task_sequence, start=1):
            env.set_task(task_name)
            test_env = env._wrap_env(env._make_base_env(), eval_mode=True)
            initial_reset_kwargs = (
                {"seed": env.current_seed} if env.current_seed is not None else {}
            )
            subsequent_reset_kwargs = (
                initial_reset_kwargs if args.rollout_reseed_each_episode else {}
            )
            task_successes = []
            try:
                for episode_idx in range(args.rollout_episodes):
                    reset_kwargs = (
                        initial_reset_kwargs if episode_idx == 0 else subsequent_reset_kwargs
                    )
                    observation, _ = test_env.reset(**reset_kwargs)
                    if env._uses_obs_normalization():
                        observation = env._normalize_obs(observation)
                    episode_return = 0.0
                    episode_success = False
                    episode_steps = 0
                    while True:
                        actor_observation = observation_vector(observation)
                        with torch.inference_mode():
                            if args.rollout_sample_action:
                                action = agent.act(actor_observation, sample=True)
                            else:
                                action = env._evaluate_action(agent, observation)
                        next_observation, reward, terminated, truncated, info = test_env.step(
                            action
                        )
                        if env._uses_obs_normalization():
                            next_observation = env._normalize_obs(next_observation)
                        step_success = bool(info.get("success", False))

                        observations.append(actor_observation)
                        next_observations.append(observation_vector(next_observation))
                        actions.append(np.asarray(action, dtype=np.float32))
                        rewards.append(float(reward))
                        successes.append(float(step_success))
                        task_ids.append(task_id)
                        episode_ids.append(global_episode_id)

                        episode_return += float(reward)
                        episode_success = episode_success or step_success
                        episode_steps += 1
                        observation = next_observation
                        if terminated or truncated:
                            break

                    task_successes.append(float(episode_success))
                    episode_rows.append(
                        {
                            "task_id": task_id,
                            "task": str(task_name),
                            "episode": episode_idx,
                            "global_episode_id": global_episode_id,
                            "steps": episode_steps,
                            "return": episode_return,
                            "success": episode_success,
                            "rollout_seed": args.rollout_seed,
                            "sample_action": bool(args.rollout_sample_action),
                        }
                    )
                    global_episode_id += 1
            finally:
                test_env.close()
            print(
                f"rollout task {task_id}/{len(task_sequence)} {task_name}: "
                f"episodes={args.rollout_episodes} "
                f"success={np.mean(task_successes):.3f}"
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    replay = {
        "obses": np.asarray(observations, dtype=np.float32),
        "next_obses": np.asarray(next_observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "task_ids": np.asarray(task_ids, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
    }
    np.savez_compressed(f"{output_prefix}_rollouts.npz", **replay)
    pd.DataFrame(episode_rows).to_csv(
        f"{output_prefix}_rollout_episodes.csv", index=False
    )
    return (
        replay,
        {task_id: str(name) for task_id, name in enumerate(task_sequence, start=1)},
        encoder_checkpoint,
        checkpoint_stage,
        {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "model_dir": str(model_dir),
            "model_name": model_name,
        },
    )


def evenly_spaced(items, count):
    if len(items) <= count:
        return list(items)
    positions = np.rint(np.linspace(0, len(items) - 1, count)).astype(int)
    return [items[position] for position in positions]


def sample_episode_indices(indices, count, trim_fraction):
    trim = int(np.floor(indices.size * trim_fraction))
    candidates = indices[trim : indices.size - trim if trim else None]
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    count = min(count, candidates.size)
    positions = np.rint(np.linspace(0, candidates.size - 1, count)).astype(int)
    return candidates[np.unique(positions)]


def extract_waypoints(replay, args):
    task_ids = sorted(int(value) for value in np.unique(replay["task_ids"]) if value >= 0)
    if not task_ids:
        raise ValueError("Replay snapshot contains no nonnegative task IDs")

    waypoint_observations = {}
    audit_rows = []
    for task_id in task_ids:
        task_indices = np.flatnonzero(replay["task_ids"] == task_id)
        episodes = []
        for episode_id in sorted(np.unique(replay["episode_ids"][task_indices])):
            indices = task_indices[replay["episode_ids"][task_indices] == episode_id]
            episodes.append(
                {
                    "episode_id": int(episode_id),
                    "indices": indices,
                    "success": bool(np.any(replay["successes"][indices] > 0.5)),
                    "return": float(replay["rewards"][indices].sum()),
                }
            )

        successful = [episode for episode in episodes if episode["success"]]
        if args.episode_selection == "all":
            selected = evenly_spaced(episodes, args.episodes_per_task)
            selection = "all"
        elif successful:
            selected = evenly_spaced(successful, args.episodes_per_task)
            selection = "successful"
        elif args.allow_failed_fallback:
            selected = sorted(
                episodes,
                key=lambda episode: (-episode["return"], episode["episode_id"]),
            )[: args.episodes_per_task]
            selection = "top_return_fallback"
        else:
            selected = []
            selection = "missing_no_success"

        selected_indices = []
        if selected:
            base_count, remainder = divmod(args.waypoints_per_task, len(selected))
            for position, episode in enumerate(selected):
                episode_count = base_count + int(position < remainder)
                selected_indices.append(
                    sample_episode_indices(
                        episode["indices"], episode_count, args.trim_fraction
                    )
                )
        if selected_indices:
            waypoint_indices = np.concatenate(selected_indices)
            waypoint_observations[task_id] = np.asarray(
                replay["next_obses"][waypoint_indices], dtype=np.float32
            )

        audit_rows.append(
            {
                "task_id": task_id,
                "transitions": int(task_indices.size),
                "episodes": len(episodes),
                "successful_episodes": len(successful),
                "selected_episodes": len(selected),
                "selection": selection,
                "waypoints": int(
                    waypoint_observations.get(task_id, np.empty((0,))).shape[0]
                ),
                "selected_episode_ids": ";".join(
                    str(episode["episode_id"]) for episode in selected
                ),
            }
        )
    return task_ids, waypoint_observations, pd.DataFrame(audit_rows)


def infer_task_names(buffer_path, task_ids, ft_frame=None):
    names = {task_id: f"task-{task_id}" for task_id in task_ids}
    candidate_directories = [buffer_path.parent, buffer_path.parent.parent]
    for directory in candidate_directories:
        for task_id in task_ids:
            prefix = f"task_{task_id:02d}_"
            matches = sorted(directory.glob(f"{prefix}*")) if directory.is_dir() else []
            if len(matches) == 1:
                names[task_id] = matches[0].name[len(prefix) :]

    if buffer_path.stem.endswith("_rollouts"):
        rollout_prefix = buffer_path.stem[: -len("_rollouts")]
        episode_path = buffer_path.with_name(f"{rollout_prefix}_rollout_episodes.csv")
        if episode_path.is_file():
            episode_frame = pd.read_csv(episode_path, usecols=["task_id", "task"])
            for task_id, task_name in episode_frame.drop_duplicates().itertuples(
                index=False, name=None
            ):
                if int(task_id) in names:
                    names[int(task_id)] = str(task_name)

    if ft_frame is not None and "task" in ft_frame.columns:
        for target_id, task_name in ft_frame[["target_task_idx", "task"]].drop_duplicates().itertuples(
            index=False, name=None
        ):
            target_id = int(target_id)
            if target_id in names:
                names[target_id] = str(task_name)
    return names


def discover_final_checkpoint(model_dir):
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    checkpoints = {}
    for path in model_dir.glob("*_meta_quasimetric.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if not match:
            continue
        stage = int(match.group("stage"))
        if stage in checkpoints:
            raise ValueError(f"Multiple quasimetric checkpoints found for stage {stage}")
        checkpoints[stage] = path
    if not checkpoints:
        raise FileNotFoundError(f"No meta quasimetric checkpoints found in {model_dir}")
    final_stage = max(checkpoints)
    return checkpoints[final_stage], final_stage


def load_encoder(checkpoint_path, obs_dim, action_dim, device):
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if "quasimetric" not in payload or "quasimetric_cfg" not in payload:
        raise ValueError(f"Unexpected quasimetric checkpoint format: {checkpoint_path}")

    config_fields = {field.name for field in fields(QuasimetricConfig)}
    config = QuasimetricConfig(
        **{
            key: value
            for key, value in payload["quasimetric_cfg"].items()
            if key in config_fields
        }
    )
    learner = MultistepQuasimetricLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        config=config,
    )
    learner.load_state_dict(payload["quasimetric"]["model_state"])
    learner.eval()
    return learner


def encode_waypoints(learner, waypoint_observations, device, batch_size=1024):
    encoded = {}
    with torch.inference_mode():
        for task_id, observations in waypoint_observations.items():
            parts = []
            for start in range(0, observations.shape[0], batch_size):
                batch = torch.as_tensor(
                    observations[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                parts.append(learner.state_encoder(batch))
            encoded[task_id] = torch.cat(parts, dim=0)
    return encoded


def symmetric_pairwise_distance(latent_a, latent_b, components, batch_size):
    parts = []
    same_tensor = latent_a.data_ptr() == latent_b.data_ptr()
    with torch.inference_mode():
        for start in range(0, latent_a.shape[0], batch_size):
            left = latent_a[start : start + batch_size]
            forward = mrn_distance(
                left[:, None, :], latent_b[None, :, :], components=components
            )
            if same_tensor:
                backward = mrn_distance(
                    latent_b[None, :, :], left[:, None, :], components=components
                )
            else:
                backward = mrn_distance(
                    latent_b[None, :, :], left[:, None, :], components=components
                )
            parts.append(((forward + backward) * 0.5).cpu())
    return torch.cat(parts, dim=0).numpy()


def local_knn_radii(latents, components, knn_k, batch_size):
    if latents.shape[0] <= knn_k:
        raise ValueError(
            f"Need more than k={knn_k} waypoints, found {latents.shape[0]}"
        )
    distances = symmetric_pairwise_distance(latents, latents, components, batch_size)
    np.fill_diagonal(distances, np.inf)
    finite_distances = distances[np.isfinite(distances)]
    positive_distances = finite_distances[finite_distances > np.finfo(np.float32).eps]
    if positive_distances.size == 0:
        raise ValueError("All within-task latent waypoint distances are zero")

    duplicate_tolerance = max(
        float(np.median(positive_distances)) * 1e-7,
        float(np.finfo(np.float32).eps),
    )
    radii = []
    for row in distances:
        distinct_distances = row[np.isfinite(row) & (row > duplicate_tolerance)]
        if distinct_distances.size == 0:
            radii.append(float(np.median(positive_distances)))
            continue
        effective_k = min(knn_k, distinct_distances.size)
        radii.append(
            float(np.partition(distinct_distances, effective_k - 1)[effective_k - 1])
        )
    return np.asarray(radii, dtype=np.float32)


def coverage_row(
    source_task_id,
    task_ids,
    encoded,
    learner,
    checkpoint_path,
    args,
    task_names,
):
    rows = []
    checkpoint_match = CHECKPOINT_PATTERN.fullmatch(Path(checkpoint_path).name)
    encoder_stage = (
        int(checkpoint_match.group("stage")) if checkpoint_match else np.nan
    )
    if source_task_id not in encoded:
        for target_task_id in task_ids:
            if source_task_id != target_task_id:
                rows.append(
                    empty_pair_row(
                        source_task_id,
                        target_task_id,
                        checkpoint_path,
                        task_names,
                    )
                )
        return rows

    components = learner.config.components
    radii = {
        task_id: local_knn_radii(
            latents, components, args.knn_k, args.distance_batch_size
        )
        for task_id, latents in encoded.items()
    }
    source_latents = encoded[source_task_id]
    for target_task_id in task_ids:
        if target_task_id == source_task_id:
            continue
        if target_task_id not in encoded:
            rows.append(
                empty_pair_row(
                    source_task_id,
                    target_task_id,
                    checkpoint_path,
                    task_names,
                )
            )
            continue

        target_latents = encoded[target_task_id]
        distances = symmetric_pairwise_distance(
            source_latents,
            target_latents,
            components,
            args.distance_batch_size,
        )
        target_ratios = distances.min(axis=0) / radii[target_task_id]
        source_ratios = distances.min(axis=1) / radii[source_task_id]
        target_covered = int(np.count_nonzero(target_ratios <= args.alpha))
        source_covered = int(np.count_nonzero(source_ratios <= args.alpha))
        directed_coverage = float(target_covered / target_latents.shape[0])
        reverse_coverage = float(source_covered / source_latents.shape[0])
        rows.append(
            {
                "source_task_idx": source_task_id,
                "source_task": task_names[source_task_id],
                "target_task_idx": target_task_id,
                "target_task": task_names[target_task_id],
                "encoder_checkpoint": str(checkpoint_path),
                "encoder_stage": encoder_stage,
                "source_waypoints": int(source_latents.shape[0]),
                "target_waypoints": int(target_latents.shape[0]),
                "covered_target_waypoints": target_covered,
                "covered_source_waypoints": source_covered,
                "shared_waypoint_count": 0.5 * (target_covered + source_covered),
                "directed_coverage": directed_coverage,
                "reverse_coverage_same_encoder": reverse_coverage,
                "symmetric_overlap_same_encoder": 0.5
                * (directed_coverage + reverse_coverage),
                "soft_directed_coverage": float(
                    np.exp(-target_ratios / args.alpha).mean()
                ),
                "median_target_distance_ratio": float(np.median(target_ratios)),
                "median_cross_distance": float(np.median(distances.min(axis=0))),
                "median_target_knn_radius": float(np.median(radii[target_task_id])),
                "alpha": args.alpha,
                "knn_k": args.knn_k,
            }
        )
    return rows


def empty_pair_row(source_task_id, target_task_id, checkpoint_path, task_names):
    match = CHECKPOINT_PATTERN.fullmatch(Path(checkpoint_path).name)
    return {
        "source_task_idx": source_task_id,
        "source_task": task_names[source_task_id],
        "target_task_idx": target_task_id,
        "target_task": task_names[target_task_id],
        "encoder_checkpoint": str(checkpoint_path),
        "encoder_stage": int(match.group("stage")) if match else np.nan,
        "source_waypoints": 0,
        "target_waypoints": 0,
        "covered_target_waypoints": np.nan,
        "covered_source_waypoints": np.nan,
        "shared_waypoint_count": np.nan,
        "directed_coverage": np.nan,
        "reverse_coverage_same_encoder": np.nan,
        "symmetric_overlap_same_encoder": np.nan,
        "soft_directed_coverage": np.nan,
        "median_target_distance_ratio": np.nan,
        "median_cross_distance": np.nan,
        "median_target_knn_radius": np.nan,
        "alpha": np.nan,
        "knn_k": np.nan,
    }


def compute_coverage(
    replay,
    task_ids,
    waypoint_observations,
    checkpoint_path,
    args,
    task_names,
    device,
):
    rows = []
    learner = load_encoder(
        checkpoint_path,
        replay["next_obses"].shape[1],
        replay["actions"].shape[1],
        device,
    )
    encoded = encode_waypoints(learner, waypoint_observations, device)

    for source_task_id in task_ids:
        rows.extend(
            coverage_row(
                source_task_id,
                task_ids,
                encoded,
                learner,
                checkpoint_path,
                args,
                task_names,
            )
        )
    return pd.DataFrame(rows)


def spearman_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def task_label_permutation_test(frame, pair_frame, metric, task_ids, permutations, seed):
    available = frame.dropna(subset=[metric, "forward_transfer"]).copy()
    observed = spearman_correlation(available[metric], available["forward_transfer"])
    result = {
        "n_pairs": int(len(available)),
        "spearman_rho": observed,
        "qap_permutations": permutations,
        "qap_p_greater": np.nan,
        "qap_p_two_sided": np.nan,
    }
    if not np.isfinite(observed) or permutations == 0:
        return result

    matrix = pair_frame.pivot(
        index="source_task_idx", columns="target_task_idx", values=metric
    )
    eligible = [
        task_id
        for task_id in task_ids
        if task_id in matrix.index
        and task_id in matrix.columns
        and np.isfinite(matrix.loc[task_id].drop(labels=task_id, errors="ignore")).any()
    ]
    eligible_set = set(eligible)
    available = available[
        available["source_task_idx"].isin(eligible_set)
        & available["target_task_idx"].isin(eligible_set)
    ]
    if len(available) < 3:
        result["n_pairs"] = int(len(available))
        result["spearman_rho"] = np.nan
        return result

    observed = spearman_correlation(available[metric], available["forward_transfer"])
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(permutations):
        permutation = rng.permutation(eligible)
        mapping = dict(zip(eligible, permutation))
        permuted_values = np.asarray(
            [
                matrix.loc[mapping[int(source)], mapping[int(target)]]
                for source, target in available[
                    ["source_task_idx", "target_task_idx"]
                ].itertuples(index=False, name=None)
            ],
            dtype=float,
        )
        rho = spearman_correlation(permuted_values, available["forward_transfer"])
        if np.isfinite(rho):
            null_values.append(rho)

    null_values = np.asarray(null_values, dtype=float)
    result.update(
        {
            "n_pairs": int(len(available)),
            "spearman_rho": observed,
            "valid_qap_permutations": int(null_values.size),
            "qap_p_greater": float(
                (1 + np.count_nonzero(null_values >= observed))
                / (1 + null_values.size)
            ),
            "qap_p_two_sided": float(
                (1 + np.count_nonzero(np.abs(null_values) >= abs(observed)))
                / (1 + null_values.size)
            ),
        }
    )
    return result


def merge_forward_transfer(pair_frame, ft_path):
    ft_path = Path(ft_path).expanduser().resolve()
    if not ft_path.is_file():
        raise FileNotFoundError(f"Forward-transfer CSV does not exist: {ft_path}")
    ft_frame = pd.read_csv(ft_path)
    required = {"source_task_idx", "target_task_idx", "forward_transfer"}
    missing = required.difference(ft_frame.columns)
    if missing:
        raise ValueError(f"{ft_path} is missing columns: {sorted(missing)}")
    if ft_frame.duplicated(["source_task_idx", "target_task_idx"]).any():
        raise ValueError("Forward-transfer CSV contains duplicate source-target pairs")
    ft_frame["source_task_idx"] = pd.to_numeric(
        ft_frame["source_task_idx"], errors="raise"
    ).astype(int)
    ft_frame["target_task_idx"] = pd.to_numeric(
        ft_frame["target_task_idx"], errors="raise"
    ).astype(int)
    ft_frame["forward_transfer"] = pd.to_numeric(
        ft_frame["forward_transfer"], errors="raise"
    )
    return ft_frame.merge(
        pair_frame,
        on=["source_task_idx", "target_task_idx"],
        how="left",
        suffixes=("_ft", ""),
    ), ft_frame


def plot_coverage_matrix(pair_frame, task_ids, task_names, output_path):
    matrix = pair_frame.pivot(
        index="source_task_idx", columns="target_task_idx", values="directed_coverage"
    ).reindex(index=task_ids, columns=task_ids)
    tasks_with_waypoints = set(
        pair_frame.loc[pair_frame["source_waypoints"] > 0, "source_task_idx"].astype(int)
    )
    for task_id in tasks_with_waypoints:
        matrix.loc[task_id, task_id] = 1.0
    labels = [f"{task_id}. {task_names[task_id].removesuffix('-v3')}" for task_id in task_ids]
    matrix.index = labels
    matrix.columns = labels

    figure, axis = plt.subplots(
        figsize=(max(10.0, len(task_ids) * 1.1), max(7.0, len(task_ids) * 0.75)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        mask=matrix.isna(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Target waypoint coverage"},
        ax=axis,
    )
    axis.set_xlabel("Target task")
    axis.set_ylabel("Source task")
    axis.set_title("Directed latent waypoint coverage")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_transfer_scatter(frame, statistic, output_path):
    available = frame.dropna(subset=["directed_coverage", "forward_transfer"])
    if available.empty:
        return False
    figure, axis = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    axis.scatter(
        available["directed_coverage"],
        available["forward_transfer"],
        s=58,
        color="#197278",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for position, row in enumerate(
        available.sort_values("forward_transfer").itertuples(index=False)
    ):
        axis.annotate(
            f"{int(row.source_task_idx)}->{int(row.target_task_idx)}",
            (row.directed_coverage, row.forward_transfer),
            xytext=(5, 7 if position % 2 == 0 else -12),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.0, color="#6c757d", linewidth=0.9, linestyle="--")
    axis.set_xlim(-0.03, 1.03)
    axis.set_xlabel("Directed target waypoint coverage")
    axis.set_ylabel("Normalized AUC forward transfer")
    rho = statistic["spearman_rho"]
    p_value = statistic["qap_p_greater"]
    axis.set_title(
        f"Waypoint coverage vs. forward transfer\n"
        f"Spearman rho={rho:.3f}, task-label permutation p={p_value:.4f}, "
        f"n={statistic['n_pairs']}"
    )
    axis.grid(color="#d9d9d9", linewidth=0.6, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return True


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if args.rollout_run_dir:
        (
            replay,
            task_names,
            encoder_checkpoint,
            encoder_stage,
            source_metadata,
        ) = collect_meta_rollouts(args, device, output_prefix)
        buffer_path = None
    else:
        buffer_path = resolve_buffer_path(args.buffer)
        replay = load_replay_snapshot(buffer_path)
        ft_for_names = pd.read_csv(args.ft_components) if args.ft_components else None
        buffer_task_ids = sorted(
            int(value) for value in np.unique(replay["task_ids"]) if value >= 0
        )
        task_names = infer_task_names(buffer_path, buffer_task_ids, ft_for_names)
        source_metadata = {}
        if args.model_dir:
            encoder_checkpoint, encoder_stage = discover_final_checkpoint(args.model_dir)
        else:
            encoder_checkpoint = Path(args.checkpoint).expanduser().resolve()
            if not encoder_checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {encoder_checkpoint}")
            match = CHECKPOINT_PATTERN.fullmatch(encoder_checkpoint.name)
            encoder_stage = int(match.group("stage")) if match else None

    task_ids, waypoint_observations, audit_frame = extract_waypoints(replay, args)
    audit_frame["task"] = audit_frame["task_id"].map(task_names)
    audit_frame.to_csv(f"{output_prefix}_waypoint_audit.csv", index=False)

    print(
        f"using shared final meta encoder: {encoder_checkpoint} "
        f"(stage={encoder_stage})"
    )
    pair_frame = compute_coverage(
        replay,
        task_ids,
        waypoint_observations,
        encoder_checkpoint,
        args,
        task_names,
        device,
    )
    pair_frame.to_csv(f"{output_prefix}_overlap_pairs.csv", index=False)
    plot_coverage_matrix(
        pair_frame,
        task_ids,
        task_names,
        f"{output_prefix}_overlap_heatmap.png",
    )

    statistics = {}
    ft_pair_count = 0
    if args.ft_components:
        merged, _ = merge_forward_transfer(pair_frame, args.ft_components)
        merged.to_csv(f"{output_prefix}_ft_pairs.csv", index=False)
        ft_pair_count = len(merged)
        for metric in (
            "directed_coverage",
            "soft_directed_coverage",
            "symmetric_overlap_same_encoder",
        ):
            statistics[metric] = task_label_permutation_test(
                merged,
                pair_frame,
                metric,
                task_ids,
                args.permutations,
                args.seed,
            )
        plot_transfer_scatter(
            merged,
            statistics["directed_coverage"],
            f"{output_prefix}_ft_scatter.png",
        )

    saved_meta_rollout = bool(
        buffer_path is not None and buffer_path.stem.endswith("_rollouts")
    )
    if args.rollout_run_dir:
        data_source = "meta_rollout"
    elif saved_meta_rollout:
        data_source = "saved_meta_rollout"
        source_metadata["rollout_file"] = str(buffer_path)
    else:
        data_source = "replay_buffer"
    recorded_rollout_episodes = None
    if data_source != "replay_buffer" and audit_frame["episodes"].nunique() == 1:
        recorded_rollout_episodes = int(audit_frame["episodes"].iloc[0])

    report = {
        "data_source": data_source,
        "buffer": str(buffer_path) if buffer_path else None,
        "rollout_run_dir": str(Path(args.rollout_run_dir).expanduser().resolve())
        if args.rollout_run_dir
        else None,
        "rollout_episodes": recorded_rollout_episodes,
        "rollout_seed": args.rollout_seed if args.rollout_run_dir else None,
        "rollout_reseed_each_episode": bool(args.rollout_reseed_each_episode)
        if args.rollout_run_dir
        else None,
        "rollout_sample_action": bool(args.rollout_sample_action)
        if args.rollout_run_dir
        else None,
        "source_metadata": source_metadata,
        "encoder_mode": "final_meta_shared",
        "model_dir": source_metadata.get("model_dir")
        or (str(Path(args.model_dir).expanduser().resolve()) if args.model_dir else None),
        "checkpoint": str(encoder_checkpoint),
        "encoder_stage": encoder_stage,
        "ft_components": str(Path(args.ft_components).expanduser().resolve())
        if args.ft_components
        else None,
        "task_count": len(task_ids),
        "tasks_with_waypoints": len(waypoint_observations),
        "ft_pair_count": ft_pair_count,
        "episodes_per_task": args.episodes_per_task,
        "episode_selection": args.episode_selection,
        "waypoints_per_task": args.waypoints_per_task,
        "trim_fraction": args.trim_fraction,
        "knn_k": args.knn_k,
        "alpha": args.alpha,
        "allow_failed_fallback": args.allow_failed_fallback,
        "statistics": statistics,
    }
    with open(f"{output_prefix}_stats.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2)

    print(f"saved waypoint audit: {output_prefix}_waypoint_audit.csv")
    print(f"saved overlap pairs: {output_prefix}_overlap_pairs.csv")
    if statistics:
        primary = statistics["directed_coverage"]
        print(
            "directed coverage vs FT: "
            f"n={primary['n_pairs']} rho={primary['spearman_rho']:.4f} "
            f"QAP p(one-sided)={primary['qap_p_greater']:.4f}"
        )


if __name__ == "__main__":
    main()