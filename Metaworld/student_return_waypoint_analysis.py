"""Correlate task-student cross-task returns with latent waypoint overlap."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from agent.sac import SACAgent
from waypoint_transfer_analysis import (
    json_ready,
    load_awr_evaluator,
    spearman_correlation,
)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Roll out every trained task student on all other tasks and correlate "
            "mean episodic return with latent waypoint overlap."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overlap-pairs", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--return-pairs",
        default=None,
        help="Reuse an existing return-pair CSV and skip environment rollouts.",
    )
    parser.add_argument("--episodes", type=positive_int, default=20)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--reseed-each-episode", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--sample-action", action="store_true")
    parser.add_argument("--permutations", type=nonnegative_int, default=10000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def discover_student_checkpoints(model_dir, prefix, task_count):
    pattern = re.compile(rf"^{re.escape(prefix)}_(?P<stage>\d+)_actor\.pt$")
    checkpoints = {}
    for path in model_dir.glob(f"{prefix}_*_actor.pt"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        stage = int(match.group("stage"))
        if stage in checkpoints:
            raise ValueError(f"Multiple student actor checkpoints found for stage {stage}")
        checkpoints[stage] = path

    expected = set(range(task_count))
    missing = sorted(expected.difference(checkpoints))
    if missing:
        raise FileNotFoundError(f"Missing student actor checkpoints for stages {missing}")
    extra = sorted(set(checkpoints).difference(expected))
    if extra:
        raise ValueError(f"Unexpected student actor checkpoint stages {extra}")
    return checkpoints


def build_student_agent(obs_dim, action_dim, device, config, actor_path, evaluator):
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=[-1.0, 1.0],
        device=device,
        batch_size=int(config.get("batch_size", 256)),
        discount=float(config.get("discount", 0.99)),
        init_temperature=float(config.get("init_temperature", 0.1)),
        actor_lr=float(config.get("actor_lr", 1e-4)),
        critic_lr=float(config.get("critic_lr", 1e-4)),
        alpha_lr=float(config.get("alpha_lr", 1e-4)),
        critic_tau=float(config.get("critic_tau", 0.005)),
        actor_update_frequency=int(config.get("actor_update_frequency", 1)),
        critic_target_update_frequency=int(
            config.get("critic_target_update_frequency", 1)
        ),
    )
    actor_state = evaluator.load_torch(str(actor_path), device, weights_only=True)
    agent.actor.load_state_dict(actor_state)
    agent.eval()
    return agent


def evaluate_students(args, device):
    evaluator = load_awr_evaluator()
    run_dir = Path(evaluator.resolve_existing_path(args.run_dir))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    config_path = evaluator.find_training_config(str(run_dir), args.config)
    config = evaluator.load_training_config(config_path)
    model_dir = (
        Path(evaluator.resolve_existing_path(args.model_dir))
        if args.model_dir
        else run_dir / "model"
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    evaluator.set_seed_everywhere(args.eval_seed)
    env = evaluator.MetaWorldSingleEnvSequence(
        **evaluator.make_env_kwargs(config, args.eval_seed)
    )
    try:
        task_names = list(env.env_list)
        obs_space = evaluator.vector_observation_space(env.env.observation_space)
        action_dim = int(env.env.action_space.shape[0])
        prefix = evaluator.expected_log_name(config)
        checkpoints = discover_student_checkpoints(
            model_dir, prefix, len(task_names)
        )

        pair_rows = []
        episode_rows = []
        for source_stage in range(len(task_names)):
            source_task_idx = source_stage + 1
            source_task = str(task_names[source_stage])
            actor_path = checkpoints[source_stage]
            student = build_student_agent(
                obs_space.shape[0],
                action_dim,
                device,
                config,
                actor_path,
                evaluator,
            )
            for target_stage, target_task in enumerate(task_names):
                target_task_idx = target_stage + 1
                if target_task_idx == source_task_idx:
                    continue



                evaluator.set_seed_everywhere(args.eval_seed + target_task_idx)
                env.set_task(target_task)
                if args.sample_action:
                    returns = rollout_sampled_student(
                        env,
                        student,
                        args.episodes,
                        bool(args.reseed_each_episode),
                    )
                else:
                    eval_results = env.evaluate_agent(
                        student,
                        args.episodes,
                        reseed_each_episode=bool(args.reseed_each_episode),
                    )
                    returns = [
                        float(np.asarray(value).reshape(-1)[0])
                        for value in eval_results["episodic_returns"]
                    ]
                if len(returns) != args.episodes:
                    raise RuntimeError(
                        f"Expected {args.episodes} returns for "
                        f"{source_task_idx}->{target_task_idx}, got {len(returns)}"
                    )

                return_array = np.asarray(returns, dtype=float)
                if not np.isfinite(return_array).all():
                    raise ValueError(
                        f"Non-finite return for pair {source_task_idx}->{target_task_idx}"
                    )
                pair_rows.append(
                    {
                        "source_task_idx": source_task_idx,
                        "source_task": source_task,
                        "target_task_idx": target_task_idx,
                        "target_task": str(target_task),
                        "checkpoint_stage": source_stage,
                        "checkpoint": str(actor_path),
                        "episodes": args.episodes,
                        "return_mean": float(return_array.mean()),
                        "return_std": float(return_array.std(ddof=0)),
                        "return_sem": float(
                            return_array.std(ddof=1) / np.sqrt(return_array.size)
                        )
                        if return_array.size > 1
                        else np.nan,
                        "return_median": float(np.median(return_array)),
                    }
                )
                episode_rows.extend(
                    {
                        "source_task_idx": source_task_idx,
                        "source_task": source_task,
                        "target_task_idx": target_task_idx,
                        "target_task": str(target_task),
                        "episode": episode_idx,
                        "return": episode_return,
                    }
                    for episode_idx, episode_return in enumerate(returns)
                )
                print(
                    f"student {source_task_idx}/{len(task_names)} -> "
                    f"task {target_task_idx}/{len(task_names)}: "
                    f"return={return_array.mean():.3f} +/- {return_array.std(ddof=0):.3f}"
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    pairs = pd.DataFrame(pair_rows)
    episodes = pd.DataFrame(episode_rows)
    expected_pairs = len(task_names) * (len(task_names) - 1)
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} cross-task pairs, got {len(pairs)}")
    return pairs, episodes, task_names, model_dir, config_path


def rollout_sampled_student(env, student, episodes, reseed_each_episode):
    test_env = env._wrap_env(env._make_base_env(), eval_mode=True)
    initial_reset_kwargs = {"seed": env.current_seed} if env.current_seed is not None else {}
    subsequent_reset_kwargs = initial_reset_kwargs if reseed_each_episode else {}
    returns = []
    student.eval()
    try:
        for episode_idx in range(episodes):
            reset_kwargs = initial_reset_kwargs if episode_idx == 0 else subsequent_reset_kwargs
            observation, _ = test_env.reset(**reset_kwargs)
            if env._uses_obs_normalization():
                observation = env._normalize_obs(observation)
            episode_return = 0.0
            while True:
                if isinstance(observation, dict):
                    actor_observation = observation["observation"]
                else:
                    actor_observation = observation
                with torch.inference_mode():
                    action = student.act(actor_observation, sample=True)
                next_observation, reward, terminated, truncated, _ = test_env.step(action)
                if env._uses_obs_normalization():
                    next_observation = env._normalize_obs(next_observation)
                episode_return += float(reward)
                observation = next_observation
                if terminated or truncated:
                    returns.append(episode_return)
                    break
    finally:
        student.train()
        test_env.close()
    return returns


def two_way_fixed_effect_design(source_ids, target_ids):
    source_ids = np.asarray(source_ids)
    target_ids = np.asarray(target_ids)
    if source_ids.size != target_ids.size or source_ids.size == 0:
        raise ValueError("Source and target task IDs must have the same nonzero length")
    design_columns = [np.ones(source_ids.size, dtype=float)]
    for ids in (source_ids, target_ids):
        categories = np.unique(ids)
        design_columns.extend((ids == category).astype(float) for category in categories[1:])
    return np.column_stack(design_columns)


def fixed_effect_rank_residual(values, design, design_pseudoinverse=None):
    values = np.asarray(values, dtype=float)
    if values.size != design.shape[0]:
        raise ValueError("Values and fixed-effect design must have the same row count")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Fixed-effect normalization requires finite values")
    ranked = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    if design_pseudoinverse is None:
        design_pseudoinverse = np.linalg.pinv(design)
    return ranked - design @ (design_pseudoinverse @ ranked)


def two_way_fixed_effect_rank_residual(values, source_ids, target_ids):
    design = two_way_fixed_effect_design(source_ids, target_ids)
    return fixed_effect_rank_residual(values, design)


def pearson_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 3:
        return np.nan
    left = left - left.mean()
    right = right - right.mean()
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    tolerance = np.sqrt(np.finfo(float).eps) * max(1.0, np.sqrt(left.size))
    if left_norm <= tolerance or right_norm <= tolerance:
        return np.nan
    return float(np.dot(left, right) / (left_norm * right_norm))


def two_way_fixed_effect_partial_spearman(
    left, right, source_ids, target_ids
):
    design = two_way_fixed_effect_design(source_ids, target_ids)
    design_pseudoinverse = np.linalg.pinv(design)
    left_residual = fixed_effect_rank_residual(
        left, design, design_pseudoinverse
    )
    right_residual = fixed_effect_rank_residual(
        right, design, design_pseudoinverse
    )
    return pearson_correlation(left_residual, right_residual)


def fixed_effect_rank_z_score(values, source_ids, target_ids):
    residual = two_way_fixed_effect_rank_residual(
        values, source_ids, target_ids
    )
    residual_std = residual.std(ddof=0)
    return residual / residual_std if residual_std > 0.0 else np.full_like(residual, np.nan)


def add_target_normalized_returns(pairs):
    pairs = pairs.copy()
    grouped = pairs.groupby("target_task_idx")["return_mean"]
    target_mean = grouped.transform("mean")
    target_std = grouped.transform(lambda values: values.std(ddof=0))
    pairs["target_z_return"] = (pairs["return_mean"] - target_mean) / target_std.replace(
        0.0, np.nan
    )
    pairs["target_rank_return"] = grouped.rank(method="average", pct=True)
    pairs["source_target_fe_rank_z_return"] = fixed_effect_rank_z_score(
        pairs["return_mean"],
        pairs["source_task_idx"],
        pairs["target_task_idx"],
    )
    return pairs


def add_fixed_effect_normalized_overlaps(merged):
    merged = merged.copy()
    for overlap_metric in (
        "directed_coverage",
        "soft_directed_coverage",
        "symmetric_overlap_same_encoder",
    ):
        merged[f"source_target_fe_rank_z_{overlap_metric}"] = (
            fixed_effect_rank_z_score(
                merged[overlap_metric],
                merged["source_task_idx"],
                merged["target_task_idx"],
            )
        )
    return merged


def load_return_pairs(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Return-pair CSV does not exist: {path}")
    pairs = pd.read_csv(path)
    required = {
        "source_task_idx",
        "source_task",
        "target_task_idx",
        "target_task",
        "episodes",
        "return_mean",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if pairs.duplicated(["source_task_idx", "target_task_idx"]).any():
        raise ValueError("Return-pair CSV contains duplicate task pairs")
    if not np.isfinite(pairs["return_mean"].to_numpy(dtype=float)).all():
        raise ValueError("Return-pair CSV contains non-finite mean returns")

    task_rows = pd.concat(
        [
            pairs[["source_task_idx", "source_task"]].rename(
                columns={"source_task_idx": "task_idx", "source_task": "task"}
            ),
            pairs[["target_task_idx", "target_task"]].rename(
                columns={"target_task_idx": "task_idx", "target_task": "task"}
            ),
        ],
        ignore_index=True,
    )
    if (task_rows.groupby("task_idx")["task"].nunique() != 1).any():
        raise ValueError("Return-pair CSV maps a task index to multiple task names")
    task_lookup = task_rows.drop_duplicates().set_index("task_idx")["task"].to_dict()
    task_ids = sorted(int(task_id) for task_id in task_lookup)
    if task_ids != list(range(1, len(task_ids) + 1)):
        raise ValueError(f"Task indices must be contiguous from 1, found {task_ids}")
    expected_pairs = {
        (source, target)
        for source in task_ids
        for target in task_ids
        if source != target
    }
    actual_pairs = set(
        pairs[["source_task_idx", "target_task_idx"]].itertuples(
            index=False, name=None
        )
    )
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Expected {len(expected_pairs)} off-diagonal return pairs, "
            f"found {len(actual_pairs)}"
        )
    return pairs, [str(task_lookup[task_id]) for task_id in task_ids], path


def load_overlap_pairs(path, task_count):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Waypoint overlap CSV does not exist: {path}")
    frame = pd.read_csv(path)
    required = {
        "source_task_idx",
        "target_task_idx",
        "directed_coverage",
        "soft_directed_coverage",
        "symmetric_overlap_same_encoder",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(["source_task_idx", "target_task_idx"]).any():
        raise ValueError("Waypoint overlap CSV contains duplicate task pairs")
    expected_pairs = task_count * (task_count - 1)
    if len(frame) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} off-diagonal overlap pairs, found {len(frame)}"
        )
    return frame


def qap_test(frame, overlap_frame, overlap_metric, outcome, task_ids, permutations, seed):
    available = frame.dropna(subset=[overlap_metric, outcome]).copy()
    observed = spearman_correlation(available[overlap_metric], available[outcome])
    fixed_effect_design = two_way_fixed_effect_design(
        available["source_task_idx"],
        available["target_task_idx"],
    )
    design_pseudoinverse = np.linalg.pinv(fixed_effect_design)
    outcome_fixed_effect_residual = fixed_effect_rank_residual(
        available[outcome], fixed_effect_design, design_pseudoinverse
    )
    overlap_fixed_effect_residual = fixed_effect_rank_residual(
        available[overlap_metric], fixed_effect_design, design_pseudoinverse
    )
    observed_fixed_effect = pearson_correlation(
        overlap_fixed_effect_residual, outcome_fixed_effect_residual
    )
    result = {
        "n_pairs": int(len(available)),
        "spearman_rho": observed,
        "qap_permutations": permutations,
        "qap_p_greater": np.nan,
        "qap_p_two_sided": np.nan,
        "two_way_fe_partial_spearman_rho": observed_fixed_effect,
        "two_way_fe_qap_p_greater": np.nan,
        "two_way_fe_qap_p_two_sided": np.nan,
    }
    if permutations == 0 or not (
        np.isfinite(observed) or np.isfinite(observed_fixed_effect)
    ):
        return result

    task_to_position = {task_id: position for position, task_id in enumerate(task_ids)}
    matrix = np.full((len(task_ids), len(task_ids)), np.nan, dtype=float)
    for source, target, value in overlap_frame[
        ["source_task_idx", "target_task_idx", overlap_metric]
    ].itertuples(index=False, name=None):
        matrix[task_to_position[int(source)], task_to_position[int(target)]] = float(value)
    source_positions = available["source_task_idx"].map(task_to_position).to_numpy()
    target_positions = available["target_task_idx"].map(task_to_position).to_numpy()
    outcome_values = available[outcome].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    null_values = []
    fixed_effect_null_values = []
    for _ in range(permutations):
        permutation = rng.permutation(len(task_ids))
        permuted_overlap = matrix[
            permutation[source_positions], permutation[target_positions]
        ]
        rho = spearman_correlation(permuted_overlap, outcome_values)
        if np.isfinite(rho):
            null_values.append(rho)
        permuted_fixed_effect_residual = fixed_effect_rank_residual(
            permuted_overlap,
            fixed_effect_design,
            design_pseudoinverse,
        )
        fixed_effect_rho = pearson_correlation(
            permuted_fixed_effect_residual,
            outcome_fixed_effect_residual,
        )
        if np.isfinite(fixed_effect_rho):
            fixed_effect_null_values.append(fixed_effect_rho)
    null_values = np.asarray(null_values, dtype=float)
    fixed_effect_null_values = np.asarray(fixed_effect_null_values, dtype=float)
    result["valid_qap_permutations"] = int(null_values.size)
    result["valid_two_way_fe_qap_permutations"] = int(
        fixed_effect_null_values.size
    )
    if np.isfinite(observed) and null_values.size:
        result.update(
            {
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
    if np.isfinite(observed_fixed_effect) and fixed_effect_null_values.size:
        result.update(
            {
                "two_way_fe_qap_p_greater": float(
                    (
                        1
                        + np.count_nonzero(
                            fixed_effect_null_values >= observed_fixed_effect
                        )
                    )
                    / (1 + fixed_effect_null_values.size)
                ),
                "two_way_fe_qap_p_two_sided": float(
                    (
                        1
                        + np.count_nonzero(
                            np.abs(fixed_effect_null_values)
                            >= abs(observed_fixed_effect)
                        )
                    )
                    / (1 + fixed_effect_null_values.size)
                ),
            }
        )
    return result


def compute_statistics(merged, overlap_frame, task_ids, args):
    subsets = {
        "all_other": merged,
        "forward_only": merged[merged["source_task_idx"] < merged["target_task_idx"]],
        "backward_only": merged[merged["source_task_idx"] > merged["target_task_idx"]],
        "adjacent_forward": merged[
            merged["target_task_idx"] == merged["source_task_idx"] + 1
        ],
    }
    statistics = {}
    for subset_name, subset in subsets.items():
        statistics[subset_name] = {}
        for outcome in ("return_mean", "target_z_return", "target_rank_return"):
            statistics[subset_name][outcome] = {}
            for overlap_metric in (
                "directed_coverage",
                "soft_directed_coverage",
                "symmetric_overlap_same_encoder",
            ):
                statistics[subset_name][outcome][overlap_metric] = qap_test(
                    subset,
                    overlap_frame,
                    overlap_metric,
                    outcome,
                    task_ids,
                    args.permutations,
                    args.permutation_seed,
                )
    return statistics


def clean_task_name(name):
    return re.sub(r"-v\d+$", "", str(name))


def plot_return_matrix(
    pairs,
    task_names,
    output_path,
    value_column,
    colorbar_label,
    title,
    value_format,
    color_map,
    center=None,
):
    task_ids = list(range(1, len(task_names) + 1))
    matrix = pairs.pivot(
        index="source_task_idx", columns="target_task_idx", values=value_column
    ).reindex(index=task_ids, columns=task_ids)
    labels = [
        f"{task_id}. {clean_task_name(task_names[task_id - 1])}"
        for task_id in task_ids
    ]
    matrix.index = labels
    matrix.columns = labels
    figure, axis = plt.subplots(
        figsize=(max(11.0, len(task_ids) * 1.1), max(7.0, len(task_ids) * 0.75)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        mask=matrix.isna(),
        annot=True,
        fmt=value_format,
        cmap=color_map,
        center=center,
        linewidths=0.5,
        cbar_kws={"label": colorbar_label},
        ax=axis,
    )
    axis.set_xlabel("Evaluation task")
    axis.set_ylabel("Student's training task")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def statistic_title(statistic, fixed_effect=False):
    if fixed_effect:
        rho = statistic["two_way_fe_partial_spearman_rho"]
        p_value = statistic["two_way_fe_qap_p_greater"]
        prefix = "partial rho"
    else:
        rho = statistic["spearman_rho"]
        p_value = statistic["qap_p_greater"]
        prefix = "rho"
    rho_text = "nan" if not np.isfinite(rho) else f"{rho:.3f}"
    p_text = "nan" if not np.isfinite(p_value) else f"{p_value:.4f}"
    return f"{prefix}={rho_text}, QAP p={p_text}, n={statistic['n_pairs']}"


def plot_correlations(merged, statistics, task_names, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.4), constrained_layout=True)
    colors = plt.get_cmap("tab10", len(task_names))
    for axis, x_column, outcome, label, statistic_outcome, fixed_effect in (
        (
            axes[0],
            "directed_coverage",
            "return_mean",
            "Mean episodic return",
            "return_mean",
            False,
        ),
        (
            axes[1],
            "directed_coverage",
            "target_z_return",
            "Target-wise z-scored return",
            "target_z_return",
            False,
        ),
        (
            axes[2],
            "source_target_fe_rank_z_directed_coverage",
            "source_target_fe_rank_z_return",
            "Source + target FE return rank residual (z)",
            "return_mean",
            True,
        ),
    ):
        for source_task_idx in range(1, len(task_names) + 1):
            subset = merged[merged["source_task_idx"] == source_task_idx]
            axis.scatter(
                subset[x_column],
                subset[outcome],
                s=38,
                alpha=0.78,
                color=colors(source_task_idx - 1),
                edgecolor="white",
                linewidth=0.5,
                label=f"{source_task_idx}. {clean_task_name(task_names[source_task_idx - 1])}",
            )
        statistic = statistics["all_other"][statistic_outcome]["directed_coverage"]
        axis.set_title(statistic_title(statistic, fixed_effect=fixed_effect))
        axis.set_xlabel(
            "Directed target waypoint coverage"
            if not fixed_effect
            else "Source + target FE coverage rank residual (z)"
        )
        axis.set_ylabel(label)
        axis.grid(color="#d9d9d9", linewidth=0.6, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=8,
    )
    figure.suptitle("Waypoint overlap vs. student cross-task return", fontsize=14)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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

    if args.return_pairs:
        pairs, task_names, return_pairs_source = load_return_pairs(args.return_pairs)
        episodes = None
        model_dir = (
            Path(args.model_dir).expanduser().resolve()
            if args.model_dir
            else Path(args.run_dir).expanduser().resolve() / "model"
        )
        config_path = (
            Path(args.config).expanduser().resolve() if args.config else None
        )
    else:
        pairs, episodes, task_names, model_dir, config_path = evaluate_students(
            args, device
        )
        return_pairs_source = None
    pairs = add_target_normalized_returns(pairs)
    overlap_frame = load_overlap_pairs(args.overlap_pairs, len(task_names))
    merged = pairs.merge(
        overlap_frame,
        on=["source_task_idx", "target_task_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_overlap"),
    )
    if merged["directed_coverage"].isna().any():
        missing = merged.loc[
            merged["directed_coverage"].isna(),
            ["source_task_idx", "target_task_idx"],
        ].values.tolist()
        raise ValueError(f"Missing waypoint overlap for task pairs {missing}")
    merged = add_fixed_effect_normalized_overlaps(merged)

    task_ids = list(range(1, len(task_names) + 1))
    statistics = compute_statistics(merged, overlap_frame, task_ids, args)
    episode_counts = pairs["episodes"].dropna().astype(int).unique()
    if episode_counts.size != 1:
        raise ValueError("All return pairs must use the same episode count")
    episodes_per_pair = int(episode_counts[0])
    pairs.to_csv(f"{output_prefix}_return_pairs.csv", index=False)
    if episodes is not None:
        episodes.to_csv(f"{output_prefix}_return_episodes.csv", index=False)
    merged.to_csv(f"{output_prefix}_return_overlap_pairs.csv", index=False)
    plot_return_matrix(
        pairs,
        task_names,
        f"{output_prefix}_return_matrix.png",
        "return_mean",
        "Mean episodic return",
        f"Student cross-task return ({episodes_per_pair} episodes per pair)",
        ".0f",
        "viridis",
    )
    plot_return_matrix(
        pairs,
        task_names,
        f"{output_prefix}_normalized_return_matrix.png",
        "source_target_fe_rank_z_return",
        "Source + target FE return rank residual (z)",
        "Source- and target-normalized student cross-task return",
        ".2f",
        "vlag",
        center=0.0,
    )
    plot_correlations(
        merged,
        statistics,
        task_names,
        f"{output_prefix}_return_overlap_scatter.png",
    )

    report = {
        "metric_interpretation": (
            "Direct zero-shot student cross-task episodic return; this is not "
            "learning-curve forward transfer."
        ),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "model_dir": str(model_dir),
        "config": str(config_path) if config_path is not None else None,
        "overlap_pairs": str(Path(args.overlap_pairs).expanduser().resolve()),
        "reused_return_pairs": (
            str(return_pairs_source) if return_pairs_source is not None else None
        ),
        "task_count": len(task_names),
        "cross_task_pair_count": len(pairs),
        "episodes_per_pair": episodes_per_pair,
        "eval_seed": args.eval_seed,
        "reseed_each_episode": bool(args.reseed_each_episode),
        "sample_action": bool(args.sample_action),
        "normalization": {
            "target_z_return": "z-score across source students within each target",
            "target_rank_return": "percentile rank across source students within each target",
            "source_target_fe_rank_z_return": (
                "global return ranks residualized on categorical source and target "
                "fixed effects, then standardized"
            ),
            "partial_correlation": (
                "Pearson correlation between overlap-rank and return-rank residuals "
                "after the same source and target fixed effects"
            ),
        },
        "statistics": statistics,
    }
    with open(f"{output_prefix}_stats.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, indent=2)

    primary = statistics["all_other"]["return_mean"]["directed_coverage"]
    normalized = statistics["all_other"]["target_z_return"]["directed_coverage"]
    fixed_effect = statistics["all_other"]["return_mean"]["directed_coverage"]
    forward_fixed_effect = statistics["forward_only"]["return_mean"][
        "directed_coverage"
    ]
    print(
        "directed coverage vs raw cross-task return: "
        f"n={primary['n_pairs']} rho={primary['spearman_rho']:.4f} "
        f"QAP p(one-sided)={primary['qap_p_greater']:.4f}"
    )
    print(
        "directed coverage vs target-z cross-task return: "
        f"n={normalized['n_pairs']} rho={normalized['spearman_rho']:.4f} "
        f"QAP p(one-sided)={normalized['qap_p_greater']:.4f}"
    )
    print(
        "directed coverage vs return, source + target FE partial Spearman: "
        f"n={fixed_effect['n_pairs']} "
        f"rho={fixed_effect['two_way_fe_partial_spearman_rho']:.4f} "
        f"QAP p(one-sided)={fixed_effect['two_way_fe_qap_p_greater']:.4f}"
    )
    print(
        "forward-only source + target FE partial Spearman: "
        f"n={forward_fixed_effect['n_pairs']} "
        f"rho={forward_fixed_effect['two_way_fe_partial_spearman_rho']:.4f} "
        f"QAP p(one-sided)={forward_fixed_effect['two_way_fe_qap_p_greater']:.4f}"
    )
    print(f"saved return pairs: {output_prefix}_return_pairs.csv")


if __name__ == "__main__":
    main()