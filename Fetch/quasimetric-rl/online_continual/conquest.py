"""Continual QRL with configurable students and a quasimetric meta teacher.

This entrypoint keeps the online QRL and structural meta learner from
``online_continual.main``, while adopting FAME's task-boundary policy
selection and transient policy regularization.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

if __package__:
    from . import main as base
else:
    import main as base


DEFAULT_SAVE_PATH = os.path.join(base.DEFAULT_SAVE_PATH, "cqrl")


def base_normal_distribution(distribution):
    current = distribution
    for _ in range(8):
        if isinstance(current, Normal):
            return current
        if hasattr(current, "_dist"):
            current = current._dist
        elif hasattr(current, "base_dist"):
            current = current.base_dist
        else:
            break
    raise TypeError(
        f"Cannot find a Normal base distribution in {type(distribution).__name__}."
    )


class CQRLFastAgent(base.OnlineQRLStudentAgent):
    """Online QRL learner with optional transient meta-policy guidance."""

    def _optimize_actor_loss(self, actor_loss):
        actor_loss_module = self.qrl_losses.actor_loss
        with actor_loss_module.actor_optim.update_context(optimize=True):
            actor_loss.backward()
        actor_loss_module.actor_sched.step()

    def update(
        self,
        replay_buffer,
        step,
        teacher_agent=None,
        regularization_weight=0.0,
        goal_buffer=None,
        distill_loss_type="kl",
        distill_goal_source="current_replay",
    ):
        metrics = super().update(replay_buffer, step)
        regularization_loss = torch.zeros((), device=self.device)

        if teacher_agent is not None and regularization_weight > 0.0:
            if distill_goal_source == "current_replay":
                batch = self._sample_batch(replay_buffer)
                obs = batch["obses"]
                goal_obs = batch["goals"]
            elif distill_goal_source == "quasimetric_buffer":
                obs, _, _, _, _, _ = replay_buffer.sample(self.batch_size)
                obs = torch.as_tensor(obs, device=self.device).float()

                goal_obs = None
                if goal_buffer is not None and hasattr(
                    goal_buffer, "sample_behavior_goal"
                ):
                    goal_obs = goal_buffer.sample_behavior_goal(
                        discount=getattr(
                            teacher_agent, "goal_discount", self.goal_discount
                        ),
                        success_only=getattr(
                            teacher_agent,
                            "behavior_goal_success_only",
                            True,
                        ),
                        batch_size=self.batch_size,
                    )
                if goal_obs is None:
                    goal_obs = getattr(teacher_agent, "behavior_goal", None)
            else:
                raise ValueError(
                    f"Unsupported distillation goal source: {distill_goal_source}"
                )

            student_dist = self.actor_distribution(
                obs,
                goal_obs=goal_obs,
                detach_goal=True,
            )
            with torch.no_grad():
                teacher_dist = base.actor_distribution(
                    teacher_agent,
                    obs,
                    goal_obs=goal_obs,
                    detach_goal=True,
                )

            student_normal = base_normal_distribution(student_dist)
            teacher_normal = base_normal_distribution(teacher_dist)
            if distill_loss_type == "kl":
                regularization_loss = (
                    kl_divergence(
                        student_normal,
                        teacher_normal,
                    )
                    .sum(-1)
                    .mean()
                )
            elif distill_loss_type == "wd":
                regularization_loss = (
                    torch.square(student_normal.loc - teacher_normal.loc).sum(-1)
                    + torch.square(student_normal.scale - teacher_normal.scale).sum(-1)
                ).mean()
            else:
                raise ValueError(
                    f"Unsupported distillation loss type: {distill_loss_type}"
                )
            self._optimize_actor_loss(
                float(regularization_weight) * regularization_loss
            )

        loss_value = float(regularization_loss.item())
        metrics["cqrl_distill_loss"] = loss_value
        metrics["cqrl_kl"] = loss_value if distill_loss_type == "kl" else 0.0
        metrics["cqrl_wd"] = loss_value if distill_loss_type == "wd" else 0.0
        metrics["cqrl_distill_is_wd"] = float(distill_loss_type == "wd")
        metrics["cqrl_goal_source_is_buffer"] = float(
            distill_goal_source == "quasimetric_buffer"
        )
        metrics["cqrl_regularized"] = float(
            teacher_agent is not None and regularization_weight > 0.0
        )
        return metrics


def build_fast_agent(observation_space, action_space, device, args, total_optim_steps):
    return CQRLFastAgent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        batch_size=args.batch_size,
        total_optim_steps=total_optim_steps,
        args=args,
    )


def student_slot(task_number, student_mode):
    if task_number < 1:
        raise ValueError("task_number must be positive")
    if student_mode == "shared":
        return 0
    if student_mode == "independent":
        return task_number - 1
    raise ValueError(f"Unsupported student mode: {student_mode}")


def copy_replay_buffer(source_buffer, target_buffer):
    indices = base.chronological_indices(source_buffer)
    for source_idx in indices:
        target_buffer.add(
            source_buffer.obses[source_idx],
            source_buffer.actions[source_idx],
            source_buffer.rewards[source_idx],
            source_buffer.successes[source_idx],
            source_buffer.next_obses[source_idx],
            not bool(source_buffer.not_dones[source_idx, 0]),
            not bool(source_buffer.not_dones_no_max[source_idx, 0]),
        )
    return int(indices.size)


def returns_are_better(candidate_returns, reference_returns, use_ttest):
    candidate_returns = np.asarray(candidate_returns, dtype=np.float64)
    reference_returns = np.asarray(reference_returns, dtype=np.float64)
    candidate_mean = float(np.mean(candidate_returns))
    reference_mean = float(np.mean(reference_returns))
    if not use_ttest or candidate_returns.size < 2 or reference_returns.size < 2:
        return candidate_mean > reference_mean

    from scipy import stats

    result = stats.ttest_ind(
        candidate_returns,
        reference_returns,
        alternative="greater",
        equal_var=False,
    )
    return bool(np.isfinite(result.pvalue) and result.pvalue < 0.05)


def detect_initialization(
    args,
    eval_env,
    task_name,
    fast_agent,
    meta_agent,
    random_agent,
    meta_ready,
    writer,
    step,
):
    forced_selection = args.task_switch_selection
    if forced_selection != "auto":
        if forced_selection == "meta" and not meta_ready:
            raise RuntimeError(
                "--task_switch_selection meta requires an updated meta actor. "
                "Increase stored trajectories or meta updates so the quasimetric buffer "
                "reaches --qm_min_buffer_size."
            )
        selection_id = {"random": 0, "fast": 1, "meta": 2}[forced_selection]
        base.log_scalar(writer, "detection/selection", selection_id, step)
        base.log_scalar(writer, "detection/forced", 1, step)
        print(f"CQRL task-switch selection forced: {forced_selection}")
        return forced_selection, {}

    eval_env.set_task(task_name)
    candidates = {"fast": fast_agent, "random": random_agent}
    if meta_ready:
        candidates["meta"] = meta_agent

    returns = {}
    for name, candidate in candidates.items():
        eval_results = eval_env.evaluate_agent(candidate, args.detection_episodes)
        candidate_returns = np.asarray(
            eval_results["episodic_returns"],
            dtype=np.float64,
        )
        returns[name] = candidate_returns
        mean_return = float(np.mean(candidate_returns))
        base.log_scalar(writer, f"detection/{name}_return", mean_return, step)
        print(f"CQRL detection {name}: return {mean_return:.3f}")

    fast_mean = float(np.mean(returns["fast"]))
    random_mean = float(np.mean(returns["random"]))
    if "meta" not in returns:
        selection = "fast" if fast_mean > random_mean else "random"
    else:
        meta_mean = float(np.mean(returns["meta"]))
        meta_is_better = returns_are_better(
            returns["meta"],
            returns["fast"],
            args.use_ttest,
        )
        fast_is_better = returns_are_better(
            returns["fast"],
            returns["meta"],
            args.use_ttest,
        )
        if meta_is_better and meta_mean > random_mean:
            selection = "meta"
        elif fast_is_better and fast_mean > random_mean:
            selection = "fast"
        else:
            selection = "random"

    selection_id = {"random": 0, "fast": 1, "meta": 2}[selection]
    base.log_scalar(writer, "detection/selection", selection_id, step)
    base.log_scalar(writer, "detection/forced", 0, step)
    print(f"CQRL detection selected: {selection}")
    return selection, {
        name: float(np.mean(candidate_returns))
        for name, candidate_returns in returns.items()
    }


def update_meta_from_recent(
    args,
    meta_agent,
    quasimetric_buffer,
    recent_buffer,
    completed_task_count,
    writer,
    step,
):
    if args.meta_update_steps == 0 or len(quasimetric_buffer) < args.qm_min_buffer_size:
        return {}, {}, 0

    structure_metrics = {}
    actor_metrics = {}
    recent_updates = 0
    recent_is_ready = len(recent_buffer) >= args.qm_min_buffer_size
    print(
        "CQRL meta integration:",
        f"memory={len(quasimetric_buffer)}",
        f"recent={len(recent_buffer)}",
        f"updates={args.meta_update_steps}",
    )

    for update_idx in range(args.meta_update_steps):
        use_recent = (
            recent_is_ready
            and completed_task_count > 1
            and update_idx % completed_task_count == 0
        )
        update_buffer = recent_buffer if use_recent else quasimetric_buffer
        structure_metrics, actor_metrics = meta_agent.update_structure_and_actor(
            update_buffer
        )
        recent_updates += int(use_recent)

    base.log_metric_dict(writer, "metric_structure", structure_metrics, step)
    base.log_metric_dict(writer, "meta_actor", actor_metrics, step)
    base.log_scalar(writer, "meta/recent_updates", recent_updates, step)
    return structure_metrics, actor_metrics, recent_updates


def load_resume_config(resume_run_dir):
    if resume_run_dir is None:
        return {}

    config_path = os.path.join(os.path.abspath(resume_run_dir), "run_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Resume config not found: {config_path}")
    with open(config_path, "r") as handle:
        config = json.load(handle)
    for key in (
        "start_task",
        "resume_run_dir",
        "resume_model_name",
        "resume_buffer_path",
        "update_meta_on_resume",
    ):
        config.pop(key, None)
    return config


def set_env_task_position(env, task_position, change_freq):
    env.task_counter = task_position
    env.timestep_counter = (task_position - 1) * int(change_freq)
    env.set_task(env.env_list[task_position - 1])


def resolve_resume_checkpoint(args):
    source_run_name = os.path.basename(os.path.normpath(args.resume_run_dir))
    model_name = (
        args.resume_model_name or f"{source_run_name}_task{args.start_task - 1}"
    )
    return os.path.join(args.resume_run_dir, "model"), model_name


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_env_state(env):
    return {
        "task_counter": env.task_counter,
        "timestep_counter": env.timestep_counter,
        "base_task_name": env.base_task_name,
        "obs_mean": None if env.obs_mean is None else np.array(env.obs_mean, copy=True),
        "obs_var": None if env.obs_var is None else np.array(env.obs_var, copy=True),
        "obs_count": env.obs_count,
    }


def restore_env_state(env, state, expected_task):
    if not state:
        return
    if int(state["task_counter"]) != expected_task:
        raise ValueError(
            "Resume state targets task "
            f"{state['task_counter']}, but --start_task is {expected_task}."
        )
    if state["base_task_name"] != env.base_task_name:
        raise ValueError(
            "Resume task name does not match the configured sequence: "
            f"{state['base_task_name']} != {env.base_task_name}."
        )
    env.timestep_counter = int(state["timestep_counter"])
    env.obs_mean = (
        None if state["obs_mean"] is None else np.array(state["obs_mean"], copy=True)
    )
    env.obs_var = (
        None if state["obs_var"] is None else np.array(state["obs_var"], copy=True)
    )
    env.obs_count = float(state["obs_count"])


def save_resume_state(
    model_dir,
    model_name,
    env,
    completed_task_count,
    step,
    meta_ready,
    count_success,
    buffer_path,
):
    resume_path = os.path.join(model_dir, f"{model_name}_resume.pt")
    temporary_path = f"{resume_path}.tmp"
    payload = {
        "version": 1,
        "completed_task": completed_task_count,
        "next_task": env.task_counter,
        "step": step,
        "meta_ready": meta_ready,
        "count_success": count_success,
        "buffer_path": os.path.abspath(buffer_path),
        "env": capture_env_state(env),
        "rng": capture_rng_state(),
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, resume_path)
    print("saved resume state:", resume_path)


def load_resume_training_state(
    args,
    env,
    fast_agent,
    meta_agent,
    quasimetric_buffer,
):
    model_dir, model_name = resolve_resume_checkpoint(args)
    fast_model_name = f"{model_name}_fast"
    meta_model_name = f"{model_name}_meta"
    fast_path = os.path.join(model_dir, f"{fast_model_name}_online_qrl.pt")
    required_paths = [
        ("Fast", fast_path),
        ("Meta actor", os.path.join(model_dir, f"{meta_model_name}_actor.pt")),
        ("Meta critic", os.path.join(model_dir, f"{meta_model_name}_critic.pt")),
        (
            "Meta target critic",
            os.path.join(model_dir, f"{meta_model_name}_critic_target.pt"),
        ),
        (
            "Meta quasimetric",
            os.path.join(model_dir, f"{meta_model_name}_quasimetric.pt"),
        ),
    ]
    for label, path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    resume_path = os.path.join(model_dir, f"{model_name}_resume.pt")
    resume_state = {}
    if os.path.isfile(resume_path):
        resume_state = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(resume_state["completed_task"]) != args.start_task - 1:
            raise ValueError(
                f"Resume checkpoint completed task {resume_state['completed_task']}, "
                f"but --start_task {args.start_task} requires task {args.start_task - 1}."
            )
        if int(resume_state["next_task"]) != args.start_task:
            raise ValueError(
                f"Resume checkpoint targets task {resume_state['next_task']}, "
                f"but --start_task is {args.start_task}."
            )
    else:
        print(
            "resume state marker not found; using inferred step and freshly seeded "
            "environment state for this legacy checkpoint"
        )

    fast_agent.load(model_dir, fast_model_name)
    meta_agent.load(model_dir, meta_model_name)
    print("loaded fast checkpoint:", fast_path)
    print("loaded meta checkpoint:", meta_model_name)

    source_save_path = os.path.dirname(os.path.normpath(args.resume_run_dir))
    source_log_name = os.path.basename(os.path.normpath(args.resume_run_dir))
    inferred_buffer_path = base.quasimetric_buffer_path(
        source_save_path,
        source_log_name,
        env,
        args.start_task - 1,
    )
    if args.resume_buffer_path is not None:
        buffer_candidates = [args.resume_buffer_path]
    else:
        buffer_candidates = [
            resume_state.get("buffer_path"),
            inferred_buffer_path,
        ]
    buffer_path = next(
        (
            os.path.abspath(path)
            for path in buffer_candidates
            if path is not None
            and os.path.isfile(os.path.join(os.path.abspath(path), "offline_data.npz"))
        ),
        None,
    )
    if buffer_path is None or not quasimetric_buffer.load_data(buffer_path):
        raise FileNotFoundError(
            "Unable to load the accumulated quasimetric buffer. Checked: "
            + ", ".join(str(path) for path in buffer_candidates if path is not None)
        )
    print("loaded quasimetric buffer:", buffer_path)

    restore_env_state(env, resume_state.get("env"), args.start_task)
    restore_rng_state(resume_state.get("rng"))
    return {
        "step": int(resume_state.get("step", (args.start_task - 1) * args.change_freq)),
        "meta_ready": bool(
            resume_state.get(
                "meta_ready",
                args.meta_update_steps > 0
                and len(quasimetric_buffer) >= args.qm_min_buffer_size,
            )
        ),
        "count_success": int(resume_state.get("count_success", -1)),
    }


def parse_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--resume_run_dir", type=base.str2none, default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=(
            "Run continual QRL with configurable students and a quasimetric meta teacher "
            "on Fetch task streams"
        )
    )
    base.add_common_args(parser)
    base.add_sac_args(parser)
    base.add_student_qrl_args(parser)
    base.add_quasimetric_args(parser)

    parser.add_argument(
        "--quasimetric_buffer_capacity",
        "--meta_buffer_capacity",
        dest="quasimetric_buffer_capacity",
        type=int,
        default=1000000,
    )
    parser.add_argument("--meta_update_steps", type=int, default=10000)
    parser.add_argument("--detection_episodes", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=50000)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument(
        "--student_mode",
        type=str,
        default="shared",
        choices=["shared", "independent"],
        help=(
            "Use one student across all tasks or initialize an independent "
            "student for each task."
        ),
    )
    parser.add_argument(
        "--distill_loss_type",
        type=str,
        default="kl",
        choices=["kl", "wd"],
        help=(
            "Policy distillation loss used during meta-guided warmup: KL divergence "
            "or squared 2-Wasserstein distance between diagonal Gaussians."
        ),
    )
    parser.add_argument(
        "--distill_goal_source",
        type=str,
        default="current_replay",
        choices=["current_replay", "quasimetric_buffer"],
        help=(
            "Goal source used during meta-guided warmup: episode-paired future "
            "goals from the current task replay or behavior goals from the "
            "accumulated quasimetric buffer."
        ),
    )
    parser.add_argument(
        "--start_task",
        type=int,
        default=1,
        help="1-based task position in the full sequence at which training starts.",
    )
    parser.add_argument(
        "--resume_run_dir",
        type=base.str2none,
        default=None,
        help="Previous run directory containing run_config.json and task checkpoints.",
    )
    parser.add_argument(
        "--resume_model_name",
        type=base.str2none,
        default=None,
        help=(
            "Task checkpoint prefix without _fast/_meta suffixes; defaults to "
            "the checkpoint immediately before start_task."
        ),
    )
    parser.add_argument(
        "--resume_buffer_path",
        type=base.str2none,
        default=None,
        help="Accumulated quasimetric buffer directory; inferred from resume_run_dir by default.",
    )
    parser.add_argument(
        "--update_meta_on_resume",
        action="store_true",
        help=(
            "Update the restored meta agent from the accumulated quasimetric "
            "buffer before task-switch policy selection."
        ),
    )
    parser.add_argument(
        "--task_switch_selection",
        type=str,
        default="auto",
        choices=["auto", "meta", "fast", "random"],
        help=(
            "Policy used at every task boundary. 'auto' compares candidates; "
            "'meta' forces transient meta-actor distillation guidance."
        ),
    )

    method_action = next(
        action for action in parser._actions if action.dest == "method"
    )
    method_action.choices = ["cqrl"]
    method_action.default = "cqrl"
    parser.set_defaults(
        save_path=DEFAULT_SAVE_PATH,
        store_traj_num=20,
        wandb_project_name="cqrl-fetch",
    )
    parser.set_defaults(**load_resume_config(bootstrap_args.resume_run_dir))

    args = parser.parse_args()
    try:
        args.log_backends = base.normalize_log_backends(args.log_backends)
    except ValueError as error:
        parser.error(str(error))

    if args.random_steps <= 0:
        parser.error("--random_steps must be positive")
    if args.store_traj_num <= 0:
        parser.error("--store_traj_num must be positive")
    if args.quasimetric_buffer_capacity <= 0:
        parser.error("--quasimetric_buffer_capacity must be positive")
    if args.meta_update_steps < 0:
        parser.error("--meta_update_steps cannot be negative")
    if args.detection_episodes <= 0:
        parser.error("--detection_episodes must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps cannot be negative")
    if args.lambda_reg < 0.0:
        parser.error("--lambda_reg cannot be negative")
    if args.start_task < 1:
        parser.error("--start_task must be positive")
    if args.start_task > 1 and args.resume_run_dir is None:
        parser.error("--resume_run_dir is required when --start_task is greater than 1")
    if args.start_task == 1 and args.resume_run_dir is not None:
        parser.error("--start_task must be greater than 1 when --resume_run_dir is set")
    if args.resume_run_dir is None and (
        args.resume_model_name is not None or args.resume_buffer_path is not None
    ):
        parser.error(
            "--resume_model_name and --resume_buffer_path require --resume_run_dir"
        )
    if args.update_meta_on_resume and args.resume_run_dir is None:
        parser.error("--update_meta_on_resume requires --resume_run_dir")
    if args.task_switch_selection == "meta" and args.meta_update_steps == 0:
        parser.error(
            "--task_switch_selection meta requires --meta_update_steps to be positive"
        )
    if args.resume_run_dir is not None:
        args.resume_run_dir = os.path.abspath(args.resume_run_dir)
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.save_path, exist_ok=True)
    base.set_seed_everywhere(args.seed)

    env_kwargs = base.make_env_kwargs(args)
    env = base.FetchGoalEnvSequence(**env_kwargs)
    eval_env = base.FetchGoalEnvSequence(**env_kwargs)
    print("env_list:", env.env_list)
    if not 1 <= args.start_task <= len(env.env_list):
        raise ValueError(
            f"start_task must be between 1 and {len(env.env_list)}, got {args.start_task}"
        )
    if args.start_task > 1:
        set_env_task_position(env, args.start_task, args.change_freq)
        eval_env.set_task(env.env_list[args.start_task - 1])
        print("starting task:", args.start_task, env.base_task_name)

    num_steps_per_run = len(env.env_list) * args.change_freq
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()

    order_name = (
        args.task_order
        or env_kwargs["env_sequence"]
        or env_kwargs["base_task_name"]
        or args.env
    )
    task_tag = base.safe_tag(str(order_name).replace(",", "-"))
    log_name = f"fetch_cqrl_{task_tag}_seed{args.seed}_gc-{args.gc_reward_type}"
    if args.slide_goal_scale != 1.0:
        log_name += f"_slide-scale{base.safe_tag(args.slide_goal_scale)}"
    if args.start_task > 1:
        log_name += f"_from-task{args.start_task}"

    run_dir = os.path.join(args.save_path, log_name)
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    writer, log_info = base.create_experiment_logger(args, log_name, run_dir)

    if log_info["active_backends"]:
        print("log_backends:", ", ".join(log_info["active_backends"]))
    else:
        print("log_backends: disabled")
    for key in ("tensorboard_path", "wandb_path", "wandb_url"):
        if log_info[key] is not None:
            print(f"{key}:", log_info[key])
    print("run_dir:", run_dir)

    obs_space = base.vector_observation_space(env.env.observation_space)
    obs_dim = obs_space.shape[0]
    action_space = env.env.action_space
    action_shape = action_space.shape
    replay_buffer_capacity = int(args.change_freq) + int(args.random_steps)
    replay_buffer_cls = (
        base.ReplayBufferMetric
        if args.replay_buffer_mode == "her"
        else base.ReplayBufferMetricNoHER
    )
    replay_buffer = replay_buffer_cls(
        obs_space.shape,
        action_shape,
        replay_buffer_capacity,
        device,
    )
    recent_buffer = replay_buffer_cls(
        obs_space.shape,
        action_shape,
        replay_buffer_capacity,
        device,
    )
    quasimetric_buffer = replay_buffer_cls(
        obs_space.shape,
        action_shape,
        max(
            args.quasimetric_buffer_capacity,
            args.batch_size,
            args.qm_min_buffer_size,
        ),
        device,
    )

    student_count = 1 if args.student_mode == "shared" else len(env.env_list)
    student_agents = [
        build_fast_agent(
            obs_space,
            action_space,
            device,
            args,
            num_steps_per_run,
        )
        for _ in range(student_count)
    ]
    active_student_idx = student_slot(args.start_task, args.student_mode)
    fast_agent = student_agents[active_student_idx]
    meta_agent = base.build_meta_agent(obs_dim, action_shape[0], device, args)
    collector = base.Collector(env, replay_buffer)
    if args.start_task == 1:
        env.reset()

    base.log_scalar(writer, "config/obs_dim", obs_dim, 0)
    base.log_scalar(writer, "config/action_dim", action_shape[0], 0)
    base.log_scalar(writer, "config/task_count", len(env.env_list), 0)
    base.log_scalar(
        writer,
        "config/task_specific_students",
        int(args.student_mode == "independent"),
        0,
    )
    base.log_scalar(
        writer,
        "config/student_mode",
        {"shared": 0, "independent": 1}[args.student_mode],
        0,
    )
    base.log_scalar(writer, "config/lambda_reg", args.lambda_reg, 0)
    base.log_scalar(
        writer,
        "config/distill_loss_type",
        {"kl": 0, "wd": 1}[args.distill_loss_type],
        0,
    )
    base.log_scalar(
        writer,
        "config/distill_goal_source",
        {"current_replay": 0, "quasimetric_buffer": 1}[args.distill_goal_source],
        0,
    )
    base.log_scalar(writer, "config/warmup_steps", args.warmup_steps, 0)
    base.log_scalar(
        writer,
        "config/task_switch_selection",
        {"auto": 0, "random": 1, "fast": 2, "meta": 3}[args.task_switch_selection],
        0,
    )

    intermediate_stats = defaultdict(list)
    task_counter = env.task_counter
    task_start_step = 0
    selection = "fast"
    detection_scores = {}
    meta_ready = False
    count_success = -1
    step = 0

    if args.start_task > 1:
        resume_student_idx = student_slot(
            args.start_task - 1,
            args.student_mode,
        )
        resume_agent = student_agents[resume_student_idx]
        resume_state = load_resume_training_state(
            args,
            env,
            resume_agent,
            meta_agent,
            quasimetric_buffer,
        )
        step = resume_state["step"]
        task_start_step = step
        meta_ready = resume_state["meta_ready"]
        count_success = resume_state["count_success"]
        if args.update_meta_on_resume:
            _, actor_metrics, _ = update_meta_from_recent(
                args,
                meta_agent,
                quasimetric_buffer,
                recent_buffer,
                args.start_task - 1,
                writer,
                step,
            )
            if not actor_metrics:
                raise RuntimeError(
                    "Unable to update the resumed meta agent. Ensure "
                    "--meta_update_steps is positive and the quasimetric buffer "
                    "contains at least --qm_min_buffer_size transitions."
                )
            meta_ready = True
        random_agent = build_fast_agent(
            obs_space,
            action_space,
            device,
            args,
            num_steps_per_run,
        )
        selection, detection_scores = detect_initialization(
            args,
            eval_env,
            env.base_task_name,
            fast_agent,
            meta_agent,
            random_agent,
            meta_ready,
            writer,
            step,
        )
        if selection == "random":
            fast_agent = random_agent
            student_agents[active_student_idx] = fast_agent

    collector.initial_collect(args.random_steps)

    while step <= num_steps_per_run:
        if task_counter != env.task_counter:
            completed_task_count = task_counter
            task_counter = env.task_counter

            recent_buffer.reset()
            copied_transitions, count_success = base.copy_recent_trajectories(
                replay_buffer,
                recent_buffer,
                args.store_traj_num,
            )
            integrated_transitions = copy_replay_buffer(
                recent_buffer,
                quasimetric_buffer,
            )
            buffer_path = base.quasimetric_buffer_path(
                args.save_path,
                log_name,
                env,
                completed_task_count,
            )
            quasimetric_buffer.save_data(buffer_path)
            _, actor_metrics, recent_updates = update_meta_from_recent(
                args,
                meta_agent,
                quasimetric_buffer,
                recent_buffer,
                completed_task_count,
                writer,
                step,
            )
            meta_ready = meta_ready or bool(actor_metrics)

            base.log_scalar(writer, "meta/copied_transitions", copied_transitions, step)
            base.log_scalar(
                writer, "meta/integrated_transitions", integrated_transitions, step
            )
            base.log_scalar(
                writer,
                "meta/buffer_size",
                len(quasimetric_buffer),
                step,
            )
            base.log_scalar(writer, "meta/count_success", count_success, step)
            base.log_scalar(
                writer,
                "meta/recent_update_fraction",
                recent_updates / max(args.meta_update_steps, 1),
                step,
            )
            checkpoint_name = f"{log_name}_task{completed_task_count}"
            fast_agent.save(model_dir, f"{checkpoint_name}_fast")
            meta_agent.save(model_dir, f"{checkpoint_name}_meta")
            save_resume_state(
                model_dir,
                checkpoint_name,
                env,
                completed_task_count,
                step,
                meta_ready,
                count_success,
                buffer_path,
            )

            if step == num_steps_per_run:
                break

            replay_buffer.reset()
            student_idx = student_slot(task_counter, args.student_mode)
            fast_agent = student_agents[student_idx]
            random_agent = build_fast_agent(
                obs_space,
                action_space,
                device,
                args,
                num_steps_per_run,
            )
            selection, detection_scores = detect_initialization(
                args,
                eval_env,
                env.base_task_name,
                fast_agent,
                meta_agent,
                random_agent,
                meta_ready,
                writer,
                step,
            )
            if selection == "random":
                fast_agent = random_agent
            student_agents[student_idx] = fast_agent

            task_start_step = step
            collector.initial_collect(args.random_steps)
            base.log_scalar(writer, "task/task_idx", task_counter, step)

        warmup_active = (
            selection == "meta"
            and meta_ready
            and args.lambda_reg > 0.0
            and step - task_start_step < args.warmup_steps
        )
        train_metrics = fast_agent.update(
            replay_buffer,
            step,
            teacher_agent=meta_agent if warmup_active else None,
            regularization_weight=args.lambda_reg if warmup_active else 0.0,
            goal_buffer=quasimetric_buffer if warmup_active else None,
            distill_loss_type=args.distill_loss_type,
            distill_goal_source=args.distill_goal_source,
        )
        collector.run_one_step(step, fast_agent)
        base.log_metric_dict(writer, "train", train_metrics, step)
        base.log_scalar(writer, "train/task_idx", task_counter, step)
        current_student_idx = student_slot(task_counter, args.student_mode)
        base.log_scalar(writer, "train/student_idx", current_student_idx, step)
        base.log_scalar(writer, "train/replay_buffer_size", len(replay_buffer), step)
        base.log_scalar(writer, "train/meta_warmup", int(warmup_active), step)

        if step % args.save_freq == 0:
            elapsed = time.perf_counter() - start_time
            sps = int(step / elapsed) if elapsed > 0 else 0
            print(
                "step:",
                step,
                "time:",
                round(elapsed / 60, 3),
                "SPS:",
                sps,
                "task:",
                (task_counter, env.base_task_name),
                "selection:",
                selection,
                "warmup:",
                warmup_active,
            )

            eval_metrics = base.evaluate_and_log(
                env,
                fast_agent,
                writer,
                "eval",
                step,
                args.num_eval_runs,
            )
            base.append_eval_stats(intermediate_stats, eval_metrics)
            intermediate_stats["steps"].append(step)
            intermediate_stats["task"].append(env.base_task_name)
            intermediate_stats["seed"].append(args.seed)
            intermediate_stats["task_idx"].append(task_counter)
            intermediate_stats["student_idx"].append(current_student_idx)
            intermediate_stats["student_mode"].append(args.student_mode)
            intermediate_stats["method"].append(args.method)
            intermediate_stats["distill_goal_source"].append(args.distill_goal_source)
            intermediate_stats["selection"].append(selection)
            intermediate_stats["meta_warmup"].append(warmup_active)
            intermediate_stats["time"].append(round(elapsed / 3600, 3))
            intermediate_stats["count_success"].append(count_success)
            for candidate_name in ("fast", "meta", "random"):
                intermediate_stats[f"detection_{candidate_name}_return"].append(
                    detection_scores.get(candidate_name, np.nan)
                )

            base.log_scalar(writer, "charts/SPS", sps, step)
            base.log_scalar(writer, "eval/task_idx", task_counter, step)
            base.log_scalar(writer, "eval/count_success", count_success, step)
            print(
                f"success {eval_metrics['success_mean']:.3f} +/- {eval_metrics['success_std']:.3f}, "
                f"gc_success {eval_metrics['gc_success_mean']:.3f} +/- {eval_metrics['gc_success_std']:.3f}, "
                f"eval return {eval_metrics['return_mean']:.3f} +/- {eval_metrics['return_std']:.3f}"
            )
            writer.flush()

        if args.save_model_freq > 0 and step > 0 and step % args.save_model_freq == 0:
            fast_agent.save(model_dir, f"{log_name}_step{step}_fast")
            meta_agent.save(model_dir, f"{log_name}_step{step}_meta")

        step += 1

    base.write_stats_csv(
        os.path.join(run_dir, f"{log_name}.csv"),
        intermediate_stats,
    )
    fast_agent.save(model_dir, f"{log_name}_final_fast")
    meta_agent.save(model_dir, f"{log_name}_final_meta")
    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    sys.exit(main())
