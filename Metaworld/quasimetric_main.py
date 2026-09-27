import argparse
import importlib
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from agent.quasimetric import (
    ContinualQuasimetricAgentConfig,
    ContinualQuasimetricSACAgent,
    QuasimetricConfig,
)
from replay_buffer import Collector
from replay_buffer_metric import ReplayBufferMetric, ReplayBufferMetricNoHER


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


class ExperimentLogger:
    def __init__(self, tensorboard_writer=None, wandb_run=None):
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run
        self._wandb_step = None
        self._wandb_buffer = {}

    def _flush_wandb(self):
        if self.wandb_run is None or self._wandb_step is None or not self._wandb_buffer:
            return
        self.wandb_run.log(self._wandb_buffer, step=self._wandb_step)
        self._wandb_buffer = {}
        self._wandb_step = None

    def add_scalar(self, tag, value, step):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(tag, value, step)

        if self.wandb_run is not None:
            if self._wandb_step is not None and step != self._wandb_step:
                self._flush_wandb()
            self._wandb_step = step
            self._wandb_buffer[tag] = value

    def add_text(self, tag, text_string, global_step=None):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(tag, text_string, global_step)

        if self.wandb_run is not None:
            self._flush_wandb()
            summary_key = tag.replace("/", "_")
            self.wandb_run.summary[summary_key] = text_string
            if global_step is not None:
                self.wandb_run.summary[f"{summary_key}_step"] = global_step

    def flush(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
        self._flush_wandb()

    def close(self):
        self.flush()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


def set_seed_everywhere(seed_value):
    seed_value = int(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class ConfigDictConverter:
    def __init__(self, config_dict):
        self.config_dict = config_dict.copy()

        if "num_repeats" in self.config_dict:
            del self.config_dict["num_repeats"]
        if "num_runs_per_group" in self.config_dict:
            del self.config_dict["num_runs_per_group"]

        self.agent_dict = self.config_dict.copy()
        self.env_dict = self.config_dict.copy()
        self.repeat_idx = config_dict["repeat_idx"]

        env_key_lst = [
            "env",
            "base_task_name",
            "seed",
            "goal_hidden",
            "normalize_obs",
            "normalize_rewards",
            "capture_video",
            "save_name",
            "change_freq",
            "env_sequence",
            "obs_drift_mean",
            "obs_drift_std",
            "obs_scale_drift",
            "obs_noise_std",
            "normalize_avg_coef",
            "reset_obs_stats",
            "change_when_solved",
            "goal_conditioned",
            "gc_reward_type",
            "gc_success_threshold",
            "gc_achieved_goal",
            "task_order",
            "fetch_env_version",
            "max_episode_steps",
        ]
        env = config_dict["env"].lower()

        if env.startswith("fetch_sequence_"):
            import envs.fetch_env

            self.env_class = envs.fetch_env.FetchGoalEnvSequence
            env_suffix = env[len("fetch_sequence_") :]
            if env_suffix and env_suffix != "custom":
                if env_suffix.startswith("set") or env_suffix in envs.fetch_env.FETCH_SEQUENCE_PRESETS:
                    self.env_dict["env_sequence"] = env_suffix
                else:
                    self.env_dict["base_task_name"] = env_suffix
        else:
            import envs.metaworld_env

            self.env_class = envs.metaworld_env.MetaWorldSingleEnvSequence

        if env[0:19] == "metaworld_sequence_":
            if env[19:22] == "set":
                self.env_dict["env_sequence"] = env[19:]
            else:
                self.env_dict["base_task_name"] = f"{env[19:]}-v2"

        self.env_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
        self.env_dict["env_type"] = "rl"

        print("env_dict keys", self.env_dict.keys())
        print("env_dict", self.env_dict)

        self.agent_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.env_dict["seed"] += self.repeat_idx * 1


def str2none(value):
    if value.lower() in {"none", ""}:
        return None
    return value


def str_choice(value):
    return value.strip().strip("'\"‘’“”").lower()


def normalize_log_backends(backends):
    normalized = []
    for backend in backends:
        backend = backend.lower()
        if backend == "none":
            if len(backends) != 1:
                raise ValueError("'none' cannot be combined with other logging backends")
            return []
        if backend not in {"tensorboard", "wandb"}:
            raise ValueError(f"Unsupported logging backend: {backend}")
        if backend not in normalized:
            normalized.append(backend)
    return normalized


def load_wandb_module():
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def vector_observation_space(observation_space):
    spaces = getattr(observation_space, "spaces", None)
    if spaces is not None and "observation" in spaces:
        return spaces["observation"]
    return observation_space


def build_agent(obs_dim, action_dim, device, args):
    quasimetric_cfg = QuasimetricConfig(
        latent_dim=args.qm_latent_dim,
        hidden_dim=args.qm_hidden_dim,
        hidden_depth=args.qm_hidden_depth,
        transition_input=args.qm_transition_input,
        components=args.qm_components,
        batch_size=args.qm_batch_size,
        lr=args.qm_lr,
        discount=args.qm_discount,
        lambda_=args.qm_lambda,
        next_state_sample=args.qm_next_state_sample,
        backup_clip=args.qm_backup_clip,
        action_invariance_coef=args.qm_action_invariance_coef,
        transition_consistency_coef=args.qm_transition_consistency_coef,
        contrastive_coef=args.qm_contrastive_coef,
        nce_mode=args.qm_nce_mode,
        target_tau=args.qm_target_tau,
        current_batch_ratio=args.qm_current_batch_ratio,
        max_grad_norm=args.qm_max_grad_norm,
        min_buffer_size=args.qm_min_buffer_size,
    )
    continual_cfg = ContinualQuasimetricAgentConfig(
        structure_update_frequency=args.structure_update_frequency,
        structure_updates_per_step=args.structure_updates_per_step,
        structure_bonus_coef=args.structure_bonus_coef,
        bc_alpha=args.bc_alpha,
        memory_max_tasks=args.memory_max_tasks,
        memory_max_transitions_per_task=args.memory_max_transitions_per_task,
        share_sac_batch=bool(args.qm_share_sac_batch),
        shared_batch_size=args.qm_shared_batch_size,
        goal_reward_scale=args.goal_reward_scale,
        task_reward_scale=args.task_reward_scale,
        goal_reward_type=args.goal_reward_type,
        behavior_goal_success_only=bool(args.behavior_goal_success_only),
        encode_actor_critic_goal=bool(args.encode_actor_critic_goal),
    )

    return ContinualQuasimetricSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=[-1.0, 1.0],
        device=device,
        batch_size=args.batch_size,
        discount=args.discount,
        init_temperature=args.init_temperature,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        critic_tau=args.critic_tau,
        actor_update_frequency=args.actor_update_frequency,
        critic_target_update_frequency=args.critic_target_update_frequency,
        quasimetric_cfg=quasimetric_cfg,
        continual_cfg=continual_cfg,
    )


def save_model(agent, model_dir, model_name, step, save_model_freq):
    if save_model_freq <= 0:
        return
    if step == 0 or step % save_model_freq != 0:
        return
    agent.save(model_dir, f"{model_name}_{step}")


def mean_std(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def summarize_eval_results(eval_results):
    return_mean, return_std = mean_std(eval_results["episodic_returns"])
    metaworld_success_mean, metaworld_success_std = mean_std(eval_results["successes"])
    gc_success_mean, gc_success_std = mean_std(eval_results.get("goal_successes", []))
    gc_final_distance_mean, gc_final_distance_std = mean_std(eval_results.get("final_goal_distances", []))
    gc_min_distance_mean, gc_min_distance_std = mean_std(eval_results.get("min_goal_distances", []))
    gc_mean_distance_mean, gc_mean_distance_std = mean_std(eval_results.get("mean_goal_distances", []))
    return {
        "return_mean": return_mean,
        "return_std": return_std,
        "metaworld_success_mean": metaworld_success_mean,
        "metaworld_success_std": metaworld_success_std,
        "gc_success_mean": gc_success_mean,
        "gc_success_std": gc_success_std,
        "gc_final_goal_distance_mean": gc_final_distance_mean,
        "gc_final_goal_distance_std": gc_final_distance_std,
        "gc_min_goal_distance_mean": gc_min_distance_mean,
        "gc_min_goal_distance_std": gc_min_distance_std,
        "gc_mean_goal_distance_mean": gc_mean_distance_mean,
        "gc_mean_goal_distance_std": gc_mean_distance_std,
    }


def evaluate_current_task(env, agent, num_eval_runs):
    eval_results = env.evaluate_agent(agent, num_eval_runs)
    return summarize_eval_results(eval_results)


def create_experiment_logger(args, log_name):
    wandb_module = None
    tb_writer = None
    tb_path = None
    wandb_run = None
    wandb_path = None
    wandb_url = None
    active_backends = []

    if "tensorboard" in args.log_backends:
        if SummaryWriter is None:
            print("TensorBoard is unavailable. Install 'tensorboard' to enable event logging.")
        else:
            tb_path = os.path.join(args.save_path, "tensorboard", args.env, log_name)
            os.makedirs(tb_path, exist_ok=True)
            tb_writer = SummaryWriter(tb_path)
            active_backends.append("tensorboard")

    if "wandb" in args.log_backends:
        wandb_module = load_wandb_module()
        if wandb_module is None:
            print("W&B is unavailable. Install 'wandb' to enable Weights & Biases logging.")
        else:
            os.makedirs(args.save_path, exist_ok=True)
            try:
                wandb_run = wandb_module.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    config=vars(args),
                    name=log_name,
                    dir=args.save_path,
                    mode=args.wandb_mode,
                )
                wandb_path = os.path.join(args.save_path, "wandb")
                wandb_url = getattr(wandb_run, "url", None)
                active_backends.append("wandb")
            except Exception as error:
                print(f"W&B initialization failed: {error}")

    logger = None
    if tb_writer is not None or wandb_run is not None:
        logger = ExperimentLogger(tensorboard_writer=tb_writer, wandb_run=wandb_run)
        logger.add_text(
            "config/args",
            "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())),
            0,
        )

    return logger, {
        "active_backends": active_backends,
        "tensorboard_path": tb_path,
        "wandb_path": wandb_path,
        "wandb_url": wandb_url,
    }


def log_scalar(writer, tag, value, step):
    if writer is None or value is None:
        return

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return
        value = value.item()
    elif isinstance(value, (np.integer, np.floating)):
        value = value.item()

    if isinstance(value, (int, float)) and np.isfinite(value):
        writer.add_scalar(tag, value, step)


def log_metric_dict(writer, prefix, metrics, step):
    if writer is None:
        return

    for key, value in metrics.items():
        log_scalar(writer, f"{prefix}/{key}", value, step)


def append_intermediate_stats(
    intermediate_stats,
    args,
    env,
    task_counter,
    step,
    start_time,
    eval_return_mean,
    eval_success_mean,
    train_metrics,
    agent,
    eval_metrics=None,
):
    intermediate_stats["mean_return"].append(eval_return_mean)
    intermediate_stats["mean_success"].append(eval_success_mean)
    intermediate_stats["metaworld_success"].append(eval_success_mean)
    intermediate_stats["gc_success"].append(
        np.nan if eval_metrics is None else eval_metrics.get("gc_success_mean", np.nan)
    )
    intermediate_stats["gc_final_goal_distance"].append(
        np.nan if eval_metrics is None else eval_metrics.get("gc_final_goal_distance_mean", np.nan)
    )
    intermediate_stats["gc_min_goal_distance"].append(
        np.nan if eval_metrics is None else eval_metrics.get("gc_min_goal_distance_mean", np.nan)
    )
    intermediate_stats["gc_mean_goal_distance"].append(
        np.nan if eval_metrics is None else eval_metrics.get("gc_mean_goal_distance_mean", np.nan)
    )
    intermediate_stats["steps"].append(step)
    intermediate_stats["task"].append(env.base_task_name)
    intermediate_stats["seed"].append(args.seed)
    intermediate_stats["task_idx"].append(task_counter)
    intermediate_stats["method"].append("continual_quasimetric")
    intermediate_stats["time"].append(round((time.perf_counter() - start_time) / 3600, 3))
    intermediate_stats["critic_loss"].append(train_metrics.get("critic", np.nan))
    intermediate_stats["actor_loss"].append(train_metrics.get("actor", np.nan))
    intermediate_stats["alpha"].append(train_metrics.get("alpha", np.nan))
    intermediate_stats["goal_reward"].append(train_metrics.get("goal_reward", np.nan))
    intermediate_stats["goal_reached_ratio"].append(train_metrics.get("goal_reached_ratio", np.nan))
    intermediate_stats["goal_steps"].append(train_metrics.get("goal_steps", np.nan))
    intermediate_stats["structure_bonus"].append(train_metrics.get("structure_bonus", 0.0))
    intermediate_stats["quasimetric_loss"].append(train_metrics.get("quasimetric/loss", np.nan))
    intermediate_stats["quasimetric_backup_loss"].append(
        train_metrics.get("quasimetric/backup_loss", np.nan)
    )
    intermediate_stats["quasimetric_alignment"].append(
        train_metrics.get("quasimetric/structure_alignment", np.nan)
    )
    intermediate_stats["memory_tasks"].append(agent.structure_memory.num_tasks)
    intermediate_stats["memory_transitions"].append(len(agent.structure_memory))


def evaluate_all_tasks(env, agent, num_eval_runs, seed):
    final_stats = defaultdict(list)
    for task_idx, task_name in enumerate(env.env_list, start=1):
        env.set_task(task_name)
        eval_results = env.evaluate_agent(agent, num_eval_runs)
        eval_metrics = summarize_eval_results(eval_results)

        print(
            f"Final task {task_name} success {round(eval_metrics['metaworld_success_mean'], 3)} "
            f"gc_success {round(eval_metrics['gc_success_mean'], 3)} "
            f"return {round(eval_metrics['return_mean'], 3)}"
        )

        final_stats["mean_return"].append(eval_metrics["return_mean"])
        final_stats["mean_success"].append(eval_metrics["metaworld_success_mean"])
        final_stats["metaworld_success"].append(eval_metrics["metaworld_success_mean"])
        final_stats["gc_success"].append(eval_metrics["gc_success_mean"])
        final_stats["gc_final_goal_distance"].append(eval_metrics["gc_final_goal_distance_mean"])
        final_stats["gc_min_goal_distance"].append(eval_metrics["gc_min_goal_distance_mean"])
        final_stats["gc_mean_goal_distance"].append(eval_metrics["gc_mean_goal_distance_mean"])
        final_stats["task"].append(task_name)
        final_stats["task_idx"].append(task_idx)
        final_stats["seed"].append(seed)
        final_stats["method"].append("continual_quasimetric")
    return pd.DataFrame(final_stats)


def main():
    parser = argparse.ArgumentParser(description="Run continual quasimetric RL experiments")
    parser.add_argument("--repeat_idx", type=int, default=0, help="Index of the repeat")
    parser.add_argument("--env", type=str, default="metaworld_sequence_set6", help="Environment to run")
    parser.add_argument(
        "--change_freq",
        type=int,
        default=1e6,
        help="Frequency to change tasks in the environment",
    )
    parser.add_argument(
        "--normalize_obs",
        type=str2none,
        default=None,
        help="Normalize observations (pass 'none' to keep None)",
    )
    # -------------
    parser.add_argument(
        "--goal_conditioned",
        type=int,
        default=1,
        help="Whether MetaWorld returns dict observations with observation/achieved_goal/desired_goal",
    )
    parser.add_argument(
        "--gc_reward_type",
        type=str_choice,
        default="sparse",
        choices=["sparse", "dense", "success"],
        help="Reward type used by the goal-conditioned MetaWorld wrapper",
    )
    parser.add_argument(
        "--gc_success_threshold",
        type=float,
        default=0.05,
        help="Goal distance threshold for goal-conditioned success",
    )
    parser.add_argument(
        "--gc_achieved_goal",
        type=str_choice,
        default="auto",
        choices=["auto", "object", "tcp"],
        help="Source for the achieved_goal vector in goal-conditioned observations",
    )
    parser.add_argument(
        "--task_order",
        type=str2none,
        default=None,
        help="Custom Fetch task order, e.g. reach,push,pick-and-place,slide or a txt/json file path",
    )
    parser.add_argument(
        "--fetch_env_version",
        type=str2none,
        default="auto",
        help="Gymnasium Robotics Fetch env version to use, e.g. auto, 3, or 2",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=50,
        help="Episode time limit for Fetch envs; MetaWorld keeps its internal 200-step limit",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_path", type=str, default="results/", help="Path prefix for logs and models")
    parser.add_argument("--save_freq", type=int, default=25000, help="Evaluation frequency")
    parser.add_argument("--save_model_freq", type=int, default=-1, help="Model save frequency")
    parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU IDs")
    parser.add_argument("--random_steps", type=int, default=10000, help="Random warmup steps for each task")
    parser.add_argument(
        "--log_backends",
        type=str.lower,
        nargs="+",
        default=["wandb"],
        choices=["tensorboard", "wandb", "none"],
        help="Logging backends to enable. Use one or both of: tensorboard wandb, or pass none.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="continual-quasimetric-rl",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str2none,
        default=None,
        help="Optional Weights & Biases entity/team",
    )
    parser.add_argument(
        "--wandb_group",
        type=str2none,
        default=None,
        help="Optional Weights & Biases group name",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="Weights & Biases run mode when wandb logging is enabled",
    )

    parser.add_argument("--batch_size", type=int, default=256, help="SAC batch size")
    parser.add_argument("--discount", type=float, default=0.99, help="SAC discount")
    parser.add_argument("--init_temperature", type=float, default=0.1, help="Initial SAC temperature")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic learning rate")
    parser.add_argument("--alpha_lr", type=float, default=1e-4, help="Temperature learning rate")
    parser.add_argument("--critic_tau", type=float, default=0.005, help="Target critic tau")
    parser.add_argument(
        "--actor_update_frequency",
        type=int,
        default=1,
        help="Actor update frequency",
    )
    parser.add_argument(
        "--critic_target_update_frequency",
        type=int,
        default=1,
        help="Critic target update frequency",
    )

    parser.add_argument("--qm_latent_dim", type=int, default=256, help="Quasimetric latent dim")
    parser.add_argument(
        "--qm_transition_input",
        "--qm-transition-input",
        choices=("state", "latent"),
        default="state",
        help="Use raw observations (T(s,a)) or latent states (T(z,a)).",
    )
    parser.add_argument("--qm_hidden_dim", type=int, default=256, help="Quasimetric hidden dim")
    parser.add_argument("--qm_hidden_depth", type=int, default=2, help="Quasimetric MLP depth")
    parser.add_argument("--qm_components", type=int, default=8, help="MRN components")
    parser.add_argument("--qm_batch_size", type=int, default=256, help="Structure learner batch size")
    parser.add_argument(
        "--qm_share_sac_batch",
        type=int,
        default=0,
        help="Whether quasimetric reuses the SAC batch instead of sampling its own batch",
    )
    parser.add_argument(
        "--qm_shared_batch_size",
        type=int,
        default=None,
        help="Optional shared batch size used when SAC and quasimetric reuse the same batch",
    )
    parser.add_argument("--qm_lr", type=float, default=1e-4, help="Structure learner lr")
    parser.add_argument("--qm_discount", type=float, default=0.995, help="Goal sampling discount")
    parser.add_argument("--qm_lambda", type=float, default=0.95, help="Intermediate goal sampling ratio")
    parser.add_argument("--qm_next_state_sample", type=float, default=0.2, help="Probability of one-step backup")
    parser.add_argument("--qm_backup_clip", type=float, default=5.0, help="LINEX backup clip")
    parser.add_argument(
        "--qm_action_invariance_coef",
        type=float,
        default=1.0,
        help="Action invariance coefficient",
    )
    parser.add_argument(
        "--qm_transition_consistency_coef",
        type=float,
        default=1.0,
        help="Latent transition consistency coefficient",
    )
    parser.add_argument(
        "--qm_contrastive_coef",
        type=float,
        default=0.05,
        help="Contrastive auxiliary coefficient",
    )
    parser.add_argument(
        "--qm_nce_mode",
        choices=("forward_nce", "backward_nce"),
        default="forward_nce",
        help="Direction used by the contrastive NCE objective",
    )
    parser.add_argument("--qm_target_tau", type=float, default=0.01, help="Target encoder tau")
    parser.add_argument(
        "--qm_current_batch_ratio",
        type=float,
        default=0.5,
        help="Fraction of structure batches drawn from the current task",
    )
    parser.add_argument(
        "--qm_max_grad_norm",
        type=float,
        default=None,
        help="Optional gradient clipping for structure learner",
    )
    parser.add_argument(
        "--qm_min_buffer_size",
        type=int,
        default=256,
        help="Minimum current-task buffer size before structure updates start",
    )

    parser.add_argument(
        "--structure_update_frequency",
        type=int,
        default=1,
        help="How often to update the slow quasimetric structure",
    )
    # TODO: kl between z?
    parser.add_argument(
        "--structure_updates_per_step",
        type=int,
        default=1,
        help="How many structure updates to run each time",
    )
    parser.add_argument(
        "--structure_bonus_coef",
        type=float,
        default=0.0,   #### try 0.5   without bonus
        help="Reward shaping coefficient from quasimetric structure bonus",
    )
    parser.add_argument("--bc_alpha", type=float, default=0.1, help="Behavior-cloning loss coefficient")
    parser.add_argument(
        "--memory_max_tasks",
        type=int,
        default=None,
        help="Maximum number of past tasks stored in quasimetric memory",
    )
    parser.add_argument(
        "--memory_max_transitions_per_task",
        type=int,
        default=None,
        help="Optional cap on stored transitions per task",
    )
    parser.add_argument(
        "--goal_reward_scale",
        type=float,
        default=1.0,
        help="Weight of the goal reward term in metric SAC updates",
    )
    parser.add_argument(
        "--task_reward_scale",
        type=float,
        default=0.0,
        help="Weight of the environment reward in metric SAC updates",
    )
    parser.add_argument(
        "--goal_reward_type",
        type=str,
        default="step_cost",
        choices=["binary", "step_cost"],
        help="Reward type used when future-goal relabeling is enabled",
    )
    parser.add_argument(
        "--behavior_goal_success_only",
        type=int,
        default=0,
        help="Prefer successful achieved states when sampling the behavior goal",
    )
    parser.add_argument(
        "--encode_actor_critic_goal",
        type=int,
        default=1,
        help="Whether to encode goal observations before feeding actor and critic",
    )
    parser.add_argument(
        "--replay_buffer_mode",
        type=str,
        default="her",
        choices=["her", "no_her"],
        help="Metric replay buffer mode: HER reward relabeling or future-goal conditioning without HER rewards",
    )

    args = parser.parse_args()
    try:
        args.log_backends = normalize_log_backends(args.log_backends)
    except ValueError as error:
        parser.error(str(error))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config_obj = ConfigDictConverter(vars(args))
    env_parameters = config_obj.env_dict
    env = config_obj.env_class(**env_parameters)
    print("env_list:", env.env_list)

    num_steps_per_run = len(env.env_list) * args.change_freq
    num_eval_runs = 10
    set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()

    replay_buffer_capacity = args.change_freq  #### 1e6
    replay_buffer_cls = ReplayBufferMetric if args.replay_buffer_mode == "her" else ReplayBufferMetricNoHER
    obs_space = vector_observation_space(env.env.observation_space)
    replay_buffer = replay_buffer_cls(
        obs_space.shape,
        env.env.action_space.shape,
        int(replay_buffer_capacity) + args.random_steps,
        device,
    )
    collector = Collector(env, replay_buffer)

    agent = build_agent(
        obs_dim=obs_space.shape[0],
        action_dim=env.env.action_space.shape[0],
        device=device,
        args=args,
    )

    collector.initial_collect(args.random_steps)

    log_path = os.path.join(args.save_path, "log", args.env)
    model_path = os.path.join(args.save_path, "model")
    gc_log_suffix = f"_gc-{args.gc_reward_type}" if bool(args.goal_conditioned) else ""
    log_name = f"quasimetric_{args.env}_{args.seed}{gc_log_suffix}"
    writer, log_info = create_experiment_logger(args, log_name)
    if log_info["active_backends"]:
        print("log_backends:", ", ".join(log_info["active_backends"]))
    else:
        print("log_backends: disabled")
    if log_info["tensorboard_path"] is not None:
        print("tensorboard_path:", log_info["tensorboard_path"])
    if log_info["wandb_path"] is not None:
        print("wandb_path:", log_info["wandb_path"])
    if log_info["wandb_url"] is not None:
        print("wandb_url:", log_info["wandb_url"])
    log_scalar(writer, "config/goal_conditioned", int(bool(args.goal_conditioned)), 0)
    log_scalar(writer, "config/obs_dim", obs_space.shape[0], 0)
    log_scalar(writer, "config/gc_success_threshold", args.gc_success_threshold, 0)

    intermediate_stats = defaultdict(list)
    task_counter = env.task_counter
    last_train_metrics = {}
    last_finished_task_idx = None

    i_step = 0
    try:
        while i_step <= num_steps_per_run:
            if task_counter != env.task_counter:
                finished_task_idx = (task_counter - 1) % len(env.env_list)
                last_finished_task_idx = finished_task_idx
                finished_task_name = env.env_list[finished_task_idx]
                memory_info = agent.finish_task(finished_task_idx, replay_buffer)   #  write structured memory; taskid for task logging
                print(
                    f"finish task {finished_task_name}: "
                    f"memory_tasks={memory_info['memory_tasks']} "
                    f"memory_transitions={memory_info['memory_transitions']}"
                )
                log_scalar(writer, "memory/num_tasks", memory_info["memory_tasks"], i_step)
                log_scalar(writer, "memory/num_transitions", memory_info["memory_transitions"], i_step)
                if writer is not None:
                    writer.add_text("task/finished", finished_task_name, i_step)

                if i_step == num_steps_per_run:
                    break

                task_counter = env.task_counter
                replay_buffer.reset()  #### TODO: not reset?
                collector.initial_collect(args.random_steps)

            last_train_metrics = agent.update(replay_buffer, i_step)
            collector.run_one_step(i_step, agent)  ####
            save_model(agent, model_path, log_name, i_step, args.save_model_freq)

            log_metric_dict(writer, "train", last_train_metrics, i_step)
            log_scalar(writer, "train/task_idx", task_counter, i_step)
            log_scalar(writer, "train/replay_buffer_size", len(replay_buffer), i_step)
            log_scalar(writer, "train/memory_tasks", agent.structure_memory.num_tasks, i_step)
            log_scalar(writer, "train/memory_transitions", len(agent.structure_memory), i_step)

            if i_step % args.save_freq == 0:
                elapsed = time.perf_counter() - start_time
                sps = int(i_step / elapsed) if elapsed > 0 else 0
                print(
                    "step:",
                    i_step,
                    "time:",
                    round(elapsed / 60, 3),
                    "SPS:",
                    sps,
                    "task_counter",
                    (task_counter, env.base_task_name),
                    "method",
                    "continual_quasimetric",
                    "seed",
                    args.seed,
                )
                if last_train_metrics:
                    print(
                        "train:",
                        {
                            key: round(value, 4)
                            for key, value in last_train_metrics.items()
                            if isinstance(value, (int, float))
                        },
                    )

                eval_metrics = evaluate_current_task(
                    env,
                    agent,
                    num_eval_runs,
                )
                append_intermediate_stats(
                    intermediate_stats,
                    args,
                    env,
                    task_counter,
                    i_step,
                    start_time,
                    eval_metrics["return_mean"],
                    eval_metrics["metaworld_success_mean"],
                    last_train_metrics,
                    agent,
                    eval_metrics,
                )
                log_scalar(writer, "charts/SPS", sps, i_step)
                log_scalar(writer, "eval/mean_return", eval_metrics["return_mean"], i_step)
                log_scalar(writer, "eval/return_std", eval_metrics["return_std"], i_step)
                log_scalar(writer, "eval/mean_success", eval_metrics["metaworld_success_mean"], i_step)
                log_scalar(writer, "eval/success_std", eval_metrics["metaworld_success_std"], i_step)
                log_scalar(writer, "eval/metaworld_success", eval_metrics["metaworld_success_mean"], i_step)
                log_scalar(writer, "eval/metaworld_success_std", eval_metrics["metaworld_success_std"], i_step)
                log_scalar(writer, "eval/gc_success", eval_metrics["gc_success_mean"], i_step)
                log_scalar(writer, "eval/gc_success_std", eval_metrics["gc_success_std"], i_step)
                log_scalar(writer, "eval/gc_final_goal_distance", eval_metrics["gc_final_goal_distance_mean"], i_step)
                log_scalar(writer, "eval/gc_final_goal_distance_std", eval_metrics["gc_final_goal_distance_std"], i_step)
                log_scalar(writer, "eval/gc_min_goal_distance", eval_metrics["gc_min_goal_distance_mean"], i_step)
                log_scalar(writer, "eval/gc_min_goal_distance_std", eval_metrics["gc_min_goal_distance_std"], i_step)
                log_scalar(writer, "eval/gc_mean_goal_distance", eval_metrics["gc_mean_goal_distance_mean"], i_step)
                log_scalar(writer, "eval/gc_mean_goal_distance_std", eval_metrics["gc_mean_goal_distance_std"], i_step)
                log_scalar(writer, "eval/task_idx", task_counter, i_step)
                print(
                    f"success {round(eval_metrics['metaworld_success_mean'], 3)} +/- "
                    f"{round(eval_metrics['metaworld_success_std'], 3)}, "
                    f"gc_success {round(eval_metrics['gc_success_mean'], 3)} +/- "
                    f"{round(eval_metrics['gc_success_std'], 3)}, "
                    f"eval return {round(eval_metrics['return_mean'], 3)} +/- "
                    f"{round(eval_metrics['return_std'], 3)}"
                )
                if writer is not None:
                    writer.flush()
            i_step += 1

        current_task_idx = (task_counter - 1) % len(env.env_list)
        if len(replay_buffer) > 0 and last_finished_task_idx != current_task_idx:
            finished_task_idx = (task_counter - 1) % len(env.env_list)
            memory_info = agent.finish_task(finished_task_idx, replay_buffer)
            log_scalar(writer, "memory/num_tasks", memory_info["memory_tasks"], i_step)
            log_scalar(writer, "memory/num_transitions", memory_info["memory_transitions"], i_step)

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        intermediate_stats = pd.DataFrame(intermediate_stats)
        intermediate_stats.to_csv(os.path.join(log_path, f"{log_name}.csv"), index=False)

        print("---")
        final_stats = evaluate_all_tasks(env, agent, num_eval_runs, args.seed)  ####
        final_stats.to_csv(os.path.join(log_path, f"{log_name}_final.csv"), index=False)

        if writer is not None:
            for row in final_stats.itertuples(index=False):
                task_tag = str(row.task).replace(" ", "_")
                log_scalar(writer, f"final/{task_tag}/mean_return", row.mean_return, row.task_idx)
                log_scalar(writer, f"final/{task_tag}/mean_success", row.mean_success, row.task_idx)
                log_scalar(writer, f"final/{task_tag}/gc_success", row.gc_success, row.task_idx)
                log_scalar(
                    writer,
                    f"final/{task_tag}/gc_final_goal_distance",
                    row.gc_final_goal_distance,
                    row.task_idx,
                )
            writer.flush()

        agent.save(model_path, f"{log_name}_final")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()


# python quasimetric_main.py --seed 0 --gpu 1 --env metaworld_sequence_set6 --log_backends tensorboard wandb

