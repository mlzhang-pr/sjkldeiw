import argparse
import csv
import importlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch


ONLINE_CONTINUAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUASIMETRIC_RL_DIR = os.path.dirname(ONLINE_CONTINUAL_DIR)
FETCH_DIR = os.path.dirname(QUASIMETRIC_RL_DIR)

if FETCH_DIR not in sys.path:
    sys.path.insert(0, FETCH_DIR)
if QUASIMETRIC_RL_DIR not in sys.path:
    sys.path.insert(0, QUASIMETRIC_RL_DIR)

from agent.quasimetric import ( 
    ContinualQuasimetricAgentConfig,
    ContinualQuasimetricSACAgent,
    QuasimetricConfig,
)
from fetch_env import FetchGoalEnvSequence  
from quasimetric_rl.data import BatchData, EnvSpec 
from quasimetric_rl.modules import QRLConf  
from replay_buffer import Collector  
from replay_buffer_metric import ReplayBufferMetric, ReplayBufferMetricNoHER


DEFAULT_SAVE_PATH = os.path.join(ONLINE_CONTINUAL_DIR, "results")
FETCH_TASK_NAMES = {"reach", "push", "pick-and-place", "slide"}


class WandbSink:
    def __init__(self, wandb_run=None):
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
        if self.wandb_run is None:
            return
        if self._wandb_step is not None and step != self._wandb_step:
            self._flush_wandb()
        self._wandb_step = step
        self._wandb_buffer[tag] = value

    def add_text(self, tag, text_string, global_step=None):
        if self.wandb_run is None:
            return
        self._flush_wandb()
        summary_key = tag.replace("/", "_")
        self.wandb_run.summary[summary_key] = text_string
        if global_step is not None:
            self.wandb_run.summary[f"{summary_key}_step"] = global_step

    def flush(self):
        self._flush_wandb()

    def close(self):
        self.flush()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


class TensorboardSink:
    def __init__(self, log_dir):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("TensorBoard is unavailable. Install tensorboard to enable TensorBoard logging.")
            return
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag, text_string, global_step=None):
        if self.writer is not None:
            self.writer.add_text(tag, text_string, global_step=global_step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class ExperimentLogger:
    def __init__(self, sinks=None):
        self.sinks = [sink for sink in (sinks or []) if sink is not None]

    def add_scalar(self, tag, value, step):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            value = value.item()
        elif isinstance(value, (np.integer, np.floating)):
            value = value.item()

        if not isinstance(value, (int, float)) or not np.isfinite(value):
            return
        for sink in self.sinks:
            sink.add_scalar(tag, value, step)

    def add_text(self, tag, text_string, global_step=None):
        for sink in self.sinks:
            sink.add_text(tag, text_string, global_step=global_step)

    def flush(self):
        for sink in self.sinks:
            sink.flush()

    def close(self):
        for sink in self.sinks:
            sink.close()
        self.sinks = []


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


def str2none(value):
    if value is None:
        return None
    if str(value).lower() in {"none", ""}:
        return None
    return value


def str_choice(value):
    return value.strip().strip("'\"").lower()


def normalize_key(value):
    value = str(value).strip().strip("'\"").lower()
    value = value.replace("_", "-").replace(" ", "-")
    value = re.sub(r"-v\d+$", "", value)
    return re.sub(r"-+", "-", value).strip("-")


def safe_tag(value):
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return value or "run"


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


def create_experiment_logger(args, log_name, run_dir):
    sinks = []
    log_info = {
        "active_backends": [],
        "tensorboard_path": None,
        "wandb_path": None,
        "wandb_url": None,
    }

    if "tensorboard" in args.log_backends:
        tensorboard_path = os.path.join(run_dir, "tensorboard")
        tensorboard_sink = TensorboardSink(tensorboard_path)
        if tensorboard_sink.writer is not None:
            sinks.append(tensorboard_sink)
            log_info["active_backends"].append("tensorboard")
            log_info["tensorboard_path"] = tensorboard_path

    if "wandb" in args.log_backends:
        wandb_module = load_wandb_module()
        if wandb_module is None:
            print("W&B is unavailable. Install wandb to enable Weights & Biases logging.")
        else:
            try:
                wandb_run = wandb_module.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    config=vars(args),
                    name=log_name,
                    dir=run_dir,
                    mode=args.wandb_mode,
                )
                sinks.append(WandbSink(wandb_run=wandb_run))
                log_info["active_backends"].append("wandb")
                log_info["wandb_path"] = os.path.join(run_dir, "wandb")
                log_info["wandb_url"] = getattr(wandb_run, "url", None)
            except Exception as error:
                print(f"W&B initialization failed: {error}")

    logger = ExperimentLogger(sinks=sinks)
    logger.add_text(
        "config/args",
        "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())),
        0,
    )
    return logger, log_info


def log_scalar(writer, tag, value, step):
    if writer is not None:
        writer.add_scalar(tag, value, step)


def log_metric_dict(writer, prefix, metrics, step):
    if writer is None or not metrics:
        return
    for key, value in metrics.items():
        tag = f"{prefix}/{key}"
        if isinstance(value, dict):
            log_metric_dict(writer, tag, value, step)
        else:
            log_scalar(writer, tag, value, step)


def actor_distribution(agent, obs, goal_obs=None, detach_goal=True):
    if hasattr(agent, "actor_distribution"):
        return agent.actor_distribution(obs, goal_obs=goal_obs, detach_goal=detach_goal)
    if hasattr(agent, "_prepare_goal_rep") and hasattr(agent, "_augment_obs"):
        goal_rep = agent._prepare_goal_rep(obs.shape[0], goal_obs=goal_obs, detach=detach_goal)
        return agent.actor(agent._augment_obs(obs, goal_rep))
    return agent.actor(obs)


def distribution_mean_action(distribution):
    if hasattr(distribution, "mean"):
        return distribution.mean
    if hasattr(distribution, "mode"):
        return distribution.mode
    if hasattr(distribution, "loc"):
        return distribution.loc
    raise AttributeError(f"Cannot extract a deterministic action from {type(distribution).__name__}.")


def distill_meta_actor_to_agent(agent, meta_agent, replay_buffer, goal_buffer, update_num):
    if update_num <= 0:
        return {}

    if hasattr(agent, "distill_actor_from_agent"):
        return agent.distill_actor_from_agent(meta_agent, replay_buffer, goal_buffer, update_num)

    print_interval = max(update_num // 5, 1)
    last_goal_obs = None
    last_metrics = {}
    for idx in range(update_num):
        obs, _, _, _, _, _ = replay_buffer.sample(agent.batch_size)
        obs = torch.as_tensor(obs, device=agent.device).float()

        goal_obs = None
        if hasattr(goal_buffer, "sample_behavior_goal"):
            goal_obs = goal_buffer.sample_behavior_goal(
                discount=getattr(meta_agent, "goal_discount", 0.995),
                success_only=getattr(meta_agent, "behavior_goal_success_only", True),
                batch_size=agent.batch_size,
            )
        if goal_obs is None:
            goal_obs = getattr(meta_agent, "behavior_goal", None)
        else:
            last_goal_obs = goal_obs

        with torch.no_grad():
            teacher_dist = actor_distribution(meta_agent, obs, goal_obs=goal_obs, detach_goal=True)
            teacher_action = distribution_mean_action(teacher_dist).detach()

        student_dist = actor_distribution(agent, obs, goal_obs=goal_obs, detach_goal=True)
        student_action = distribution_mean_action(student_dist)
        actor_loss = torch.square(student_action - teacher_action).sum(-1).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        last_metrics = {
            "actor_distill_loss": float(actor_loss.item()),
            "teacher_action": float(teacher_action.mean().item()),
            "student_action": float(student_action.mean().item()),
            "teacher_goal_ready": float(goal_obs is not None),
        }
        if idx % print_interval == 0:
            print("actor_distill:", idx, last_metrics)

    if last_goal_obs is not None and hasattr(meta_agent, "set_behavior_goal"):
        meta_agent.set_behavior_goal(
            last_goal_obs[0] if np.asarray(last_goal_obs).ndim > 1 else last_goal_obs
        )
    return last_metrics


def mean_std(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def summarize_eval_results(eval_results):
    return_mean, return_std = mean_std(eval_results["episodic_returns"])
    success_mean, success_std = mean_std(eval_results["successes"])
    gc_success_mean, gc_success_std = mean_std(eval_results.get("goal_successes", []))
    final_distance_mean, final_distance_std = mean_std(eval_results.get("final_goal_distances", []))
    min_distance_mean, min_distance_std = mean_std(eval_results.get("min_goal_distances", []))
    mean_distance_mean, mean_distance_std = mean_std(eval_results.get("mean_goal_distances", []))
    return {
        "return_mean": return_mean,
        "return_std": return_std,
        "success_mean": success_mean,
        "success_std": success_std,
        "gc_success_mean": gc_success_mean,
        "gc_success_std": gc_success_std,
        "gc_final_goal_distance_mean": final_distance_mean,
        "gc_final_goal_distance_std": final_distance_std,
        "gc_min_goal_distance_mean": min_distance_mean,
        "gc_min_goal_distance_std": min_distance_std,
        "gc_mean_goal_distance_mean": mean_distance_mean,
        "gc_mean_goal_distance_std": mean_distance_std,
    }


def append_eval_stats(stats, eval_metrics):
    stats["mean_return"].append(eval_metrics["return_mean"])
    stats["mean_success"].append(eval_metrics["success_mean"])
    stats["fetch_success"].append(eval_metrics["success_mean"])
    stats["gc_success"].append(eval_metrics["gc_success_mean"])
    stats["gc_final_goal_distance"].append(eval_metrics["gc_final_goal_distance_mean"])
    stats["gc_min_goal_distance"].append(eval_metrics["gc_min_goal_distance_mean"])
    stats["gc_mean_goal_distance"].append(eval_metrics["gc_mean_goal_distance_mean"])


def log_eval_metrics(writer, prefix, eval_metrics, step):
    log_scalar(writer, f"{prefix}/mean_return", eval_metrics["return_mean"], step)
    log_scalar(writer, f"{prefix}/return_std", eval_metrics["return_std"], step)
    log_scalar(writer, f"{prefix}/mean_success", eval_metrics["success_mean"], step)
    log_scalar(writer, f"{prefix}/success_std", eval_metrics["success_std"], step)
    log_scalar(writer, f"{prefix}/fetch_success", eval_metrics["success_mean"], step)
    log_scalar(writer, f"{prefix}/fetch_success_std", eval_metrics["success_std"], step)
    log_scalar(writer, f"{prefix}/gc_success", eval_metrics["gc_success_mean"], step)
    log_scalar(writer, f"{prefix}/gc_success_std", eval_metrics["gc_success_std"], step)
    log_scalar(writer, f"{prefix}/gc_final_goal_distance", eval_metrics["gc_final_goal_distance_mean"], step)
    log_scalar(writer, f"{prefix}/gc_final_goal_distance_std", eval_metrics["gc_final_goal_distance_std"], step)
    log_scalar(writer, f"{prefix}/gc_min_goal_distance", eval_metrics["gc_min_goal_distance_mean"], step)
    log_scalar(writer, f"{prefix}/gc_min_goal_distance_std", eval_metrics["gc_min_goal_distance_std"], step)
    log_scalar(writer, f"{prefix}/gc_mean_goal_distance", eval_metrics["gc_mean_goal_distance_mean"], step)
    log_scalar(writer, f"{prefix}/gc_mean_goal_distance_std", eval_metrics["gc_mean_goal_distance_std"], step)


def vector_observation_space(observation_space):
    spaces = getattr(observation_space, "spaces", None)
    if spaces is not None and "observation" in spaces:
        return spaces["observation"]
    return observation_space


def make_student_qrl_conf(args):
    qrl_conf = QRLConf()
    qrl_conf.num_critics = args.student_qrl_num_critics
    qrl_conf.actor.losses.actor_optim.lr = args.student_qrl_actor_lr
    qrl_conf.actor.losses.entropy_weight_optim.lr = args.student_qrl_entropy_lr
    qrl_conf.actor.losses.min_dist.adaptive_entropy_regularizer = bool(args.student_qrl_adaptive_entropy)
    qrl_conf.actor.losses.min_dist.add_goal_as_future_state = bool(args.student_qrl_add_goal_as_future_state)
    qrl_conf.actor.losses.behavior_cloning.weight = args.student_qrl_bc_weight
    qrl_conf.quasimetric_critic.model.encoder.latent_size = args.student_qrl_encoder_latent_size
    qrl_conf.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec = args.student_qrl_head_spec
    qrl_conf.quasimetric_critic.losses.critic_optim.lr = args.student_qrl_critic_lr
    qrl_conf.quasimetric_critic.losses.lagrange_mult_optim.lr = args.student_qrl_lagrange_lr
    qrl_conf.quasimetric_critic.losses.local_constraint.epsilon = args.student_qrl_local_epsilon
    qrl_conf.quasimetric_critic.losses.local_constraint.step_cost = args.student_qrl_step_cost
    qrl_conf.quasimetric_critic.losses.global_push.softplus_offset = args.student_qrl_global_push_offset
    qrl_conf.quasimetric_critic.losses.latent_dynamics.weight = args.student_qrl_dynamics_weight
    return qrl_conf


class OnlineQRLStudentAgent:
    def __init__(self, observation_space, action_space, device, batch_size, total_optim_steps, args):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.goal_discount = args.student_qrl_future_discount
        self.goal_next_state_sample = args.student_qrl_next_state_sample
        self.goal_reward_type = args.goal_reward_type
        self.behavior_goal_success_only = bool(args.behavior_goal_success_only)
        self.exploration_eps = args.student_qrl_exploration_eps
        self.actor_critic_goal_dim = observation_space.shape[0]
        self.behavior_goal = None

        env_spec = EnvSpec(
            observation_space=observation_space,
            observation_space_is_dict=False,
            action_space=action_space,
        )
        self.qrl_conf = make_student_qrl_conf(args)
        self.qrl_agent, self.qrl_losses = self.qrl_conf.make(
            env_spec=env_spec,
            total_optim_steps=max(1, int(total_optim_steps)),
        )
        self.qrl_agent.to(self.device)
        self.qrl_losses.to(self.device)
        self.actor = self.qrl_agent.actor
        self.action_low = torch.as_tensor(action_space.low, device=self.device).float()
        self.action_high = torch.as_tensor(action_space.high, device=self.device).float()
        self.train()

    def train(self, training=True):
        self.training = training
        self.qrl_agent.train(training)
        self.qrl_losses.train(training)
        return self

    def eval(self):
        return self.train(False)

    def set_behavior_goal(self, goal_obs):
        if goal_obs is None:
            self.behavior_goal = None
        else:
            self.behavior_goal = np.array(goal_obs, copy=True)

    def _to_tensor(self, value):
        value = torch.as_tensor(value, device=self.device).float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    def _prepare_goal(self, batch_size, goal_obs=None, detach=True):
        if goal_obs is None:
            goal_obs = self.behavior_goal
        if goal_obs is None:
            goal = torch.zeros(batch_size, self.actor_critic_goal_dim, device=self.device)
        else:
            goal = self._to_tensor(goal_obs)
            if goal.shape[-1] != self.actor_critic_goal_dim:
                raise ValueError(
                    f"Goal dim {goal.shape[-1]} does not match online QRL observation dim "
                    f"{self.actor_critic_goal_dim}. Use --fetch_goal_format online for Fetch task streams."
                )
            if goal.shape[0] == 1 and batch_size != 1:
                goal = goal.repeat(batch_size, 1)
        return goal.detach() if detach else goal

    def actor_distribution(self, obs, goal_obs=None, detach_goal=True):
        obs = torch.as_tensor(obs, device=self.device).float()
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        goal = self._prepare_goal(obs.shape[0], goal_obs=goal_obs, detach=detach_goal)
        return self.actor(obs, goal)

    def act(self, obs, sample=False, goal_obs=None):
        with torch.inference_mode(False), torch.no_grad():
            obs = self._to_tensor(obs)
            goal = self._prepare_goal(obs.shape[0], goal_obs=goal_obs, detach=True)
            dist = self.actor(obs, goal)
            action = dist.sample() if sample else distribution_mean_action(dist)
            if sample and self.exploration_eps > 0:
                action = action + torch.randn_like(action) * self.exploration_eps
            action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.detach().cpu().numpy()[0]

    def _sample_batch(self, replay_buffer):
        if hasattr(replay_buffer, "sample_metric_batch"):
            batch = replay_buffer.sample_metric_batch(
                batch_size=self.batch_size,
                discount=self.goal_discount,
                next_state_sample=self.goal_next_state_sample,
                reward_type=self.goal_reward_type,
            )
        else:
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(self.batch_size)
            del success
            obs, action, reward, _, next_obs, not_done_no_max = replay_buffer.as_torch(
                obs,
                action,
                reward,
                np.zeros_like(reward),
                next_obs,
                not_done_no_max,
            )
            batch = {
                "obses": obs,
                "actions": action,
                "env_rewards": reward,
                "next_obses": next_obs,
                "not_dones_no_max": not_done_no_max,
                "goals": next_obs,
                "goal_steps": torch.ones_like(reward),
            }
        return {
            key: value.to(self.device).float() if torch.is_tensor(value) else torch.as_tensor(value, device=self.device).float()
            for key, value in batch.items()
        }

    def _to_batch_data(self, batch):
        terminals = 1.0 - batch["not_dones_no_max"]
        timeouts = torch.zeros_like(terminals)
        return BatchData(
            observations=batch["obses"],
            actions=batch["actions"],
            next_observations=batch["next_obses"],
            rewards=batch["env_rewards"],
            terminals=terminals,
            timeouts=timeouts,
            future_observations=batch["goals"],
        )

    def _refresh_behavior_goal(self, replay_buffer, batch):
        goal_obs = None
        if hasattr(replay_buffer, "sample_behavior_goal"):
            goal_obs = replay_buffer.sample_behavior_goal(
                discount=self.goal_discount,
                success_only=self.behavior_goal_success_only,
            )
        if goal_obs is None and batch.get("goals") is not None and batch["goals"].shape[0] > 0:
            goal_obs = batch["goals"][0].detach().cpu().numpy()
        if goal_obs is not None:
            self.set_behavior_goal(goal_obs)

    def update(self, replay_buffer, step):
        del step
        batch = self._sample_batch(replay_buffer)
        result = self.qrl_losses(self.qrl_agent, self._to_batch_data(batch), optimize=True)
        self._refresh_behavior_goal(replay_buffer, batch)
        info = dict(result.info)
        info["loss"] = result.loss.detach() if torch.is_tensor(result.loss) else result.loss
        info["behavior_goal_ready"] = float(self.behavior_goal is not None)
        if "goal_steps" in batch:
            info["goal_steps"] = batch["goal_steps"].mean().detach()
        return info

    def sac_update(self, replay_buffer, step):
        return self.update(replay_buffer, step)

    def distill_actor_from_agent(self, teacher_agent, replay_buffer, goal_buffer, update_num):
        print_interval = max(update_num // 5, 1)
        last_goal_obs = None
        last_metrics = {}
        for idx in range(update_num):
            obs, _, _, _, _, _ = replay_buffer.sample(self.batch_size)
            obs = torch.as_tensor(obs, device=self.device).float()

            goal_obs = None
            if hasattr(goal_buffer, "sample_behavior_goal"):
                goal_obs = goal_buffer.sample_behavior_goal(
                    discount=getattr(teacher_agent, "goal_discount", self.goal_discount),
                    success_only=getattr(teacher_agent, "behavior_goal_success_only", True),
                    batch_size=self.batch_size,
                )
            if goal_obs is None:
                goal_obs = getattr(teacher_agent, "behavior_goal", None)
            else:
                last_goal_obs = goal_obs

            with torch.no_grad():
                teacher_dist = actor_distribution(teacher_agent, obs, goal_obs=goal_obs, detach_goal=True)
                teacher_action = distribution_mean_action(teacher_dist).detach()

            student_dist = self.actor_distribution(obs, goal_obs=goal_obs, detach_goal=True)
            student_action = distribution_mean_action(student_dist)
            actor_loss = torch.square(student_action - teacher_action).sum(-1).mean()

            actor_loss_module = self.qrl_losses.actor_loss
            with actor_loss_module.actor_optim.update_context(optimize=True):
                actor_loss.backward()
            actor_loss_module.actor_sched.step()

            last_metrics = {
                "actor_distill_loss": float(actor_loss.item()),
                "teacher_action": float(teacher_action.mean().item()),
                "student_action": float(student_action.mean().item()),
                "teacher_goal_ready": float(goal_obs is not None),
            }
            if idx % print_interval == 0:
                print("actor_distill:", idx, last_metrics)

        if last_goal_obs is not None:
            behavior_goal = last_goal_obs[0] if np.asarray(last_goal_obs).ndim > 1 else last_goal_obs
            if hasattr(teacher_agent, "set_behavior_goal"):
                teacher_agent.set_behavior_goal(behavior_goal)
            self.set_behavior_goal(behavior_goal)
        return last_metrics

    def copy_state_from(self, source_agent):
        self.qrl_agent.load_state_dict(source_agent.qrl_agent.state_dict())
        self.qrl_losses.load_state_dict(source_agent.qrl_losses.state_dict())
        self.set_behavior_goal(source_agent.behavior_goal)

    def copy_actor_state_from(self, source_agent):
        self.qrl_agent.actor.load_state_dict(source_agent.qrl_agent.actor.state_dict())
        self.qrl_losses.actor_loss.actor_optim.load_state_dict(source_agent.qrl_losses.actor_loss.actor_optim.state_dict())
        self.qrl_losses.actor_loss.actor_sched.load_state_dict(source_agent.qrl_losses.actor_loss.actor_sched.state_dict())
        self.qrl_losses.actor_loss.entropy_weight_optim.load_state_dict(
            source_agent.qrl_losses.actor_loss.entropy_weight_optim.state_dict()
        )
        self.qrl_losses.actor_loss.entropy_weight_sched.load_state_dict(
            source_agent.qrl_losses.actor_loss.entropy_weight_sched.state_dict()
        )
        self.set_behavior_goal(source_agent.behavior_goal)

    def save(self, model_dir, model_name):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "agent": self.qrl_agent.state_dict(),
                "losses": self.qrl_losses.state_dict(),
                "behavior_goal": self.behavior_goal,
            },
            os.path.join(model_dir, f"{model_name}_online_qrl.pt"),
        )

    def load(self, model_dir, model_name):
        payload = torch.load(os.path.join(model_dir, f"{model_name}_online_qrl.pt"), map_location=self.device, weights_only=False)
        self.qrl_agent.load_state_dict(payload["agent"])
        self.qrl_losses.load_state_dict(payload["losses"])
        self.set_behavior_goal(payload.get("behavior_goal"))


def build_student_agent(observation_space, action_space, device, args, total_optim_steps):
    return OnlineQRLStudentAgent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        batch_size=args.batch_size,
        total_optim_steps=total_optim_steps,
        args=args,
    )


def build_meta_agent(obs_dim, action_dim, device, args):
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
        backup_coef=getattr(args, "qm_backup_coef", 1.0),
        diag_backup=getattr(args, "qm_diag_backup", 1.0),
        action_invariance_coef=args.qm_action_invariance_coef,
        transition_consistency_coef=args.qm_transition_consistency_coef,
        contrastive_coef=args.qm_contrastive_coef,
        ranking_coef=args.qm_ranking_coef,
        ranking_margin=args.qm_ranking_margin,
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
        q_loss_coef=getattr(args, "qm_q_loss_coef", 1.0),
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


def copy_sac_state(source_agent, target_agent):
    if hasattr(target_agent, "copy_state_from"):
        target_agent.copy_state_from(source_agent)
        return
    target_agent.critic.load_state_dict(source_agent.critic.state_dict())
    target_agent.critic_target.load_state_dict(source_agent.critic_target.state_dict())
    target_agent.actor.load_state_dict(source_agent.actor.state_dict())
    if hasattr(source_agent, "goal_encoder") and hasattr(target_agent, "goal_encoder"):
        if source_agent.goal_encoder is not None and target_agent.goal_encoder is not None:
            target_agent.goal_encoder.load_state_dict(source_agent.goal_encoder.state_dict())
    target_agent.log_alpha.data.copy_(source_agent.log_alpha.data)
    target_agent.critic_optimizer.load_state_dict(source_agent.critic_optimizer.state_dict())
    target_agent.actor_optimizer.load_state_dict(source_agent.actor_optimizer.state_dict())
    target_agent.log_alpha_optimizer.load_state_dict(source_agent.log_alpha_optimizer.state_dict())
    if hasattr(source_agent, "behavior_goal") and hasattr(target_agent, "set_behavior_goal"):
        target_agent.set_behavior_goal(source_agent.behavior_goal)


def copy_actor_state(source_agent, target_agent):
    if hasattr(target_agent, "copy_actor_state_from"):
        target_agent.copy_actor_state_from(source_agent)
        return
    target_agent.actor.load_state_dict(source_agent.actor.state_dict())
    target_agent.actor_optimizer.load_state_dict(source_agent.actor_optimizer.state_dict())
    if hasattr(source_agent, "behavior_goal") and hasattr(target_agent, "set_behavior_goal"):
        target_agent.set_behavior_goal(source_agent.behavior_goal)


def chronological_indices(replay_buffer):
    size = len(replay_buffer)
    if size == 0:
        return np.array([], dtype=np.int64)
    if not replay_buffer.full:
        return np.arange(size, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(replay_buffer.idx, replay_buffer.capacity, dtype=np.int64),
            np.arange(0, replay_buffer.idx, dtype=np.int64),
        ]
    )


def copy_recent_trajectories(source_buffer, target_buffer, store_traj_num): 
    indices = chronological_indices(source_buffer)
    if indices.size == 0:
        return 0, 0

    done_positions = np.flatnonzero(source_buffer.not_dones[indices].reshape(-1) < 0.5)
    if done_positions.size >= store_traj_num + 1:
        start_pos = int(done_positions[-store_traj_num - 1] + 1)
    else:
        start_pos = 0
    selected_indices = indices[start_pos:]

    count_success = 0
    episode_success = False
    for source_idx in selected_indices:
        episode_success = episode_success or bool(source_buffer.successes[source_idx, 0] > 0.5)
        target_buffer.add(
            source_buffer.obses[source_idx],
            source_buffer.actions[source_idx],
            source_buffer.rewards[source_idx],
            source_buffer.successes[source_idx],
            source_buffer.next_obses[source_idx],
            not bool(source_buffer.not_dones[source_idx, 0]),
            not bool(source_buffer.not_dones_no_max[source_idx, 0]),
        )
        if source_buffer.not_dones[source_idx, 0] < 0.5:
            count_success += int(episode_success)
            episode_success = False

    return int(selected_indices.size), count_success


def write_stats_csv(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(stats.keys())
    if not keys:
        open(path, "w").close()
        return

    row_count = max((len(values) for values in stats.values()), default=0)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row_idx in range(row_count):
            row = {}
            for key in keys:
                values = stats[key]
                row[key] = values[row_idx] if row_idx < len(values) else ""
            writer.writerow(row)


def resolve_env_selection(args):
    env_sequence = args.env_sequence
    base_task_name = args.base_task_name
    env_key = normalize_key(args.env)

    if args.task_order is not None:
        return env_sequence, base_task_name
    if base_task_name is not None:
        return env_sequence, base_task_name
    if env_key.startswith("fetch-sequence-"):
        env_suffix = env_key[len("fetch-sequence-") :]
        if env_suffix and env_suffix != "custom":
            env_sequence = env_suffix
    elif env_key.startswith("fetch-"):
        base_task_name = env_key[len("fetch-") :]
    elif env_key in FETCH_TASK_NAMES:
        base_task_name = env_key
    return env_sequence, base_task_name


def make_env_kwargs(args):
    env_sequence, base_task_name = resolve_env_selection(args)
    return {
        "env_sequence": env_sequence,
        "base_task_name": base_task_name,
        "task_order": args.task_order,
        "change_freq": args.change_freq,
        "goal_conditioned": bool(args.goal_conditioned),
        "gc_reward_type": args.gc_reward_type,
        "gc_success_threshold": args.gc_success_threshold,
        "fetch_goal_format": args.fetch_goal_format,
        "fetch_env_version": args.fetch_env_version,
        "max_episode_steps": args.max_episode_steps,
        "slide_goal_scale": args.slide_goal_scale,
        "normalize_obs": args.normalize_obs,
        "normalize_avg_coef": args.normalize_avg_coef,
        "reset_obs_stats": bool(args.reset_obs_stats),
        "change_when_solved": bool(args.change_when_solved),
        "seed": args.seed,
    }


def evaluate_and_log(env, agent, writer, prefix, step, num_eval_runs, reseed_each_episode=True):
    eval_results = env.evaluate_agent(
        agent,
        num_eval_runs,
        reseed_each_episode=reseed_each_episode,
    )
    eval_metrics = summarize_eval_results(eval_results)
    log_eval_metrics(writer, prefix, eval_metrics, step)
    return eval_metrics


def add_common_args(parser):
    parser.add_argument("--method", type=str, default="buffer", choices=["independent", "continue", "buffer", "buffer_wd", "average"])
    parser.add_argument("--store_traj_num", type=int, default=2000)
    parser.add_argument(
        "--transfer_traj_num",
        type=int,
        default=20,
        help="Trajectory-equivalent budget used when distilling the meta actor into a new task actor.",
    )
    parser.add_argument("--meta_updates_per_traj", type=int, default=100)
    parser.add_argument("--use_ttest", type=int, default=0)

    parser.add_argument("--env", type=str, default="fetch_sequence_set1")
    parser.add_argument("--env_sequence", type=str2none, default="set1")
    parser.add_argument("--base_task_name", type=str2none, default=None)
    parser.add_argument("--task_order", type=str2none, default=None)
    parser.add_argument("--change_freq", type=int, default=1000000)
    parser.add_argument("--normalize_obs", type=str2none, default=None)
    parser.add_argument("--normalize_avg_coef", type=float, default=0.0001)
    parser.add_argument("--reset_obs_stats", type=int, default=0)
    parser.add_argument("--change_when_solved", type=int, default=0)
    parser.add_argument("--goal_conditioned", type=int, default=1)
    parser.add_argument("--gc_reward_type", type=str_choice, default="sparse", choices=["sparse", "dense", "success"])
    parser.add_argument("--gc_success_threshold", type=float, default=0.05)
    parser.add_argument("--fetch_goal_format", type=str_choice, default="online", choices=["online", "native"])
    parser.add_argument("--fetch_env_version", type=str2none, default="auto")
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument(
        "--slide_goal_scale",
        type=float,
        default=1.0,
        help="Scale FetchSlide target distance toward the initial object position; use 0.5 for an easier task.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--save_freq", type=int, default=25000)
    parser.add_argument("--save_model_freq", type=int, default=-1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--random_steps", type=int, default=10000)
    parser.add_argument("--num_eval_runs", type=int, default=10)
    parser.add_argument("--log_backends", type=str.lower, nargs="+", default=["wandb"], choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--wandb_project_name", type=str, default="continual-quasimetric-fetch")
    parser.add_argument("--wandb_entity", type=str2none, default=None)
    parser.add_argument("--wandb_group", type=str2none, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])


def add_sac_args(parser):
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--alpha_lr", type=float, default=1e-4)
    parser.add_argument("--critic_tau", type=float, default=0.005)
    parser.add_argument("--actor_update_frequency", type=int, default=1)
    parser.add_argument("--critic_target_update_frequency", type=int, default=1)


def add_student_qrl_args(parser):
    parser.add_argument("--student_qrl_num_critics", type=int, default=2)
    parser.add_argument("--student_qrl_actor_lr", type=float, default=3e-5)
    parser.add_argument("--student_qrl_entropy_lr", type=float, default=3e-4)
    parser.add_argument("--student_qrl_critic_lr", type=float, default=1e-4)
    parser.add_argument("--student_qrl_lagrange_lr", type=float, default=1e-2)
    parser.add_argument("--student_qrl_encoder_latent_size", type=int, default=128)
    parser.add_argument("--student_qrl_head_spec", type=str, default="iqe(dim=2048,components=64)")
    parser.add_argument("--student_qrl_local_epsilon", type=float, default=0.25)
    parser.add_argument("--student_qrl_step_cost", type=float, default=1.0)
    parser.add_argument("--student_qrl_global_push_offset", type=float, default=15.0)
    parser.add_argument("--student_qrl_dynamics_weight", type=float, default=0.1)
    parser.add_argument("--student_qrl_adaptive_entropy", type=int, default=1)
    parser.add_argument("--student_qrl_add_goal_as_future_state", type=int, default=1)
    parser.add_argument("--student_qrl_bc_weight", type=float, default=0.0)
    parser.add_argument("--student_qrl_future_discount", type=float, default=0.995)
    parser.add_argument("--student_qrl_next_state_sample", type=float, default=0.2)
    parser.add_argument("--student_qrl_exploration_eps", type=float, default=0.3)


def add_quasimetric_args(parser):
    parser.add_argument("--qm_latent_dim", type=int, default=256)
    parser.add_argument(
        "--qm_transition_input",
        "--qm-transition-input",
        choices=("state", "latent"),
        default="state",
        help="Use raw observations (T(s,a)) or latent states (T(z,a)).",
    )
    parser.add_argument("--qm_hidden_dim", type=int, default=256)
    parser.add_argument("--qm_hidden_depth", type=int, default=2)
    parser.add_argument("--qm_components", type=int, default=8)
    parser.add_argument("--qm_batch_size", type=int, default=256)
    parser.add_argument("--qm_share_sac_batch", type=int, default=0)
    parser.add_argument("--qm_shared_batch_size", type=int, default=None)
    parser.add_argument("--qm_lr", type=float, default=1e-4)
    parser.add_argument("--qm_discount", type=float, default=0.995)
    parser.add_argument("--qm_lambda", type=float, default=0.95)
    parser.add_argument("--qm_next_state_sample", type=float, default=0.2)
    parser.add_argument("--qm_backup_clip", type=float, default=5.0)
    parser.add_argument("--qm_backup_coef", type=float, default=1.0)
    parser.add_argument("--qm_diag_backup", type=float, default=1.0)
    parser.add_argument("--qm_action_invariance_coef", type=float, default=1.0)
    parser.add_argument("--qm_transition_consistency_coef", type=float, default=1.0)
    parser.add_argument("--qm_contrastive_coef", type=float, default=0.05)
    parser.add_argument("--qm_ranking_coef", type=float, default=0.0)
    parser.add_argument("--qm_ranking_margin", type=float, default=0.1)
    parser.add_argument(
        "--qm_nce_mode",
        choices=("forward_nce", "backward_nce"),
        default="forward_nce",
    )
    parser.add_argument("--qm_target_tau", type=float, default=0.01)
    parser.add_argument("--qm_current_batch_ratio", type=float, default=0.5)
    parser.add_argument("--qm_max_grad_norm", type=float, default=None)
    parser.add_argument("--qm_min_buffer_size", type=int, default=256)

    parser.add_argument("--structure_update_frequency", type=int, default=1)
    parser.add_argument("--structure_updates_per_step", type=int, default=1)
    parser.add_argument("--structure_bonus_coef", type=float, default=0.0)
    parser.add_argument("--qm_q_loss_coef", type=float, default=1.0)
    parser.add_argument("--bc_alpha", type=float, default=1.0)
    parser.add_argument("--memory_max_tasks", type=int, default=None)
    parser.add_argument("--memory_max_transitions_per_task", type=int, default=None)
    parser.add_argument("--goal_reward_scale", type=float, default=1.0)
    parser.add_argument("--task_reward_scale", type=float, default=0.0)
    parser.add_argument("--goal_reward_type", type=str, default="step_cost", choices=["binary", "step_cost"])
    parser.add_argument("--behavior_goal_success_only", type=int, default=1)
    parser.add_argument("--encode_actor_critic_goal", type=int, default=0)
    parser.add_argument("--replay_buffer_mode", type=str, default="her", choices=["her", "no_her"])


def parse_args():
    parser = argparse.ArgumentParser(description="Run online continual quasimetric RL on Fetch task streams")
    add_common_args(parser)
    add_sac_args(parser)
    add_student_qrl_args(parser)
    add_quasimetric_args(parser)

    args = parser.parse_args()
    try:
        args.log_backends = normalize_log_backends(args.log_backends)
    except ValueError as error:
        parser.error(str(error))
    return args


def quasimetric_buffer_path(save_path, log_name, env, completed_task_idx):
    completed_task_name = str(
        env.env_list[(completed_task_idx - 1) % len(env.env_list)]
    ).replace("/", "_")
    return os.path.join(
        save_path,
        "quasimetric_buffers",
        log_name,
        f"task_{completed_task_idx:02d}_{completed_task_name}",
    )


def handle_task_switch(
    args,
    env,
    eval_env,
    writer,
    replay_buffer,
    quasimetric_buffer,
    agent,
    meta_agent,
    previous_task_counter,
    step,
    model_dir,
    log_name,
):
    copied_transitions, count_success = copy_recent_trajectories(
        replay_buffer,
        quasimetric_buffer,
        args.store_traj_num,
    )
    print(
        "quasimetric_buffer idx:",
        quasimetric_buffer.idx,
        "copied_transitions:",
        copied_transitions,
        "count_success:",
        count_success,
    )
    buffer_path = quasimetric_buffer_path(
        args.save_path,
        log_name,
        env,
        previous_task_counter,
    )
    quasimetric_buffer.save_data(buffer_path)
    log_scalar(writer, "meta/buffer_size", len(quasimetric_buffer), step)
    log_scalar(writer, "meta/copied_transitions", copied_transitions, step)
    log_scalar(writer, "meta/count_success", count_success, step)

    update_num = max(1, args.store_traj_num * previous_task_counter * args.meta_updates_per_traj)
    if args.method == "buffer":
        print("---- updating quasimetric critic and actor ----")
        structure_metrics = {}
        meta_actor_log = {}
        for _ in range(update_num):
            structure_metrics, meta_actor_log = meta_agent.update_structure_and_actor(
                quasimetric_buffer
            )

        log_metric_dict(writer, "metric_structure", structure_metrics, step)
        log_metric_dict(writer, "meta_actor", meta_actor_log, step)

    eval_env.set_task(env.env_list[(previous_task_counter - 1) % len(env.env_list)])
    eval_metrics = evaluate_and_log(eval_env, meta_agent, writer, "meta_eval", step, args.num_eval_runs)
    print(
        f"meta_agent: task {eval_env.base_task_name}, "
        f"success {eval_metrics['success_mean']:.3f} +/- {eval_metrics['success_std']:.3f}, "
        f"gc_success {eval_metrics['gc_success_mean']:.3f} +/- {eval_metrics['gc_success_std']:.3f}, "
        f"return {eval_metrics['return_mean']:.3f} +/- {eval_metrics['return_std']:.3f}"
    )

    agent.save(model_dir, f"{log_name}_task{previous_task_counter}")
    meta_agent.save(model_dir, f"{log_name}_task{previous_task_counter}_meta")
    return count_success


def train_average_agent(env, replay_buffer, agent, agent_list, task_counter, step, writer):
    if not hasattr(agent, "compute_target_q") or not hasattr(agent, "update_with_target_q"):
        raise NotImplementedError("The average method requires SAC-style Q targets and is not defined for online QRL students.")
    obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(agent.batch_size)
    del success
    obs, action, reward, _, next_obs, not_done_no_max = replay_buffer.as_torch(
        obs,
        action,
        reward,
        np.zeros_like(reward),
        next_obs,
        not_done_no_max,
    )
    agent_num = len(env.env_list) if task_counter >= len(env.env_list) else task_counter
    q_target = agent.compute_target_q(reward, next_obs, not_done_no_max) / agent_num
    for agent_idx in range(agent_num - 1):
        q_target += agent_list[agent_idx].compute_target_q(reward, next_obs, not_done_no_max) / agent_num
    train_metrics = agent.update_with_target_q(obs, action, q_target, step)
    log_scalar(writer, "train/average_agent_num", agent_num, step)
    return train_metrics


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    os.makedirs(args.save_path, exist_ok=True)
    set_seed_everywhere(args.seed)

    env_kwargs = make_env_kwargs(args)
    env = FetchGoalEnvSequence(**env_kwargs)
    eval_env = FetchGoalEnvSequence(**env_kwargs)
    print("env_list:", env.env_list)

    num_steps_per_run = len(env.env_list) * args.change_freq
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()

    order_name = args.task_order or env_kwargs["env_sequence"] or env_kwargs["base_task_name"] or args.env
    task_tag = safe_tag(str(order_name).replace(",", "-"))
    log_name = f"fetch_continual_{task_tag}_seed{args.seed}_{args.method}_gc-{args.gc_reward_type}"
    if args.slide_goal_scale != 1.0:
        log_name += f"_slide-scale{safe_tag(args.slide_goal_scale)}"
    run_dir = os.path.join(args.save_path, log_name)
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    writer, log_info = create_experiment_logger(args, log_name, run_dir)

    if log_info["active_backends"]:
        print("log_backends:", ", ".join(log_info["active_backends"]))
    else:
        print("log_backends: disabled")
    for key in ("tensorboard_path", "wandb_path", "wandb_url"):
        if log_info[key] is not None:
            print(f"{key}:", log_info[key])
    print("run_dir:", run_dir)

    obs_space = vector_observation_space(env.env.observation_space)
    obs_dim = obs_space.shape[0]
    action_shape = env.env.action_space.shape
    action_dim = action_shape[0]
    replay_buffer_capacity = int(args.change_freq) + int(args.random_steps)
    replay_buffer = ReplayBufferMetric(obs_space.shape, action_shape, replay_buffer_capacity, device)

    log_scalar(writer, "config/goal_conditioned", int(bool(args.goal_conditioned)), 0)
    log_scalar(writer, "config/obs_dim", obs_dim, 0)
    log_scalar(writer, "config/action_dim", action_dim, 0)
    log_scalar(writer, "config/gc_success_threshold", args.gc_success_threshold, 0)
    log_scalar(writer, "config/slide_goal_scale", args.slide_goal_scale, 0)
    log_scalar(writer, "config/task_count", len(env.env_list), 0)

    agent_list = [
        build_student_agent(
            observation_space=obs_space,
            action_space=env.env.action_space,
            device=device,
            args=args,
            total_optim_steps=num_steps_per_run,
        )
        for _ in range(len(env.env_list))
    ]

    use_meta = args.method in {"buffer", "buffer_wd"}
    meta_agent_list = []
    quasimetric_buffer = None
    meta_agent = None
    from_meta = False
    if use_meta:
        meta_agent_list = [
            build_meta_agent(obs_dim=obs_dim, action_dim=action_dim, device=device, args=args)
            for _ in range(len(env.env_list))
        ]
        meta_buffer_cls = ReplayBufferMetric if args.replay_buffer_mode == "her" else ReplayBufferMetricNoHER
        quasimetric_buffer = meta_buffer_cls(obs_space.shape, action_shape, replay_buffer_capacity, device)

    collector = Collector(env, replay_buffer)
    env.reset()

    task_counter = env.task_counter
    agent = agent_list[task_counter - 1]
    if use_meta:
        meta_agent = meta_agent_list[task_counter - 1]
        if hasattr(agent, "set_quasimetric_agent"):
            agent.set_quasimetric_agent(meta_agent)

    collector.initial_collect(args.random_steps)

    intermediate_stats = defaultdict(list)
    step = 0
    count_success = -1
    task_start_step = 0

    while step <= num_steps_per_run:
        if task_counter != env.task_counter:
            previous_task_counter = task_counter
            task_counter = env.task_counter
            task_start_step = step
            log_scalar(writer, "task/task_idx", task_counter, step)

            if use_meta:
                count_success = handle_task_switch(
                    args,
                    env,
                    eval_env,
                    writer,
                    replay_buffer,
                    quasimetric_buffer,
                    agent,
                    meta_agent,
                    previous_task_counter,
                    step,
                    model_dir,
                    log_name,
                )
            else:
                agent.save(model_dir, f"{log_name}_task{previous_task_counter}")

            if step == num_steps_per_run:
                break

            replay_buffer.reset()

            if args.method == "continue":
                next_agent = agent_list[(task_counter - 1) % len(env.env_list)]
                copy_sac_state(agent, next_agent)
                agent = next_agent
                collector.initial_collect(args.random_steps)
            elif use_meta:
                agent = agent_list[(task_counter - 1) % len(env.env_list)]

                if hasattr(agent, "set_quasimetric_agent"):
                    agent.set_quasimetric_agent(meta_agent)
                if hasattr(agent, "set_task_start_step"):
                    agent.set_task_start_step(task_start_step)

                collector.initial_agent_collect(args.random_steps, [agent], args.num_eval_runs)
                print("---- distill goal-conditioned meta actor into the task-specific actor ----")
                distill_update_num = max(1, args.transfer_traj_num * args.meta_updates_per_traj * task_counter)
                print(
                    "distillation trajectory budget:",
                    args.transfer_traj_num,
                    "updates:",
                    distill_update_num,
                )
                log_scalar(writer, "meta_distill/trajectory_budget", args.transfer_traj_num, step)
                log_scalar(writer, "meta_distill/update_budget", distill_update_num, step)
                actor_distill_metrics = distill_meta_actor_to_agent(
                    agent,
                    meta_agent,
                    replay_buffer,
                    quasimetric_buffer,
                    distill_update_num,
                )
                log_metric_dict(writer, "meta_distill", actor_distill_metrics, step)
                from_meta = True
                print("--------- distillation done ---------")
                print("--------- reset buffer then collect by new agent (planning with structure prior) ---------")

                replay_buffer.reset()
                collector.initial_agent_collect(args.random_steps, [agent], args.num_eval_runs)
                log_scalar(writer, "meta/from_meta", int(from_meta), step)

                next_meta_agent = meta_agent_list[(task_counter - 1) % len(env.env_list)]
                copy_actor_state(meta_agent, next_meta_agent)
                meta_agent = next_meta_agent
            else:
                agent = agent_list[(task_counter - 1) % len(env.env_list)]
                collector.initial_collect(args.random_steps)

        if args.method in {"independent", "continue", "buffer", "buffer_wd"}:
            train_metrics = agent.update(replay_buffer, step)
        elif args.method == "average":
            train_metrics = train_average_agent(env, replay_buffer, agent, agent_list, task_counter, step, writer)
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        collector.run_one_step(step, agent)
        log_metric_dict(writer, "train", train_metrics, step)
        log_scalar(writer, "train/task_idx", task_counter, step)
        log_scalar(writer, "train/agent_idx", (task_counter - 1) % len(env.env_list), step)
        log_scalar(writer, "train/replay_buffer_size", len(replay_buffer), step)

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
                "task_counter",
                (task_counter, env.base_task_name),
                "agent",
                (task_counter - 1) % len(env.env_list),
                "method",
                args.method,
                "seed",
                args.seed,
            )

            eval_metrics = evaluate_and_log(env, agent, writer, "eval", step, args.num_eval_runs)
            append_eval_stats(intermediate_stats, eval_metrics)
            intermediate_stats["steps"].append(step)
            intermediate_stats["task"].append(env.base_task_name)
            intermediate_stats["seed"].append(args.seed)
            intermediate_stats["task_idx"].append(task_counter)
            intermediate_stats["method"].append(args.method)
            intermediate_stats["time"].append(round(elapsed / 3600, 3))
            intermediate_stats["count_success"].append(count_success)
            if use_meta:
                intermediate_stats["from_meta"].append(from_meta)

            log_scalar(writer, "charts/SPS", sps, step)
            log_scalar(writer, "eval/task_idx", task_counter, step)
            log_scalar(writer, "eval/agent_idx", (task_counter - 1) % len(env.env_list), step)
            log_scalar(writer, "eval/count_success", count_success, step)
            if use_meta:
                log_scalar(writer, "eval/from_meta", int(from_meta), step)
            print(
                f"success {eval_metrics['success_mean']:.3f} +/- {eval_metrics['success_std']:.3f}, "
                f"gc_success {eval_metrics['gc_success_mean']:.3f} +/- {eval_metrics['gc_success_std']:.3f}, "
                f"eval return {eval_metrics['return_mean']:.3f} +/- {eval_metrics['return_std']:.3f}"
            )
            writer.flush()

        if args.save_model_freq > 0 and step > 0 and step % args.save_model_freq == 0:
            agent.save(model_dir, f"{log_name}_step{step}")
            if use_meta and meta_agent is not None:
                meta_agent.save(model_dir, f"{log_name}_step{step}_meta")

        step += 1

    write_stats_csv(os.path.join(run_dir, f"{log_name}.csv"), intermediate_stats)
    agent.save(model_dir, f"{log_name}_final")
    if use_meta and meta_agent is not None:
        meta_agent.save(model_dir, f"{log_name}_final_meta")
    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    sys.exit(main())
