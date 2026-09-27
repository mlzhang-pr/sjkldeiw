import argparse
import importlib
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import time
from agent.sac import SACAgent
from replay_buffer import Collector
from replay_buffer_metric import ReplayBufferMetric, ReplayBufferMetricNoHER
from agent.quasimetric import (
    ContinualQuasimetricAgentConfig,
    ContinualQuasimetricSACAgent,
    QuasimetricConfig,
)
from torch.distributions import Normal, kl_divergence
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import random
import copy
import shutil
import sys


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
            summary_key = tag.replace('/', '_')
            self.wandb_run.summary[summary_key] = text_string
            if global_step is not None:
                self.wandb_run.summary[f'{summary_key}_step'] = global_step

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

def set_seed_everywhere(seed_value):
    seed_value = int(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

class ConfigDictConverter:
    def __init__(self, config_dict):
        '''
        This class takes a config_dict which contains all the variables needed to do one run
        and converts it into the variables needed to run the RL experiments
        We assume that the config file has certain variables and is organized in the proper way
        For the env and agent parameters, we will pass config_dict on to them and assume that they handle it properly
        Note that we *cannot* use the same variable names for env and agent parameters. If the agent or env expect the
        same name, we will need to write it down different in the config file and then convert here

        Attributes:
        agent_dict:
        env_dict:
        '''
        # Improvement: possible to split agent and env parameters here. That is this class contains
        # two dicts for agent_parameters and env_parameters.
        # This would help with passing only the required arguments for envs that are already created (e.g gym)
        # Also, it could help with dealing with parameters that have the same name but are different for agent and env

        self.config_dict = config_dict.copy()

        # training shouldn't need these variables
        if 'num_repeats' in self.config_dict.keys():
            del self.config_dict['num_repeats']
        if 'num_runs_per_group' in self.config_dict.keys():
            del self.config_dict['num_runs_per_group']

        self.agent_dict = self.config_dict.copy()
        self.env_dict = self.config_dict.copy()

        # TODO remove maybe?
        self.repeat_idx = config_dict['repeat_idx']

        # environment
        # reinforcement learning
        import envs.metaworld_env
        self.env_class = envs.metaworld_env.MetaWorldSingleEnvSequence

        env_key_lst = ['env', 'base_task_name', 'seed', 'goal_hidden', 'normalize_obs', 'normalize_rewards',
                       'capture_video', 'save_name', 'change_freq', 'env_sequence',
                       'obs_drift_mean', 'obs_drift_std', 'obs_scale_drift', 'obs_noise_std', 'normalize_avg_coef',
                       'reset_obs_stats', 'change_when_solved', 'freeze_rand_vec',
                       'goal_conditioned', 'gc_reward_type', 'gc_success_threshold', 'gc_achieved_goal']
        env = config_dict['env'].lower()

        if env[0:19] == 'metaworld_sequence_':  # e.g. 'metaworld_sequence_reach'
            if env[19:22] == 'set':  # e.g. "metaworld_sequence_set1"
                self.env_dict['env_sequence'] = env[19:]  # e.g. "set1"
            else:
                self.env_dict['base_task_name'] = f'{env[19:]}-v2'

        self.env_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
        self.env_dict['env_type'] = 'rl'

        print('env_dict keys', self.env_dict.keys())
        print('env_dict', self.env_dict)

        # Other params
        self.agent_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Adjust the seed based on repeat
        self.env_dict['seed'] += self.repeat_idx * 1


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
        if backend == 'none':
            if len(backends) != 1:
                raise ValueError("'none' cannot be combined with other logging backends")
            return []
        if backend not in {'tensorboard', 'wandb'}:
            raise ValueError(f'Unsupported logging backend: {backend}')
        if backend not in normalized:
            normalized.append(backend)
    return normalized


def load_wandb_module():
    try:
        return importlib.import_module('wandb')
    except ImportError:
        return None


def create_experiment_logger(args, log_name):
    tensorboard_writer = None
    tensorboard_path = None
    wandb_run = None
    wandb_path = None
    wandb_url = None
    active_backends = []

    if 'tensorboard' in args.log_backends:
        if SummaryWriter is None:
            print("TensorBoard is unavailable. Install 'tensorboard' to enable event logging.")
        else:
            tensorboard_path = os.path.join(
                args.save_path,
                'tensorboard',
                args.env,
                log_name,
            )
            os.makedirs(tensorboard_path, exist_ok=True)
            tensorboard_writer = SummaryWriter(tensorboard_path)
            active_backends.append('tensorboard')

    if 'wandb' in args.log_backends:
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
                wandb_path = os.path.join(args.save_path, 'wandb')
                wandb_url = getattr(wandb_run, 'url', None)
                active_backends.append('wandb')
            except Exception as error:
                print(f'W&B initialization failed: {error}')

    logger = None
    if tensorboard_writer is not None or wandb_run is not None:
        logger = ExperimentLogger(
            tensorboard_writer=tensorboard_writer,
            wandb_run=wandb_run,
        )
        logger.add_text(
            'config/args',
            '\n'.join(f'{key}: {value}' for key, value in sorted(vars(args).items())),
            0,
        )

    return logger, {
        'active_backends': active_backends,
        'tensorboard_path': tensorboard_path,
        'wandb_path': wandb_path,
        'wandb_url': wandb_url,
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
    if writer is None or not metrics:
        return

    for key, value in metrics.items():
        log_scalar(writer, f'{prefix}/{key}', value, step)


def actor_distribution(agent, obs, goal_obs=None, detach_goal=True):
    if hasattr(agent, "actor_distribution"):
        return agent.actor_distribution(obs, goal_obs=goal_obs, detach_goal=detach_goal)
    if hasattr(agent, "_prepare_goal_rep") and hasattr(agent, "_augment_obs"):
        goal_rep = agent._prepare_goal_rep(obs.shape[0], goal_obs=goal_obs, detach=detach_goal)
        return agent.actor(agent._augment_obs(obs, goal_rep))
    return agent.actor(obs)


def mean_std(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def summarize_eval_results(eval_results):
    return_mean, return_std = mean_std(eval_results['episodic_returns'])
    metaworld_success_mean, metaworld_success_std = mean_std(eval_results['successes'])
    gc_success_mean, gc_success_std = mean_std(eval_results.get('goal_successes', []))
    gc_final_distance_mean, gc_final_distance_std = mean_std(eval_results.get('final_goal_distances', []))
    gc_min_distance_mean, gc_min_distance_std = mean_std(eval_results.get('min_goal_distances', []))
    gc_mean_distance_mean, gc_mean_distance_std = mean_std(eval_results.get('mean_goal_distances', []))
    return {
        'return_mean': return_mean,
        'return_std': return_std,
        'metaworld_success_mean': metaworld_success_mean,
        'metaworld_success_std': metaworld_success_std,
        'gc_success_mean': gc_success_mean,
        'gc_success_std': gc_success_std,
        'gc_final_goal_distance_mean': gc_final_distance_mean,
        'gc_final_goal_distance_std': gc_final_distance_std,
        'gc_min_goal_distance_mean': gc_min_distance_mean,
        'gc_min_goal_distance_std': gc_min_distance_std,
        'gc_mean_goal_distance_mean': gc_mean_distance_mean,
        'gc_mean_goal_distance_std': gc_mean_distance_std,
    }


def append_eval_stats(stats, eval_metrics):
    stats['mean_return'].append(eval_metrics['return_mean'])
    stats['mean_success'].append(eval_metrics['metaworld_success_mean'])
    stats['metaworld_success'].append(eval_metrics['metaworld_success_mean'])
    stats['gc_success'].append(eval_metrics['gc_success_mean'])
    stats['gc_final_goal_distance'].append(eval_metrics['gc_final_goal_distance_mean'])
    stats['gc_min_goal_distance'].append(eval_metrics['gc_min_goal_distance_mean'])
    stats['gc_mean_goal_distance'].append(eval_metrics['gc_mean_goal_distance_mean'])


def log_eval_metrics(writer, prefix, eval_metrics, step):
    log_scalar(writer, f'{prefix}/mean_return', eval_metrics['return_mean'], step)
    log_scalar(writer, f'{prefix}/return_std', eval_metrics['return_std'], step)
    log_scalar(writer, f'{prefix}/mean_success', eval_metrics['metaworld_success_mean'], step)
    log_scalar(writer, f'{prefix}/success_std', eval_metrics['metaworld_success_std'], step)
    log_scalar(writer, f'{prefix}/metaworld_success', eval_metrics['metaworld_success_mean'], step)
    log_scalar(writer, f'{prefix}/metaworld_success_std', eval_metrics['metaworld_success_std'], step)
    log_scalar(writer, f'{prefix}/gc_success', eval_metrics['gc_success_mean'], step)
    log_scalar(writer, f'{prefix}/gc_success_std', eval_metrics['gc_success_std'], step)
    log_scalar(writer, f'{prefix}/gc_final_goal_distance', eval_metrics['gc_final_goal_distance_mean'], step)
    log_scalar(writer, f'{prefix}/gc_final_goal_distance_std', eval_metrics['gc_final_goal_distance_std'], step)
    log_scalar(writer, f'{prefix}/gc_min_goal_distance', eval_metrics['gc_min_goal_distance_mean'], step)
    log_scalar(writer, f'{prefix}/gc_min_goal_distance_std', eval_metrics['gc_min_goal_distance_std'], step)
    log_scalar(writer, f'{prefix}/gc_mean_goal_distance', eval_metrics['gc_mean_goal_distance_mean'], step)
    log_scalar(writer, f'{prefix}/gc_mean_goal_distance_std', eval_metrics['gc_mean_goal_distance_std'], step)


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


class CQRLFastAgent(SACAgent):
    """Shared SAC learner with transient quasimetric-teacher guidance."""

    def update(self, replay_buffer, step, teacher_agent=None, regularization_weight=0.0):
        metrics = super().sac_update(replay_buffer, step)
        regularization_loss = torch.zeros((), device=self.device)

        if teacher_agent is not None and regularization_weight > 0.0:
            if hasattr(replay_buffer, "sample_metric_batch"):
                batch = replay_buffer.sample_metric_batch(
                    batch_size=self.batch_size,
                    discount=getattr(teacher_agent, "goal_discount", 0.995),
                    next_state_sample=getattr(teacher_agent, "goal_next_state_sample", 0.2),
                    reward_type=getattr(teacher_agent, "goal_reward_type", "step_cost"),
                )
                obs = batch["obses"]
                goal_obs = batch.get("goals")
            else:
                obs, _, _, _, _, _ = replay_buffer.sample(self.batch_size)
                obs = torch.as_tensor(obs, device=self.device).float()
                goal_obs = None

            student_dist = self.actor(obs)
            with torch.no_grad():
                teacher_dist = actor_distribution(
                    teacher_agent,
                    obs,
                    goal_obs=goal_obs,
                    detach_goal=True,
                )

            regularization_loss = kl_divergence(
                base_normal_distribution(student_dist),
                base_normal_distribution(teacher_dist),
            ).sum(-1).mean()
            self.actor_optimizer.zero_grad()
            (float(regularization_weight) * regularization_loss).backward()
            self.actor_optimizer.step()

        metrics["cqrl_kl"] = float(regularization_loss.item())
        metrics["cqrl_regularized"] = float(
            teacher_agent is not None and regularization_weight > 0.0
        )
        return metrics


def build_fast_agent(obs_dim, action_dim, device, args):
    return CQRLFastAgent(
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
    )


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


def copy_replay_buffer(source_buffer, target_buffer):
    indices = chronological_indices(source_buffer)
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
                "Increase --store_traj_num or --meta_update_steps so the quasimetric "
                "buffer reaches --qm_min_buffer_size."
            )
        selection_id = {"random": 0, "fast": 1, "meta": 2}[forced_selection]
        log_scalar(writer, "detection/selection", selection_id, step)
        log_scalar(writer, "detection/forced", 1, step)
        print(f"CQRL task-switch selection forced: {forced_selection}")
        return forced_selection, {}

    eval_env.set_task(task_name)
    candidates = {"fast": fast_agent, "random": random_agent}
    if meta_ready:
        candidates["meta"] = meta_agent

    returns = {}
    for name, candidate in candidates.items():
        eval_results = eval_env.evaluate_agent(
            candidate,
            args.detection_episodes,
            reseed_each_episode=bool(args.reseed_each_episode),
        )
        candidate_returns = np.asarray(eval_results["episodic_returns"], dtype=np.float64)
        returns[name] = candidate_returns
        mean_return = float(np.mean(candidate_returns))
        log_scalar(writer, f"detection/{name}_return", mean_return, step)
        print(f"CQRL detection {name}: return {mean_return:.3f}")

    fast_mean = float(np.mean(returns["fast"]))
    random_mean = float(np.mean(returns["random"]))
    if "meta" not in returns:
        selection = "fast" if fast_mean > random_mean else "random"
    else:
        meta_mean = float(np.mean(returns["meta"]))
        meta_is_better = returns_are_better(
            returns["meta"], returns["fast"], args.use_ttest
        )
        fast_is_better = returns_are_better(
            returns["fast"], returns["meta"], args.use_ttest
        )
        if meta_is_better and meta_mean > random_mean:
            selection = "meta"
        elif fast_is_better and fast_mean > random_mean:
            selection = "fast"
        else:
            selection = "random"

    selection_id = {"random": 0, "fast": 1, "meta": 2}[selection]
    log_scalar(writer, "detection/selection", selection_id, step)
    log_scalar(writer, "detection/forced", 0, step)
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
        structure_metrics, actor_metrics = meta_agent.update_structure_and_actor(update_buffer)
        recent_updates += int(use_recent)

    behavior_goal = recent_buffer.sample_behavior_goal(
        discount=meta_agent.goal_discount,
        success_only=meta_agent.behavior_goal_success_only,
    )
    if behavior_goal is None:
        behavior_goal = quasimetric_buffer.sample_behavior_goal(
            discount=meta_agent.goal_discount,
            success_only=meta_agent.behavior_goal_success_only,
        )
    meta_agent.set_behavior_goal(behavior_goal)
    actor_metrics = dict(actor_metrics)
    actor_metrics["behavior_goal_ready"] = float(behavior_goal is not None)

    log_metric_dict(writer, "metric_structure", structure_metrics, step)
    log_metric_dict(writer, "meta_actor", actor_metrics, step)
    log_scalar(writer, "meta/recent_updates", recent_updates, step)
    return structure_metrics, actor_metrics, recent_updates


def agent_training_state(agent):
    state = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "log_alpha_optimizer": agent.log_alpha_optimizer.state_dict(),
    }
    if getattr(agent, "goal_encoder", None) is not None:
        state["goal_encoder"] = agent.goal_encoder.state_dict()
    if hasattr(agent, "behavior_goal"):
        state["behavior_goal"] = copy.deepcopy(agent.behavior_goal)
    if hasattr(agent, "quasimetric"):
        state["quasimetric"] = agent.quasimetric.checkpoint()
    if hasattr(agent, "structure_memory"):
        state["structure_memory"] = agent.structure_memory.state_dict()
    return state


def load_agent_training_state(agent, state):
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    agent.critic_target.load_state_dict(state["critic_target"])
    agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
    with torch.no_grad():
        agent.log_alpha.copy_(state["log_alpha"].to(agent.device))
    agent.log_alpha_optimizer.load_state_dict(state["log_alpha_optimizer"])
    if "goal_encoder" in state and getattr(agent, "goal_encoder", None) is not None:
        agent.goal_encoder.load_state_dict(state["goal_encoder"])
    if hasattr(agent, "set_behavior_goal"):
        agent.set_behavior_goal(state.get("behavior_goal"))
    if "quasimetric" in state and hasattr(agent, "quasimetric"):
        agent.quasimetric.load_checkpoint(state["quasimetric"])
    if "structure_memory" in state and hasattr(agent, "structure_memory"):
        agent.structure_memory.load_state_dict(state["structure_memory"])


def environment_training_state(env):
    return {
        "base_task_name": env.base_task_name,
        "current_seed": env.current_seed,
        "task_counter": env.task_counter,
        "timestep_counter": env.timestep_counter,
        "obs_mean": np.array(env.obs_mean, copy=True),
        "obs_var": np.array(env.obs_var, copy=True),
        "obs_count": env.obs_count,
        "bias_correction": env.bias_correction,
        "eval_success_history": list(env.eval_success_history),
        "change_task_next_step": env._change_task_next_step,
        "rng_state": env.rng.get_state(),
    }


def load_environment_training_state(env, state):
    env.current_seed = state["current_seed"]
    env.set_task(state["base_task_name"])
    env.task_counter = int(state["task_counter"])
    env.timestep_counter = int(state["timestep_counter"])
    env.obs_mean = np.array(state["obs_mean"], copy=True)
    env.obs_var = np.array(state["obs_var"], copy=True)
    env.obs_count = state["obs_count"]
    env.bias_correction = bool(state["bias_correction"])
    env.eval_success_history = list(state["eval_success_history"])
    env._change_task_next_step = bool(state["change_task_next_step"])
    env.rng.set_state(state["rng_state"])


def rng_training_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_training_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_checkpoint_path(path):
    checkpoint_path = os.path.abspath(path)
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.isfile(state_path):
        return checkpoint_path

    latest_path = os.path.join(checkpoint_path, "latest_checkpoint.txt")
    if os.path.isfile(latest_path):
        with open(latest_path, "r") as handle:
            latest_checkpoint = handle.read().strip()
        checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
        if os.path.isfile(os.path.join(checkpoint_path, "training_state.pt")):
            return checkpoint_path
    raise FileNotFoundError(f"Training checkpoint not found: {path}")


def restore_checkpoint_args(args):
    checkpoint_path = resolve_checkpoint_path(args.resume_checkpoint)
    state = torch.load(
        os.path.join(checkpoint_path, "training_state.pt"),
        map_location="cpu",
        weights_only=False,
    )
    runtime_keys = (
        "resume_checkpoint",
        "gpu",
        "save_path",
        "save_freq",
        "save_model_freq",
        "checkpoint_freq",
        "keep_periodic_checkpoints",
        "log_backends",
        "wandb_project_name",
        "wandb_entity",
        "wandb_group",
        "wandb_mode",
    )
    runtime_values = {
        key: getattr(args, key)
        for key in runtime_keys
    }
    for key, value in state["args"].items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in runtime_values.items():
        setattr(args, key, value)
    args.resume_checkpoint = checkpoint_path
    print(f"resume configuration loaded: {checkpoint_path}")
    return args


def build_run_state(
    step,
    task_counter,
    task_start_step,
    selection,
    detection_scores,
    meta_ready,
    count_success,
    intermediate_stats,
    collector,
):
    return {
        "step": int(step),
        "task_counter": int(task_counter),
        "task_start_step": int(task_start_step),
        "selection": selection,
        "detection_scores": dict(detection_scores),
        "meta_ready": bool(meta_ready),
        "count_success": int(count_success),
        "intermediate_stats": {
            key: list(values)
            for key, values in intermediate_stats.items()
        },
        "collector_return_list": list(collector.return_list),
        "collector_success_list": list(collector.success_list),
    }


def save_training_checkpoint(
    checkpoint_root,
    checkpoint_name,
    args,
    fast_agents,
    meta_agent,
    replay_buffer,
    recent_buffer,
    quasimetric_buffer,
    env,
    run_state,
):
    checkpoint_root = os.path.abspath(checkpoint_root)
    checkpoint_path = os.path.join(checkpoint_root, checkpoint_name)
    suffix = 1
    while os.path.isfile(os.path.join(checkpoint_path, "training_state.pt")):
        checkpoint_path = os.path.join(checkpoint_root, f"{checkpoint_name}_{suffix}")
        suffix += 1
    checkpoint_name = os.path.basename(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    replay_buffer.save_data(os.path.join(checkpoint_path, "replay_buffer"))
    recent_buffer.save_data(os.path.join(checkpoint_path, "recent_buffer"))
    quasimetric_buffer.save_data(os.path.join(checkpoint_path, "quasimetric_buffer"))

    state = {
        "version": 2,
        "args": vars(args),
        "fast_agents": [agent_training_state(agent) for agent in fast_agents],
        "meta_agent": agent_training_state(meta_agent),
        "environment": environment_training_state(env),
        "rng": rng_training_state(),
        "run_state": run_state,
    }
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    temporary_state_path = f"{state_path}.tmp"
    torch.save(state, temporary_state_path)
    os.replace(temporary_state_path, state_path)

    os.makedirs(checkpoint_root, exist_ok=True)
    latest_path = os.path.join(checkpoint_root, "latest_checkpoint.txt")
    temporary_latest_path = f"{latest_path}.tmp"
    with open(temporary_latest_path, "w") as handle:
        handle.write(checkpoint_name)
    os.replace(temporary_latest_path, latest_path)
    print(f"training checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path,
    fast_agents,
    meta_agent,
    replay_buffer,
    recent_buffer,
    quasimetric_buffer,
    env,
    eval_env,
    collector,
):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    state = torch.load(
        os.path.join(checkpoint_path, "training_state.pt"),
        map_location=fast_agents[0].device,
        weights_only=False,
    )
    if state.get("version") not in (1, 2):
        raise ValueError(f"Unsupported checkpoint version: {state.get('version')}")

    if "fast_agents" in state:
        if len(state["fast_agents"]) != len(fast_agents):
            raise ValueError(
                "Checkpoint fast-agent count does not match the environment task count: "
                f"{len(state['fast_agents'])} != {len(fast_agents)}"
            )
        for fast_agent, fast_agent_state in zip(fast_agents, state["fast_agents"]):
            load_agent_training_state(fast_agent, fast_agent_state)
    else:
        current_agent_idx = (
            int(state["run_state"]["task_counter"]) - 1
        ) % len(fast_agents)
        load_agent_training_state(fast_agents[current_agent_idx], state["fast_agent"])
        print(
            "warning: legacy checkpoint contains only the current fast agent; "
            "earlier task-specific fast agents cannot be restored"
        )
    load_agent_training_state(meta_agent, state["meta_agent"])
    load_environment_training_state(env, state["environment"])
    load_environment_training_state(eval_env, state["environment"])

    buffers = (
        (replay_buffer, "replay_buffer"),
        (recent_buffer, "recent_buffer"),
        (quasimetric_buffer, "quasimetric_buffer"),
    )
    for buffer, directory_name in buffers:
        if not buffer.load_data(os.path.join(checkpoint_path, directory_name)):
            raise RuntimeError(f"Failed to restore {directory_name} from {checkpoint_path}")

    replay_buffer.start_new_episode()
    collector.obs, _ = collector._reset_env()
    load_rng_training_state(state["rng"])
    print(f"training checkpoint loaded: {checkpoint_path}")
    return state["run_state"]


def prune_periodic_checkpoints(checkpoint_root, keep):
    if keep <= 0 or not os.path.isdir(checkpoint_root):
        return
    periodic_paths = sorted(
        os.path.join(checkpoint_root, name)
        for name in os.listdir(checkpoint_root)
        if name.startswith("step_")
        and os.path.isdir(os.path.join(checkpoint_root, name))
    )
    for checkpoint_path in periodic_paths[:-keep]:
        shutil.rmtree(checkpoint_path)



def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    # General experiment arguments
    # parser.add_argument('--repeat_idx', type=int, default=0, help='Index of the repeat (for multiple runs)')
    # parser.add_argument('--env', type=str, default='metaworld_sequence_set6', help='Environment to run')

    # parser.add_argument("--normalize_obs", type=str2none, default=None, help="Normalize observations (pass 'none' to keep None)")
    # parser.add_argument('--goal_conditioned', type=int, default=0, help='Whether MetaWorld returns goal-conditioned dict observations')
    # parser.add_argument('--gc_reward_type', type=str_choice, default='sparse', choices=['sparse', 'dense', 'success'], help='Goal-conditioned reward type')
    # parser.add_argument('--gc_success_threshold', type=float, default=0.05, help='Goal-conditioned success distance threshold')
    # parser.add_argument('--gc_achieved_goal', type=str_choice, default='auto', choices=['auto', 'object', 'tcp'], help='Goal-conditioned achieved goal source')
    # parser.add_argument('--seed', type=int, default=0, help='Num steps to run')
    # parser.add_argument('--save_path', type=str, default='results/', help='Path to the folder to be saved in')
    # parser.add_argument('--save_freq', type=int, default=25000, help='Number steps between recording metrics')
    # parser.add_argument('--save_model_freq', type=int, default=-1,help='Number of steps between saving the model. Set to -1 for never. ')
    parser.add_argument('--method', type=str, default='cqrl', choices=['cqrl'], help='Continual learning method')
    parser.add_argument('--store_traj_num', type=int, default=20, help='Number of recent trajectories integrated at each task boundary')
    parser.add_argument('--quasimetric_buffer_capacity', type=int, default=1000000, help='Capacity of the cross-task quasimetric replay buffer')
    parser.add_argument('--meta_update_steps', type=int, default=10000, help='Joint quasimetric and meta-actor updates at each task boundary')
    parser.add_argument('--detection_episodes', type=int, default=10, help='Episodes per candidate during task-boundary policy detection')
    parser.add_argument('--warmup_steps', type=int, default=50000, help='Steps of transient meta-policy regularization')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='Meta-policy KL regularization weight')
    parser.add_argument(
        '--task_switch_selection',
        type=str,
        default='auto',
        choices=['auto', 'meta', 'fast', 'random'],
        help='Policy initialization selected at each task boundary',
    )
    parser.add_argument('--use_ttest', type=int, default=0, help='Whether policy detection uses Welch t-tests')

    parser.add_argument("--repeat_idx", type=int, default=0, help="Index of the repeat")
    parser.add_argument("--env", type=str, default="metaworld_sequence_set6", help="Environment to run")
    parser.add_argument(
        "--change_freq",
        type=int,
        default=1000000,
        help="Frequency to change tasks in the environment",
    )
    parser.add_argument(
        "--normalize_obs",
        type=str2none,
        default=None,
        help="Normalize observations (pass 'none' to keep None)",
    )
    parser.add_argument(
        "--freeze_rand_vec",
        type=int,
        default=0,
        choices=[0, 1],
        help="Reuse one MetaWorld task instance across episode resets",
    )
    parser.add_argument(
        "--reseed_each_episode",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to reset every evaluation episode with the current environment seed",
    )
    # ------------- goal-conditioned setting
    parser.add_argument(
        "--goal_conditioned",
        type=int,
        default=0,
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_path", type=str, default="results/", help="Path prefix for logs and models")
    parser.add_argument("--save_freq", type=int, default=25000, help="Evaluation frequency")
    parser.add_argument("--save_model_freq", type=int, default=-1, help="Model save frequency")
    parser.add_argument("--checkpoint_freq", type=int, default=25000, help="Full training checkpoint frequency; use 0 to disable periodic checkpoints")
    parser.add_argument("--keep_periodic_checkpoints", type=int, default=2, help="Number of periodic full checkpoints to retain; use 0 to keep all")
    parser.add_argument("--resume_checkpoint", type=str2none, default=None, help="Checkpoint directory or its parent containing latest_checkpoint.txt")
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
        type = int,
        default = 1,
        help = "How often to update the slow quasimetric structure",
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
        default=1,
        help="Prefer successful achieved states when sampling the behavior goal",
    )
    parser.add_argument(
        "--encode_actor_critic_goal",
        type=int,
        default = 0,
        help="Whether to encode goal observations before feeding actor and critic",
    )
    parser.add_argument(
        "--replay_buffer_mode",
        type=str,
        default="her",
        choices=["her", "no_her"],
        help="Metric replay buffer mode: HER reward relabeling or future-goal conditioning without HER rewards",
    )

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    try:
        args.log_backends = normalize_log_backends(args.log_backends)
    except ValueError as error:
        parser.error(str(error))
    if args.random_steps <= 0:
        parser.error('--random_steps must be positive')
    if args.store_traj_num <= 0:
        parser.error('--store_traj_num must be positive')
    if args.quasimetric_buffer_capacity <= 0:
        parser.error('--quasimetric_buffer_capacity must be positive')
    if args.meta_update_steps < 0:
        parser.error('--meta_update_steps cannot be negative')
    if args.detection_episodes <= 0:
        parser.error('--detection_episodes must be positive')
    if args.warmup_steps < 0:
        parser.error('--warmup_steps cannot be negative')
    if args.lambda_reg < 0.0:
        parser.error('--lambda_reg cannot be negative')
    if args.task_switch_selection == 'meta' and args.meta_update_steps == 0:
        parser.error('--task_switch_selection meta requires --meta_update_steps to be positive')
    if args.checkpoint_freq < 0:
        parser.error('--checkpoint_freq cannot be negative')
    if args.keep_periodic_checkpoints < 0:
        parser.error('--keep_periodic_checkpoints cannot be negative')
    if args.resume_checkpoint is not None:
        args = restore_checkpoint_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config_obj = ConfigDictConverter(vars(args))
    env_parameters = config_obj.env_dict

    env = config_obj.env_class(**env_parameters)
    eval_env = copy.deepcopy(env)
    print('env_list:',env.env_list)
    # print(env_parameters)

    num_steps_per_run = len(env.env_list) * args.change_freq
    # print('num_steps_per_run:',num_steps_per_run, env.normalize_obs)

    num_eval_runs = 10
    set_seed_everywhere(args.seed)

    method = args.method
    log_path = 'log/' + args.env + '/'
    gc_log_suffix = f'_gc-{args.gc_reward_type}' if bool(args.goal_conditioned) else ''
    log_name = 'cqrl_' + args.env + '_' + str(args.seed) + '_' + method + gc_log_suffix
    writer, log_info = create_experiment_logger(args, log_name)
    if log_info['active_backends']:
        print('log_backends:', ', '.join(log_info['active_backends']))
    else:
        print('log_backends: disabled')
    if log_info['tensorboard_path'] is not None:
        print('tensorboard_path:', log_info['tensorboard_path'])
    if log_info['wandb_path'] is not None:
        print('wandb_path:', log_info['wandb_path'])
    if log_info['wandb_url'] is not None:
        print('wandb_url:', log_info['wandb_url'])

    # sys.stdout = Logger(log_path + log_name + ".txt")
    # sys.stderr = sys.stdout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()

    # print(env.env.action_space)
    # print(env.env.observation_space)
    # env.env.env.env.env.env.env.env.base_task_name
    # print('---')
    # print(env.env.env.env.env.env.env.action_space)

    random_steps = args.random_steps
    replay_buffer_capacity = int(args.change_freq) + int(random_steps)
    obs_space = vector_observation_space(env.env.observation_space)
    log_scalar(writer, 'config/goal_conditioned', int(bool(args.goal_conditioned)), 0)
    log_scalar(writer, 'config/obs_dim', obs_space.shape[0], 0)
    log_scalar(writer, 'config/gc_success_threshold', args.gc_success_threshold, 0)
    log_scalar(writer, 'config/task_count', len(env.env_list), 0)
    log_scalar(writer, 'config/lambda_reg', args.lambda_reg, 0)
    log_scalar(writer, 'config/warmup_steps', args.warmup_steps, 0)

    replay_buffer_cls = (
        ReplayBufferMetric
        if args.replay_buffer_mode == 'her'
        else ReplayBufferMetricNoHER
    )
    replay_buffer = replay_buffer_cls(
        obs_space.shape,
        env.env.action_space.shape,
        replay_buffer_capacity,
        device)
    recent_buffer = replay_buffer_cls(
        obs_space.shape,
        env.env.action_space.shape,
        replay_buffer_capacity,
        device,
    )
    quasimetric_buffer = replay_buffer_cls(
        obs_space.shape,
        env.env.action_space.shape,
        max(
            args.quasimetric_buffer_capacity,
            args.batch_size,
            args.qm_min_buffer_size,
        ),
        device,
    )

    fast_agents = [
        build_fast_agent(
            obs_dim=obs_space.shape[0],
            action_dim=env.env.action_space.shape[0],
            device=device,
            args=args,
        )
        for _ in env.env_list
    ]
    meta_agent = build_agent(
        obs_dim=obs_space.shape[0],
        action_dim=env.env.action_space.shape[0],
        device=device,
        args=args,
    )

    collector = Collector(env, replay_buffer)
    intermediate_stats = defaultdict(list)
    final_stats = defaultdict(list)
    i_step = 0
    task_counter = env.task_counter
    fast_agent = fast_agents[(task_counter - 1) % len(fast_agents)]
    count_success = -1
    task_start_step = 0
    selection = 'fast'
    detection_scores = {}
    meta_ready = False
    model_dir = os.path.join(args.save_path, 'model', log_name)
    checkpoint_root = os.path.join(model_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    if args.resume_checkpoint is None:
        collector.initial_collect(random_steps)
        initial_checkpoint_state = build_run_state(
            i_step,
            task_counter,
            task_start_step,
            selection,
            detection_scores,
            meta_ready,
            count_success,
            intermediate_stats,
            collector,
        )
        save_training_checkpoint(
            checkpoint_root,
            f'task_{task_counter:02d}_step_{i_step:012d}',
            args,
            fast_agents,
            meta_agent,
            replay_buffer,
            recent_buffer,
            quasimetric_buffer,
            env,
            initial_checkpoint_state,
        )
    else:
        run_state = load_training_checkpoint(
            args.resume_checkpoint,
            fast_agents,
            meta_agent,
            replay_buffer,
            recent_buffer,
            quasimetric_buffer,
            env,
            eval_env,
            collector,
        )
        i_step = int(run_state['step'])
        task_counter = int(run_state['task_counter'])
        fast_agent = fast_agents[(task_counter - 1) % len(fast_agents)]
        task_start_step = int(run_state['task_start_step'])
        selection = run_state['selection']
        detection_scores = dict(run_state['detection_scores'])
        meta_ready = bool(run_state['meta_ready'])
        count_success = int(run_state['count_success'])
        intermediate_stats = defaultdict(list, run_state['intermediate_stats'])
        collector.return_list = list(run_state['collector_return_list'])
        collector.success_list = list(run_state['collector_success_list'])
        log_scalar(writer, 'resume/step', i_step, i_step)
        log_scalar(writer, 'resume/task_idx', env.task_counter, i_step)
        log_scalar(writer, 'resume/behavior_goal_ready', int(meta_agent.behavior_goal is not None), i_step)
        print(
            f"resuming at step {i_step}, task {env.task_counter} "
            f"({env.base_task_name}), behavior_goal={meta_agent.behavior_goal is not None}"
        )

    while i_step <= num_steps_per_run:
        if task_counter != env.task_counter:
            completed_task_count = task_counter
            task_counter = env.task_counter

            recent_buffer.reset()
            copied_transitions, count_success = copy_recent_trajectories(
                replay_buffer,
                recent_buffer,
                args.store_traj_num,
            )
            integrated_transitions = copy_replay_buffer(
                recent_buffer,
                quasimetric_buffer,
            )
            completed_task_name = str(
                env.env_list[(completed_task_count - 1) % len(env.env_list)]
            ).replace('/', '_')
            quasimetric_buffer_path = os.path.join(
                args.save_path,
                'quasimetric_buffers',
                log_name,
                f'task_{completed_task_count:02d}_{completed_task_name}',
            )
            quasimetric_buffer.save_data(quasimetric_buffer_path)
            _, actor_metrics, recent_updates = update_meta_from_recent(
                args,
                meta_agent,
                quasimetric_buffer,
                recent_buffer,
                completed_task_count,
                writer,
                i_step,
            )
            meta_ready = meta_ready or bool(actor_metrics)

            log_scalar(writer, 'meta/copied_transitions', copied_transitions, i_step)
            log_scalar(writer, 'meta/integrated_transitions', integrated_transitions, i_step)
            log_scalar(writer, 'meta/buffer_size', len(quasimetric_buffer), i_step)
            log_scalar(writer, 'meta/count_success', count_success, i_step)
            log_scalar(
                writer,
                'meta/recent_update_fraction',
                recent_updates / max(args.meta_update_steps, 1),
                i_step,
            )
            fast_agent.save(model_dir, f'{log_name}_task{completed_task_count}_fast')
            meta_agent.save(model_dir, f'{log_name}_task{completed_task_count}_meta')

            if i_step == num_steps_per_run:
                break

            replay_buffer.reset()
            previous_fast_agent = fast_agent
            fast_agent = fast_agents[(task_counter - 1) % len(fast_agents)]
            selection, detection_scores = detect_initialization(
                args,
                eval_env,
                env.env_list[(task_counter - 1) % len(env.env_list)],
                previous_fast_agent,
                meta_agent,
                fast_agent,
                meta_ready,
                writer,
                i_step,
            )
            if selection == 'fast':
                load_agent_training_state(
                    fast_agent,
                    agent_training_state(previous_fast_agent),
                )

            task_start_step = i_step
            collector.initial_collect(random_steps)
            log_scalar(writer, 'task/task_idx', task_counter, i_step)
            task_checkpoint_state = build_run_state(
                i_step,
                task_counter,
                task_start_step,
                selection,
                detection_scores,
                meta_ready,
                count_success,
                intermediate_stats,
                collector,
            )
            save_training_checkpoint(
                checkpoint_root,
                f'task_{task_counter:02d}_step_{i_step:012d}',
                args,
                fast_agents,
                meta_agent,
                replay_buffer,
                recent_buffer,
                quasimetric_buffer,
                env,
                task_checkpoint_state,
            )

        warmup_active = (
            selection == 'meta'
            and meta_ready
            and args.lambda_reg > 0.0
            and i_step - task_start_step < args.warmup_steps
        )
        train_metrics = fast_agent.update(
            replay_buffer,
            i_step,
            teacher_agent=meta_agent if warmup_active else None,
            regularization_weight=args.lambda_reg if warmup_active else 0.0,
        )

        collector.run_one_step(i_step, fast_agent)
        log_metric_dict(writer, 'train', train_metrics, i_step)
        log_scalar(writer, 'train/task_idx', task_counter, i_step)
        log_scalar(writer, 'train/replay_buffer_size', len(replay_buffer), i_step)
        log_scalar(writer, 'train/meta_warmup', int(warmup_active), i_step)

        if i_step % args.save_freq == 0:
            elapsed = time.perf_counter() - start_time
            sps = int(i_step / elapsed) if elapsed > 0 else 0
            print('step:',i_step, 'time:', round(elapsed / 60, 3), "SPS:", sps,
                  'task_counter', (task_counter,env.base_task_name), 'selection', selection,
                  'warmup', warmup_active, 'method', method, 'seed', args.seed)

            eval_results = env.evaluate_agent(
                fast_agent,
                num_eval_runs,
                reseed_each_episode=bool(args.reseed_each_episode),
            )
            eval_metrics = summarize_eval_results(eval_results)

            append_eval_stats(intermediate_stats, eval_metrics)
            intermediate_stats['steps'].append(i_step)
            intermediate_stats['task'].append(env.base_task_name)
            intermediate_stats['seed'].append(args.seed)
            intermediate_stats['task_idx'].append(task_counter)
            intermediate_stats['method'].append(method)
            intermediate_stats['time'].append(round(elapsed / 3600, 3))
            intermediate_stats['selection'].append(selection)
            intermediate_stats['meta_warmup'].append(warmup_active)
            intermediate_stats['count_success'].append(count_success)
            for candidate_name in ('fast', 'meta', 'random'):
                intermediate_stats[f'detection_{candidate_name}_return'].append(
                    detection_scores.get(candidate_name, np.nan)
                )

            log_scalar(writer, 'charts/SPS', sps, i_step)
            log_eval_metrics(writer, 'eval', eval_metrics, i_step)
            log_scalar(writer, 'eval/task_idx', task_counter, i_step)
            log_scalar(writer, 'eval/count_success', count_success, i_step)
            print(f"success {round(eval_metrics['metaworld_success_mean'], 3)} +/- {round(eval_metrics['metaworld_success_std'], 3)},",
                  f"gc_success {round(eval_metrics['gc_success_mean'], 3)} +/- {round(eval_metrics['gc_success_std'], 3)},",
                  f"eval return {round(eval_metrics['return_mean'], 3)} +/- {round(eval_metrics['return_std'], 3)}")
            if writer is not None:
                writer.flush()
        if args.save_model_freq > 0 and i_step > 0 and i_step % args.save_model_freq == 0:
            fast_agent.save(model_dir, f'{log_name}_step{i_step}_fast')
            meta_agent.save(model_dir, f'{log_name}_step{i_step}_meta')
        next_step = i_step + 1
        if args.checkpoint_freq > 0 and next_step % args.checkpoint_freq == 0:
            periodic_checkpoint_state = build_run_state(
                next_step,
                task_counter,
                task_start_step,
                selection,
                detection_scores,
                meta_ready,
                count_success,
                intermediate_stats,
                collector,
            )
            save_training_checkpoint(
                checkpoint_root,
                f'step_{next_step:012d}',
                args,
                fast_agents,
                meta_agent,
                replay_buffer,
                recent_buffer,
                quasimetric_buffer,
                env,
                periodic_checkpoint_state,
            )
            prune_periodic_checkpoints(
                checkpoint_root,
                args.keep_periodic_checkpoints,
            )
        i_step += 1

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('len:',len(intermediate_stats['count_success']), len(intermediate_stats['mean_return']), len(intermediate_stats['mean_success']))
    intermediate_stats['count_success'].extend([count_success] * (len(intermediate_stats['mean_return']) - len(intermediate_stats['count_success'])))
    intermediate_stats = pd.DataFrame(intermediate_stats)
    intermediate_stats.to_csv(log_path + "/" + log_name + ".csv", index=False)

    print('---')
    for task_idx, (task_name, task_fast_agent) in enumerate(
        zip(env.env_list, fast_agents)
    ):
        env.set_task(task_name)
        eval_results = env.evaluate_agent(
            task_fast_agent,
            num_eval_runs,
            reseed_each_episode=bool(args.reseed_each_episode),
        )
        eval_metrics = summarize_eval_results(eval_results)

        print(f"Final task {task_name} success {round(eval_metrics['metaworld_success_mean'], 3)} "
              f"gc_success {round(eval_metrics['gc_success_mean'], 3)} "
              f"return {round(eval_metrics['return_mean'], 3)}")

        append_eval_stats(final_stats, eval_metrics)
        final_stats['task'].append(env.base_task_name)
        final_stats['task_idx'].append(task_idx + 1)
        final_stats['seed'].append(args.seed)
        final_stats['method'].append(method)
        task_tag = str(env.base_task_name).replace(' ', '_')
        log_eval_metrics(writer, f'final/{task_tag}/fast', eval_metrics, task_idx + 1)

    final_stats = pd.DataFrame(final_stats)
    final_stats.to_csv(log_path + "/" + log_name + "_final.csv", index=False)
    for task_idx, task_fast_agent in enumerate(fast_agents, start=1):
        task_fast_agent.save(model_dir, f'{log_name}_final_task{task_idx}_fast')
    fast_agent.save(model_dir, f'{log_name}_final_fast')
    meta_agent.save(model_dir, f'{log_name}_final_meta')
    if writer is not None:
        writer.close()

'''
python continual_quasimetric_main4.py \
  --seed 0 \
  --gpu 0 \
  --env metaworld_sequence_set12 \
  --task_switch_selection meta \
  --meta_update_steps 10000 \
  --warmup_steps 50000 \
  --store_traj_num 20 \
  --lambda_reg 1.0

python continual_quasimetric_main5.py \
  --seed 0 \
  --gpu 0 \
  --env metaworld_sequence_set12 \
  --task_switch_selection meta \
  --meta_update_steps 10000 \
  --warmup_steps 50000 \
  --store_traj_num 20 \
  --lambda_reg 1.0 \
    --log_backends tensorboard wandb \
  --wandb_mode offline

# Resume the latest complete checkpoint under this run.
python continual_quasimetric_main4.py \
    --resume_checkpoint results/model/cqrl_metaworld_sequence_set12_0_cqrl/checkpoints \
    --gpu 0

# Resume an explicit task-boundary checkpoint.
python continual_quasimetric_main4.py \
    --resume_checkpoint results/model/cqrl_metaworld_sequence_set12_0_cqrl/checkpoints/task_07_step_000006000000 \
    --gpu 0
'''
'''
独立student  共享teacher  pass


'''
if __name__ == "__main__":
    main()