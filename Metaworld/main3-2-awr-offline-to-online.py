import argparse
import importlib
import torch
import time
from agent.sac_shaping import SACAgentIR
from agent.sac import SACAgent
from replay_buffer import ReplayBuffer, Collector
from replay_buffer_metric import ReplayBufferMetric
from agent.quasimetric import QuasimetricConfig
from agent.quasimetric.awr_agent import ContinualQuasimetricAWRAgent
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import random
import copy
import sys


class WandbLogger:
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
        summary_key = tag.replace('/', '_')
        self.wandb_run.summary[summary_key] = text_string
        if global_step is not None:
            self.wandb_run.summary[f'{summary_key}_step'] = global_step

    def flush(self):
        self._flush_wandb()

    def close(self):
        self.flush()
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
        if backend != 'wandb':
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
    wandb_run = None
    wandb_path = None
    wandb_url = None
    active_backends = []

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
    if wandb_run is not None:
        logger = WandbLogger(wandb_run=wandb_run)
        logger.add_text(
            'config/args',
            '\n'.join(f'{key}: {value}' for key, value in sorted(vars(args).items())),
            0,
        )

    return logger, {
        'active_backends': active_backends,
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
        action_invariance_coef=args.qm_action_invariance_coef,
        transition_consistency_coef=args.qm_transition_consistency_coef,
        contrastive_coef=args.qm_contrastive_coef,
        nce_mode=args.qm_nce_mode,
        target_tau=args.qm_target_tau,
        current_batch_ratio=args.qm_current_batch_ratio,
        max_grad_norm=args.qm_max_grad_norm,
        min_buffer_size=args.qm_min_buffer_size,
    )
    return ContinualQuasimetricAWRAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=[-1.0, 1.0],
        device=device,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        awr_beta=args.awr_beta,
        awr_weight_clip=args.awr_weight_clip,
        awr_num_goals=args.awr_num_goals,
        quasimetric_cfg=quasimetric_cfg,
    )


def resolve_meta_update_num(args, completed_task_idx):
    if args.meta_update_steps is not None:
        return args.meta_update_steps
    return max(
        1,
        args.store_traj_num * completed_task_idx * args.meta_updates_per_traj,
    )


def resolve_warmup_distill_num(args, task_counter):
    if args.warmup_distill_steps is not None:
        return args.warmup_distill_steps
    return max(
        1,
        args.store_traj_num * task_counter * args.meta_updates_per_traj,
    )


def distill_meta_actor_to_agent(agent, meta_agent, replay_buffer, update_num):
    print_interval = max(update_num // 5, 1)
    last_metrics = {}
    for update_index in range(update_num):
        obs, _, _, _, _, _ = replay_buffer.sample(agent.batch_size)
        obs = torch.as_tensor(obs, device=agent.device).float()

        with torch.no_grad():
            teacher_action = meta_agent.actor(obs).mean.detach()
        student_action = agent.actor(obs).mean
        actor_loss = torch.square(student_action - teacher_action).sum(-1).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        last_metrics = {
            'actor_distill_loss': float(actor_loss.item()),
            'teacher_action': float(teacher_action.mean().item()),
            'student_action': float(student_action.mean().item()),
        }
        if update_index % print_interval == 0:
            print('actor_distill:', update_index, last_metrics)
    return last_metrics


def initialize_new_task_agent(
    agent,
    meta_agent,
    collector,
    replay_buffer,
    random_steps,
    num_eval_runs,
    init_mode,
    distill_update_num,
):
    replay_buffer.reset()
    if init_mode == 'copy':
        agent.actor.load_state_dict(meta_agent.actor.state_dict())
        distill_metrics = {}
    elif init_mode == 'warmup':
        collector.initial_agent_collect(random_steps, [agent], num_eval_runs)
        distill_metrics = distill_meta_actor_to_agent(
            agent,
            meta_agent,
            replay_buffer,
            distill_update_num,
        )
        replay_buffer.reset()
    else:
        raise ValueError(f'Unsupported new-task initialization mode: {init_mode}')

    collector.initial_agent_collect(random_steps, [agent], num_eval_runs)
    return (
        {
            'actor_transferred': int(init_mode == 'copy'),
            'meta_warmup': int(init_mode == 'warmup'),
        },
        distill_metrics,
    )


def task_model_name(log_name, completed_task_idx, use_meta=False):
    model_name = f'{log_name}_{completed_task_idx - 1}'
    if use_meta:
        model_name += '_meta'
    return model_name


def set_start_task(env, task_idx):
    task_name = env.env_list[task_idx - 1]
    env.set_task(task_name)
    env.task_counter = task_idx
    env.timestep_counter = int((task_idx - 1) * env.change_freq)
    print(f'start from task {task_idx}: {task_name}')


def quasimetric_buffer_path(save_path, log_name, env, completed_task_idx):
    completed_task_name = str(
        env.env_list[(completed_task_idx - 1) % len(env.env_list)]
    ).replace('/', '_')
    return os.path.join(
        save_path,
        'quasimetric_buffers',
        log_name,
        f'task_{completed_task_idx:02d}_{completed_task_name}',
    )


def load_previous_task_artifacts(
    previous_agent,
    meta_agent,
    quasimetric_buffer,
    resume_path,
    log_name,
    env,
    current_task_idx,
):
    completed_task_idx = current_task_idx - 1
    model_dir = os.path.join(resume_path, 'model')
    agent_model_name = task_model_name(log_name, completed_task_idx)
    meta_model_name = task_model_name(log_name, completed_task_idx, use_meta=True)
    required_paths = [
        os.path.join(model_dir, f'{agent_model_name}_{suffix}.pt')
        for suffix in ('actor', 'critic', 'critic_target')
    ] + [
        os.path.join(model_dir, f'{meta_model_name}_{suffix}.pt')
        for suffix in ('actor', 'awr', 'quasimetric')
    ]
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            f'previous task {completed_task_idx} is missing model files: {missing_paths}'
        )

    previous_agent.load(model_dir, agent_model_name)
    meta_agent.load(model_dir, meta_model_name)

    buffer_path = quasimetric_buffer_path(
        resume_path,
        log_name,
        env,
        completed_task_idx,
    )
    if not quasimetric_buffer.load_data(buffer_path):
        raise FileNotFoundError(
            f'failed to load previous task buffer from {buffer_path}'
        )
    print(
        f'loaded task {completed_task_idx} model and buffer from '
        f'{os.path.abspath(resume_path)}'
    )
    return completed_task_idx



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
    parser.add_argument('--method', type=str, default='independent', help='Method to use for multitask learning') # 'independent', 'average', 'continue', 'buffer', 'buffer_wd'
    parser.add_argument('--store_traj_num', type=int, default=10, help='Number of trajectories to store in the buffer for each task, only for buffer method')
    parser.add_argument('--meta_updates_per_traj', type=int, default=100, help='Number of joint meta updates per stored trajectory')
    parser.add_argument('--meta_update_steps', type=int, default=None, help='Fixed number of joint meta updates at each task boundary; overrides --meta_updates_per_traj')
    parser.add_argument('--use_ttest', type=int, default=0, help='Whether to use t-test for agent selection (0: False, 1: True)')
    parser.add_argument(
        '--new_task_init',
        choices=('copy', 'warmup'),
        default='warmup',
        help='Initialize a new task by copying the meta actor or distilling it into the SAC actor',
    )
    parser.add_argument(
        '--warmup_distill_steps',
        type=int,
        default=None,
        help='Actor distillation updates in warmup mode; defaults to store_traj_num * task_counter * meta_updates_per_traj',
    )
   
    parser.add_argument("--repeat_idx", type=int, default=0, help="Index of the repeat")
    parser.add_argument("--env", type=str, default="metaworld_sequence_set6", help="Environment to run")
    parser.add_argument(
        "--current_task_idx",
        type=int,
        default=None,
        help="1-based task index to continue from using the previous task artifacts",
    )
    parser.add_argument(
        "--resume_path",
        type=str2none,
        default=None,
        help="Previous run root containing model/ and quasimetric_buffers/; defaults to --save_path",
    )
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
    parser.add_argument("--awr_beta", type=float, default=1.0, help="Quasimetric advantage temperature")
    parser.add_argument("--awr_weight_clip", type=float, default=20.0, help="Maximum quasimetric AWR weight")
    parser.add_argument(
        "--awr_num_goals",
        type=int,
        default=4,
        help="Number of successful terminal goals sampled per transition",
    )
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
    if args.resume_path is None:
        args.resume_path = args.save_path
    if args.current_task_idx is not None and args.method not in ('buffer', 'buffer_wd'):
        parser.error('--current_task_idx currently supports only buffer and buffer_wd methods')
    if args.meta_update_steps is not None and args.meta_update_steps < 0:
        parser.error('--meta_update_steps cannot be negative')
    if args.awr_beta <= 0:
        parser.error('--awr_beta must be positive')
    if args.awr_weight_clip < 1:
        parser.error('--awr_weight_clip must be at least 1')
    if args.awr_num_goals <= 0:
        parser.error('--awr_num_goals must be positive')
    if args.random_steps <= 0:
        parser.error('--random_steps must be positive')
    if args.warmup_distill_steps is not None and args.warmup_distill_steps <= 0:
        parser.error('--warmup_distill_steps must be positive')

    model_dir = os.path.join(args.save_path, 'model')
    distilled_model_dir = os.path.join(args.save_path, 'model_distilledagent')
    print('model_save_path:', os.path.abspath(model_dir))
    print('distilled_model_save_path:', os.path.abspath(distilled_model_dir))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config_obj = ConfigDictConverter(vars(args))
    env_parameters = config_obj.env_dict

    env = config_obj.env_class(**env_parameters)
    eval_env = copy.deepcopy(env)
    print('env_list:',env.env_list)
    if args.current_task_idx is not None:
        if args.current_task_idx < 2 or args.current_task_idx > len(env.env_list):
            parser.error(f'--current_task_idx must be in [2, {len(env.env_list)}]')
        set_start_task(env, args.current_task_idx)
        set_start_task(eval_env, args.current_task_idx)
    # print(env_parameters)

    num_steps_per_run = len(env.env_list) * args.change_freq
    # print('num_steps_per_run:',num_steps_per_run, env.normalize_obs)

    num_eval_runs = 10
    set_seed_everywhere(args.seed)

    method = args.method
    log_path = 'log/' + args.env + '/'
    gc_log_suffix = f'_gc-{args.gc_reward_type}' if bool(args.goal_conditioned) else ''
    log_name = 'sac_' + args.env + '_' + str(args.seed) + '_' + method + gc_log_suffix
    run_log_name = log_name
    if args.current_task_idx is not None:
        run_log_name += f'_from_task{args.current_task_idx:02d}'
    writer, log_info = create_experiment_logger(args, run_log_name)
    if log_info['active_backends']:
        print('log_backends:', ', '.join(log_info['active_backends']))
    else:
        print('log_backends: disabled')
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
    replay_buffer_capacity = args.change_freq
    obs_space = vector_observation_space(env.env.observation_space)
    log_scalar(writer, 'config/goal_conditioned', int(bool(args.goal_conditioned)), 0)
    log_scalar(writer, 'config/obs_dim', obs_space.shape[0], 0)
    log_scalar(writer, 'config/gc_success_threshold', args.gc_success_threshold, 0)

    replay_buffer = ReplayBuffer(
        obs_space.shape,
        env.env.action_space.shape,
        int(replay_buffer_capacity) + random_steps,
        device)

    agent_list = []
    for i in range(len(env.env_list)):
        if i == 0:
            agent_list.append(SACAgent(obs_dim=obs_space.shape[0],
                         action_dim=env.env.action_space.shape[0],
                         action_range=[-1., 1.],
                         device=device, ))  # normal sac   but update with reward shaping
        else:
            # agent_list.append(SACAgentIR(obs_dim=obs_space.shape[0],
            #              action_dim=env.env.action_space.shape[0],
            #              action_range=[-1., 1.],
            #              device=device, ))
            agent_list.append(SACAgent(obs_dim=obs_space.shape[0],
                         action_dim=env.env.action_space.shape[0],
                         action_range=[-1., 1.],
                         device=device, ))

    if method == 'buffer' or method == 'buffer_wd':
        meta_agent = build_meta_agent(
            obs_dim=obs_space.shape[0],
            action_dim=env.env.action_space.shape[0],
            device=device,
            args=args,
        )
        
        # meta_buffer = ReplayBuffer(
        #     obs_space.shape,
        #     env.env.action_space.shape,
        #     int(replay_buffer_capacity) + random_steps,
        #     device) 
        quasimetric_buffer = ReplayBufferMetric(
            obs_space.shape,
            env.env.action_space.shape,
            int(replay_buffer_capacity) + random_steps,
            device)

        from_meta = False

    collector = Collector(env, replay_buffer)

    task_counter = env.task_counter
    agent = agent_list[task_counter-1] # task_counter starts from 1, agent_list from 0

    intermediate_stats = defaultdict(list)
    final_stats = defaultdict(list)

    resumed_from_task_boundary = args.current_task_idx is not None
    i_step = int((task_counter - 1) * args.change_freq) if resumed_from_task_boundary else 0
    count_success = -1
    task_start_step = i_step

    if resumed_from_task_boundary:
        completed_task_idx = load_previous_task_artifacts(
            agent_list[task_counter - 2],
            meta_agent,
            quasimetric_buffer,
            args.resume_path,
            log_name,
            env,
            task_counter,
        )
        distill_update_num = resolve_warmup_distill_num(args, task_counter)
        init_metrics, distill_metrics = initialize_new_task_agent(
            agent,
            meta_agent,
            collector,
            replay_buffer,
            random_steps,
            num_eval_runs,
            args.new_task_init,
            distill_update_num,
        )
        from_meta = True
        log_scalar(writer, 'resume/step', i_step, i_step)
        log_scalar(writer, 'resume/task_idx', task_counter, i_step)
        log_scalar(writer, 'meta/init_copy', init_metrics['actor_transferred'], i_step)
        log_scalar(writer, 'meta/init_warmup', init_metrics['meta_warmup'], i_step)
        log_scalar(writer, 'meta/distill_updates', distill_update_num if distill_metrics else 0, i_step)
        log_metric_dict(writer, 'meta_distill', distill_metrics, i_step)
        agent.save(
            distilled_model_dir,
            task_model_name(log_name, completed_task_idx),
        )
        print(
            f'resuming at step {i_step}, task {task_counter} '
            f'({env.base_task_name}) from completed task {completed_task_idx}'
        )
    else:
        collector.initial_collect(random_steps)

    while i_step <= num_steps_per_run:
        if task_counter != env.task_counter:  # when task change
            task_counter = env.task_counter
            task_start_step = i_step
            log_scalar(writer, 'task/task_idx', task_counter, i_step)

            if method == 'buffer' or method == 'buffer_wd':
                assert replay_buffer.full == True, f"Replay buffer not full! Current idx: {replay_buffer.idx}"
                completed_task_idx = task_counter - 1
                quasimetric_buffer.start_new_task(completed_task_idx)
                done_indices = np.where(replay_buffer.not_dones == 0)[0][-args.store_traj_num-1:]
                count_success = 0
                for i in range(len(done_indices)-1):
                    if np.sum(replay_buffer.successes[done_indices[i]+1:done_indices[i+1]+1]) > 0:
                        count_success += 1
                done_ind = done_indices[0]+1

    
                while done_ind < replay_buffer.capacity:   
                    quasimetric_buffer.add(replay_buffer.obses[done_ind],
                                    replay_buffer.actions[done_ind],
                                    replay_buffer.rewards[done_ind],
                                    replay_buffer.successes[done_ind],
                                    replay_buffer.next_obses[done_ind],
                                    not replay_buffer.not_dones[done_ind],
                                    not replay_buffer.not_dones_no_max[done_ind])
                    done_ind += 1
                print('quasimetric_buffer idx:', quasimetric_buffer.idx, 'count_success:', count_success)
                current_buffer_path = quasimetric_buffer_path(
                    args.save_path,
                    log_name,
                    env,
                    completed_task_idx,
                )
                quasimetric_buffer.save_data(current_buffer_path)
                log_scalar(writer, 'meta/buffer_size', quasimetric_buffer.idx, i_step)
                log_scalar(writer, 'meta/count_success', count_success, i_step)

                update_num = resolve_meta_update_num(args, completed_task_idx)
                if method == 'buffer':
                    print(f"---- updating quasimetric structure and AWR metapolicy ({update_num} steps) ----")
                    structure_metrics = {}
                    awr_metrics = {}
                    meta_agent.train()
                    for _ in range(update_num):
                        structure_metrics, awr_metrics = meta_agent.update_structure_and_actor(
                            quasimetric_buffer
                        )

                    log_metric_dict(writer, 'metric_structure', structure_metrics, i_step)
                    log_metric_dict(writer, 'meta_awr', awr_metrics, i_step)

                eval_env.set_task(env.env_list[(task_counter-1-1)%len(env.env_list)])
                eval_results = eval_env.evaluate_agent(
                    meta_agent,
                    num_eval_runs,
                    reseed_each_episode=bool(args.reseed_each_episode),
                )
                eval_metrics = summarize_eval_results(eval_results)
                print(f"meta_agent: task {eval_env.base_task_name}, success {round(eval_metrics['metaworld_success_mean'], 3)} +/- {round(eval_metrics['metaworld_success_std'], 3)}, "
                      f"return {round(eval_metrics['return_mean'], 3)} +/- {round(eval_metrics['return_std'], 3)}")
                log_eval_metrics(writer, 'meta_eval', eval_metrics, i_step)

                agent.save(model_dir, task_model_name(log_name, completed_task_idx))
                meta_agent.save(
                    model_dir,
                    task_model_name(log_name, completed_task_idx, use_meta=True),
                )

            if i_step == num_steps_per_run:
                break

            replay_buffer.reset() #####

            if method == 'continue':
                agent_list[(task_counter-1)%len(env.env_list)].critic.load_state_dict(agent.critic.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].critic_target.load_state_dict(agent.critic_target.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].actor.load_state_dict(agent.actor.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].log_alpha = agent.log_alpha
                agent_list[(task_counter-1)%len(env.env_list)].critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].actor_optimizer.load_state_dict(agent.actor_optimizer.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].log_alpha_optimizer.load_state_dict(agent.log_alpha_optimizer.state_dict())

            if method == 'buffer' or method == 'buffer_wd':
                agent = agent_list[(task_counter - 1) % len(env.env_list)]   #### task specific agent
                if hasattr(agent, 'set_task_start_step'):
                    agent.set_task_start_step(task_start_step)
                if args.new_task_init == 'copy':
                    print("---- copy observation-only AWR metapolicy to task actor ----")
                else:
                    print("---- distill observation-only AWR metapolicy into task actor ----")
                distill_update_num = resolve_warmup_distill_num(args, task_counter)
                init_metrics, distill_metrics = initialize_new_task_agent(
                    agent,
                    meta_agent,
                    collector,
                    replay_buffer,
                    random_steps,
                    num_eval_runs,
                    args.new_task_init,
                    distill_update_num,
                )
                from_meta = True
                log_scalar(writer, 'meta/init_copy', init_metrics['actor_transferred'], i_step)
                log_scalar(writer, 'meta/init_warmup', init_metrics['meta_warmup'], i_step)
                log_scalar(writer, 'meta/actor_transferred', init_metrics['actor_transferred'], i_step)
                log_scalar(writer, 'meta/distill_updates', distill_update_num if distill_metrics else 0, i_step)
                log_metric_dict(writer, 'meta_distill', distill_metrics, i_step)
                agent.save(
                    distilled_model_dir,
                    task_model_name(log_name, task_counter - 1),
                )
                print("---------new-task initialization done")

                # if args.use_ttest:
                #     from scipy import stats
                #     t_stat, p_value = stats.ttest_ind(return_dict[1], return_dict[0], alternative='greater')
                #     print(f'T-test: t_stat={t_stat:.3f}, p_value={p_value:.3f}')
                #     log_scalar(writer, 'meta/ttest_t_stat', t_stat, i_step)
                #     log_scalar(writer, 'meta/ttest_p_value', p_value, i_step)
                #     if p_value < 0.05:
                #         agent.actor.load_state_dict(meta_agent.actor.state_dict())  #####
                #         from_meta = True
                #     else:
                #         from_meta = False
                # else:
                #     if np.mean(return_dict[1]) > np.mean(return_dict[0]):   ######
                #         agent.actor.load_state_dict(meta_agent.actor.state_dict())
                #         from_meta = True
                #     else:
                #         from_meta = False
                # print('return_dict: {:.3f} {:.3f}, success_dict: {:.3f} {:.3f}, from_meta: {}'.format(np.mean(return_dict[0]),np.mean(return_dict[1]),np.mean(success_dict[0]),np.mean(success_dict[1]), from_meta))
                # log_scalar(writer, 'meta/current_agent_return', np.mean(return_dict[0]), i_step)
                # log_scalar(writer, 'meta/meta_agent_return', np.mean(return_dict[1]), i_step)
                # log_scalar(writer, 'meta/current_agent_success', np.mean(success_dict[0]), i_step)
                # log_scalar(writer, 'meta/meta_agent_success', np.mean(success_dict[1]), i_step)

                log_scalar(writer, 'meta/from_meta', int(from_meta), i_step)
            else:
                agent = agent_list[(task_counter-1)%len(env.env_list)]
                collector.initial_collect(random_steps)

        train_metrics = {}
        if method == 'independent' or method == 'continue' or method == 'buffer' or method == 'buffer_wd':
            train_metrics = agent.sac_update(replay_buffer,i_step)


        elif method == 'average':
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(agent.batch_size)
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.as_torch(obs, action, reward, success, next_obs, not_done_no_max)
            if task_counter >= len(env.env_list):
                agent_num = len(env.env_list)
            else:
                agent_num = task_counter
            q_target = agent.compute_target_q(reward, next_obs, not_done_no_max)
            q_target /= agent_num
            for agent_idx in range(agent_num-1):
                q_target += agent_list[agent_idx].compute_target_q(reward, next_obs, not_done_no_max) / agent_num
            train_metrics = agent.update_with_target_q(obs, action, q_target, i_step)
            log_scalar(writer, 'train/average_agent_num', agent_num, i_step)

        collector.run_one_step(i_step, agent)
        log_metric_dict(writer, 'train', train_metrics, i_step)
        log_scalar(writer, 'train/task_idx', task_counter, i_step)
        log_scalar(writer, 'train/agent_idx', (task_counter-1)%len(env.env_list), i_step)
        log_scalar(writer, 'train/replay_buffer_size', len(replay_buffer), i_step)

        if i_step % args.save_freq == 0:
            elapsed = time.perf_counter() - start_time
            sps = int(i_step / elapsed) if elapsed > 0 else 0
            print('step:',i_step, 'time:', round(elapsed / 60, 3), "SPS:", sps,
                  'task_counter', (task_counter,env.base_task_name), 'agent', (task_counter-1)%len(env.env_list), 'method', method, 'seed', args.seed)

            eval_results = env.evaluate_agent(
                agent,
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
            if method == 'buffer' or method == 'buffer_wd':
                intermediate_stats['from_meta'].append(from_meta)
            if -1 in intermediate_stats['count_success']:
                intermediate_stats['count_success'][intermediate_stats['count_success'].index(-1)] = count_success
            else:
                intermediate_stats['count_success'].append(count_success)

            log_scalar(writer, 'charts/SPS', sps, i_step)
            log_eval_metrics(writer, 'eval', eval_metrics, i_step)
            log_scalar(writer, 'eval/task_idx', task_counter, i_step)
            log_scalar(writer, 'eval/agent_idx', (task_counter-1)%len(env.env_list), i_step)
            log_scalar(writer, 'eval/count_success', count_success, i_step)
            if method == 'buffer' or method == 'buffer_wd':
                log_scalar(writer, 'eval/from_meta', int(from_meta), i_step)
            print(f"success {round(eval_metrics['metaworld_success_mean'], 3)} +/- {round(eval_metrics['metaworld_success_std'], 3)},",
                  f"gc_success {round(eval_metrics['gc_success_mean'], 3)} +/- {round(eval_metrics['gc_success_std'], 3)},",
                  f"eval return {round(eval_metrics['return_mean'], 3)} +/- {round(eval_metrics['return_std'], 3)}")
            if writer is not None:
                writer.flush()
        i_step += 1

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('len:',len(intermediate_stats['count_success']), len(intermediate_stats['mean_return']), len(intermediate_stats['mean_success']))
    intermediate_stats['count_success'].extend([count_success] * (len(intermediate_stats['mean_return']) - len(intermediate_stats['count_success'])))
    intermediate_stats = pd.DataFrame(intermediate_stats)
    intermediate_stats.to_csv(log_path + "/" + log_name + ".csv", index=False)
    # evaluate all agents on all previous tasks
    print('---')
    if method == 'buffer' or method == 'buffer_wd':
        eval_agent_specs = [(-1, meta_agent, range(len(env.env_list)))]
    else:
        eval_agent_specs = [
            (agent_idx + 1, eval_agent, range(agent_idx + 1))
            for agent_idx, eval_agent in enumerate(agent_list)
        ]
    final_step = 0
    for agent_idx, eval_agent, task_indices in eval_agent_specs:
        for i in task_indices:
            env.set_task(env.env_list[i])
            eval_results = env.evaluate_agent(
                eval_agent,
                num_eval_runs,
                reseed_each_episode=bool(args.reseed_each_episode),
            )
            eval_metrics = summarize_eval_results(eval_results)

            # print(f"Final task {env.env_list[i]} success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)}")
            print(f"Final task {env.env_list[i]} success {round(eval_metrics['metaworld_success_mean'], 3)} "
                  f"gc_success {round(eval_metrics['gc_success_mean'], 3)} "
                  f"return {round(eval_metrics['return_mean'], 3)}")

            append_eval_stats(final_stats, eval_metrics)
            final_stats['task'].append(env.base_task_name)
            final_stats['task_idx'].append(i + 1)
            final_stats['seed'].append(args.seed)
            final_stats['method'].append(method)
            final_stats['agent_idx'].append(agent_idx)
            task_tag = str(env.base_task_name).replace(' ', '_')
            agent_tag = 'meta' if agent_idx == -1 else f'agent_{agent_idx}'
            final_step += 1
            log_eval_metrics(writer, f'final/{task_tag}/{agent_tag}', eval_metrics, final_step)

    # evaluate the meta agent on all tasks
    # if method == 'buffer':
    #     agent = meta_agent
    #     for i in range(len(env.env_list)):
    #         env.set_task(env.env_list[i])
    #         eval_results = env.evaluate_agent(agent, num_eval_runs)
    #         eval_episode_returns = eval_results['episodic_returns']
    #         eval_successes = eval_results['successes']
    #
    #         print(f"Final task {env.env_list[i]} success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)}")
    #
    #         final_stats['mean_return'].append(np.mean(eval_episode_returns))
    #         final_stats['mean_success'].append(np.mean(eval_successes))
    #         final_stats['task'].append(env.base_task_name)
    #         final_stats['task_idx'].append(i+1)
    #         final_stats['seed'].append(args.seed)
    #         final_stats['method'].append(method)
    #         final_stats['agent_idx'].append(-1) # meta agent

    final_stats = pd.DataFrame(final_stats)
    final_stats.to_csv(log_path + "/" + log_name + "_final.csv", index=False)
    if writer is not None:
        writer.close()

'''
python test_main.py --seed 0 --method independent --gpu 1 --env metaworld_sequence_set6
python continual_quasimetric_main2.py --seed 0 --method buffer --gpu 0 --log_backends wandb  
--env metaworld_sequence_set6 

python continual_quasimetric_main2.py --seed 0 --method buffer --gpu 0 --log_backends wandb  --env metaworld_sequence_set12 --change_freq 1000000
python continual_quasimetric_main2.py --seed 1 --method buffer --gpu 2 --log_backends wandb  --env metaworld_sequence_set12 --change_freq 1000000
python continual_quasimetric_main3.py --seed 0 --method buffer --gpu 0 --log_backends wandb  --env metaworld_sequence_set19 --freeze_rand_vec 0
python continual_quasimetric_main3.py --seed 0 --method buffer --gpu 0 --log_backends wandb  --env metaworld_sequence_set12 --freeze_rand_vec 0 --store_traj_num 10 --behavior_goal_success_only 0 --qm_nce_mode backward_nce
python continual_quasimetric_main3.py --seed 0 --method buffer --gpu 0 --log_backends wandb  --env metaworld_sequence_set12 --freeze_rand_vec 0 --store_traj_num 10 --behavior_goal_success_only 1 --qm_nce_mode backward_nce

stable version:
python continual_quasimetric_main3-2.py --seed 0 --method buffer --gpu 1 --log_backends wandb  --env metaworld_sequence_set12 --freeze_rand_vec 0 --store_traj_num 20 --behavior_goal_success_only 1 --qm_nce_mode backward_nce \
--reseed_each_episode 0  --save_path results/main3-2/set12_trj20  --meta_update_steps 10000 

seq6
python continual_quasimetric_main3-2.py --seed 1 --method buffer --gpu 0 --log_backends wandb  --env metaworld_sequence_set6 --freeze_rand_vec 0 --store_traj_num 20 --behavior_goal_success_only 1 --qm_nce_mode backward_nce \
--reseed_each_episode 0  --save_path results/main3-2/set6_trj20_seed1_test2  --meta_update_steps 10000  --bc_alpha 1.0 --qm_contrastive_coef 0.5

seq12
python continual_quasimetric_main3-2.py --seed 1 --method buffer --gpu 1 --log_backends wandb  --env metaworld_sequence_set12 --freeze_rand_vec 0 --store_traj_num 20 --behavior_goal_success_only 1 --qm_nce_mode backward_nce \
--reseed_each_episode 0  --save_path results/main3-2/set12_trj20_seed1_test  --meta_update_steps 10000  --bc_alpha 1.0 --qm_contrastive_coef 0.5

contrastive ablation:
python continual_quasimetric_main3-2.py --seed 0 --method buffer --gpu 1 --log_backends wandb  --env metaworld_sequence_set12 --freeze_rand_vec 0 --store_traj_num 20 --behavior_goal_success_only 1 --qm_nce_mode backward_nce \
--reseed_each_episode 0  --save_path results/main3-2/set12_trj20_contra_abl  --meta_update_steps 10000 --qm_contrastive_coef 0.0


python \
  Metaworld/continual_quasimetric_main3-2-awr.py \
  --seed 0 \
  --method buffer \
  --gpu 0 \
  --env metaworld_sequence_set12 \
  --freeze_rand_vec 0 \
  --reseed_each_episode 0 \
  --store_traj_num 20 \
  --meta_update_steps 10000 \
  --new_task_init warmup \
  --warmup_distill_steps 2000 \
  --awr_beta 1.0 \
  --awr_weight_clip 20 \
  --awr_num_goals 16 \
  --qm_transition_input state \
  --qm_nce_mode backward_nce \
  --qm_contrastive_coef 0.5 \
  --bc_alpha 1.0 \
  --save_path Metaworld/results/main3-2-awr/set12_warmup_seed0 \
  --log_backends wandb \
  --wandb_project_name metaworld-main3-2-awr

  
  python \
  Metaworld/main3-2-awr-offline-to-online.py \
  --seed 0 \
  --method buffer \
  --gpu 2 \
  --env metaworld_sequence_set12 \
  --freeze_rand_vec 0 \
  --reseed_each_episode 0 \
  --store_traj_num 20 \
  --meta_update_steps 10000 \
  --new_task_init warmup \
  --warmup_distill_steps 2000 \
  --awr_beta 1.0 \
  --awr_weight_clip 20 \
  --awr_num_goals 16 \
  --qm_transition_input state \
  --qm_nce_mode backward_nce \
  --qm_contrastive_coef 0.5 \
  --bc_alpha 1.0 \
  --current_task_idx 8 \
  --resume_path Metaworld/results/main3-2-awr/set12_warmup_seed0 \
  --save_path Metaworld/results/main3-2-awr/set12_warmup_seed0 \
  --log_backends wandb \
  --wandb_project_name metaworld-main3-2-awr

'''

'''
difference: quasimetric function share across tasks
'''

if __name__ == "__main__":
    main()