import argparse
import importlib
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import time
from agent.sac import SACAgent
from replay_buffer import ReplayBuffer, Collector
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import random
import copy


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

def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    # General experiment arguments
    parser.add_argument('--repeat_idx', type=int, default=0, help='Index of the repeat (for multiple runs)')
    parser.add_argument('--env', type=str, default='metaworld_sequence_set6', help='Environment to run')
    parser.add_argument('--change_freq', type=int, default=1e6,help='Frequency to change tasks in the environment')  # note this is overriden below per environment
    parser.add_argument("--normalize_obs", type=str2none, default=None, help="Normalize observations (pass 'none' to keep None)")
    parser.add_argument('--freeze_rand_vec', type=int, default=0, choices=[0, 1], help='Reuse one MetaWorld task instance across episode resets')
    parser.add_argument('--reseed_each_episode', type=int, default=0, choices=[0, 1], help='Whether to reset every evaluation episode with the current environment seed')
    parser.add_argument('--goal_conditioned', type=int, default=0, help='Whether MetaWorld returns goal-conditioned dict observations')
    parser.add_argument('--gc_reward_type', type=str_choice, default='sparse', choices=['sparse', 'dense', 'success'], help='Goal-conditioned reward type')
    parser.add_argument('--gc_success_threshold', type=float, default=0.05, help='Goal-conditioned success distance threshold')
    parser.add_argument('--gc_achieved_goal', type=str_choice, default='auto', choices=['auto', 'object', 'tcp'], help='Goal-conditioned achieved goal source')
    parser.add_argument('--seed', type=int, default=0, help='Num steps to run')
    parser.add_argument('--save_path', type=str, default='results/', help='Path for local W&B files')
    parser.add_argument('--results_path', type=str, default='log', help='Path for CSV result files')
    parser.add_argument('--model_path', type=str, default='model', help='Path for saved model files')
    parser.add_argument('--save_freq', type=int, default=25000, help='Number steps between recording metrics')
    parser.add_argument('--save_model_freq', type=int, default=-1,help='Number of steps between saving the model. Set to -1 for never. ')
    parser.add_argument('--method', type=str, default='independent', help='Method to use for multitask learning') # 'independent', 'average', 'continue', 'buffer', 'buffer_wd'
    parser.add_argument('--store_traj_num', type=int, default=10, help='Number of trajectories to store in the buffer for each task, only for buffer method')
    parser.add_argument('--use_ttest', type=int, default=0, help='Whether to use t-test for agent selection (0: False, 1: True)')
    parser.add_argument('--gpu', type=str, default='0', help='Comma separated list of GPU IDs')
    parser.add_argument('--log_backends', type=str.lower, nargs='+', default=['wandb'], choices=['tensorboard', 'wandb', 'none'], help='Logging backends to enable. Use one or both of: tensorboard wandb, or pass none.')
    parser.add_argument('--wandb_project_name', type=str, default='continual-rl-progress', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str2none, default=None, help='Optional Weights & Biases entity/team')
    parser.add_argument('--wandb_group', type=str2none, default=None, help='Optional Weights & Biases group name')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline'], help='Weights & Biases run mode when wandb logging is enabled')

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    try:
        args.log_backends = normalize_log_backends(args.log_backends)
    except ValueError as error:
        parser.error(str(error))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "0,1"

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
    log_path = os.path.join(args.results_path, args.env)
    gc_log_suffix = f'_gc-{args.gc_reward_type}' if bool(args.goal_conditioned) else ''
    log_name = 'sac_' + args.env + '_' + str(args.seed) + '_' + method + gc_log_suffix
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

    random_steps = 10000
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
        agent_list.append(SACAgent(obs_dim=obs_space.shape[0],
                         action_dim=env.env.action_space.shape[0],
                         action_range=[-1., 1.],
                         device=device, ))

    if method == 'buffer' or method == 'buffer_wd':
        meta_agent_list = []
        for i in range(len(env.env_list)):
            meta_agent_list.append(SACAgent(obs_dim=obs_space.shape[0],
                                  action_dim=env.env.action_space.shape[0],
                                  action_range=[-1., 1.],
                                  device=device, )) # only the last one will be used finally, save all just for monitoring the training process
        meta_buffer = ReplayBuffer(
            obs_space.shape,
            env.env.action_space.shape,
            int(replay_buffer_capacity) + random_steps,
            device) # no need this large buffer, just store a few trajectories for each task
        from_meta = False

    collector = Collector(env, replay_buffer)

    obs, _ = env.reset()  # match the gymnasium interface

    task_counter = env.task_counter
    agent = agent_list[task_counter-1] # task_counter starts from 1, agent_list from 0
    if method == 'buffer' or method == 'buffer_wd':
        meta_agent = meta_agent_list[task_counter-1]
    collector.initial_collect(random_steps)

    intermediate_stats = defaultdict(list)
    final_stats = defaultdict(list)

    i_step = 0
    count_success = -1

    while i_step <= num_steps_per_run:
        if task_counter != env.task_counter:
            task_counter = env.task_counter
            log_scalar(writer, 'task/task_idx', task_counter, i_step)

            if method == 'buffer' or method == 'buffer_wd':
                assert replay_buffer.full == True, f"Replay buffer not full! Current idx: {replay_buffer.idx}"
                done_indices = np.where(replay_buffer.not_dones == 0)[0][-args.store_traj_num-1:]
                count_success = 0
                for i in range(len(done_indices)-1):
                    if np.sum(replay_buffer.successes[done_indices[i]+1:done_indices[i+1]+1]) > 0:
                        count_success += 1
                done_ind = done_indices[0]+1

                if method == 'buffer_wd':
                    if task_counter - 1 - 1 - 1 >= 0:
                        meta_agent.actor_wd_loss(agent,meta_agent_list[task_counter-1-1-1],replay_buffer,done_ind, meta_buffer, args.store_traj_num * task_counter * 100)
                    else:
                        meta_agent.actor_wd_loss(agent, None, replay_buffer, done_ind, meta_buffer, args.store_traj_num * task_counter * 100)

                while done_ind < replay_buffer.capacity:
                    meta_buffer.add(replay_buffer.obses[done_ind],
                                    replay_buffer.actions[done_ind],
                                    replay_buffer.rewards[done_ind],
                                    replay_buffer.successes[done_ind],
                                    replay_buffer.next_obses[done_ind],
                                    not replay_buffer.not_dones[done_ind],
                                    not replay_buffer.not_dones_no_max[done_ind])
                    done_ind += 1
                print('meta_buffer idx:', meta_buffer.idx, 'count_success:', count_success)
                log_scalar(writer, 'meta/buffer_size', meta_buffer.idx, i_step)
                log_scalar(writer, 'meta/count_success', count_success, i_step)



                if method == 'buffer':
                    meta_actor_loss = meta_agent.actor_nll(meta_buffer,args.store_traj_num * task_counter * 100)
                    log_scalar(writer, 'meta/actor_nll', meta_actor_loss, i_step)

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

                agent.save(args.model_path, log_name + '_' + str(task_counter-1-1))
                meta_agent.save(args.model_path, log_name + '_' + str(task_counter-1-1) + '_meta')

            if i_step == num_steps_per_run:
                break

            replay_buffer.reset()

            if method == 'continue':
                agent_list[(task_counter-1)%len(env.env_list)].critic.load_state_dict(agent.critic.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].critic_target.load_state_dict(agent.critic_target.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].actor.load_state_dict(agent.actor.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].log_alpha = agent.log_alpha
                agent_list[(task_counter-1)%len(env.env_list)].critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].actor_optimizer.load_state_dict(agent.actor_optimizer.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].log_alpha_optimizer.load_state_dict(agent.log_alpha_optimizer.state_dict())

            if method == 'buffer' or method == 'buffer_wd':
                agent = agent_list[(task_counter - 1) % len(env.env_list)]
                return_dict, success_dict = collector.initial_agent_collect(random_steps, [agent,meta_agent], num_eval_runs)

                if args.use_ttest:
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(return_dict[1], return_dict[0], alternative='greater')
                    print(f'T-test: t_stat={t_stat:.3f}, p_value={p_value:.3f}')
                    log_scalar(writer, 'meta/ttest_t_stat', t_stat, i_step)
                    log_scalar(writer, 'meta/ttest_p_value', p_value, i_step)
                    if p_value < 0.05:
                        agent.actor.load_state_dict(meta_agent.actor.state_dict())
                        from_meta = True
                    else:
                        from_meta = False
                else:
                    if np.mean(return_dict[1]) > np.mean(return_dict[0]):
                        agent.actor.load_state_dict(meta_agent.actor.state_dict())
                        from_meta = True
                    else:
                        from_meta = False
                print('return_dict: {:.3f} {:.3f}, success_dict: {:.3f} {:.3f}, from_meta: {}'.format(np.mean(return_dict[0]),np.mean(return_dict[1]),np.mean(success_dict[0]),np.mean(success_dict[1]), from_meta))
                log_scalar(writer, 'meta/current_agent_return', np.mean(return_dict[0]), i_step)
                log_scalar(writer, 'meta/meta_agent_return', np.mean(return_dict[1]), i_step)
                log_scalar(writer, 'meta/current_agent_success', np.mean(success_dict[0]), i_step)
                log_scalar(writer, 'meta/meta_agent_success', np.mean(success_dict[1]), i_step)
                log_scalar(writer, 'meta/from_meta', int(from_meta), i_step)
                meta_agent_list[(task_counter-1)%len(env.env_list)].actor.load_state_dict(meta_agent.actor.state_dict())
                meta_agent_list[(task_counter-1)%len(env.env_list)].actor_optimizer.load_state_dict(meta_agent.actor_optimizer.state_dict())
                meta_agent = meta_agent_list[task_counter-1]
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
        eval_agent_list = meta_agent_list
    else:
        eval_agent_list = agent_list
    for agent_idx, agent in enumerate(eval_agent_list):
        for i in range(len(env.env_list[:agent_idx+1])):
            env.set_task(env.env_list[i])
            eval_results = env.evaluate_agent(
                agent,
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
            final_stats['agent_idx'].append(agent_idx+1)
            task_tag = str(env.base_task_name).replace(' ', '_')
            final_step = agent_idx * len(env.env_list) + i + 1
            log_eval_metrics(writer, f'final/{task_tag}/agent_{agent_idx+1}', eval_metrics, final_step)

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
python test_main.py --seed 1 --method independent --gpu 1 --env metaworld_sequence_set6

python test_main.py --seed 0 --method independent --gpu 0 --store_traj_num 10 --use_ttest 1 --env metaworld_sequence_set12

python test_main.py \
  --seed 0 \
  --method buffer \
  --gpu 1 \
  --env metaworld_sequence_set12 \
  --change_freq 1000000 \
    --freeze_rand_vec 0 \
    --reseed_each_episode 0 \
  --store_traj_num 20 \
  --use_ttest 1 \
  --log_backends tensorboard wandb \
  --wandb_mode offline \
  --model_path results/fame/fame_models_seq12_seed0_traj20 \
  --results_path results/fame/fame_res_seq12_seed0_traj20 \
  --save_path results/fame/fame_seq12_seed0_wandb_traj20


  python test_main.py \
  --seed 0 \
  --method buffer \
  --gpu 0 \
  --env metaworld_sequence_set12 \
  --change_freq 1000000 \
    --freeze_rand_vec 0 \
    --reseed_each_episode 0 \
  --store_traj_num 50 \
  --use_ttest 1 \
  --log_backends tensorboard wandb \
  --wandb_mode online \
  --model_path results/fame/fame_models_seq12_seed0_traj50 \
  --results_path results/fame/fame_res_seq12_seed0_traj50 \
  --save_path results/fame/fame_seq12_seed0_wandb_traj50

  python test_main.py \
    --seed 1 \
    --method buffer \
    --gpu 1 \
    --env metaworld_sequence_set6 \
    --change_freq 1000000 \
      --freeze_rand_vec 0 \
      --reseed_each_episode 0 \
    --store_traj_num 50 \
    --use_ttest 1 \
    --log_backends tensorboard wandb \
    --wandb_mode online \
    --model_path results/fame/fame_models_seq6_seed1_traj50 \
    --results_path results/fame/fame_res_seq6_seed1_traj50 \
    --save_path results/fame/fame_seq6_seed1_wandb_traj50

    python test_main.py \
        --seed 1 \
        --method buffer \
        --gpu 0 \
        --env metaworld_sequence_set6 \
        --change_freq 1000000 \
          --freeze_rand_vec 0 \
          --reseed_each_episode 0 \
        --store_traj_num 20 \
        --use_ttest 1 \
        --log_backends tensorboard wandb \
        --wandb_mode online \
        --model_path results/fame/fame_models_seq6_seed1_traj20 \
        --results_path results/fame/fame_res_seq6_seed1_traj20 \
        --save_path results/fame/fame_seq6_seed1_wandb_traj20
'''

if __name__ == "__main__":
    main()