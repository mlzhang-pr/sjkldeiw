import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pathlib
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional, Tuple
from models import shared, SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent
from tasks import get_task, get_task_name, RPO10_SEQ
from stable_baselines3.common.buffers import ReplayBuffer
from collections import defaultdict
import pandas as pd


class SuccessCounter(gym.Wrapper):
    """From Continual World's Codebase"""

    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        return obs, reward, terminated, truncated, info

    def pop_successes(self):
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)


@dataclass
class Args:
    model_type: Literal["simple", "finetune", "componet", "packnet", "prognet"]

    save_dir: Optional[str] = None

    prev_units: Tuple[pathlib.Path, ...] = ()

    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    seed: int = 1

    torch_deterministic: bool = True

    cuda: bool = True

    track: bool = False

    wandb_project_name: str = "cw-sac"

    wandb_entity: str = None

    capture_video: bool = False

    task_id: int = 0

    task_sequence: int = 6
    cuda_device: int = 0
    eval_every: int = 10_000

    num_evals: int = 10

    total_timesteps: int = int(1e6)

    buffer_size: int = int(1e6)

    gamma: float = 0.99

    tau: float = 0.005

    batch_size: int = 128

    learning_starts: int = 5_000

    random_actions_end: int = 10_000

    policy_lr: float = 1e-3

    q_lr: float = 1e-3

    policy_frequency: int = 2

    target_network_frequency: int = 1

    noise_clip: float = 0.5

    alpha: float = 0.2

    autotune: bool = True


def make_env(task_id, task_sequence, eval_mode=False):
    def thunk(eval_mode=eval_mode):
        env = get_task(task_id, task_sequence)
        if not eval_mode:
            env = gym.wrappers.TransformReward(env, lambda r: r / 500)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = SuccessCounter(env)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc = shared(
            np.array(envs.observation_space.shape).prod()
            + np.prod(envs.action_space.shape)
        )
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, envs, model):
        super().__init__()
        self.model = model

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, **kwargs):
        mean, log_std = self.model(x, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x, **kwargs):

        mean, log_std = self(x, **kwargs)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


@torch.no_grad()
def evaluate_agent(agent, task_id, task_sequence, num_evals, device):
    """Runs and evaluation of the agent
    It runs on the current seed i.e. the current task"""

    test_env = gym.vector.SyncVectorEnv(
        [make_env(task_id, task_sequence, eval_mode=True)]
    ).envs[0]
    obs, _ = test_env.reset()
    step = 0
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    eval_results = {}
    episodic_returns = []

    while len(episodic_returns) < num_evals:
        obs = torch.Tensor(obs).to(device).unsqueeze(0)
        action, _ = agent(obs)

        next_obs, reward, terminated, truncated, info = test_env.step(
            action[0].cpu().numpy()
        )
        step += 1
        ep_ret += reward

        if "episode" in info:
            episodic_returns.append(info["episode"]["r"].item())
        obs = next_obs
        if terminated or truncated:
            obs, _ = test_env.reset()
            step = 0

    eval_results["episodic_returns"] = episodic_returns
    eval_results["successes"] = test_env.pop_successes()

    return eval_results


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"task_sequence_{args.task_sequence}_task_{args.task_id}_{args.model_type}_{args.exp_name}_{args.seed}"
    print(f"\n*** Run name: {run_name} ***\n")
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda:" + str(args.cuda_device))
    else:
        device = torch.device("cpu")

    print(f"*** Device: {device}")

    envs = gym.vector.SyncVectorEnv([make_env(args.task_id, args.task_sequence)])
    task_name = RPO10_SEQ[args.task_sequence - 1][args.task_id]
    print(f"Loading environment: {task_name}")
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_action = float(envs.single_action_space.high[0])

    print(
        "single_observation_space:",
        envs.single_observation_space.shape,
        np.array(envs.single_observation_space.shape).prod(),
    )
    print("single_action_space:", envs.single_action_space.shape)
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    print(f"*** Loading model `{args.model_type}` ***")
    if args.model_type in ["finetune", "componet"]:
        assert len(args.prev_units) > 0, (
            f"Model type {args.model_type} requires at least one previous unit"
        )

    if args.model_type == "simple":
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    elif args.model_type == "finetune":
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)

    elif args.model_type == "componet":
        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "packnet":
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
        if len(args.prev_units) == 0:
            model = PackNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                task_id=args.task_id,
                total_task_num=10,
                device=device,
            ).to(device)
        else:
            model = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.task_id + 1,
                restart_heads=True,
                freeze_bias=True,
                map_location=device,
            ).to(device)
    elif args.model_type == "prognet":
        model = ProgressiveNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)

    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    intermediate_stats = defaultdict(list)

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            if args.model_type == "componet" and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    torch.Tensor(obs).to(device),
                    global_step=global_step,
                )
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.model_type == "packnet":
                        if global_step >= packnet_retrain_start:
                            actor.model.start_retraining()
                        actor.model.before_update()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

        if global_step % 25000 == 0:
            print(
                "step:",
                global_step,
                "time:",
                round((time.time() - start_time) / 60, 3),
                "SPS:",
                int(global_step / (time.time() - start_time)),
                "task_id",
                args.task_id,
                "method",
                args.model_type,
                "seed",
                args.seed,
            )

            eval_results = evaluate_agent(
                actor, args.task_id, args.task_sequence, args.num_evals, device
            )

            eval_episode_returns = eval_results["episodic_returns"]
            eval_successes = eval_results["successes"]
            print(
                f"success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)},",
                f"eval return {round(np.mean(eval_episode_returns), 3)} +/- {round(np.std(eval_episode_returns), 3)}",
            )

            intermediate_stats["mean_return"].append(np.mean(eval_episode_returns))
            intermediate_stats["mean_success"].append(np.mean(eval_successes))
            intermediate_stats["steps"].append(
                global_step + args.task_id * args.total_timesteps
            )
            intermediate_stats["task"].append(task_name)
            intermediate_stats["seed"].append(args.seed)
            intermediate_stats["task_idx"].append(args.task_id + 1)
            intermediate_stats["method"].append(args.model_type)
            intermediate_stats["time"].append(
                round((time.time() - start_time) / 3600, 3)
            )
            intermediate_stats["count_success"].append(-1)

    log_path = "log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    intermediate_stats = pd.DataFrame(intermediate_stats)

    file_path = (
        log_path
        + "/sac_metaworld_sequence_set"
        + str(args.task_sequence)
        + "_"
        + str(args.seed)
        + "_"
        + args.model_type
        + ".csv"
    )
    intermediate_stats.to_csv(
        file_path, mode="a", header=not os.path.exists(file_path), index=False
    )

    envs.close()

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        actor.model.save(dirname=f"{args.save_dir}/{run_name}")
