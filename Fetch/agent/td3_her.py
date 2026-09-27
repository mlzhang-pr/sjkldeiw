import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agent import utils


class TD3Actor(nn.Module):
    def __init__(self, input_dim, action_dim, action_low, action_high, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.trunk = utils.mlp(input_dim, hidden_dim, action_dim, hidden_depth, nn.Tanh())
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self.apply(utils.weight_init)

    def forward(self, features):
        return self.trunk(features) * self.action_scale + self.action_bias


class TD3Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.q1 = utils.mlp(input_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.q2 = utils.mlp(input_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, features, actions):
        features_actions = torch.cat([features, actions], dim=-1)
        return self.q1(features_actions), self.q2(features_actions)

    def q1_forward(self, features, actions):
        return self.q1(torch.cat([features, actions], dim=-1))


class TD3HERAgent:
    """TD3 agent for Fetch goals with HER relabeled replay samples."""

    def __init__(
        self,
        observation_dim,
        goal_dim,
        action_space,
        device,
        actor_lr=1e-3,
        critic_lr=1e-3,
        hidden_dim=256,
        hidden_depth=2,
        gamma=0.95,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        batch_size=256,
    ):
        self.observation_dim = int(observation_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(np.prod(action_space.shape))
        self.feature_dim = self.observation_dim + self.goal_dim
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.batch_size = int(batch_size)
        self.num_updates = 0

        self.action_low_np = np.asarray(action_space.low, dtype=np.float32)
        self.action_high_np = np.asarray(action_space.high, dtype=np.float32)
        self.action_low = torch.as_tensor(self.action_low_np, device=self.device).float()
        self.action_high = torch.as_tensor(self.action_high_np, device=self.device).float()
        self.actor_critic_goal_dim = self.goal_dim

        self.actor = TD3Actor(
            self.feature_dim,
            self.action_dim,
            self.action_low_np,
            self.action_high_np,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.actor_target = TD3Actor(
            self.feature_dim,
            self.action_dim,
            self.action_low_np,
            self.action_high_np,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.critic = TD3Critic(
            self.feature_dim,
            self.action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.critic_target = TD3Critic(
            self.feature_dim,
            self.action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        return self

    def eval(self):
        return self.train(False)

    def act(self, obs, sample=False, goal_obs=None, noise_std=0.0):
        del sample
        obs_array, goal_array = self._split_obs(obs, goal_obs)
        obs_tensor = torch.as_tensor(obs_array, device=self.device).float().unsqueeze(0)
        goal_tensor = torch.as_tensor(goal_array, device=self.device).float().unsqueeze(0)
        features = self._features(obs_tensor, goal_tensor)
        with torch.no_grad():
            action = self.actor(features)[0].cpu().numpy()
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape).astype(np.float32)
        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32)

    def update(self, replay_buffer, gradient_steps=1):
        if len(replay_buffer) < self.batch_size:
            return {}

        metrics = {}
        for _ in range(int(gradient_steps)):
            batch = replay_buffer.sample(self.batch_size)
            metrics = self._update_from_batch(batch)
        return metrics

    def save(self, model_dir, model_name):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "num_updates": self.num_updates,
            },
            os.path.join(model_dir, f"{model_name}_td3_her.pt"),
        )

    def load(self, model_dir, model_name):
        checkpoint = torch.load(os.path.join(model_dir, f"{model_name}_td3_her.pt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.num_updates = int(checkpoint.get("num_updates", 0))

    def _update_from_batch(self, batch):
        obs_features = self._features(batch["observations"], batch["desired_goals"])
        next_features = self._features(batch["next_observations"], batch["next_desired_goals"])
        actions = batch["actions"]
        rewards = batch["rewards"]
        not_dones = batch["not_dones"]

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(next_features) + noise).clamp(self.action_low, self.action_high)
            target_q1, target_q2 = self.critic_target(next_features, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + not_dones * self.gamma * target_q

        current_q1, current_q2 = self.critic(obs_features, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.zeros((), device=self.device)
        if self.num_updates % self.policy_delay == 0:
            actor_actions = self.actor(obs_features)
            actor_loss = -self.critic.q1_forward(obs_features, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            utils.soft_update_params(self.actor, self.actor_target, self.tau)
            utils.soft_update_params(self.critic, self.critic_target, self.tau)

        self.num_updates += 1
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
            "target_q": float(target_q.mean().item()),
            "reward": float(rewards.mean().item()),
            "her_ratio": float(batch["her_mask"].mean().item()),
        }

    def _features(self, observations, desired_goals):
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if desired_goals.ndim == 1:
            desired_goals = desired_goals.unsqueeze(0)
        return torch.cat([observations, desired_goals], dim=-1)

    def _split_obs(self, obs, goal_obs=None):
        if isinstance(obs, dict):
            goal_obs = obs["desired_goal"] if goal_obs is None else goal_obs
            obs = obs["observation"]
        if goal_obs is None:
            raise ValueError("TD3HERAgent.act requires goal_obs when obs is not a dict observation.")
        return np.asarray(obs, dtype=np.float32), np.asarray(goal_obs, dtype=np.float32)