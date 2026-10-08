import abc
import os

import numpy as np
import torch
import torch.nn.functional as F

from agent import utils
from agent.actor import DiagGaussianActor
from agent.critic import DoubleQCritic_metrtic


class Agent(object):
    def reset(self):
        pass

    @abc.abstractmethod
    def train(self, training=True):
        pass

    @abc.abstractmethod
    def update(self, replay_buffer, step):
        pass

    @abc.abstractmethod
    def act(self, obs, sample=False):
        pass


class SACAgent(Agent):
    """Goal-conditioned SAC with HER-style future goal relabeling."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        rep_dim,
        action_range,
        device,
        critic_cfg=DoubleQCritic_metrtic,
        actor_cfg=DiagGaussianActor,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=1,
        batch_size=256,
        learnable_temperature=True,
        normalize_state_entropy=True,
        learn_goal_encoder=True,
        goal_discount=0.995,
        goal_next_state_sample=0.2,
        goal_reward_scale=0.1,
        task_reward_scale=1.0,
        goal_reward_type="binary",
        behavior_goal_success_only=True,
        encode_actor_critic_goal=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rep_dim = rep_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.normalize_state_entropy = normalize_state_entropy

        self.critic_cfg = critic_cfg
        self.actor_cfg = actor_cfg
        self.goal_discount = goal_discount
        self.goal_next_state_sample = goal_next_state_sample
        self.goal_reward_scale = goal_reward_scale
        self.task_reward_scale = task_reward_scale
        self.goal_reward_type = goal_reward_type
        self.behavior_goal_success_only = behavior_goal_success_only
        self.encode_actor_critic_goal = encode_actor_critic_goal
        self.actor_critic_goal_dim = rep_dim if encode_actor_critic_goal else obs_dim
        self.behavior_goal = None

        self.goal_encoder = None
        if learn_goal_encoder:
            self.goal_encoder = utils.MLP(obs_dim, 256, rep_dim, 2).to(self.device)

        self.critic = self.critic_cfg(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rep_dim=self.actor_critic_goal_dim,
            hidden_dim=256,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target = self.critic_cfg(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rep_dim=self.actor_critic_goal_dim,
            hidden_dim=256,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = self.actor_cfg(
            obs_dim=obs_dim + self.actor_critic_goal_dim,
            action_dim=action_dim,
            hidden_dim=256,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        self.log_alpha = torch.tensor(np.log(init_temperature), device=self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        critic_parameters = list(self.critic.parameters())
        if self.goal_encoder is not None and self.encode_actor_critic_goal:
            critic_parameters.extend(self.goal_encoder.parameters())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=critic_lr, betas=critic_betas
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.train()
        self.critic_target.train()

    def eval(self):
        self.train(False)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.goal_encoder is not None:
            self.goal_encoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _to_goal_tensor(self, goal):
        goal = torch.as_tensor(goal, device=self.device).float()
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        return goal

    def _zero_goal_rep(self, batch_size):
        return torch.zeros(batch_size, self.actor_critic_goal_dim, device=self.device)

    def encode_goal(self, goal_obs):
        goal_obs = self._to_goal_tensor(goal_obs)

        if self.goal_encoder is not None:
            return self.goal_encoder(goal_obs)
        if goal_obs.shape[-1] == self.rep_dim:
            return goal_obs
        raise NotImplementedError(
            "Goal encoding is undefined. Provide a goal encoder or override encode_goal()."
        )

    def _goal_input_from_obs(self, goal_obs):
        goal_obs = self._to_goal_tensor(goal_obs)
        goal_input = (
            self.encode_goal(goal_obs) if self.encode_actor_critic_goal else goal_obs
        )
        if goal_input.shape[-1] != self.actor_critic_goal_dim:
            raise ValueError(
                f"Goal input has dim {goal_input.shape[-1]}, expected {self.actor_critic_goal_dim}."
            )
        return goal_input

    def _prepare_goal_rep(self, batch_size, goal_obs=None, goal_rep=None, detach=False):
        if goal_rep is None:
            if goal_obs is not None:
                goal_rep = self._goal_input_from_obs(goal_obs)
            elif self.behavior_goal is not None:
                goal_rep = self._goal_input_from_obs(self.behavior_goal)
            else:
                goal_rep = self._zero_goal_rep(batch_size)
        else:
            goal_rep = self._to_goal_tensor(goal_rep)

        if goal_rep.ndim == 1:
            goal_rep = goal_rep.unsqueeze(0)
        if batch_size is not None and goal_rep.shape[0] == 1 and batch_size != 1:
            goal_rep = goal_rep.repeat(batch_size, 1)
        if goal_rep.shape[-1] != self.actor_critic_goal_dim:
            raise ValueError(
                f"Goal input has dim {goal_rep.shape[-1]}, expected {self.actor_critic_goal_dim}."
            )
        return goal_rep.detach() if detach else goal_rep

    def _augment_obs(self, obs, goal_rep):
        return torch.cat([obs, goal_rep], dim=-1)

    def act(self, obs, sample=False, goal_obs=None, goal_rep=None):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        goal_rep = self._prepare_goal_rep(
            obs.shape[0], goal_obs=goal_obs, goal_rep=goal_rep, detach=True
        )
        dist = self.actor(self._augment_obs(obs, goal_rep))
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def shape_reward(self, reward, obs, action, next_obs, metrics):
        return reward

    def update_critic(self, obs, action, reward, next_obs, not_done, goal_rep):
        target_goal_rep = goal_rep.detach()
        dist = self.actor(self._augment_obs(next_obs, target_goal_rep))
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(
            next_obs, next_action, target_goal_rep
        )
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + not_done * self.discount * target_V
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action, goal_rep)
        critic_loss = 0.5 * (
            F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def compute_target_q(self, reward, next_obs, not_done, goal_rep=None):
        goal_rep = self._prepare_goal_rep(
            next_obs.shape[0], goal_rep=goal_rep, detach=True
        )
        dist = self.actor(self._augment_obs(next_obs, goal_rep))
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action, goal_rep)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        return (reward + not_done * self.discount * target_V).detach()

    def update_with_target_q(self, obs, action, target_Q, step, goal_rep=None):
        goal_rep = self._prepare_goal_rep(obs.shape[0], goal_rep=goal_rep, detach=True)
        current_Q1, current_Q2 = self.critic(obs, action, goal_rep)
        critic_loss = 0.5 * (
            F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        alpha_loss = None
        if step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, goal_rep.detach())

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return {
            "alpha": float(alpha_loss.item())
            if alpha_loss is not None
            else float(self.alpha.item()),
            "actor": float(actor_loss.item()) if actor_loss is not None else 0.0,
            "critic": float(critic_loss.item()),
        }

    def save(self, model_dir, model_name):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.actor.state_dict(), f"{model_dir}/{model_name}_actor.pt")
        torch.save(self.critic.state_dict(), f"{model_dir}/{model_name}_critic.pt")
        torch.save(
            self.critic_target.state_dict(),
            f"{model_dir}/{model_name}_critic_target.pt",
        )
        if self.goal_encoder is not None:
            torch.save(
                self.goal_encoder.state_dict(),
                f"{model_dir}/{model_name}_goal_encoder.pt",
            )

    def load(self, model_dir, model_name):
        self.actor.load_state_dict(
            torch.load(f"{model_dir}/{model_name}_actor.pt", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{model_dir}/{model_name}_critic.pt", map_location=self.device)
        )
        self.critic_target.load_state_dict(
            torch.load(
                f"{model_dir}/{model_name}_critic_target.pt", map_location=self.device
            )
        )
        goal_encoder_path = f"{model_dir}/{model_name}_goal_encoder.pt"
        if self.goal_encoder is not None and os.path.exists(goal_encoder_path):
            self.goal_encoder.load_state_dict(
                torch.load(goal_encoder_path, map_location=self.device)
            )

    def update_actor_and_alpha(self, obs, goal_rep):
        dist = self.actor(self._augment_obs(obs, goal_rep))
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, goal_rep)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss

    def _batch_to_torch(self, batch):
        return {
            key: value.to(self.device).float()
            if torch.is_tensor(value)
            else torch.as_tensor(value, device=self.device).float()
            for key, value in batch.items()
        }

    def _default_metric_batch(self, replay_buffer):
        obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )
        obs, action, reward, success, next_obs, not_done_no_max = (
            replay_buffer.as_torch(
                obs,
                action,
                reward,
                success,
                next_obs,
                not_done_no_max,
            )
        )
        ones = torch.ones_like(reward)
        return {
            "obses": obs,
            "actions": action,
            "env_rewards": reward,
            "successes": success,
            "next_obses": next_obs,
            "not_dones_no_max": not_done_no_max,
            "goals": next_obs,
            "goal_rewards": reward,
            "goal_reached": ones,
            "goal_not_dones": not_done_no_max,
            "goal_steps": ones,
        }

    def _sample_metric_batch(self, replay_buffer):
        if hasattr(replay_buffer, "sample_metric_batch"):
            batch = replay_buffer.sample_metric_batch(
                batch_size=self.batch_size,
                discount=self.goal_discount,
                next_state_sample=self.goal_next_state_sample,
                reward_type=self.goal_reward_type,
            )
            return self._batch_to_torch(batch)
        return self._default_metric_batch(replay_buffer)

    def set_behavior_goal(self, goal_obs):
        if goal_obs is None:
            self.behavior_goal = None
        else:
            self.behavior_goal = np.array(goal_obs, copy=True)

    def _refresh_behavior_goal(self, replay_buffer, batch):
        goal_obs = None
        if hasattr(replay_buffer, "sample_behavior_goal"):
            goal_obs = replay_buffer.sample_behavior_goal(
                discount=self.goal_discount,
                success_only=self.behavior_goal_success_only,
            )

        if (
            goal_obs is None
            and batch.get("goals") is not None
            and batch["goals"].shape[0] > 0
        ):
            goal_idx = 0
            if "goal_reached" in batch:
                goal_idx = int(torch.argmax(batch["goal_reached"].reshape(-1)).item())
            goal_obs = utils.to_np(batch["goals"][goal_idx])

        if goal_obs is not None:
            self.set_behavior_goal(goal_obs)

    def _update_from_batch(self, replay_buffer, batch, step):
        obs = batch["obses"]
        action = batch["actions"]
        next_obs = batch["next_obses"]
        goal_rep = self._prepare_goal_rep(
            obs.shape[0], goal_obs=batch.get("goals"), detach=False
        )

        metrics = {
            "env_reward": float(batch["env_rewards"].mean().item()),
            "goal_reward": float(batch["goal_rewards"].mean().item()),
            "goal_reached_ratio": float(batch["goal_reached"].mean().item()),
            "goal_steps": float(batch["goal_steps"].mean().item()),
        }
        use_env_reward_only = batch.get("use_env_reward_only")
        if use_env_reward_only is not None and torch.any(use_env_reward_only > 0.5):
            reward = batch["env_rewards"]
        else:
            reward = (
                self.task_reward_scale * batch["env_rewards"]
                + self.goal_reward_scale * batch["goal_rewards"]
            )
        reward = self.shape_reward(reward, obs, action, next_obs, metrics)

        critic_loss = self.update_critic(
            obs, action, reward, next_obs, batch["goal_not_dones"], goal_rep
        )

        actor_loss = None
        alpha_loss = None
        if step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, goal_rep.detach())

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        self._refresh_behavior_goal(replay_buffer, batch)
        metrics.update(
            {
                "alpha": float(alpha_loss.item())
                if alpha_loss is not None
                else float(self.alpha.item()),
                "actor": float(actor_loss.item()) if actor_loss is not None else 0.0,
                "critic": float(critic_loss.item()),
                "behavior_goal_ready": float(self.behavior_goal is not None),
            }
        )
        return metrics

    def actor_nll(self, replay_buffer, update_num):
        print_interval = max(update_num // 5, 1)

        for i in range(update_num):
            batch = self._sample_metric_batch(replay_buffer)
            obs = batch["obses"]
            action = batch["actions"]
            goal_obs = batch.get("actor_goals", batch.get("goals"))
            goal_rep = self._prepare_goal_rep(
                obs.shape[0], goal_obs=goal_obs, detach=True
            )
            eps = 1e-6
            action = torch.clamp(action, min=-1.0 + eps, max=1.0 - eps)
            assert not torch.isnan(obs).any(), f"obs nan: {obs}"
            dist = self.actor(self._augment_obs(obs, goal_rep))
            assert not torch.isnan(dist.loc).any(), f"dist: {dist.loc}"
            assert torch.all(action > -1) and torch.all(action < 1), (
                f"action out of range: {action}"
            )
            assert not torch.isnan(action).any(), f"action nan: {action}"
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_loss = -log_prob.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if i % print_interval == 0:
                print("actor_loss:", i, actor_loss)
        return actor_loss.item()

    def update(self, replay_buffer, step):
        batch = self._sample_metric_batch(replay_buffer)
        return self._update_from_batch(replay_buffer, batch, step)

    def sac_update(self, replay_buffer, step):
        return self.update(replay_buffer, step)


MetricSACAgent = SACAgent
