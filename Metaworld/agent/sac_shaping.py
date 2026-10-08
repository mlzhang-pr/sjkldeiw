import numpy as np
import torch
import torch.nn.functional as F
from agent import utils
import abc
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor
import os


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


class SACAgentIR(Agent):
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg=DoubleQCritic,
        actor_cfg=DiagGaussianActor,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=[0.9, 0.999],
        actor_lr=1e-4,
        actor_betas=[0.9, 0.999],
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=[0.9, 0.999],
        critic_tau=0.005,
        critic_target_update_frequency=1,
        batch_size=256,
        learnable_temperature=True,
        normalize_state_entropy=True,
        quasimetric_agent=None,
        intrinsic_reward_coef=0.01,
        state_delta_reduction="sum",
        obs_history_max_size=8,
        metric_type="123",
        intrinsic_reward_steps=100000,
    ):
        super().__init__()

        self.action_range = action_range

        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic_cfg = critic_cfg
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.normalize_state_entropy = normalize_state_entropy
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas
        self.actor_cfg = actor_cfg
        self.actor_betas = actor_betas
        self.alpha_lr = alpha_lr
        self.quasimetric_agent = quasimetric_agent
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.state_delta_reduction = state_delta_reduction
        self.obs_history_max_size = obs_history_max_size
        self.obs_history = None
        self.metric_type = metric_type
        self.intrinsic_reward_steps = intrinsic_reward_steps
        self.task_start_step = 0

        self.critic = self.critic_cfg(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, hidden_depth=2
        ).to(self.device)
        self.critic_target = self.critic_cfg(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, hidden_depth=2
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = self.actor_cfg(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -action_dim

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
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

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def set_quasimetric_agent(self, quasimetric_agent):
        self.quasimetric_agent = quasimetric_agent
        self.reset()

    def set_task_start_step(self, step):
        self.task_start_step = int(step)
        self.reset()

    def reset(self):
        self.obs_history = None

    def _intrinsic_reward_task_step(self, step):
        return max(0, int(step) - int(self.task_start_step))

    def _intrinsic_reward_active(self, step):
        if self.intrinsic_reward_steps is None:
            return True
        return self._intrinsic_reward_task_step(step) < int(self.intrinsic_reward_steps)

    def _quasimetric_module(self):
        if self.quasimetric_agent is None:
            return None
        return getattr(self.quasimetric_agent, "quasimetric", self.quasimetric_agent)

    def _state_encoder(self, quasimetric):
        if hasattr(quasimetric, "state_encoder"):
            return quasimetric.state_encoder
        raise AttributeError(
            "The quasimetric module must expose state_encoder for reward shaping."
        )

    def _transition_encoder(self, quasimetric):
        if hasattr(quasimetric, "latent_transition_encoder"):
            return quasimetric.latent_transition_encoder
        raise AttributeError(
            "metric_type=mqe requires latent_transition_encoder for action-conditioned reward shaping."
        )

    def _use_action_for_intrinsic_reward(self):
        return str(self.metric_type).lower() == "mqe"

    def _reduce_state_delta(self, state_delta):
        if self.state_delta_reduction == "sum":
            return state_delta.sum(dim=-1, keepdim=True)
        if self.state_delta_reduction == "mean":
            return state_delta.mean(dim=-1, keepdim=True)
        if self.state_delta_reduction == "norm":
            return torch.linalg.norm(state_delta, dim=-1, keepdim=True)
        raise ValueError(
            f"Unsupported state_delta_reduction: {self.state_delta_reduction}"
        )

    def _ensure_obs_history(self, batch_size, obs_history=None):
        if obs_history is None:
            if self.obs_history is None or len(self.obs_history) != batch_size:
                self.obs_history = [None for _ in range(batch_size)]
            return self.obs_history

        if len(obs_history) != batch_size:
            raise ValueError(
                f"obs_history length {len(obs_history)} does not match batch size {batch_size}."
            )
        return obs_history

    def _obs_history_limit(self):
        if self.obs_history_max_size is None:
            return max(2, int(self.batch_size))
        return max(2, int(self.obs_history_max_size))

    def _history_transition_limit(self):
        return max(1, self._obs_history_limit() - 1)

    def _append_obs_history(self, obs_history, batch_idx, curr_rep, next_rep):
        history = obs_history[batch_idx]
        if history is None:
            history = torch.cat([curr_rep, next_rep], dim=0)
        else:
            history = torch.as_tensor(history, device=self.device).float()
            if history.ndim == curr_rep.ndim - 1:
                history = history.unsqueeze(0)
            history = torch.cat([history, next_rep], dim=0)

        history_limit = self._obs_history_limit()
        if history.shape[0] > history_limit:
            history = history[-history_limit:]
        obs_history[batch_idx] = history.detach()
        return history

    def _quasimetric_distance(self, quasimetric, history_rep, next_rep):

        return quasimetric.distance(history_rep, next_rep)

    def _history_obs_to_rep(self, state_encoder, history_obs):
        history_obs = torch.as_tensor(history_obs, device=self.device).float()
        if history_obs.ndim == 1:
            history_obs = history_obs.unsqueeze(0)
        history_limit = self._obs_history_limit()
        if history_obs.shape[0] > history_limit:
            history_obs = history_obs[-history_limit:]
        return state_encoder(history_obs)

    def _batch_history_obs_to_rep(self, state_encoder, obs_history):
        history_limit = self._obs_history_limit()
        batch_size = len(obs_history)
        first_history = np.asarray(obs_history[0])
        obs_shape = first_history.shape[1:]

        padded_history = np.zeros(
            (batch_size, history_limit, *obs_shape), dtype=np.float32
        )
        valid_mask = np.zeros((batch_size, history_limit), dtype=bool)
        history_lengths = np.zeros(batch_size, dtype=np.float32)

        for batch_idx, history in enumerate(obs_history):
            history = np.asarray(history, dtype=np.float32)
            if history.shape[0] > history_limit:
                history = history[-history_limit:]
            history_len = history.shape[0]
            padded_history[batch_idx, -history_len:] = history
            valid_mask[batch_idx, -history_len:] = True
            history_lengths[batch_idx] = history_len

        padded_history = torch.as_tensor(padded_history, device=self.device).float()
        flat_history = padded_history.reshape(batch_size * history_limit, *obs_shape)
        history_rep = state_encoder(flat_history).reshape(batch_size, history_limit, -1)
        valid_mask = torch.as_tensor(valid_mask, device=self.device)
        return history_rep, valid_mask, history_lengths

    def _batch_history_transition_to_rep(
        self, state_encoder, transition_encoder, obs_history, action_history
    ):
        history_limit = self._obs_history_limit()
        transition_limit = self._history_transition_limit()
        batch_size = len(obs_history)
        first_history = np.asarray(obs_history[0])
        first_action_history = np.asarray(action_history[0])
        obs_shape = first_history.shape[1:]
        action_shape = first_action_history.shape[1:]

        padded_history = np.zeros(
            (batch_size, history_limit, *obs_shape), dtype=np.float32
        )
        padded_actions = np.zeros(
            (batch_size, transition_limit, *action_shape), dtype=np.float32
        )
        valid_transition_mask = np.zeros((batch_size, transition_limit), dtype=bool)
        history_lengths = np.zeros(batch_size, dtype=np.float32)
        action_lengths = np.zeros(batch_size, dtype=np.float32)

        for batch_idx, history in enumerate(obs_history):
            history = np.asarray(history, dtype=np.float32)
            actions = np.asarray(action_history[batch_idx], dtype=np.float32)
            if history.shape[0] > history_limit:
                removed = history.shape[0] - history_limit
                history = history[-history_limit:]
                actions = actions[removed:]
            action_len = min(actions.shape[0], history.shape[0] - 1, transition_limit)
            if action_len > 0:
                actions = actions[-action_len:]
                padded_actions[batch_idx, -action_len:] = actions
                valid_transition_mask[batch_idx, -action_len:] = True
            history_len = min(history.shape[0], history_limit)
            padded_history[batch_idx, -history_len:] = history[-history_len:]
            history_lengths[batch_idx] = history_len
            action_lengths[batch_idx] = action_len

        padded_history = torch.as_tensor(padded_history, device=self.device).float()
        flat_history = padded_history.reshape(batch_size * history_limit, *obs_shape)
        state_rep = state_encoder(flat_history).reshape(batch_size, history_limit, -1)
        source_state_rep = state_rep[:, :-1].reshape(batch_size * transition_limit, -1)

        padded_actions = torch.as_tensor(padded_actions, device=self.device).float()
        flat_actions = padded_actions.reshape(
            batch_size * transition_limit, *action_shape
        )
        transition_rep = transition_encoder(source_state_rep, flat_actions).reshape(
            batch_size, transition_limit, -1
        )
        valid_transition_mask = torch.as_tensor(
            valid_transition_mask, device=self.device
        )
        return transition_rep, valid_transition_mask, history_lengths, action_lengths

    def _compute_buffer_history_intrinsic_reward(
        self,
        quasimetric,
        state_encoder,
        obs_history,
        obs=None,
        action=None,
        action_history=None,
    ):
        if self._use_action_for_intrinsic_reward():
            if action_history is None:
                raise ValueError(
                    "metric_type=mqe requires action_history for action-conditioned reward shaping."
                )
            transition_encoder = self._transition_encoder(quasimetric)
            history_transition_rep, transition_mask, history_lengths, action_lengths = (
                self._batch_history_transition_to_rep(
                    state_encoder,
                    transition_encoder,
                    obs_history,
                    action_history,
                )
            )
            obs_rep = state_encoder(obs)
            query_rep = transition_encoder(obs_rep, action).unsqueeze(1)
            candidate_mask = transition_mask.clone()
            candidate_mask[:, -1] = False
            dists = self._quasimetric_distance(
                quasimetric, history_transition_rep, query_rep
            )
            dists = dists.masked_fill(~candidate_mask, float("inf"))
        else:
            history_rep, valid_mask, history_lengths = self._batch_history_obs_to_rep(
                state_encoder, obs_history
            )
            previous_rep = history_rep[:, :-1]
            previous_mask = valid_mask[:, :-1]
            next_rep = history_rep[:, -1:].contiguous()
            dists = self._quasimetric_distance(quasimetric, previous_rep, next_rep)
            dists = dists.masked_fill(~previous_mask, float("inf"))
            action_lengths = None

        intrinsic_reward = dists.min(dim=1, keepdim=True).values
        intrinsic_reward = torch.where(
            torch.isinf(intrinsic_reward),
            torch.zeros_like(intrinsic_reward),
            intrinsic_reward,
        )
        valid_mask_for_dists = torch.isfinite(dists)
        valid_dists = dists[valid_mask_for_dists]
        if valid_dists.numel() == 0:
            valid_dists = intrinsic_reward.reshape(-1)
        return intrinsic_reward, valid_dists, history_lengths, action_lengths

    def _quasimetric_q(self, quasimetric, obs, action, next_obs):
        if hasattr(quasimetric, "structure_bonus"):
            return quasimetric.structure_bonus(obs, action, next_obs).reshape(
                obs.shape[0], -1
            )

        critic = getattr(quasimetric, "critic", None)
        if critic is not None:
            q_value = critic(obs, action, next_obs)
            if isinstance(q_value, tuple):
                q_value = torch.min(q_value[0], q_value[1])
            return q_value.reshape(obs.shape[0], -1)

        if hasattr(quasimetric, "latent_transition_encoder") and hasattr(
            quasimetric, "target_state_encoder"
        ):
            state_rep = quasimetric.state_encoder(obs)
            if hasattr(quasimetric, "transition_representation"):
                transition_rep = quasimetric.transition_representation(
                    obs, action, state_rep
                )
            else:
                transition_rep = quasimetric.latent_transition_encoder(
                    state_rep, action
                )
            next_rep = quasimetric.target_state_encoder(next_obs)
            if hasattr(quasimetric, "distance"):
                return torch.exp(
                    -quasimetric.distance(transition_rep, next_rep)
                ).reshape(obs.shape[0], -1)
            return transition_rep

        raise AttributeError(
            "The quasimetric module must expose structure_bonus, critic, or latent_transition_encoder."
        )

    def compute_intrinsic_reward(
        self, obs, action, next_obs, obs_history=None, action_history=None
    ):
        quasimetric = self._quasimetric_module()
        if quasimetric is None:
            return None, {}

        obs = torch.as_tensor(obs, device=self.device).float()
        action = torch.as_tensor(action, device=self.device).float()
        next_obs = torch.as_tensor(next_obs, device=self.device).float()
        batch_size = obs.shape[0]
        use_buffer_history = obs_history is not None
        obs_history = self._ensure_obs_history(batch_size, obs_history)

        with torch.no_grad():
            state_encoder = self._state_encoder(quasimetric)
            if use_buffer_history:
                intrinsic_reward, history_dists, history_lengths, action_lengths = (
                    self._compute_buffer_history_intrinsic_reward(
                        quasimetric,
                        state_encoder,
                        obs_history,
                        obs=obs,
                        action=action,
                        action_history=action_history,
                    )
                )
            else:
                curr_rep = state_encoder(obs)
                next_rep = state_encoder(next_obs)
                if self._use_action_for_intrinsic_reward():
                    transition_encoder = self._transition_encoder(quasimetric)
                    transition_rep = transition_encoder(curr_rep, action)
                intrinsic_rewards = []
                history_lengths = []
                all_dists = []
                for batch_idx in range(batch_size):
                    history = self._append_obs_history(
                        obs_history,
                        batch_idx,
                        curr_rep[batch_idx].view(1, -1),
                        next_rep[batch_idx].view(1, -1),
                    )
                    if self._use_action_for_intrinsic_reward():
                        dists = self._quasimetric_distance(
                            quasimetric,
                            transition_rep[batch_idx].view(1, -1),
                            history[:-1],
                        )
                    else:
                        dists = self._quasimetric_distance(
                            quasimetric, history[:-1], history[-1].view(1, -1)
                        )
                    intrinsic_rewards.append(dists.min().view(1, 1))
                    history_lengths.append(history.shape[0])
                    all_dists.append(dists)

                intrinsic_reward = torch.cat(intrinsic_rewards, dim=0)
                history_dists = torch.cat(all_dists, dim=0)
                action_lengths = None

        metrics = {
            "intrinsic_reward": float(intrinsic_reward.mean().item()),
            "history_dist_mean": float(history_dists.mean().item()),
            "history_dist_min": float(history_dists.min().item()),
            "history_dist_max": float(history_dists.max().item()),
            "obs_history_len": float(np.mean(history_lengths)),
            "intrinsic_reward_uses_action": float(
                self._use_action_for_intrinsic_reward()
            ),
        }
        if action_lengths is not None:
            metrics["action_history_len"] = float(np.mean(action_lengths))
        return intrinsic_reward, metrics

    def shape_reward(
        self,
        reward,
        obs,
        action,
        next_obs,
        obs_history=None,
        action_history=None,
        intrinsic_reward_active=True,
    ):
        if self.intrinsic_reward_coef == 0.0 or not intrinsic_reward_active:
            return reward, {}
        intrinsic_reward, metrics = self.compute_intrinsic_reward(
            obs, action, next_obs, obs_history, action_history
        )
        if intrinsic_reward is None:
            return reward, {}

        shaped_reward = reward + self.intrinsic_reward_coef * intrinsic_reward
        metrics["env_reward"] = float(reward.mean().item())
        metrics["shaped_reward"] = float(shaped_reward.mean().item())
        return shaped_reward, metrics

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)

        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)

        critic_loss = (
            F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        ) * 0.5

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def compute_target_q(self, reward, next_obs, not_done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        return target_Q.detach()

    def update_with_target_q(self, obs, action, target_Q, step):
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = (
            F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        ) * 0.5
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if step % self.actor_update_frequency == 0:
            loss_pi, loss_alpha = self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return {
            "alpha": loss_alpha.item(),
            "actor": loss_pi.item(),
            "critic": critic_loss.item(),
        }

    def save(self, model_dir, model_name):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pt" % (model_dir, model_name))
        torch.save(
            self.critic.state_dict(), "%s/%s_critic.pt" % (model_dir, model_name)
        )
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pt" % (model_dir, model_name),
        )

    def load(self, model_dir, model_name):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pt" % (model_dir, model_name))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pt" % (model_dir, model_name))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pt" % (model_dir, model_name))
        )

    def update_actor_and_alpha(self, obs):

        dist = self.actor(obs)
        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss

    def sac_update(self, replay_buffer, step, obs_history=None, action_history=None):
        intrinsic_reward_active = self._intrinsic_reward_active(step)
        if (
            intrinsic_reward_active
            and obs_history is None
            and hasattr(replay_buffer, "sample_with_obs_history")
        ):
            sample = replay_buffer.sample_with_obs_history(
                self.batch_size,
                history_len=self._history_transition_limit(),
            )
            if len(sample) == 8:
                (
                    obs,
                    action,
                    reward,
                    success,
                    next_obs,
                    not_done_no_max,
                    obs_history,
                    action_history,
                ) = sample
            else:
                obs, action, reward, success, next_obs, not_done_no_max, obs_history = (
                    sample
                )
        else:
            obs, action, reward, success, next_obs, not_done_no_max = (
                replay_buffer.sample(self.batch_size)
            )

        obs, action, reward, success, next_obs, not_done_no_max = (
            replay_buffer.as_torch(
                obs, action, reward, success, next_obs, not_done_no_max
            )
        )

        reward, shaping_metrics = self.shape_reward(
            reward,
            obs,
            action,
            next_obs,
            obs_history,
            action_history,
            intrinsic_reward_active=intrinsic_reward_active,
        )

        loss_q = self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        loss_pi = torch.zeros((), device=self.device)
        loss_alpha = torch.zeros((), device=self.device)
        if step % self.actor_update_frequency == 0:
            loss_pi, loss_alpha = self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        metrics = {
            "alpha": loss_alpha.item(),
            "actor": loss_pi.item(),
            "critic": loss_q.item(),
            "intrinsic_reward_active": float(intrinsic_reward_active),
            "intrinsic_reward_task_step": float(self._intrinsic_reward_task_step(step)),
        }
        metrics.update(shaping_metrics)
        return metrics

    def actor_nll(self, replay_buffer, update_num):
        print_interval = update_num // 5

        for i in range(update_num):
            obs, action, reward, success, next_obs, not_done_no_max = (
                replay_buffer.sample(self.batch_size)
            )

            obs, action, reward, success, next_obs, not_done_no_max = (
                replay_buffer.as_torch(
                    obs, action, reward, success, next_obs, not_done_no_max
                )
            )
            eps = 1e-6
            action = torch.clamp(action, min=-1.0 + eps, max=1.0 - eps)
            assert not torch.isnan(obs).any(), f"obs nan: {obs}"
            dist = self.actor(obs)
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

    def actor_wd_loss(
        self,
        current_agent,
        pre_meta_agent,
        current_buffer,
        current_buffer_start_idx,
        meta_buffer,
        update_num,
    ):
        print_interval = update_num // 5

        for i in range(update_num):
            if pre_meta_agent is not None:
                (
                    meta_obs,
                    meta_action,
                    meta_reward,
                    meta_success,
                    meta_next_obs,
                    meta_not_done_no_max,
                ) = meta_buffer.sample(self.batch_size)
                meta_obs = torch.as_tensor(meta_obs, device=self.device).float()
                with torch.no_grad():
                    pre_meta_agent_dist = pre_meta_agent.actor(meta_obs)
                    meta_mu, meta_std = (
                        pre_meta_agent_dist.loc,
                        pre_meta_agent_dist.scale,
                    )

                dist1 = self.actor(meta_obs)
                mu1, std1 = dist1.loc, dist1.scale

                meta_loss = torch.mean(
                    torch.square(mu1 - meta_mu).sum(-1)
                    + torch.square(std1 - meta_std).sum(-1)
                )
            else:
                meta_loss = 0

            (
                current_obs,
                current_action,
                current_reward,
                current_success,
                current_next_obs,
                current_not_done_no_max,
            ) = current_buffer.sample_last(current_buffer_start_idx, self.batch_size)
            current_obs = torch.as_tensor(current_obs, device=self.device).float()
            with torch.no_grad():
                current_agent_dist = current_agent.actor(current_obs)
                current_mu, current_std = (
                    current_agent_dist.loc,
                    current_agent_dist.scale,
                )

            dist2 = self.actor(current_obs)
            mu2, std2 = dist2.loc, dist2.scale

            current_loss = torch.mean(
                torch.square(mu2 - current_mu).sum(-1)
                + torch.square(std2 - current_std).sum(-1)
            )

            actor_loss = meta_loss + current_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if i % print_interval == 0:
                print("actor_loss:", i, actor_loss)

        return actor_loss.item()
