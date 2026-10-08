import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F

from agent.sac_metric import SACAgent

from .config import ContinualQuasimetricAgentConfig, QuasimetricConfig
from .memory import TaskAwareReplayMemory
from .structure import MultistepQuasimetricLearner


def _distribution_mean_action(distribution):
    if hasattr(distribution, "mean"):
        return distribution.mean
    if hasattr(distribution, "mode"):
        return distribution.mode
    if hasattr(distribution, "loc"):
        return distribution.loc
    raise AttributeError(
        f"Cannot extract a deterministic action from {type(distribution).__name__}."
    )


class ContinualQuasimetricSACAgent(SACAgent):
    """SAC with a slow quasimetric structure module for continual RL."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        quasimetric_cfg: Optional[QuasimetricConfig] = None,
        continual_cfg: Optional[ContinualQuasimetricAgentConfig] = None,
        **kwargs,
    ):
        self.quasimetric_cfg = quasimetric_cfg or QuasimetricConfig()
        self.continual_cfg = continual_cfg or ContinualQuasimetricAgentConfig()
        if self.continual_cfg.q_loss_coef < 0.0:
            raise ValueError("q_loss_coef must be non-negative")
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rep_dim=self.quasimetric_cfg.latent_dim,
            action_range=action_range,
            device=device,
            learn_goal_encoder=False,
            goal_discount=self.quasimetric_cfg.discount,
            goal_next_state_sample=self.quasimetric_cfg.next_state_sample,
            goal_reward_scale=self.continual_cfg.goal_reward_scale,
            task_reward_scale=self.continual_cfg.task_reward_scale,
            goal_reward_type=self.continual_cfg.goal_reward_type,
            behavior_goal_success_only=self.continual_cfg.behavior_goal_success_only,
            encode_actor_critic_goal=self.continual_cfg.encode_actor_critic_goal,
            **kwargs,
        )
        self.quasimetric = MultistepQuasimetricLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            config=self.quasimetric_cfg,
        )
        self.structure_memory = TaskAwareReplayMemory(
            device=device,
            max_tasks=self.continual_cfg.memory_max_tasks,
        )

    def train(self, training=True):
        super().train(training)
        if hasattr(self, "quasimetric"):
            self.quasimetric.train(training)

    def encode_goal(self, goal_obs):
        goal_obs = torch.as_tensor(goal_obs, device=self.device).float()
        if goal_obs.ndim == 1:
            goal_obs = goal_obs.unsqueeze(0)
        with torch.no_grad():
            return self.quasimetric.target_state_encoder(goal_obs)

    def shape_reward(self, reward, obs, action, next_obs, metrics):
        coef = self.continual_cfg.structure_bonus_coef
        if coef == 0.0:
            return reward

        with torch.no_grad():
            bonus = self.quasimetric.structure_bonus(obs, action, next_obs).unsqueeze(-1)
        metrics["structure_bonus"] = float(bonus.mean().item())
        return reward + coef * bonus

    def _shared_batch_size(self):
        shared_batch_size = self.continual_cfg.shared_batch_size
        if shared_batch_size is None:
            return self.batch_size
        if shared_batch_size <= 0:
            raise ValueError("shared_batch_size must be positive when provided.")
        return int(shared_batch_size)

    def _sample_shared_batch(self, replay_buffer, batch_size=None):
        shared_batch_size = self._shared_batch_size() if batch_size is None else int(batch_size)
        if hasattr(replay_buffer, "sample_metric_batch"):
            batch = replay_buffer.sample_metric_batch(
                batch_size=shared_batch_size,
                discount=self.goal_discount,
                next_state_sample=self.goal_next_state_sample,
                reward_type=self.goal_reward_type,
                structure_discount=self.quasimetric_cfg.discount,
                structure_lambda_=self.quasimetric_cfg.lambda_,
                structure_next_state_sample=self.quasimetric_cfg.next_state_sample,
            )
            return self._batch_to_torch(batch)

        batch = self._sample_metric_batch(replay_buffer)
        shared_batch = dict(batch)
        shared_batch.update(
            {
                "dones": 1.0 - batch["not_dones_no_max"],
                "value_goals": batch["goals"],
                "value_goals_offsets": batch["goal_steps"],
                "intermediate_value_goals": batch["next_obses"],
                "intermediate_value_goals_offsets": torch.ones_like(batch["goal_steps"]),
            }
        )
        return shared_batch

    def _update_actor_offline_from_batch(
        self,
        batch,
        bc_alpha=0.1,
        normalize_q_loss=True,
        teacher_agent=None,
        distill_batch=None,
        distill_loss_weight=0.0,
    ):
        bc_alpha = self.alpha.detach() if bc_alpha is None else torch.as_tensor(bc_alpha, device=self.device).float()

        obs = batch["obses"]
        action = batch["actions"]
        goal_obs = batch.get("actor_goals", batch.get("value_goals", batch.get("goals")))
        goal_rep = self._prepare_goal_rep(obs.shape[0], goal_obs=goal_obs, detach=True)

        eps = 1e-6
        action = torch.clamp(action, min=-1.0 + eps, max=1.0 - eps)
        assert not torch.isnan(obs).any(), f'obs nan: {obs}'
        assert not torch.isnan(action).any(), f'action nan: {action}'
        assert torch.all(action > -1) and torch.all(action < 1), f'action out of range: {action}'

        dist = self.actor(self._augment_obs(obs, goal_rep))
        assert not torch.isnan(dist.loc).any(), f'dist: {dist.loc}'

        q_action = dist.mean.clamp(*self.action_range)
        quasimetric_requires_grad = [param.requires_grad for param in self.quasimetric.parameters()]
        try:
            for param in self.quasimetric.parameters():
                param.requires_grad_(False)

            state_rep = self.quasimetric.state_encoder(obs)
            transition_rep = self.quasimetric.transition_representation(
                obs, q_action, state_rep
            )
            if goal_obs is None:
                if goal_rep.shape[-1] != self.quasimetric_cfg.latent_dim:
                    raise ValueError(
                        f"Quasimetric goal input has dim {goal_rep.shape[-1]}, "
                        f"expected {self.quasimetric_cfg.latent_dim}."
                    )
                qm_goal_rep = goal_rep
            else:
                qm_goal_rep = self.encode_goal(goal_obs)
            qm_distance = self.quasimetric.distance(transition_rep, qm_goal_rep)
        finally:
            for param, requires_grad in zip(self.quasimetric.parameters(), quasimetric_requires_grad):
                param.requires_grad_(requires_grad)
        actor_Q = -qm_distance
        if normalize_q_loss:
            q_loss = -actor_Q.mean() / actor_Q.detach().abs().mean().clamp_min(1e-6)
        else:
            q_loss = -actor_Q.mean()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        bc_loss = -(bc_alpha * log_prob).mean()
        distill_loss = torch.zeros((), device=self.device)
        teacher_action = None
        meta_action = None
        if teacher_agent is not None and distill_batch is not None and distill_loss_weight > 0.0:
            distill_obs = distill_batch["obses"]
            distill_goal_obs = distill_batch.get(
                "actor_goals",
                distill_batch.get("value_goals", distill_batch.get("goals")),
            )
            distill_goal_rep = self._prepare_goal_rep(
                distill_obs.shape[0],
                goal_obs=distill_goal_obs,
                detach=True,
            )
            meta_dist = self.actor(self._augment_obs(distill_obs, distill_goal_rep))
            meta_action = _distribution_mean_action(meta_dist)
            with torch.no_grad():
                teacher_dist = teacher_agent.actor_distribution(
                    distill_obs,
                    goal_obs=distill_goal_obs,
                    detach_goal=True,
                )
                teacher_action = _distribution_mean_action(teacher_dist).detach()
            distill_loss = torch.square(meta_action - teacher_action).sum(-1).mean()

        weighted_q_loss = self.continual_cfg.q_loss_coef * q_loss
        weighted_distill_loss = float(distill_loss_weight) * distill_loss
        actor_loss = weighted_q_loss + bc_loss + weighted_distill_loss


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = {
            "actor_loss": float(actor_loss.item()),
            "q_loss": float(q_loss.item()),
            "weighted_q_loss": float(weighted_q_loss.item()),
            "q_loss_coef": float(self.continual_cfg.q_loss_coef),
            "bc_loss": float(bc_loss.item()),
            "q_mean": float(actor_Q.mean().item()),
            "q_abs_mean": float(actor_Q.detach().abs().mean().item()),
            "distance": float(qm_distance.detach().mean().item()),
            "bc_log_prob": float(log_prob.mean().item()),
            "mse": float(F.mse_loss(dist.mean, action).item()),
            "std": float(dist.scale.mean().item()),
            "actor_distill_loss": float(distill_loss.item()),
            "actor_distill_weighted_loss": float(weighted_distill_loss.item()),
            "actor_distill_weight": float(distill_loss_weight),
            "actor_distill_applied": float(teacher_action is not None),
        }
        if teacher_action is not None:
            metrics["distill_teacher_action"] = float(teacher_action.mean().item())
            metrics["distill_meta_action"] = float(meta_action.detach().mean().item())
        return metrics

    def actor_offline(self, replay_buffer, update_num=1, bc_alpha=None, normalize_q_loss=True):
        bc_alpha = self.continual_cfg.bc_alpha if bc_alpha is None else bc_alpha
        print_interval = max(update_num // 5, 1)
        last_metrics = None
        for i in range(update_num):
            batch = self._sample_metric_batch(replay_buffer)
            last_metrics = self._update_actor_offline_from_batch(
                batch,
                bc_alpha=bc_alpha,
                normalize_q_loss=normalize_q_loss,
            )
            if update_num > 1 and i % print_interval == 0:
                print('actor_offline:', i, last_metrics)

        return last_metrics

    def update_structure_and_actor(
        self,
        replay_buffer,
        bc_alpha=None,
        normalize_q_loss=True,
        teacher_agent=None,
        distill_buffer=None,
        distill_loss_weight=0.0,
    ):
        if len(replay_buffer) < self.quasimetric_cfg.min_buffer_size:
            return {}, {}
        if distill_loss_weight < 0.0:
            raise ValueError("distill_loss_weight must be non-negative")

        bc_alpha = self.continual_cfg.bc_alpha if bc_alpha is None else bc_alpha
        batch = self._sample_shared_batch(
            replay_buffer,
            batch_size=self.quasimetric_cfg.batch_size,
        )
        distill_batch = None
        if (
            teacher_agent is not None
            and distill_buffer is not None
            and len(distill_buffer) > 0
            and distill_loss_weight > 0.0
        ):
            distill_batch = self._sample_shared_batch(
                distill_buffer,
                batch_size=self.quasimetric_cfg.batch_size,
            )
        structure_metrics = self.quasimetric.update(batch)
        actor_metrics = self._update_actor_offline_from_batch(
            batch,
            bc_alpha=bc_alpha,
            normalize_q_loss=normalize_q_loss,
            teacher_agent=teacher_agent,
            distill_batch=distill_batch,
            distill_loss_weight=distill_loss_weight,
        )
        return structure_metrics, actor_metrics

    def update_structure(self, replay_buffer, current_batch=None):
        return self.quasimetric.update_from_replay_buffer(
            replay_buffer,
            memory=self.structure_memory,
            current_batch=current_batch,
        )

    def update_structure_recent(self, replay_buffer, max_transitions=None):
        return self.quasimetric.update_from_recent_replay_buffer(
            replay_buffer,
            max_transitions=max_transitions,
        )

    def update(self, replay_buffer, step):
        shared_batch = None
        if self.continual_cfg.share_sac_batch:
            shared_batch = self._sample_shared_batch(replay_buffer)
            metrics = self._update_from_batch(replay_buffer, shared_batch, step)
        else:
            metrics = super().update(replay_buffer, step)

        if step % self.continual_cfg.structure_update_frequency == 0:
            structure_metrics = {}
            for _ in range(self.continual_cfg.structure_updates_per_step):
                structure_metrics = self.update_structure(replay_buffer, current_batch=shared_batch)
            metrics.update(
                {
                    f"quasimetric/{key}": value
                    for key, value in structure_metrics.items()
                }
            )
        return metrics

    def finish_task(self, task_id, replay_buffer):
        self.structure_memory.add_task(
            task_id=task_id,
            replay_buffer=replay_buffer,
            max_transitions=self.continual_cfg.memory_max_transitions_per_task,
        )
        return {
            "memory_tasks": self.structure_memory.num_tasks,
            "memory_transitions": len(self.structure_memory),
        }

    def save(self, model_dir, model_name):
        super().save(model_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(
            {
                "quasimetric": self.quasimetric.checkpoint(),
                "quasimetric_cfg": asdict(self.quasimetric_cfg),
                "continual_cfg": asdict(self.continual_cfg),
                "structure_memory": self.structure_memory.state_dict(),
                "sac_optimizers": {
                    "actor": self.actor_optimizer.state_dict(),
                    "critic": self.critic_optimizer.state_dict(),
                    "log_alpha": self.log_alpha_optimizer.state_dict(),
                },
                "log_alpha": self.log_alpha.detach().cpu(),
                "behavior_goal": self.behavior_goal,
            },
            os.path.join(model_dir, f"{model_name}_quasimetric.pt"),
        )

    def load(self, model_dir, model_name):
        super().load(model_dir, model_name)
        structure_path = os.path.join(model_dir, f"{model_name}_quasimetric.pt")
        if not os.path.exists(structure_path):
            return

        payload = torch.load(
            structure_path,
            map_location=self.device,
            weights_only=False,
        )
        self.quasimetric.load_checkpoint(payload["quasimetric"])
        if "structure_memory" in payload:
            self.structure_memory.load_state_dict(payload["structure_memory"])
        if "sac_optimizers" in payload:
            optimizers = payload["sac_optimizers"]
            self.actor_optimizer.load_state_dict(optimizers["actor"])
            self.critic_optimizer.load_state_dict(optimizers["critic"])
            self.log_alpha_optimizer.load_state_dict(optimizers["log_alpha"])
        if "log_alpha" in payload:
            self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        if "behavior_goal" in payload:
            self.set_behavior_goal(payload["behavior_goal"])
