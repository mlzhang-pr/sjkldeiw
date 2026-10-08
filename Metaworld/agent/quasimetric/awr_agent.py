import math
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from agent import utils
from agent.actor import DiagGaussianActor

from .config import QuasimetricConfig
from .structure import MultistepQuasimetricLearner


class ContinualQuasimetricAWRAgent:
    """Quasimetric structure learner with a non-goal-conditioned AWR policy."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        quasimetric_cfg: Optional[QuasimetricConfig] = None,
        batch_size=256,
        actor_lr=1e-4,
        awr_beta=1.0,
        awr_weight_clip=20.0,
        awr_num_goals=4,
    ):
        if awr_beta <= 0:
            raise ValueError("awr_beta must be positive.")
        if awr_weight_clip < 1:
            raise ValueError("awr_weight_clip must be at least 1.")
        if awr_num_goals <= 0:
            raise ValueError("awr_num_goals must be positive.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.awr_beta = awr_beta
        self.awr_weight_clip = awr_weight_clip
        self.awr_num_goals = awr_num_goals
        self.quasimetric_cfg = quasimetric_cfg or QuasimetricConfig()

        self.actor = DiagGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.quasimetric = MultistepQuasimetricLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device,
            config=self.quasimetric_cfg,
        )

        self._task_index_cache_key = None
        self._task_index_cache = None
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.quasimetric.train(training)

    def eval(self):
        self.train(False)

    def actor_distribution(self, obs, goal_obs=None, detach_goal=True):
        del goal_obs, detach_goal
        return self.actor(obs)

    def act(self, obs, sample=False):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def _task_index_sets(self, replay_buffer):
        size = len(replay_buffer)
        cache_key = (
            id(replay_buffer),
            size,
            int(getattr(replay_buffer, "idx", len(replay_buffer))),
            bool(getattr(replay_buffer, "full", False)),
        )
        if cache_key == self._task_index_cache_key:
            return self._task_index_cache

        task_ids = np.asarray(replay_buffer.task_ids[:size]).reshape(-1)
        episode_ids = np.asarray(replay_buffer.episode_ids[:size]).reshape(-1)
        successes = np.asarray(replay_buffer.successes[:size]).reshape(-1) > 0.5
        rewards = np.asarray(replay_buffer.rewards[:size]).reshape(-1)
        task_index_sets = {}
        for task_id in np.unique(task_ids[task_ids >= 0]):
            transition_indices = np.flatnonzero(task_ids == task_id)
            successful_episode_ids = np.unique(
                episode_ids[(task_ids == task_id) & successes]
            )
            goal_indices = []
            for episode_id in successful_episode_ids:
                episode_indices = transition_indices[
                    episode_ids[transition_indices] == episode_id
                ]
                if episode_indices.size > 0:
                    goal_indices.append(int(episode_indices[-1]))
            used_fallback = not goal_indices
            if used_fallback:
                episode_returns_and_ends = []
                for episode_id in np.unique(episode_ids[transition_indices]):
                    episode_indices = transition_indices[
                        episode_ids[transition_indices] == episode_id
                    ]
                    if episode_indices.size > 0:
                        episode_returns_and_ends.append(
                            (
                                float(rewards[episode_indices].sum()),
                                int(episode_indices[-1]),
                            )
                        )
                episode_returns_and_ends.sort(reverse=True)
                goal_indices = [
                    end_idx
                    for _, end_idx in episode_returns_and_ends[: self.awr_num_goals]
                ]
            task_index_sets[int(task_id)] = (
                transition_indices,
                np.asarray(goal_indices, dtype=np.int64),
                used_fallback,
            )

        self._task_index_cache_key = cache_key
        self._task_index_cache = task_index_sets
        return self._task_index_cache

    def _sample_awr_batch(self, replay_buffer):
        task_index_sets = self._task_index_sets(replay_buffer)
        eligible_task_ids = np.asarray(
            [
                task_id
                for task_id, (_, goal_indices, _) in task_index_sets.items()
                if goal_indices.size > 0
            ],
            dtype=np.int64,
        )
        if eligible_task_ids.size == 0:
            return None

        repeat_count, remainder = divmod(self.batch_size, eligible_task_ids.size)
        sampled_task_ids = np.tile(eligible_task_ids, repeat_count)
        if remainder:
            sampled_task_ids = np.concatenate(
                [
                    sampled_task_ids,
                    np.random.choice(
                        eligible_task_ids,
                        size=remainder,
                        replace=False,
                    ),
                ]
            )
        np.random.shuffle(sampled_task_ids)

        transition_batch = np.asarray(
            [
                np.random.choice(task_index_sets[int(task_id)][0])
                for task_id in sampled_task_ids
            ],
            dtype=np.int64,
        )
        goal_batch = np.stack(
            [
                np.random.choice(
                    task_index_sets[int(task_id)][1],
                    size=self.awr_num_goals,
                    replace=True,
                )
                for task_id in sampled_task_ids
            ]
        )
        return (
            torch.as_tensor(
                replay_buffer.obses[transition_batch], device=self.device
            ).float(),
            torch.as_tensor(
                replay_buffer.actions[transition_batch], device=self.device
            ).float(),
            torch.as_tensor(
                replay_buffer.next_obses[transition_batch], device=self.device
            ).float(),
            torch.as_tensor(
                replay_buffer.next_obses[goal_batch], device=self.device
            ).float(),
            sampled_task_ids,
            len(task_index_sets),
            int(eligible_task_ids.size),
            sum(used_fallback for _, _, used_fallback in task_index_sets.values()),
        )

    def update_awr(self, replay_buffer):
        batch = self._sample_awr_batch(replay_buffer)
        if batch is None:
            return {
                "skipped_no_success_goals": 1.0,
                "buffer_task_count": float(len(self._task_index_sets(replay_buffer))),
                "eligible_task_count": 0.0,
            }
        (
            obs,
            action,
            next_obs,
            goals,
            sampled_task_ids,
            buffer_task_count,
            eligible_task_count,
            fallback_task_count,
        ) = batch

        with torch.no_grad():
            batch_size, num_goals, obs_dim = goals.shape
            goal_rep = self.quasimetric.state_encoder(
                goals.reshape(batch_size * num_goals, obs_dim)
            ).reshape(batch_size, num_goals, -1)
            state_rep = self.quasimetric.state_encoder(obs)[:, None, :]
            next_state_rep = self.quasimetric.state_encoder(next_obs)[:, None, :]
            distances = self.quasimetric.distance(state_rep, goal_rep)
            next_distances = self.quasimetric.distance(next_state_rep, goal_rep)
            goal_advantages = distances - next_distances
            advantages = goal_advantages.mean(dim=1, keepdim=True)
            max_log_weight = math.log(self.awr_weight_clip)
            weights = torch.exp((advantages / self.awr_beta).clamp(max=max_log_weight))

        action = action.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_loss = -(weights * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "distance_mean": float(distances.mean().item()),
            "next_distance_mean": float(next_distances.mean().item()),
            "weight_mean": float(weights.mean().item()),
            "weight_max": float(weights.max().item()),
            "log_prob": float(log_prob.detach().mean().item()),
            "policy_std": float(dist.scale.detach().mean().item()),
            "goals_per_transition": float(self.awr_num_goals),
            "skipped_no_success_goals": 0.0,
            "buffer_task_count": float(buffer_task_count),
            "eligible_task_count": float(eligible_task_count),
            "fallback_task_count": float(fallback_task_count),
            "sampled_task_count": float(np.unique(sampled_task_ids).size),
        }

    def update_structure_and_actor(self, replay_buffer):
        if len(replay_buffer) < self.quasimetric_cfg.min_buffer_size:
            return {}, {}

        structure_batch = replay_buffer.sample_metric_batch(
            batch_size=self.quasimetric_cfg.batch_size,
            discount=self.quasimetric_cfg.discount,
            next_state_sample=self.quasimetric_cfg.next_state_sample,
            reward_type="binary",
            structure_discount=self.quasimetric_cfg.discount,
            structure_lambda_=self.quasimetric_cfg.lambda_,
            structure_next_state_sample=self.quasimetric_cfg.next_state_sample,
        )
        structure_metrics = self.quasimetric.update(structure_batch)
        return structure_metrics, self.update_awr(replay_buffer)

    def copy_from(self, source_agent):
        self.actor.load_state_dict(source_agent.actor.state_dict())
        self.actor_optimizer.load_state_dict(source_agent.actor_optimizer.state_dict())
        self.quasimetric.load_checkpoint(source_agent.quasimetric.checkpoint())
        self._task_index_cache_key = None
        self._task_index_cache = None

    def save(self, model_dir, model_name):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            self.actor.state_dict(), os.path.join(model_dir, f"{model_name}_actor.pt")
        )
        torch.save(
            {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "awr_beta": self.awr_beta,
                "awr_weight_clip": self.awr_weight_clip,
                "awr_num_goals": self.awr_num_goals,
            },
            os.path.join(model_dir, f"{model_name}_awr.pt"),
        )
        torch.save(
            {
                "quasimetric": self.quasimetric.checkpoint(),
                "quasimetric_cfg": asdict(self.quasimetric_cfg),
            },
            os.path.join(model_dir, f"{model_name}_quasimetric.pt"),
        )

    def load(self, model_dir, model_name):
        self.actor.load_state_dict(
            torch.load(
                os.path.join(model_dir, f"{model_name}_actor.pt"),
                map_location=self.device,
            )
        )
        awr_payload = torch.load(
            os.path.join(model_dir, f"{model_name}_awr.pt"),
            map_location=self.device,
            weights_only=False,
        )
        self.actor_optimizer.load_state_dict(awr_payload["actor_optimizer"])

        quasimetric_payload = torch.load(
            os.path.join(model_dir, f"{model_name}_quasimetric.pt"),
            map_location=self.device,
            weights_only=False,
        )
        self.quasimetric.load_checkpoint(quasimetric_payload["quasimetric"])
        self._task_index_cache_key = None
        self._task_index_cache = None
