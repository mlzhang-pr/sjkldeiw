import copy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from agent import utils

from .config import QuasimetricConfig
from .distance import alignment_score, mrn_distance
from .memory import ReplayBufferView, TaskAwareReplayMemory, cat_tensor_batches
from .networks import StateEncoder, TransitionEncoder, LatentTransitionEncoder


class MultistepQuasimetricLearner(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device,
        config: Optional[QuasimetricConfig] = None,
        state_encoder: Optional[nn.Module] = None,
        transition_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config or QuasimetricConfig()
        if self.config.nce_mode not in ("forward_nce", "backward_nce"):
            raise ValueError(f"Unsupported NCE mode: {self.config.nce_mode}")
        if self.config.transition_input not in ("state", "latent"):
            raise ValueError(
                "transition_input must be either 'state' or 'latent'."
            )
        if self.config.latent_dim % self.config.components != 0:
            raise ValueError(
                f"latent_dim={self.config.latent_dim} must be divisible by "
                f"components={self.config.components}."
            )

        self.device = torch.device(device)
        self.state_encoder = state_encoder or StateEncoder(
            obs_dim=obs_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_depth=self.config.hidden_depth,
        )







        transition_state_dim = (
            obs_dim
            if self.config.transition_input == "state"
            else self.config.latent_dim
        )
        self.latent_transition_encoder = transition_encoder or LatentTransitionEncoder(
            obs_dim=transition_state_dim,
            action_dim=action_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_depth=self.config.hidden_depth,
        )
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.to(self.device)
        self.target_state_encoder.to(self.device)
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        for parameter in self.target_state_encoder.parameters():
            parameter.requires_grad_(False)




        self._trainable_parameters = list(self.state_encoder.parameters()) + list(
            self.latent_transition_encoder.parameters()
        )


        self.optimizer = torch.optim.Adam(self._trainable_parameters, lr=self.config.lr)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mrn_distance(x, y, components=self.config.components)

    def transition_representation(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state_representation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.transition_input == "state":
            transition_state = observations
        else:
            transition_state = state_representation
            if transition_state is None:
                transition_state = self.state_encoder(observations)
        return self.latent_transition_encoder(transition_state, actions)

    def _contrastive_loss(
        self,
        transition_rep: torch.Tensor,
        goal_rep: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        episode_ids: Optional[torch.Tensor] = None,
    ):
        pairwise_dist = self.distance(
            transition_rep[:, None, :],
            goal_rep[None, :, :],
        )
        logits = -pairwise_dist
        if self.config.nce_mode == "backward_nce":
            logits = logits.transpose(0, 1)
        labels = torch.arange(logits.shape[0], device=logits.device)
        stats = {
            "cross_task_contrastive": torch.zeros((), device=logits.device),
            "contrastive_task_count": torch.zeros((), device=logits.device),
            "contrastive_episode_count": torch.zeros((), device=logits.device),
            "contrastive_negative_fraction": torch.ones((), device=logits.device),
        }

        if task_ids is not None:
            task_ids = task_ids.reshape(-1).long()
            known_tasks = task_ids >= 0
            task_count = torch.unique(task_ids[known_tasks]).numel()
            stats["contrastive_task_count"] = torch.tensor(
                task_count, dtype=logits.dtype, device=logits.device
            )
            if episode_ids is not None:
                episode_ids = episode_ids.reshape(-1).long()
                known_episodes = episode_ids >= 0
                episode_count = torch.unique(episode_ids[known_episodes]).numel()
                stats["contrastive_episode_count"] = torch.tensor(
                    episode_count, dtype=logits.dtype, device=logits.device
                )

            if bool(known_tasks.all()) and task_count > 1:
                cross_task = task_ids[:, None] != task_ids[None, :]
                diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
                valid_logits = diagonal | cross_task
                logits = logits.masked_fill(~valid_logits, torch.finfo(logits.dtype).min)
                off_diagonal_count = max(logits.numel() - logits.shape[0], 1)
                stats["cross_task_contrastive"] = torch.ones((), device=logits.device)
                stats["contrastive_negative_fraction"] = cross_task.sum() / off_diagonal_count

        return F.cross_entropy(logits, labels), logits, stats

    def checkpoint(self):
        return {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def _batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        return int(next(iter(batch.values())).shape[0])

    def _subsample_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        available = self._batch_size(batch)
        if batch_size >= available:
            return batch

        device = next(iter(batch.values())).device
        indices = torch.randperm(available, device=device)[:batch_size]
        return {
            key: value.index_select(0, indices)
            for key, value in batch.items()
        }

    def _current_structure_batch(
        self,
        current_batch: Optional[Dict[str, torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if current_batch is None:
            return None

        required_keys = (
            "obses",
            "actions",
            "next_obses",
            "dones",
            "value_goals",
            "intermediate_value_goals",
            "intermediate_value_goals_offsets",
        )
        if any(key not in current_batch for key in required_keys):
            return None

        batch_keys = required_keys + tuple(
            key for key in ("task_ids", "episode_ids") if key in current_batch
        )
        batch = {
            key: current_batch[key].to(self.device).float()
            if torch.is_tensor(current_batch[key])
            else torch.as_tensor(current_batch[key], device=self.device).float()
            for key in batch_keys
        }

        for key in ("dones", "intermediate_value_goals_offsets"):
            batch[key] = batch[key].reshape(-1)
        if batch_size is None:
            return batch
        return self._subsample_batch(batch, batch_size)

    @torch.no_grad()
    def structure_bonus(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device).float()
        action = torch.as_tensor(action, device=self.device).float()
        next_obs = torch.as_tensor(next_obs, device=self.device).float()


        transition_rep = self.transition_representation(obs, action)
        next_rep = self.target_state_encoder(next_obs)
        return alignment_score(
            transition_rep,
            next_rep,
            components=self.config.components,
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        obses = batch["obses"]
        actions = batch["actions"]
        next_obses = batch["next_obses"]
        goals = batch["value_goals"]
        dones = batch["dones"].reshape(-1)
        offsets = batch["intermediate_value_goals_offsets"].reshape(-1)



        state_rep = self.state_encoder(obses)

        goal_rep = self.state_encoder(goals)
        transition_rep = self.transition_representation(obses, actions, state_rep)

        with torch.no_grad():
            next_rep_target = self.target_state_encoder(next_obses)
            intermediate_goal_rep = self.target_state_encoder(batch["intermediate_value_goals"])
            goal_rep_target = self.target_state_encoder(goals)

        dist = self.distance(transition_rep, goal_rep)
        dist_next = self.distance(intermediate_goal_rep, goal_rep_target)




        transition_consistency_loss = torch.zeros((), device=self.device)

        discount = torch.full_like(offsets, self.config.discount)
        bootstrap = torch.pow(discount, offsets) * (1.0 - dones)

        delta = dist - dist_next
        clip = torch.full_like(delta, self.config.backup_clip)
        delta_clipped = torch.minimum(delta, clip)
        backup = torch.where(
            delta > self.config.backup_clip,
            delta,
            bootstrap * torch.exp(delta_clipped) - dist,
        )
        backup_loss = backup.mean()

        action_dist = self.distance(transition_rep, state_rep)
        action_invariance_loss = ((torch.exp(-action_dist) - 1.0) ** 2).mean()

        contrastive_loss = torch.zeros((), device=self.device)
        logits = None
        contrastive_metrics = {}
        if self.config.contrastive_coef > 0:
            contrastive_loss, logits, contrastive_metrics = self._contrastive_loss(
                transition_rep,
                goal_rep,
                task_ids=batch.get("task_ids"),
                episode_ids=batch.get("episode_ids"),
            )

        total_loss = (
            backup_loss
            + self.config.action_invariance_coef * action_invariance_loss

            + self.config.contrastive_coef * contrastive_loss
        )

        metrics = {
            "loss": total_loss,
            "backup_loss": backup_loss,
            "action_invariance_loss": action_invariance_loss,
            "transition_consistency_loss": transition_consistency_loss,
            "contrastive_loss": contrastive_loss,
            "dist": dist.mean(),
            "dist_next": dist_next.mean(),
            "transition_rep_mag": transition_rep.abs().mean(),
            "state_rep_mag": state_rep.abs().mean(),
            "goal_rep_mag": goal_rep.abs().mean(),
            "structure_alignment": alignment_score(
                transition_rep.detach(),
                next_rep_target,
                components=self.config.components,
            ).mean(),
        }
        if logits is not None:
            labels = torch.arange(logits.shape[0], device=self.device)
            metrics["categorical_accuracy"] = (
                (torch.argmax(logits, dim=1) == labels).float().mean()
            )
            metrics.update(contrastive_metrics)
        return total_loss, metrics

    def compute_loss_qrl(self, batch: Dict[str, torch.Tensor]):
        obses = batch["obses"]
        actions = batch["actions"]
        next_obses = batch["next_obses"]
        goals = batch["value_goals"]
        dones = batch["dones"].reshape(-1)
        offsets = batch["intermediate_value_goals_offsets"].reshape(-1)



        state_rep = self.state_encoder(obses)

        goal_rep = self.state_encoder(goals)
        transition_rep = self.transition_representation(obses, actions, state_rep)

        with torch.no_grad():
            next_rep_target = self.target_state_encoder(next_obses)
            intermediate_goal_rep = self.target_state_encoder(batch["intermediate_value_goals"])
            goal_rep_target = self.target_state_encoder(goals)

        dist = self.distance(transition_rep, goal_rep)
        dist_next = self.distance(intermediate_goal_rep, goal_rep_target)




        transition_consistency_loss = torch.zeros((), device=self.device)

        discount = torch.full_like(offsets, self.config.discount)
        bootstrap = torch.pow(discount, offsets) * (1.0 - dones)

        delta = dist - dist_next
        clip = torch.full_like(delta, self.config.backup_clip)
        delta_clipped = torch.minimum(delta, clip)
        backup = torch.where(
            delta > self.config.backup_clip,
            delta,
            bootstrap * torch.exp(delta_clipped) - dist,
        )
        backup_loss = backup.mean()

        action_dist = self.distance(transition_rep, state_rep)
        action_invariance_loss = ((torch.exp(-action_dist) - 1.0) ** 2).mean()

        contrastive_loss = torch.zeros((), device=self.device)
        logits = None
        contrastive_metrics = {}
        if self.config.contrastive_coef > 0:
            contrastive_loss, logits, contrastive_metrics = self._contrastive_loss(
                transition_rep,
                goal_rep,
                task_ids=batch.get("task_ids"),
                episode_ids=batch.get("episode_ids"),
            )

        total_loss = (
            backup_loss
            + self.config.action_invariance_coef * action_invariance_loss

            + self.config.contrastive_coef * contrastive_loss
        )

        metrics = {
            "loss": total_loss,
            "backup_loss": backup_loss,
            "action_invariance_loss": action_invariance_loss,
            "transition_consistency_loss": transition_consistency_loss,
            "contrastive_loss": contrastive_loss,
            "dist": dist.mean(),
            "dist_next": dist_next.mean(),
            "transition_rep_mag": transition_rep.abs().mean(),
            "state_rep_mag": state_rep.abs().mean(),
            "goal_rep_mag": goal_rep.abs().mean(),
            "structure_alignment": alignment_score(
                transition_rep.detach(),
                next_rep_target,
                components=self.config.components,
            ).mean(),
        }
        if logits is not None:
            labels = torch.arange(logits.shape[0], device=self.device)
            metrics["categorical_accuracy"] = (
                (torch.argmax(logits, dim=1) == labels).float().mean()
            )
            metrics.update(contrastive_metrics)
        return total_loss, metrics

    

    def update(self, batch: Dict[str, torch.Tensor]):
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(batch)
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._trainable_parameters,
                self.config.max_grad_norm,
            )
        self.optimizer.step()
        utils.soft_update_params(
            self.state_encoder,
            self.target_state_encoder,
            self.config.target_tau,
        )

        return {
            key: float(value.detach().cpu().item())
            for key, value in metrics.items()
        }

    def _build_batch(
        self,
        replay_buffer,
        memory: Optional[TaskAwareReplayMemory] = None,
        current_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if memory is None or len(memory) == 0:
            shared_current_batch = self._current_structure_batch(
                current_batch,
                batch_size=self.config.batch_size,
            )
            if shared_current_batch is not None:
                return shared_current_batch

            current_view = ReplayBufferView.from_replay_buffer(
                replay_buffer,
                device=self.device,
            )
            if len(current_view) == 0:
                raise ValueError("Current replay buffer is empty.")

            return current_view.sample_quasimetric_batch(
                self.config.batch_size,
                self.config.discount,
                self.config.lambda_,
                self.config.next_state_sample,
            )

        current_batch_size = max(1, int(self.config.batch_size * self.config.current_batch_ratio))
        memory_batch_size = max(0, self.config.batch_size - current_batch_size)

        batches = []
        shared_current_batch = self._current_structure_batch(current_batch, batch_size=current_batch_size)
        if shared_current_batch is not None:
            batches.append(shared_current_batch)
            memory_batch_size = max(0, self.config.batch_size - self._batch_size(shared_current_batch))
        else:
            current_view = ReplayBufferView.from_replay_buffer(
                replay_buffer,
                device=self.device,
            )
            if len(current_view) == 0:
                raise ValueError("Current replay buffer is empty.")

            batches.append(
                current_view.sample_quasimetric_batch(
                    current_batch_size,
                    self.config.discount,
                    self.config.lambda_,
                    self.config.next_state_sample,
                )
            )
        if memory_batch_size > 0:
            batches.append(
                memory.sample_quasimetric_batch(
                    memory_batch_size,
                    self.config.discount,
                    self.config.lambda_,
                    self.config.next_state_sample,
                )
            )
        if len(batches) == 1:
            return batches[0]
        return cat_tensor_batches(batches)

    def update_from_replay_buffer(
        self,
        replay_buffer,
        memory: Optional[TaskAwareReplayMemory] = None,
        current_batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if len(replay_buffer) < self.config.min_buffer_size:
            return {}
        batch = self._build_batch(
            replay_buffer,
            memory=memory,
            current_batch=current_batch,
        )
        return self.update(batch)

    def update_from_recent_replay_buffer(
        self,
        replay_buffer,
        max_transitions: Optional[int] = 10000,
    ):
        if max_transitions is not None and max_transitions <= 0:
            raise ValueError("max_transitions must be positive.")
        if len(replay_buffer) < self.config.min_buffer_size:
            return {}

        recent_view = ReplayBufferView.from_replay_buffer(
            replay_buffer,
            device=self.device,
        ).tail(max_transitions)
        if len(recent_view) == 0:
            raise ValueError("Recent replay buffer view is empty.")

        batch = recent_view.sample_quasimetric_batch(
            self.config.batch_size,
            self.config.discount,
            self.config.lambda_,
            self.config.next_state_sample,
        )
        return self.update(batch)
