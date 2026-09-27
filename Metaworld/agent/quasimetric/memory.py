from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _chronological_indices(replay_buffer) -> np.ndarray:
    size = len(replay_buffer)
    if size == 0:
        return np.zeros((0,), dtype=np.int64)
    if replay_buffer.full:
        return np.concatenate(
            [
                np.arange(replay_buffer.idx, replay_buffer.capacity, dtype=np.int64),
                np.arange(0, replay_buffer.idx, dtype=np.int64),
            ]
        )
    return np.arange(size, dtype=np.int64)


def _compute_episode_end_indices(done_flags: np.ndarray) -> np.ndarray:
    size = done_flags.shape[0]
    episode_ends = np.empty(size, dtype=np.int64)
    start = 0
    for idx, done in enumerate(done_flags):
        if done:
            episode_ends[start : idx + 1] = idx
            start = idx + 1
    if start < size:
        episode_ends[start:] = size - 1
    return episode_ends


def _compute_episode_ids(done_flags: np.ndarray) -> np.ndarray:
    episode_ids = np.empty(done_flags.shape[0], dtype=np.int64)
    episode_id = 0
    for idx, done in enumerate(done_flags):
        episode_ids[idx] = episode_id
        if done:
            episode_id += 1
    return episode_ids


def _discounted_offset(max_offset: int, discount: float) -> int:
    if max_offset <= 0:
        return 0

    support = np.arange(max_offset + 1, dtype=np.int64)
    if 0.0 < discount < 1.0:
        probs = np.power(discount, support).astype(np.float64)
    else:
        probs = np.ones_like(support, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(support, p=probs))


def cat_tensor_batches(batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("Expected at least one batch to concatenate.")

    merged = {}
    for key in batches[0]:
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
    return merged


@dataclass
class ReplayBufferView:
    obses: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obses: np.ndarray
    not_dones_no_max: np.ndarray
    device: torch.device
    task_id: Optional[int] = None
    not_dones: Optional[np.ndarray] = None

    def __post_init__(self):
        done_source = self.not_dones if self.not_dones is not None else self.not_dones_no_max
        done_flags = (1.0 - done_source.reshape(-1)) > 0.5
        self.episode_ends = _compute_episode_end_indices(done_flags.astype(np.bool_))
        self.episode_ids = _compute_episode_ids(done_flags.astype(np.bool_))

    def __len__(self) -> int:
        return int(self.obses.shape[0])

    @classmethod
    def from_replay_buffer(cls, replay_buffer, device=None, task_id=None):
        order = _chronological_indices(replay_buffer)
        device = replay_buffer.device if device is None else device
        return cls(
            obses=np.array(replay_buffer.obses[order], copy=True),
            actions=np.array(replay_buffer.actions[order], copy=True),
            rewards=np.array(replay_buffer.rewards[order], copy=True),
            next_obses=np.array(replay_buffer.next_obses[order], copy=True),
            not_dones_no_max=np.array(replay_buffer.not_dones_no_max[order], copy=True),
            device=device,
            task_id=task_id,
            not_dones=np.array(replay_buffer.not_dones[order], copy=True),
        )

    @classmethod
    def from_state_dict(cls, state_dict, device):
        return cls(
            obses=np.array(state_dict["obses"], copy=True),
            actions=np.array(state_dict["actions"], copy=True),
            rewards=np.array(state_dict["rewards"], copy=True),
            next_obses=np.array(state_dict["next_obses"], copy=True),
            not_dones_no_max=np.array(state_dict["not_dones_no_max"], copy=True),
            device=device,
            task_id=state_dict.get("task_id"),
            not_dones=np.array(state_dict["not_dones"], copy=True)
            if state_dict.get("not_dones") is not None
            else None,
        )

    def state_dict(self):
        return {
            "obses": self.obses,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_obses": self.next_obses,
            "not_dones_no_max": self.not_dones_no_max,
            "task_id": self.task_id,
            "not_dones": self.not_dones,
        }

    def tail(self, max_transitions: int):
        if max_transitions is None or len(self) <= max_transitions:
            return self
        return ReplayBufferView(
            obses=self.obses[-max_transitions:].copy(),
            actions=self.actions[-max_transitions:].copy(),
            rewards=self.rewards[-max_transitions:].copy(),
            next_obses=self.next_obses[-max_transitions:].copy(),
            not_dones_no_max=self.not_dones_no_max[-max_transitions:].copy(),
            device=self.device,
            task_id=self.task_id,
            not_dones=self.not_dones[-max_transitions:].copy() if self.not_dones is not None else None,
        )

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device).float()

    def sample_transition_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay view.")
        idxs = np.random.randint(0, len(self), size=batch_size)
        not_dones = self.not_dones_no_max[idxs].reshape(-1)
        dones = 1.0 - not_dones
        batch = {
            "obses": self._to_torch(self.obses[idxs]),
            "actions": self._to_torch(self.actions[idxs]),
            "rewards": self._to_torch(self.rewards[idxs]),
            "next_obses": self._to_torch(self.next_obses[idxs]),
            "not_dones": self._to_torch(not_dones),
            "dones": self._to_torch(dones),
        }
        return batch

    def sample_quasimetric_batch(
        self,
        batch_size: int,
        discount: float,
        lambda_: float,
        next_state_sample: float,
    ) -> Dict[str, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay view.")

        idxs = np.random.randint(0, len(self), size=batch_size)
        max_goal_offsets = self.episode_ends[idxs] - idxs
        goal_offsets = np.array(
            [_discounted_offset(int(max_offset), discount) for max_offset in max_goal_offsets],
            dtype=np.int64,
        )
        goal_indices = idxs + goal_offsets

        use_next_state = np.random.rand(batch_size) < next_state_sample
        intermediate_offsets = []
        for max_offset, force_next in zip(goal_offsets, use_next_state):
            if force_next or max_offset <= 0:
                intermediate_offsets.append(0)
            else:
                intermediate_offsets.append(_discounted_offset(int(max_offset), lambda_))
        intermediate_offsets = np.array(intermediate_offsets, dtype=np.int64)
        intermediate_indices = idxs + intermediate_offsets

        step_offsets = intermediate_indices - idxs + 1
        not_dones = self.not_dones_no_max[idxs].reshape(-1)
        dones = 1.0 - not_dones

        batch = {
            "obses": self._to_torch(self.obses[idxs]),
            "actions": self._to_torch(self.actions[idxs]),
            "rewards": self._to_torch(self.rewards[idxs]),
            "next_obses": self._to_torch(self.next_obses[idxs]),
            "not_dones": self._to_torch(not_dones),
            "dones": self._to_torch(dones),
            "value_goals": self._to_torch(self.next_obses[goal_indices]),
            "intermediate_value_goals": self._to_torch(self.next_obses[intermediate_indices]),
            "intermediate_value_goals_offsets": self._to_torch(step_offsets),
            "task_ids": torch.full(
                (batch_size,),
                -1 if self.task_id is None else self.task_id,
                device=self.device,
                dtype=torch.long,
            ),
            "episode_ids": torch.as_tensor(
                self.episode_ids[idxs], device=self.device, dtype=torch.long
            ),
        }
        return batch


class TaskAwareReplayMemory:
    def __init__(self, device, max_tasks: Optional[int] = None):
        self.device = device
        self.max_tasks = max_tasks
        self._tasks: "OrderedDict[int, ReplayBufferView]" = OrderedDict()

    def __len__(self) -> int:
        return sum(len(view) for view in self._tasks.values())

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    def add_task(self, task_id, replay_buffer, max_transitions: Optional[int] = None):
        view = ReplayBufferView.from_replay_buffer(
            replay_buffer,
            device=self.device,
            task_id=task_id,
        ).tail(max_transitions)
        if len(view) == 0:
            return

        if task_id in self._tasks:
            del self._tasks[task_id]
        self._tasks[task_id] = view

        while self.max_tasks is not None and len(self._tasks) > self.max_tasks:
            self._tasks.popitem(last=False)

    def _sample(self, batch_size: int, fn_name: str, *args) -> Dict[str, torch.Tensor]:
        if not self._tasks:
            raise ValueError("Task memory is empty.")

        views = list(self._tasks.values())
        weights = np.array([len(view) for view in views], dtype=np.float64)
        weights /= weights.sum()
        counts = np.random.multinomial(batch_size, weights)

        batches = []
        for view, count in zip(views, counts):
            if count == 0:
                continue
            batches.append(getattr(view, fn_name)(count, *args))

        return cat_tensor_batches(batches)

    def sample_transition_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self._sample(batch_size, "sample_transition_batch")

    def sample_quasimetric_batch(
        self,
        batch_size: int,
        discount: float,
        lambda_: float,
        next_state_sample: float,
    ) -> Dict[str, torch.Tensor]:
        return self._sample(
            batch_size,
            "sample_quasimetric_batch",
            discount,
            lambda_,
            next_state_sample,
        )

    def state_dict(self):
        tasks: List[Tuple[int, Dict[str, np.ndarray]]] = []
        for task_id, view in self._tasks.items():
            tasks.append((task_id, view.state_dict()))
        return {
            "max_tasks": self.max_tasks,
            "tasks": tasks,
        }

    def load_state_dict(self, state_dict):
        self.max_tasks = state_dict.get("max_tasks", self.max_tasks)
        self._tasks = OrderedDict()
        for task_id, payload in state_dict.get("tasks", []):
            self._tasks[task_id] = ReplayBufferView.from_state_dict(payload, self.device)
