import os

import numpy as np
import torch

from agent.quasimetric.memory import _discounted_offset
from replay_buffer import ReplayBuffer


class ReplayBufferMetric(ReplayBuffer):
    """Replay buffer with future-goal relabeling for goal-conditioned SAC."""

    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        super().__init__(obs_shape, action_shape, capacity, device, window=window)
        self.episode_ends = np.full((self.capacity,), -1, dtype=np.int64)
        self.task_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.episode_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.current_episode_start = 0
        self.current_task_id = -1
        self.current_episode_id = 0
        self.next_episode_id = 1

    def reset(self):
        super().reset()
        self.episode_ends = np.full((self.capacity,), -1, dtype=np.int64)
        self.task_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.episode_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.current_episode_start = 0
        self.current_task_id = -1
        self.current_episode_id = 0
        self.next_episode_id = 1

    def save_data(self, path):
        os.makedirs(path, exist_ok=True)
        valid_slice = slice(None) if self.full else slice(0, self.idx)
        data_path = os.path.join(path, "offline_data.npz")
        temporary_path = f"{data_path}.tmp"
        with open(temporary_path, "wb") as handle:
            np.savez_compressed(
                handle,
                obses=self.obses[valid_slice],
                next_obses=self.next_obses[valid_slice],
                actions=self.actions[valid_slice],
                rewards=self.rewards[valid_slice],
                successes=self.successes[valid_slice],
                not_dones=self.not_dones[valid_slice],
                not_dones_no_max=self.not_dones_no_max[valid_slice],
                episode_ends=self.episode_ends[valid_slice],
                task_ids=self.task_ids[valid_slice],
                episode_ids=self.episode_ids[valid_slice],
                current_episode_start=self.current_episode_start,
                current_task_id=self.current_task_id,
                current_episode_id=self.current_episode_id,
                next_episode_id=self.next_episode_id,
                others=[self.window, self.idx, self.last_save, self.full],
            )
        os.replace(temporary_path, data_path)
        print("data saved!", path)

    def load_data(self, path):
        print("data loading ...", path)
        try:
            with np.load(os.path.join(path, "offline_data.npz")) as data:
                saved_idx = int(data["others"][1])
                saved_full = bool(data["others"][3])
                saved_size = len(data["obses"]) if saved_full else saved_idx
                if saved_size > self.capacity:
                    raise ValueError(
                        f"Saved buffer size {saved_size} exceeds capacity {self.capacity}"
                    )

                self.reset()
                for name in (
                    "obses",
                    "next_obses",
                    "actions",
                    "rewards",
                    "successes",
                    "not_dones",
                    "not_dones_no_max",
                ):
                    getattr(self, name)[:saved_size] = data[name][:saved_size]

                self.window = int(data["others"][0])
                self.last_save = min(int(data["others"][2]), saved_size)
                self.full = saved_full and saved_size == self.capacity
                self.idx = saved_idx if self.full else saved_size

                if "episode_ends" in data and "current_episode_start" in data:
                    self.episode_ends[:saved_size] = data["episode_ends"][:saved_size]
                    self.current_episode_start = int(data["current_episode_start"])
                else:
                    self._rebuild_episode_ends()

                if "task_ids" in data:
                    self.task_ids[:saved_size] = data["task_ids"][:saved_size]
                    self.current_task_id = int(data.get("current_task_id", -1))
                if "episode_ids" in data:
                    self.episode_ids[:saved_size] = data["episode_ids"][:saved_size]
                    self.current_episode_id = int(data.get("current_episode_id", 0))
                    self.next_episode_id = int(
                        data.get("next_episode_id", self.current_episode_id + 1)
                    )
                else:
                    self._rebuild_episode_ids()
            return True
        except (OSError, ValueError, KeyError, IndexError) as error:
            print(error)
            return False

    def start_new_episode(self):
        self.current_episode_start = self.idx
        self.current_episode_id = self.next_episode_id
        self.next_episode_id += 1

    def start_new_task(self, task_id):
        self.current_task_id = int(task_id)
        self.start_new_episode()

    def _rebuild_episode_ends(self):
        self.episode_ends = np.full((self.capacity,), -1, dtype=np.int64)
        size = len(self)
        if size == 0:
            self.current_episode_start = 0
            return

        episode_start = 0
        for idx in range(size):
            self.episode_ends[episode_start : idx + 1] = idx
            if (1.0 - float(self.not_dones[idx, 0])) > 0.5:
                episode_start = idx + 1
        self.current_episode_start = episode_start if episode_start < size else 0

    def _rebuild_episode_ids(self):
        self.episode_ids.fill(-1)
        episode_id = 0
        for idx in range(len(self)):
            self.episode_ids[idx] = episode_id
            if (1.0 - float(self.not_dones[idx, 0])) > 0.5:
                episode_id += 1
        self.current_episode_id = episode_id
        self.next_episode_id = episode_id + 1

    def add(self, obs, action, reward, success, next_obs, done, done_no_max):
        write_idx = self.idx
        super().add(obs, action, reward, success, next_obs, done, done_no_max)
        self.task_ids[write_idx] = self.current_task_id
        self.episode_ids[write_idx] = self.current_episode_id

        if self.current_episode_start <= write_idx:
            self.episode_ends[self.current_episode_start : write_idx + 1] = write_idx
        else:
            self.episode_ends[self.current_episode_start : self.capacity] = write_idx
            self.episode_ends[: write_idx + 1] = write_idx

        if bool(done):
            self.current_episode_start = self.idx
            self.current_episode_id = self.next_episode_id
            self.next_episode_id += 1

    def _goal_rewards(self, goal_reached, reward_type):
        if reward_type == "binary":
            return goal_reached.astype(np.float32)
        if reward_type == "step_cost":
            return goal_reached.astype(np.float32) - 1.0
        raise ValueError(f"Unsupported goal reward type: {reward_type}")

    def _sample_discounted_offsets(self, max_offsets, discount):
        return np.array(
            [_discounted_offset(int(max_offset), discount) for max_offset in max_offsets],
            dtype=np.int64,
        )

    def _sample_intermediate_offsets(self, goal_offsets, discount, next_state_sample):
        use_next_state = np.random.rand(goal_offsets.shape[0]) < next_state_sample
        intermediate_offsets = []
        for max_offset, force_next in zip(goal_offsets, use_next_state):
            if force_next or max_offset <= 0:
                intermediate_offsets.append(0)
            else:
                intermediate_offsets.append(_discounted_offset(int(max_offset), discount))
        return np.array(intermediate_offsets, dtype=np.int64)

    def _sample_task_balanced_indices(self, size, batch_size):
        task_ids = self.task_ids[:size]
        if np.any(task_ids < 0):
            return np.random.randint(0, size, size=batch_size)

        unique_tasks = np.unique(task_ids)
        if unique_tasks.size <= 1:
            return np.random.randint(0, size, size=batch_size)

        repeat_count, remainder = divmod(batch_size, unique_tasks.size)
        repeated_tasks = np.tile(unique_tasks, repeat_count)
        if remainder:
            repeated_tasks = np.concatenate(
                [repeated_tasks, np.random.choice(unique_tasks, size=remainder, replace=False)]
            )
        np.random.shuffle(repeated_tasks)
        return np.array(
            [np.random.choice(np.flatnonzero(task_ids == task_id)) for task_id in repeated_tasks],
            dtype=np.int64,
        )

    def sample_metric_batch(
        self,
        batch_size,
        discount,
        next_state_sample=0.0,
        reward_type="binary",
        structure_discount=None,
        structure_lambda_=None,
        structure_next_state_sample=None,
    ):
        if structure_discount is None:
            if structure_lambda_ is not None or structure_next_state_sample is not None:
                raise ValueError(
                    "structure_discount, structure_lambda_, and structure_next_state_sample must be set together."
                )
        elif structure_lambda_ is None:
            raise ValueError("structure_lambda_ must be provided when structure_discount is set.")

        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        idxs = self._sample_task_balanced_indices(size, batch_size)
        episode_ends = self.episode_ends[idxs]
        episode_ends = np.where(episode_ends >= idxs, episode_ends, idxs)

        max_goal_offsets = episode_ends - idxs
        goal_offsets = self._sample_discounted_offsets(max_goal_offsets, discount)
        if next_state_sample > 0.0:
            force_next = np.random.rand(batch_size) < next_state_sample
            goal_offsets = np.where(force_next, 0, goal_offsets)

        goal_indices = idxs + goal_offsets
        goal_reached = (goal_offsets == 0).astype(np.float32).reshape(-1, 1)
        goal_rewards = self._goal_rewards(goal_reached, reward_type)
        goal_not_dones = self.not_dones_no_max[idxs] * (1.0 - goal_reached)

        batch = {
            "obses": self.obses[idxs],
            "actions": self.actions[idxs],
            "env_rewards": self.rewards[idxs],
            "successes": self.successes[idxs],
            "next_obses": self.next_obses[idxs],
            "not_dones_no_max": self.not_dones_no_max[idxs],
            "goals": self.next_obses[goal_indices],
            "goal_rewards": goal_rewards,
            "goal_reached": goal_reached,
            "goal_not_dones": goal_not_dones,
            "goal_steps": (goal_offsets + 1).reshape(-1, 1).astype(np.float32),
            "task_ids": self.task_ids[idxs].reshape(-1, 1),
            "episode_ids": self.episode_ids[idxs].reshape(-1, 1),
        }

        if structure_discount is not None:
            structure_goal_offsets = self._sample_discounted_offsets(max_goal_offsets, structure_discount)
            structure_goal_indices = idxs + structure_goal_offsets
            structure_next_state_sample = (
                next_state_sample if structure_next_state_sample is None else structure_next_state_sample
            )
            intermediate_offsets = self._sample_intermediate_offsets(
                structure_goal_offsets,
                structure_lambda_,
                structure_next_state_sample,
            )
            intermediate_indices = idxs + intermediate_offsets
            batch.update(
                {
                    "dones": 1.0 - self.not_dones_no_max[idxs],
                    "value_goals": self.next_obses[structure_goal_indices],
                    "intermediate_value_goals": self.next_obses[intermediate_indices],
                    "intermediate_value_goals_offsets": (intermediate_offsets + 1)
                    .reshape(-1, 1)
                    .astype(np.float32),
                }
            )

        return {
            key: torch.as_tensor(value, device=self.device).float()
            for key, value in batch.items()
        }

    def sample_behavior_goal(self, discount=0.995, success_only=True, batch_size=1):
        size = len(self)
        if size == 0:
            return None
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if success_only:
            success_indices = np.flatnonzero(self.successes[:size].reshape(-1) > 0.5)
            if success_indices.size > 0:
                goal_indices = np.random.choice(success_indices, size=batch_size)
                goals = np.array(self.next_obses[goal_indices], copy=True)
                return goals[0] if batch_size == 1 else goals

        start_indices = np.random.randint(0, size, size=batch_size)
        episode_ends = self.episode_ends[start_indices]
        episode_ends = np.where(episode_ends >= start_indices, episode_ends, start_indices)
        goal_offsets = self._sample_discounted_offsets(episode_ends - start_indices, discount)
        goals = np.array(self.next_obses[start_indices + goal_offsets], copy=True)
        return goals[0] if batch_size == 1 else goals


class ReplayBufferMetricNoHER(ReplayBufferMetric):
    """Replay buffer with future-goal conditioning but no HER reward relabeling."""

    def sample_metric_batch(
        self,
        batch_size,
        discount,
        next_state_sample=0.0,
        reward_type="binary",
        structure_discount=None,
        structure_lambda_=None,
        structure_next_state_sample=None,
    ):
        del reward_type

        if structure_discount is None:
            if structure_lambda_ is not None or structure_next_state_sample is not None:
                raise ValueError(
                    "structure_discount, structure_lambda_, and structure_next_state_sample must be set together."
                )
        elif structure_lambda_ is None:
            raise ValueError("structure_lambda_ must be provided when structure_discount is set.")

        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        idxs = self._sample_task_balanced_indices(size, batch_size)
        episode_ends = self.episode_ends[idxs]
        episode_ends = np.where(episode_ends >= idxs, episode_ends, idxs)

        max_goal_offsets = episode_ends - idxs
        goal_offsets = self._sample_discounted_offsets(max_goal_offsets, discount)
        if next_state_sample > 0.0:
            force_next = np.random.rand(batch_size) < next_state_sample
            goal_offsets = np.where(force_next, 0, goal_offsets)

        goal_indices = idxs + goal_offsets
        goal_reached = np.zeros((batch_size, 1), dtype=np.float32)
        goal_steps = (goal_offsets + 1).reshape(-1, 1).astype(np.float32)

        batch = {
            "obses": self.obses[idxs],
            "actions": self.actions[idxs],
            "env_rewards": self.rewards[idxs],
            "successes": self.successes[idxs],
            "next_obses": self.next_obses[idxs],
            "not_dones_no_max": self.not_dones_no_max[idxs],
            "goals": self.next_obses[goal_indices],
            "goal_rewards": self.rewards[idxs],
            "goal_reached": goal_reached,
            "goal_not_dones": self.not_dones_no_max[idxs],
            "goal_steps": goal_steps,
            "use_env_reward_only": np.ones((batch_size, 1), dtype=np.float32),
            "task_ids": self.task_ids[idxs].reshape(-1, 1),
            "episode_ids": self.episode_ids[idxs].reshape(-1, 1),
        }

        if structure_discount is not None:
            structure_goal_offsets = self._sample_discounted_offsets(max_goal_offsets, structure_discount)
            structure_goal_indices = idxs + structure_goal_offsets
            structure_next_state_sample = (
                next_state_sample if structure_next_state_sample is None else structure_next_state_sample
            )
            intermediate_offsets = self._sample_intermediate_offsets(
                structure_goal_offsets,
                structure_lambda_,
                structure_next_state_sample,
            )
            intermediate_indices = idxs + intermediate_offsets
            batch.update(
                {
                    "dones": 1.0 - self.not_dones_no_max[idxs],
                    "value_goals": self.next_obses[structure_goal_indices],
                    "intermediate_value_goals": self.next_obses[intermediate_indices],
                    "intermediate_value_goals_offsets": (intermediate_offsets + 1)
                    .reshape(-1, 1)
                    .astype(np.float32),
                }
            )

        return {
            key: torch.as_tensor(value, device=self.device).float()
            for key, value in batch.items()
        }
