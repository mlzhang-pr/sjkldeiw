import numpy as np
import torch


class HerReplayBuffer:
    """Goal replay buffer with SB3-style future goal relabeling."""

    def __init__(
        self,
        observation_shape,
        achieved_goal_shape,
        desired_goal_shape,
        action_shape,
        capacity,
        device,
        env,
        n_sampled_goal=4,
        handle_timeout_termination=True,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.env = env
        self.n_sampled_goal = int(n_sampled_goal)
        self.her_ratio = 1.0 - 1.0 / (self.n_sampled_goal + 1.0) if self.n_sampled_goal > 0 else 0.0
        self.handle_timeout_termination = bool(handle_timeout_termination)

        self.observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.achieved_goals = np.empty((self.capacity, *achieved_goal_shape), dtype=np.float32)
        self.next_achieved_goals = np.empty((self.capacity, *achieved_goal_shape), dtype=np.float32)
        self.desired_goals = np.empty((self.capacity, *desired_goal_shape), dtype=np.float32)
        self.next_desired_goals = np.empty((self.capacity, *desired_goal_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.timeouts = np.empty((self.capacity, 1), dtype=np.float32)
        self.successes = np.empty((self.capacity, 1), dtype=np.float32)

        self.episode_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.episode_start_abs = np.full((self.capacity,), -1, dtype=np.int64)
        self.episode_steps = np.zeros((self.capacity,), dtype=np.int64)
        self.episode_lengths = np.zeros((self.capacity,), dtype=np.int64)
        self.abs_indices = np.full((self.capacity,), -1, dtype=np.int64)

        self.pos = 0
        self.full = False
        self.total_added = 0
        self.current_episode_id = 0
        self.current_episode_start_abs = 0
        self.current_episode_step = 0
        self.current_episode_indices = []

    def reset(self):
        self.pos = 0
        self.full = False
        self.total_added = 0
        self.current_episode_id = 0
        self.current_episode_start_abs = 0
        self.current_episode_step = 0
        self.current_episode_indices = []
        self.episode_ids.fill(-1)
        self.episode_start_abs.fill(-1)
        self.episode_steps.fill(0)
        self.episode_lengths.fill(0)
        self.abs_indices.fill(-1)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, done, info=None, timeout=False):
        info = {} if info is None else info
        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)

        write_idx = self.pos
        self._invalidate_overwritten_episode(write_idx)
        self._remove_overwritten_current_index(write_idx)

        np.copyto(self.observations[write_idx], obs["observation"])
        np.copyto(self.next_observations[write_idx], next_obs["observation"])
        np.copyto(self.achieved_goals[write_idx], obs["achieved_goal"])
        np.copyto(self.next_achieved_goals[write_idx], next_obs["achieved_goal"])
        np.copyto(self.desired_goals[write_idx], obs["desired_goal"])
        np.copyto(self.next_desired_goals[write_idx], next_obs["desired_goal"])
        np.copyto(self.actions[write_idx], np.asarray(action, dtype=np.float32))
        self.rewards[write_idx] = float(np.asarray(reward).reshape(-1)[0])
        self.dones[write_idx] = float(done)
        self.timeouts[write_idx] = float(timeout)
        self.successes[write_idx] = float(info.get("success", info.get("is_success", False)))

        self.episode_ids[write_idx] = self.current_episode_id
        self.episode_start_abs[write_idx] = self.current_episode_start_abs
        self.episode_steps[write_idx] = self.current_episode_step
        self.episode_lengths[write_idx] = 0
        self.abs_indices[write_idx] = self.total_added
        self.current_episode_indices.append(write_idx)

        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0
        self.total_added += 1
        self.current_episode_step += 1

        if done:
            self._finish_current_episode()

    def sample(self, batch_size):
        valid_indices = self._valid_indices()
        if valid_indices.size == 0:
            raise RuntimeError(
                "Unable to sample before the end of the first episode. Choose learning_starts greater than "
                "the episode horizon, as SB3's HER replay buffer recommends."
            )

        batch_indices = np.random.choice(valid_indices, size=int(batch_size), replace=True)
        observations = self.observations[batch_indices].copy()
        next_observations = self.next_observations[batch_indices].copy()
        achieved_goals = self.achieved_goals[batch_indices].copy()
        next_achieved_goals = self.next_achieved_goals[batch_indices].copy()
        desired_goals = self.desired_goals[batch_indices].copy()
        next_desired_goals = self.next_desired_goals[batch_indices].copy()
        actions = self.actions[batch_indices].copy()
        rewards = self.rewards[batch_indices].copy()
        dones = self.dones[batch_indices].copy()
        timeouts = self.timeouts[batch_indices].copy()

        her_mask = np.random.rand(batch_indices.shape[0]) < self.her_ratio
        if np.any(her_mask):
            her_indices = np.flatnonzero(her_mask)
            future_indices = self._sample_future_indices(batch_indices[her_indices])
            desired_goals[her_indices] = self.next_achieved_goals[future_indices]
            next_desired_goals[her_indices] = self.next_achieved_goals[future_indices]
            rewards[her_indices] = self._compute_reward(
                next_achieved_goals[her_indices],
                desired_goals[her_indices],
            )

        terminal_dones = dones * (1.0 - timeouts) if self.handle_timeout_termination else dones
        not_dones = 1.0 - terminal_dones

        return {
            "observations": self._as_torch(observations),
            "achieved_goals": self._as_torch(achieved_goals),
            "desired_goals": self._as_torch(desired_goals),
            "actions": self._as_torch(actions),
            "rewards": self._as_torch(rewards),
            "next_observations": self._as_torch(next_observations),
            "next_achieved_goals": self._as_torch(next_achieved_goals),
            "next_desired_goals": self._as_torch(next_desired_goals),
            "not_dones": self._as_torch(not_dones),
            "dones": self._as_torch(dones),
            "her_mask": self._as_torch(her_mask.reshape(-1, 1).astype(np.float32)),
        }

    def _finish_current_episode(self):
        episode_length = self.current_episode_step
        for idx in self.current_episode_indices:
            if self.episode_ids[idx] == self.current_episode_id:
                self.episode_lengths[idx] = episode_length

        self.current_episode_id += 1
        self.current_episode_start_abs = self.total_added
        self.current_episode_step = 0
        self.current_episode_indices = []

    def _remove_overwritten_current_index(self, write_idx):
        if write_idx in self.current_episode_indices:
            self.current_episode_indices = [idx for idx in self.current_episode_indices if idx != write_idx]

    def _invalidate_overwritten_episode(self, write_idx):
        episode_length = int(self.episode_lengths[write_idx])
        if episode_length <= 0:
            return

        episode_start_abs = int(self.episode_start_abs[write_idx])
        for offset in range(episode_length):
            idx = (episode_start_abs + offset) % self.capacity
            if self.episode_start_abs[idx] == episode_start_abs:
                self.episode_lengths[idx] = 0

    def _valid_indices(self):
        size = len(self)
        if size == 0:
            return np.array([], dtype=np.int64)
        return np.flatnonzero(self.episode_lengths[:size] > 0).astype(np.int64)

    def _sample_future_indices(self, batch_indices):
        future_indices = []
        for idx in batch_indices:
            episode_length = int(self.episode_lengths[idx])
            episode_step = int(self.episode_steps[idx])
            if episode_length <= episode_step:
                future_indices.append(idx)
                continue

            future_step = np.random.randint(episode_step, episode_length)
            future_abs = int(self.episode_start_abs[idx]) + future_step
            future_idx = future_abs % self.capacity
            if self.abs_indices[future_idx] != future_abs or self.episode_ids[future_idx] != self.episode_ids[idx]:
                future_idx = idx
            future_indices.append(future_idx)
        return np.asarray(future_indices, dtype=np.int64)

    def _compute_reward(self, achieved_goal, desired_goal):
        infos = [{} for _ in range(achieved_goal.shape[0])]
        rewards = self.env.compute_reward(achieved_goal, desired_goal, infos)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        return rewards

    def _normalize_obs(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("HerReplayBuffer expects dict observations with observation/achieved_goal/desired_goal.")
        return {
            "observation": np.asarray(obs["observation"], dtype=np.float32),
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32),
            "desired_goal": np.asarray(obs["desired_goal"], dtype=np.float32),
        }

    def _as_torch(self, value):
        return torch.as_tensor(value, device=self.device).float()