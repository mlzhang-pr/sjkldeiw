"""Continual Fetch goal-conditioned benchmark.

This module mirrors the task-stream interface used by ``metaworld_env.py`` but
uses Gymnasium Robotics Fetch tasks.  A task is a Fetch environment family
(``reach``, ``push``, ``pick-and-place``, or ``slide``), and a benchmark sequence
can be selected from presets or provided explicitly.

The default goal-conditioned Fetch observations follow the online GCRL setting
from https://github.com/quasimetric-learning/quasimetric-rl: the observation dict
contains ``observation``, ``achieved_goal``, and ``desired_goal`` tensors with the
same shape.  The state-shaped ``desired_goal`` is made by writing the native Fetch
3-D desired position into the corresponding state coordinates, while rewards and
success are still computed from the true 3-D Fetch goal distance.
"""

import inspect
import importlib
import json
import os
import re
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics


FETCH_TASK_TO_ENV_ID_PREFIX = {
    "reach": "FetchReach",
    "push": "FetchPush",
    "pick-and-place": "FetchPickAndPlace",
    "slide": "FetchSlide",
}

FETCH_TASK_ALIASES = {
    "reach": "reach",
    "fetch-reach": "reach",
    "fetchreach": "reach",
    "push": "push",
    "fetch-push": "push",
    "fetchpush": "push",
    "pick": "pick-and-place",
    "pick-place": "pick-and-place",
    "pick-place-v2": "pick-and-place",
    "pick-and-place": "pick-and-place",
    "pick_and_place": "pick-and-place",
    "pickandplace": "pick-and-place",
    "fetch-pick-and-place": "pick-and-place",
    "fetch-pickandplace": "pick-and-place",
    "fetchpickandplace": "pick-and-place",
    "slide": "slide",
    "fetch-slide": "slide",
    "fetchslide": "slide",
}

FETCH_SEQUENCE_PRESETS = {
    "set1": ["reach", "push", "pick-and-place", "slide"],
    "set2": ["push", "slide", "reach", "pick-and-place"],
    "set3": ["pick-and-place", "reach", "slide", "push"],
    "set4": ["slide", "pick-and-place", "push", "reach"],
    "easy2": ["reach", "push"],
    "manipulation3": ["push", "pick-and-place", "slide"],
    "all": ["reach", "push", "pick-and-place", "slide"],
}

FETCH_OBSERVATION_DIMS = {
    "reach": 10,
    "push": 25,
    "pick-and-place": 25,
    "slide": 25,
}

FETCH_GOAL_STATE_SLICES = {
    "reach": (0, 3),
    "push": (3, 6),
    "pick-and-place": (3, 6),
    "slide": (3, 6),
}

FETCH_STATE_GOAL_FORMAT_ALIASES = {
    "online": "online",
    "online-state": "online",
    "online_state": "online",
    "state": "online",
    "state-goal": "online",
    "state_goal": "online",
    "gcrl": "online",
    "native": "native",
    "native-goal": "native",
    "native_goal": "native",
    "fetch": "native",
    "fetch-goal": "native",
    "fetch_goal": "native",
    "3d": "native",
}


def _normalize_key(value):
    value = str(value).strip().strip("'\"").lower()
    value = value.replace("_", "-").replace(" ", "-")
    value = re.sub(r"-v\d+$", "", value)
    return re.sub(r"-+", "-", value).strip("-")


def normalize_fetch_task_name(task_name):
    key = _normalize_key(task_name)
    if key.startswith("fetch") and key not in FETCH_TASK_ALIASES:
        key = _normalize_key(key[len("fetch") :])
    if key in FETCH_TASK_ALIASES:
        return FETCH_TASK_ALIASES[key]
    raise ValueError(
        f"Unknown Fetch task '{task_name}'. Supported tasks are: "
        f"{', '.join(FETCH_TASK_TO_ENV_ID_PREFIX)}"
    )


def normalize_fetch_goal_format(goal_format):
    key = _normalize_key("online" if goal_format is None else goal_format)
    if key in FETCH_STATE_GOAL_FORMAT_ALIASES:
        return FETCH_STATE_GOAL_FORMAT_ALIASES[key]
    raise ValueError(
        f"Unknown Fetch goal format '{goal_format}'. Supported formats are: online, native."
    )


def _split_task_order(task_order):
    if isinstance(task_order, str):
        text = task_order.strip()
        if not text:
            return []
        if os.path.exists(text):
            return _load_task_order_file(text)
        separator = r"[,;|]" if re.search(r"[,;|]", text) else r"\s+"
        return [item for item in re.split(separator, text) if item]
    if isinstance(task_order, Iterable):
        return list(task_order)
    raise TypeError("task_order must be a string or an iterable of task names.")


def _load_task_order_file(path):
    with open(path, "r") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
            if isinstance(payload, dict):
                for key in ("task_order", "tasks", "sequence"):
                    if key in payload:
                        payload = payload[key]
                        break
            return _split_task_order(payload)
        lines = []
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                lines.extend(_split_task_order(line))
        return lines


def parse_fetch_task_order(task_order):
    tasks = [normalize_fetch_task_name(task) for task in _split_task_order(task_order)]
    if not tasks:
        raise ValueError("Fetch task order is empty.")
    return tasks


def resolve_fetch_sequence(env_sequence="set1", task_order=None, base_task_name=None):
    if task_order is not None:
        return parse_fetch_task_order(task_order)
    if base_task_name is not None:
        return [normalize_fetch_task_name(base_task_name)]

    sequence_name = "set1" if env_sequence is None else str(env_sequence).strip()
    if os.path.exists(sequence_name):
        return parse_fetch_task_order(sequence_name)
    if re.search(r"[,;|\s]", sequence_name.strip()):
        return parse_fetch_task_order(sequence_name)

    key = _normalize_key(sequence_name)
    if key.startswith("fetch-sequence-"):
        key = key[len("fetch-sequence-") :]
    if key in FETCH_SEQUENCE_PRESETS:
        return list(FETCH_SEQUENCE_PRESETS[key])
    return [normalize_fetch_task_name(key)]


def _import_gymnasium_robotics():
    try:
        gymnasium_robotics = importlib.import_module("gymnasium_robotics")
    except ImportError as error:
        raise ImportError(
            "FetchGoalEnvSequence requires gymnasium-robotics. Install it with "
            "`pip install gymnasium-robotics` in the training environment."
        ) from error
    if hasattr(gym, "register_envs"):
        gym.register_envs(gymnasium_robotics)


def _candidate_fetch_env_ids(task_name, fetch_env_version):
    prefix = FETCH_TASK_TO_ENV_ID_PREFIX[normalize_fetch_task_name(task_name)]
    if fetch_env_version in {None, "auto"}:
        versions = (4, 3, 2, 1)
    else:
        version_text = str(fetch_env_version).lower().lstrip("v")
        versions = (int(version_text),)
    return [f"{prefix}-v{version}" for version in versions]


def make_fetch_env(
    task_name, reward_type="sparse", fetch_env_version="auto", max_episode_steps=50
):
    _import_gymnasium_robotics()
    native_reward_type = "sparse" if reward_type == "success" else reward_type
    make_kwargs = {"reward_type": native_reward_type}
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = int(max_episode_steps)

    last_error = None
    for env_id in _candidate_fetch_env_ids(task_name, fetch_env_version):
        try:
            gym.spec(env_id)
        except gym.error.Error as error:
            last_error = error
            continue

        try:
            return gym.make(env_id, **make_kwargs), env_id
        except TypeError:
            fallback_kwargs = dict(make_kwargs)
            fallback_kwargs.pop("max_episode_steps", None)
            return gym.make(env_id, **fallback_kwargs), env_id
        except Exception as error:
            last_error = error
            break

    raise RuntimeError(
        f"Could not create a Gymnasium Robotics Fetch env for task '{task_name}'. "
        f"Tried: {', '.join(_candidate_fetch_env_ids(task_name, fetch_env_version))}. "
        f"Last error: {last_error}"
    )


class FetchGoalInfoWrapper(gym.Wrapper):
    def __init__(self, env, reward_type="sparse", success_threshold=0.05):
        super().__init__(env)
        if reward_type not in {"sparse", "dense", "success"}:
            raise ValueError(f"Unsupported Fetch reward type: {reward_type}")
        self.reward_type = reward_type
        self.success_threshold = float(success_threshold)

        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError("FetchGoalInfoWrapper expects a Dict observation space.")
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    np.asarray(space.low, dtype=np.float32),
                    np.asarray(space.high, dtype=np.float32),
                    dtype=np.float32,
                )
                for key, space in obs_space.spaces.items()
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._cast_obs(obs)
        return obs, self._goal_info(info, obs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = self._cast_obs(obs)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        return (
            obs,
            float(np.asarray(reward).reshape(-1)[0]),
            terminated,
            truncated,
            self._goal_info(info, obs),
        )

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        del info
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reached = distance <= self.success_threshold
        if self.reward_type == "dense":
            reward = -distance
        elif self.reward_type == "success":
            reward = reached.astype(np.float32)
        else:
            reward = -np.logical_not(reached).astype(np.float32)
        if np.ndim(reward) == 0:
            return np.float32(reward)
        return reward.astype(np.float32)

    def _cast_obs(self, obs):
        return {key: np.asarray(value, dtype=np.float32) for key, value in obs.items()}

    def _goal_info(self, info, obs):
        info = dict(info)
        distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        success = info.get("is_success", distance <= self.success_threshold)
        success = bool(np.asarray(success).reshape(-1)[0] > 0.5)
        info["is_success"] = np.float32(success)
        info["success"] = success
        info["goal_distance"] = distance
        info["achieved_goal"] = np.array(
            obs["achieved_goal"], dtype=np.float32, copy=True
        )
        info["desired_goal"] = np.array(
            obs["desired_goal"], dtype=np.float32, copy=True
        )
        return info


class FetchGoalScaleWrapper(gym.Wrapper):
    def __init__(self, env, goal_scale):
        super().__init__(env)
        self.goal_scale = float(goal_scale)
        if not 0.0 < self.goal_scale <= 1.0:
            raise ValueError("Fetch goal scale must be in (0, 1].")
        self.episode_goal = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        self.episode_goal = achieved_goal + self.goal_scale * (
            desired_goal - achieved_goal
        )
        return self._replace_goal(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._replace_goal(obs), reward, terminated, truncated, info

    def _replace_goal(self, obs):
        obs = dict(obs)
        obs["desired_goal"] = np.array(self.episode_goal, dtype=np.float32, copy=True)
        return obs


class FetchOuterGoalRangeWrapper(gym.Wrapper):
    def __init__(self, env, outer_range, max_reset_attempts=10000):
        super().__init__(env)
        base_env = env.unwrapped
        self.inner_range = float(base_env.target_range)
        self.outer_range = float(outer_range)
        if self.outer_range <= self.inner_range:
            raise ValueError(
                f"Outer goal range must exceed the native range "
                f"{self.inner_range}, got {self.outer_range}."
            )
        self.max_reset_attempts = int(max_reset_attempts)
        target_offset = np.asarray(base_env.target_offset, dtype=np.float32)
        if target_offset.ndim == 0:
            target_offset = np.full(3, target_offset, dtype=np.float32)
        self.goal_center = (
            np.asarray(base_env.initial_gripper_xpos[:3], dtype=np.float32)
            + target_offset
        )
        base_env.target_range = self.outer_range

    def reset(self, **kwargs):
        reset_kwargs = kwargs
        for _ in range(self.max_reset_attempts):
            obs, info = self.env.reset(**reset_kwargs)
            reset_kwargs = {}
            goal = np.asarray(obs["desired_goal"], dtype=np.float32)
            if np.max(np.abs(goal[:2] - self.goal_center[:2])) > self.inner_range:
                info = dict(info)
                info["goal_range_inner"] = self.inner_range
                info["goal_range_outer"] = self.outer_range
                return obs, info
        raise RuntimeError(
            f"Could not sample a goal outside range {self.inner_range} "
            f"within {self.max_reset_attempts} resets."
        )


class FetchOuterInitialStateRangeWrapper(gym.Wrapper):
    def __init__(self, env, outer_range, max_reset_attempts=10000):
        super().__init__(env)
        base_env = env.unwrapped
        self.inner_range = float(base_env.obj_range)
        self.outer_range = float(outer_range)
        if self.outer_range <= self.inner_range:
            raise ValueError(
                f"Outer initial-state range must exceed the native range "
                f"{self.inner_range}, got {self.outer_range}."
            )
        self.max_reset_attempts = int(max_reset_attempts)
        self.object_center = np.asarray(
            base_env.initial_gripper_xpos[:2],
            dtype=np.float32,
        )
        base_env.obj_range = self.outer_range

    def reset(self, **kwargs):
        reset_kwargs = kwargs
        for _ in range(self.max_reset_attempts):
            obs, info = self.env.reset(**reset_kwargs)
            reset_kwargs = {}
            object_position = np.asarray(obs["achieved_goal"], dtype=np.float32)
            if (
                np.max(np.abs(object_position[:2] - self.object_center))
                > self.inner_range
            ):
                info = dict(info)
                info["initial_state_range_inner"] = self.inner_range
                info["initial_state_range_outer"] = self.outer_range
                return obs, info
        raise RuntimeError(
            f"Could not sample an initial object state outside range "
            f"{self.inner_range} within {self.max_reset_attempts} resets."
        )


class FetchFixedStateGoalWrapper(gym.Wrapper):
    def __init__(self, env, task_name, initial_position=None, goal_position=None):
        super().__init__(env)
        self.task_name = normalize_fetch_task_name(task_name)
        self.initial_position = self._position(initial_position, "initial_position")
        self.goal_position = self._position(goal_position, "goal_position")
        if self.initial_position is not None and self.task_name == "reach":
            raise ValueError(
                "A fixed object initial position is not available for FetchReach."
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_env = self.env.unwrapped

        if self.initial_position is not None:
            object_qpos = np.array(
                base_env._utils.get_joint_qpos(
                    base_env.model,
                    base_env.data,
                    "object0:joint",
                ),
                dtype=np.float64,
                copy=True,
            )
            object_qpos[:3] = self.initial_position
            base_env._utils.set_joint_qpos(
                base_env.model,
                base_env.data,
                "object0:joint",
                object_qpos,
            )
            base_env._mujoco.mj_forward(base_env.model, base_env.data)

        if self.goal_position is not None:
            base_env.goal = np.array(self.goal_position, dtype=np.float64, copy=True)
            wrapped_env = self.env
            while wrapped_env is not None:
                if isinstance(wrapped_env, FetchGoalScaleWrapper):
                    wrapped_env.episode_goal = np.array(
                        self.goal_position,
                        dtype=np.float32,
                        copy=True,
                    )
                    break
                wrapped_env = getattr(wrapped_env, "env", None)

        if self.initial_position is not None or self.goal_position is not None:
            previous_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
            obs = base_env._get_obs()
            if self.goal_position is None:
                obs["desired_goal"] = np.array(previous_goal, copy=True)

        info = dict(info)
        if self.initial_position is not None:
            info["fixed_initial_position"] = np.array(
                self.initial_position,
                dtype=np.float32,
                copy=True,
            )
        if self.goal_position is not None:
            info["fixed_goal_position"] = np.array(
                self.goal_position,
                dtype=np.float32,
                copy=True,
            )
        return obs, info

    @staticmethod
    def _position(value, name):
        if value is None:
            return None
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} must contain exactly three finite coordinates.")
        return position


class FetchObservationPaddingWrapper(gym.Wrapper):
    def __init__(self, env, target_observation_dim):
        super().__init__(env)
        self.target_observation_dim = int(target_observation_dim)

        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError(
                "FetchObservationPaddingWrapper expects a Dict observation space."
            )

        observation_space = obs_space.spaces["observation"]
        self.source_observation_dim = int(np.prod(observation_space.shape))
        if self.source_observation_dim > self.target_observation_dim:
            raise ValueError(
                f"Cannot pad Fetch observation dim {self.source_observation_dim} "
                f"to smaller target dim {self.target_observation_dim}."
            )

        spaces = dict(obs_space.spaces)
        low = self._pad_observation(
            np.asarray(observation_space.low, dtype=np.float32), pad_value=0.0
        )
        high = self._pad_observation(
            np.asarray(observation_space.high, dtype=np.float32), pad_value=0.0
        )
        spaces["observation"] = gym.spaces.Box(low, high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pad_obs(obs), reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _pad_obs(self, obs):
        obs = dict(obs)
        obs["observation"] = self._pad_observation(obs["observation"])
        return obs

    def _pad_observation(self, observation, pad_value=0.0):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape[0] == self.target_observation_dim:
            return observation.astype(np.float32)
        padded = np.full(self.target_observation_dim, pad_value, dtype=np.float32)
        padded[: observation.shape[0]] = observation
        return padded


class FetchOnlineGoalWrapper(gym.Wrapper):
    def __init__(self, env, task_name, success_threshold=0.05):
        super().__init__(env)
        self.task_name = normalize_fetch_task_name(task_name)
        self.success_threshold = float(success_threshold)
        self.goal_start, self.goal_end = FETCH_GOAL_STATE_SLICES[self.task_name]

        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError("FetchOnlineGoalWrapper expects a Dict observation space.")

        observation_space = obs_space.spaces["observation"]
        state_low = np.asarray(observation_space.low, dtype=np.float32).reshape(-1)
        state_high = np.asarray(observation_space.high, dtype=np.float32).reshape(-1)
        self.state_dim = int(state_low.shape[0])
        state_space = gym.spaces.Box(state_low, state_high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": state_space,
                "achieved_goal": gym.spaces.Box(
                    state_low.copy(), state_high.copy(), dtype=np.float32
                ),
                "desired_goal": gym.spaces.Box(
                    state_low.copy(), state_high.copy(), dtype=np.float32
                ),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        online_obs = self._online_obs(obs)
        return online_obs, self._online_info(info, obs, online_obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        online_obs = self._online_obs(obs)
        return (
            online_obs,
            reward,
            terminated,
            truncated,
            self._online_info(info, obs, online_obs),
        )

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        achieved_goal = self._extract_fetch_goal(achieved_goal)
        desired_goal = self._extract_fetch_goal(desired_goal)
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _online_obs(self, obs):
        state = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
        desired_position = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(
            -1
        )[:3]
        desired_state = np.zeros_like(state, dtype=np.float32)
        if self.goal_start > 0:
            desired_state[: self.goal_start] = desired_position
        desired_state[self.goal_start : self.goal_end] = desired_position
        return {
            "observation": np.array(state, dtype=np.float32, copy=True),
            "achieved_goal": np.array(state, dtype=np.float32, copy=True),
            "desired_goal": desired_state,
        }

    def _online_info(self, info, obs, online_obs):
        info = dict(info)
        fetch_achieved_goal = np.asarray(
            obs["achieved_goal"], dtype=np.float32
        ).reshape(-1)[:3]
        fetch_desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(
            -1
        )[:3]
        distance = float(np.linalg.norm(fetch_achieved_goal - fetch_desired_goal))
        success = distance <= self.success_threshold
        info["is_success"] = np.float32(success)
        info["success"] = bool(success)
        info["goal_distance"] = distance
        info["fetch_achieved_goal"] = np.array(
            fetch_achieved_goal, dtype=np.float32, copy=True
        )
        info["fetch_desired_goal"] = np.array(
            fetch_desired_goal, dtype=np.float32, copy=True
        )
        info["achieved_goal"] = np.array(
            online_obs["achieved_goal"], dtype=np.float32, copy=True
        )
        info["desired_goal"] = np.array(
            online_obs["desired_goal"], dtype=np.float32, copy=True
        )
        return info

    def _extract_fetch_goal(self, goal):
        goal = np.asarray(goal, dtype=np.float32)
        if goal.shape[-1] == 3:
            return goal
        if goal.shape[-1] != self.state_dim:
            raise ValueError(
                f"Fetch goal has dim {goal.shape[-1]}, expected 3 or {self.state_dim}."
            )
        return goal[..., self.goal_start : self.goal_end]


class FetchFlatObservationWrapper(gym.Wrapper):
    def __init__(self, env, keys=("observation", "achieved_goal", "desired_goal")):
        super().__init__(env)
        self.keys = tuple(keys)
        spaces = env.observation_space.spaces
        lows = [
            np.asarray(spaces[key].low, dtype=np.float32).reshape(-1)
            for key in self.keys
        ]
        highs = [
            np.asarray(spaces[key].high, dtype=np.float32).reshape(-1)
            for key in self.keys
        ]
        self.observation_space = gym.spaces.Box(
            np.concatenate(lows),
            np.concatenate(highs),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _flatten(self, obs):
        return np.concatenate(
            [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in self.keys]
        ).astype(np.float32)


class SuccessCounter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False) or info.get("is_success", False):
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)

    def pop_successes(self):
        successes = self.successes
        self.successes = []
        return successes


class FetchGoalEnvSequence:
    def __init__(
        self,
        change_freq=1e7,
        base_task_name=None,
        env_sequence="set1",
        task_order=None,
        goal_conditioned=True,
        gc_reward_type="sparse",
        gc_success_threshold=0.05,
        fetch_goal_format="online",
        fetch_env_version="auto",
        max_episode_steps=50,
        slide_goal_scale=1.0,
        eval_goal_outer_range=None,
        eval_initial_state_outer_range=None,
        eval_initial_position=None,
        eval_goal_position=None,
        normalize_obs=None,
        normalize_avg_coef=0.0001,
        normalize_rewards=False,
        reset_obs_stats=False,
        change_when_solved=False,
        capture_video=False,
        seed=None,
        **kwargs,
    ):
        del capture_video, normalize_rewards, kwargs
        self.env_sequence = env_sequence
        self.env_list = resolve_fetch_sequence(env_sequence, task_order, base_task_name)
        self.base_task_name = base_task_name
        self.goal_conditioned = bool(goal_conditioned)
        self.gc_reward_type = gc_reward_type
        self.gc_success_threshold = gc_success_threshold
        self.fetch_goal_format = normalize_fetch_goal_format(fetch_goal_format)
        self.fetch_env_version = fetch_env_version
        self.max_episode_steps = max_episode_steps
        self.slide_goal_scale = float(slide_goal_scale)
        if not 0.0 < self.slide_goal_scale <= 1.0:
            raise ValueError("slide_goal_scale must be in (0, 1].")
        self.eval_goal_outer_range = (
            None if eval_goal_outer_range is None else float(eval_goal_outer_range)
        )
        self.eval_initial_state_outer_range = (
            None
            if eval_initial_state_outer_range is None
            else float(eval_initial_state_outer_range)
        )
        self.eval_initial_position = FetchFixedStateGoalWrapper._position(
            eval_initial_position,
            "eval_initial_position",
        )
        self.eval_goal_position = FetchFixedStateGoalWrapper._position(
            eval_goal_position,
            "eval_goal_position",
        )
        if (
            self.eval_initial_position is not None
            and self.eval_initial_state_outer_range is not None
        ):
            raise ValueError(
                "eval_initial_position and eval_initial_state_outer_range are "
                "mutually exclusive."
            )
        if (
            self.eval_goal_position is not None
            and self.eval_goal_outer_range is not None
        ):
            raise ValueError(
                "eval_goal_position and eval_goal_outer_range are mutually exclusive."
            )
        self.episode_length = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.fetch_observation_dim = max(
            FETCH_OBSERVATION_DIMS[task_name] for task_name in self.env_list
        )
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.reset_obs_stats = reset_obs_stats
        self.change_when_solved = change_when_solved
        self.change_freq = int(change_freq)
        self.base_seed = seed
        self.current_seed = seed
        self.current_env_id = None

        self.env = None
        self.timestep_counter = 0
        self.task_counter = 0
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 1e-4
        self.bias_correction = False
        self.eval_success_history = []
        self._change_task_next_step = False
        self._pending_reset_seed = seed

        self.make_task()

    def reset(self, seed=None, options=None):
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        elif self._pending_reset_seed is not None:
            reset_kwargs["seed"] = self._pending_reset_seed
            self._pending_reset_seed = None
        if options is not None:
            reset_kwargs["options"] = options

        obs, info = self.env.reset(**reset_kwargs)
        if self._uses_obs_normalization():
            obs = self._normalize_obs(obs)
        return obs, info

    def make_task(self):
        self.task_counter += 1
        self.base_task_name = self.env_list[
            (self.task_counter - 1) % len(self.env_list)
        ]
        self._replace_env(self.base_task_name)
        print(
            f"TASK {self.task_counter} {self.current_seed} {self.base_task_name} ({self.current_env_id})"
        )

    def set_task(self, task_name):
        self.base_task_name = normalize_fetch_task_name(task_name)
        self._replace_env(self.base_task_name)

    def _replace_env(self, task_name):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = self._wrap_env(task_name)
        self.env.action_space.seed(self.current_seed)
        self.env.observation_space.seed(self.current_seed)
        self._pending_reset_seed = self.current_seed

        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self._obs_statistics_shape(), dtype=np.float32)
            self.obs_var = (
                np.zeros(self._obs_statistics_shape(), dtype=np.float32)
                if self.bias_correction
                else np.ones(self._obs_statistics_shape(), dtype=np.float32)
            )
            self.obs_count = 1e-4

    def _wrap_env(self, task_name, eval_mode=False):
        env, env_id = make_fetch_env(
            task_name,
            reward_type=self.gc_reward_type,
            fetch_env_version=self.fetch_env_version,
            max_episode_steps=self.max_episode_steps,
        )
        self.current_env_id = env_id
        if task_name == "slide" and self.slide_goal_scale < 1.0:
            env = FetchGoalScaleWrapper(env, self.slide_goal_scale)
        if eval_mode and self.eval_goal_outer_range is not None:
            if task_name not in {"push", "pick-and-place"}:
                raise ValueError(
                    "eval_goal_outer_range only supports push and pick-and-place."
                )
            env = FetchOuterGoalRangeWrapper(env, self.eval_goal_outer_range)
        if eval_mode and self.eval_initial_state_outer_range is not None:
            if task_name not in {"push", "pick-and-place"}:
                raise ValueError(
                    "eval_initial_state_outer_range only supports push and "
                    "pick-and-place."
                )
            env = FetchOuterInitialStateRangeWrapper(
                env,
                self.eval_initial_state_outer_range,
            )
        if eval_mode and (
            self.eval_initial_position is not None
            or self.eval_goal_position is not None
        ):
            env = FetchFixedStateGoalWrapper(
                env,
                task_name,
                initial_position=self.eval_initial_position,
                goal_position=self.eval_goal_position,
            )
        env = FetchGoalInfoWrapper(
            env,
            reward_type=self.gc_reward_type,
            success_threshold=self.gc_success_threshold,
        )
        env = FetchObservationPaddingWrapper(env, self.fetch_observation_dim)
        if self.fetch_goal_format == "online":
            env = FetchOnlineGoalWrapper(
                env,
                task_name,
                success_threshold=self.gc_success_threshold,
            )
        if not self.goal_conditioned:
            env = FetchFlatObservationWrapper(env)
        env = RecordEpisodeStatistics(env)
        env = SuccessCounter(env)
        return env

    def step(self, action):
        self.timestep_counter += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._uses_obs_normalization():
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)

        if self.change_when_solved and self._change_task_next_step:
            self._change_task_next_step = False
            self.make_task()
            truncated = True
        elif self.timestep_counter % self.change_freq == 0:
            self.make_task()
            truncated = True

        return obs, reward, terminated, truncated, info

    def no_count_step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._uses_obs_normalization():
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)
        return obs, reward, terminated, truncated, info

    def evaluate_agent(self, agent, num_eval_episodes=10, reseed_each_episode=True):
        test_env = self._wrap_env(self.base_task_name, eval_mode=True)
        try:
            reset_kwargs = (
                {"seed": self.current_seed} if self.current_seed is not None else {}
            )
            subsequent_reset_kwargs = reset_kwargs if reseed_each_episode else {}
            obs, _ = test_env.reset(**reset_kwargs)
            if self._uses_obs_normalization():
                obs = self._normalize_obs(obs)

            eval_results = {}
            episodic_returns = []
            goal_successes = []
            final_goal_distances = []
            min_goal_distances = []
            mean_goal_distances = []
            current_goal_success = False
            current_goal_distances = []
            agent.eval()

            while len(episodic_returns) < num_eval_episodes:
                action = self._evaluate_action(agent, obs)
                next_obs, _, terminated, truncated, info = test_env.step(action)

                if "is_success" in info:
                    current_goal_success = current_goal_success or bool(
                        info["is_success"]
                    )
                if "goal_distance" in info:
                    current_goal_distances.append(float(info["goal_distance"]))

                if self._uses_obs_normalization():
                    next_obs = self._normalize_obs(next_obs)

                if "episode" in info:
                    episodic_returns.append(info["episode"]["r"])
                    if current_goal_distances:
                        goal_successes.append(current_goal_success)
                        final_goal_distances.append(current_goal_distances[-1])
                        min_goal_distances.append(float(np.min(current_goal_distances)))
                        mean_goal_distances.append(
                            float(np.mean(current_goal_distances))
                        )
                    current_goal_success = False
                    current_goal_distances = []

                obs = next_obs
                if terminated or truncated:
                    obs, _ = test_env.reset(**subsequent_reset_kwargs)
                    if self._uses_obs_normalization():
                        obs = self._normalize_obs(obs)

            agent.train()
            eval_results["episodic_returns"] = episodic_returns
            eval_results["successes"] = test_env.pop_successes()
            eval_results["goal_successes"] = goal_successes
            eval_results["final_goal_distances"] = final_goal_distances
            eval_results["min_goal_distances"] = min_goal_distances
            eval_results["mean_goal_distances"] = mean_goal_distances

            if self.change_when_solved:
                self.eval_success_history.append(np.mean(eval_results["successes"]))
                self._change_task_next_step = self._check_solved_task()
            return eval_results
        finally:
            test_env.close()

    def _evaluate_action(self, agent, obs):
        if isinstance(obs, dict):
            obs_vector = obs["observation"]
            act_parameters = inspect.signature(agent.act).parameters
            expected_goal_dim = getattr(agent, "actor_critic_goal_dim", None)
            desired_goal = obs.get("desired_goal")
            if (
                desired_goal is not None
                and "goal_obs" in act_parameters
                and (
                    expected_goal_dim is None
                    or desired_goal.shape[-1] == expected_goal_dim
                )
            ):
                return agent.act(obs_vector, goal_obs=desired_goal)
            return agent.act(obs_vector)
        return agent.act(obs)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        env = self.env
        while env is not None:
            compute_reward = getattr(env, "compute_reward", None)
            if compute_reward is not None:
                return compute_reward(achieved_goal, desired_goal, info)
            env = getattr(env, "env", None)
        raise AttributeError(
            "compute_reward is only available for goal-conditioned Fetch observations."
        )

    def close(self):
        if self.env is not None:
            self.env.close()

    def _uses_obs_normalization(self):
        return self.normalize_obs not in {None, False}

    def _obs_statistics_shape(self):
        observation_space = self.env.observation_space
        if isinstance(observation_space, gym.spaces.Dict):
            return observation_space.spaces["observation"].shape
        return observation_space.shape

    def _normalizable_obs(self, obs):
        if isinstance(obs, dict):
            return obs["observation"]
        return obs

    def _replace_normalizable_obs(self, obs, normalized_obs):
        if isinstance(obs, dict):
            new_obs = dict(obs)
            new_obs["observation"] = normalized_obs.astype(np.float32)
            return new_obs
        return normalized_obs.astype(np.float32)

    def _update_obs_statistics(self, obs):
        obs = np.asarray(self._normalizable_obs(obs), dtype=np.float32)
        if self.normalize_obs.lower() == "ema":
            self.obs_mean = (
                1 - self.normalize_avg_coef
            ) * self.obs_mean + self.normalize_avg_coef * obs
            self.obs_var = (
                1 - self.normalize_avg_coef
            ) * self.obs_var + self.normalize_avg_coef * (obs - self.obs_mean) ** 2
        elif self.normalize_obs.lower() == "straight":
            mean = self.obs_mean
            var = self.obs_var
            count = self.obs_count
            delta = obs - mean
            total_count = count + 1
            self.obs_mean = mean + delta / total_count
            self.obs_var = var * count / total_count + delta**2 * count / total_count**2
            self.obs_count = total_count

    def _normalize_obs(self, obs):
        if self.timestep_counter == 0:
            return obs
        obs_array = np.asarray(self._normalizable_obs(obs), dtype=np.float32)
        if self.bias_correction:
            bias_correction = 1 - (1 - self.normalize_avg_coef) ** self.timestep_counter
            obs_mean = self.obs_mean / bias_correction
            obs_var = self.obs_var / bias_correction
        else:
            obs_mean = self.obs_mean
            obs_var = self.obs_var
        normalized_obs = (obs_array - obs_mean) / (np.sqrt(obs_var) + 1e-8)
        normalized_obs = np.clip(normalized_obs, -10, 10)
        return self._replace_normalizable_obs(obs, normalized_obs)

    def _check_solved_task(self):
        return (
            len(self.eval_success_history) >= 5
            and np.min(self.eval_success_history[-5:]) >= 0.799
        )


if __name__ == "__main__":
    env = FetchGoalEnvSequence(change_freq=1000, env_sequence="set1", seed=0)
    obs, info = env.reset()
    print(env.env_list)
    print(env.base_task_name, env.current_env_id)
    print({key: value.shape for key, value in obs.items()}, info.get("success"))
