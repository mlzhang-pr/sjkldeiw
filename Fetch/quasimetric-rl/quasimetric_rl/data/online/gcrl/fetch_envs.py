# coding=utf-8
# Copyright 2022 The Google Research Authors.
# Copyright 2023 Tongzhou Wang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for loading the Gymnasium-Robotics Fetch v4 environments."""

import gymnasium as gym
from gymnasium_robotics.envs.fetch import push
from gymnasium_robotics.envs.fetch import reach
from gymnasium_robotics.envs.fetch import slide
import numpy as np


def get_reward(norm_dist, reward_mode):
    # is_success === norm_dist < 1
    is_success = float(norm_dist < 1)
    if reward_mode == 'dense':
        return np.exp(-norm_dist * np.log(2))  # 0.5 at boundary, 1 at exact
    elif reward_mode == 'positive':
        return is_success
    else:
        assert reward_mode == 'negative'
        return is_success - 1


FETCH_REACH_IMAGE_CAMERA_CONFIG = dict(
    lookat=np.array([1.2, 0.8, 0.5]),
    distance=0.8,
    azimuth=180,
    elevation=-30,
)

FETCH_PUSH_IMAGE_CAMERA_CONFIGS = dict(
    camera1=dict(
        lookat=np.array([1.2, 0.8, 0.4]),
        distance=0.9,
        azimuth=180,
        elevation=-40,
    ),
    camera2=dict(
        lookat=np.array([1.25, 0.8, 0.4]),
        distance=0.65,
        azimuth=90,
        elevation=-40,
    ),
)


def _hide_fetch_markers(env):
    env.model.geom_rgba[1:5] = 0
    target_site_id = env._model_names.site_name2id.get('target0')
    if target_site_id is not None:
        env.model.site_rgba[target_site_id, 3] = 0


def _get_joint_qpos(env, name):
    return env._utils.get_joint_qpos(env.model, env.data, name)


def _set_joint_qpos(env, name, value):
    env._utils.set_joint_qpos(env.model, env.data, name, value)
    env._mujoco.mj_forward(env.model, env.data)


def _reset_obs(reset_result):
    return reset_result[0]


def _step_obs(step_result):
    return step_result[0]



class FetchReachEnv(reach.MujocoFetchReachEnv):
    """Wrapper for the FetchReach environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchReachEnv, self).__init__(reward_type='sparse')
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((20,), -np.inf, dtype=np.float32),
            high=np.full((20,), np.inf, dtype=np.float32),
                dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self, *, seed=None, options=None):
        self.observation_space = self._old_observation_space
        s, info = super(FetchReachEnv, self).reset(seed=seed, options=options)
        self.observation_space = self._new_observation_space
        return self.observation(s), info

    def step(self, action):
        s = _step_obs(super(FetchReachEnv, self).step(action))
        terminated = False
        truncated = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, terminated, truncated, info

    def observation(self, observation):
        start_index = 0
        end_index = 3
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchPushEnv(push.MujocoFetchPushEnv):
    """Wrapper for the FetchPush environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchPushEnv, self).__init__(reward_type='sparse')
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((50,), -np.inf, dtype=np.float32),
            high=np.full((50,), np.inf, dtype=np.float32),
                dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self, *, seed=None, options=None):
        self.observation_space = self._old_observation_space
        s, info = super(FetchPushEnv, self).reset(seed=seed, options=options)
        self.observation_space = self._new_observation_space
        return self.observation(s), info

    def step(self, action):
        s = _step_obs(super(FetchPushEnv, self).step(action))
        terminated = False
        truncated = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, terminated, truncated, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchSlideEnv(slide.MujocoFetchSlideEnv):
    """Wrapper for the FetchSlide environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchSlideEnv, self).__init__(reward_type='sparse')
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((50,), -np.inf, dtype=np.float32),
            high=np.full((50,), np.inf, dtype=np.float32),
                dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self, *, seed=None, options=None):
        self.observation_space = self._old_observation_space
        s, info = super(FetchSlideEnv, self).reset(seed=seed, options=options)
        self.observation_space = self._new_observation_space
        return self.observation(s), info

    def step(self, action):
        s = _step_obs(super(FetchSlideEnv, self).step(action))
        terminated = False
        truncated = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, terminated, truncated, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchReachImageEnv(reach.MujocoFetchReachEnv):
    """Wrapper for the FetchReach environment with image observations."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        self._dist = []
        self._dist_vec = []
        super(FetchReachImageEnv, self).__init__(
            reward_type='sparse',
            render_mode='rgb_array',
            width=64,
            height=64,
            default_camera_config=FETCH_REACH_IMAGE_CAMERA_CONFIG,
        )
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
                low=np.full((64*64*6), 0),
                high=np.full((64*64*6), 255),
                dtype=np.uint8)
        self.observation_space = self._new_observation_space
        _hide_fetch_markers(self)

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def reset(self, *, seed=None, options=None):
        if self._dist:  # if len(self._dist) > 0, ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s, info = super(FetchReachImageEnv, self).reset(seed=seed, options=options)
        self.observation_space = self._new_observation_space
        self._goal = s['desired_goal'].copy()

        for _ in range(10):
            hand = s['achieved_goal']
            obj = s['desired_goal']
            delta = obj - hand
            a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
            s = _step_obs(super(FetchReachImageEnv, self).step(a))

        self._goal_img = self.observation(s)

        self.observation_space = self._old_observation_space
        s, _ = super(FetchReachImageEnv, self).reset(options=options)
        self.observation_space = self._new_observation_space
        img = self.observation(s)
        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        return np.concatenate([img, self._goal_img]), info

    def step(self, action):
        s = _step_obs(super(FetchReachImageEnv, self).step(action))
        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        terminated = False
        truncated = False
        img = self.observation(s)
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return np.concatenate([img, self._goal_img]), r, terminated, truncated, info

    def observation(self, observation):
        img = self.render()
        return img.flatten()

    def compute_reward(self, achieved_goal, goal, info):
        # just image comparison
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.reward_mode == 'positive':
            r = is_success
        else:
            assert self.reward_mode == 'negative'
            r = is_success - 1
        return r


class FetchPushImageEnv(push.MujocoFetchPushEnv):
    """Wrapper for the FetchPush environment with image observations."""

    def __init__(self, camera='camera2', start_at_obj=True, rand_y=False,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        self._start_at_obj = start_at_obj
        self._rand_y = rand_y
        self._camera_name = camera
        self._dist = []
        self._dist_vec = []
        if camera not in FETCH_PUSH_IMAGE_CAMERA_CONFIGS:
            raise NotImplementedError
        super(FetchPushImageEnv, self).__init__(
            reward_type='sparse',
            render_mode='rgb_array',
            width=64,
            height=64,
            default_camera_config=FETCH_PUSH_IMAGE_CAMERA_CONFIGS[camera],
        )
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
                low=np.full((64*64*6), 0),
                high=np.full((64*64*6), 255),
                dtype=np.uint8)
        self.observation_space = self._new_observation_space
        _hide_fetch_markers(self)

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _move_hand_to_obj(self):
        s = super(FetchPushImageEnv, self)._get_obs()
        for _ in range(100):
            hand = s['observation'][:3]
            obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
            delta = obj - hand
            if np.linalg.norm(delta) < 0.06:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            s = _step_obs(super(FetchPushImageEnv, self).step(a))

    def reset(self, *, seed=None, options=None):
        if self._dist:  # if len(self._dist) > 0 ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s, info = super(FetchPushImageEnv, self).reset(seed=seed, options=options)
        self.observation_space = self._new_observation_space
        # Randomize object position
        for _ in range(8):
            super(FetchPushImageEnv, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
        object_qpos = _get_joint_qpos(self, 'object0:joint')
        if not self._rand_y:
            object_qpos[1] = 0.75
        _set_joint_qpos(self, 'object0:joint', object_qpos)
        self._move_hand_to_obj()
        self._goal_img = self.observation(s)
        block_xyz = _get_joint_qpos(self, 'object0:joint')[:3]
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print('Bad reset, recursing.')
            return self.reset(options=options)
        self._goal = block_xyz[:2].copy()

        self.observation_space = self._old_observation_space
        s, _ = super(FetchPushImageEnv, self).reset(options=options)
        self.observation_space = self._new_observation_space
        for _ in range(8):
            super(FetchPushImageEnv, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
        object_qpos = _get_joint_qpos(self, 'object0:joint')
        object_qpos[:2] = np.array([1.15, 0.75])
        _set_joint_qpos(self, 'object0:joint', object_qpos)
        if self._start_at_obj:
            self._move_hand_to_obj()
        else:
            for _ in range(5):
                super(FetchPushImageEnv, self).step(self.action_space.sample())

        block_xyz = _get_joint_qpos(self, 'object0:joint')[:3].copy()
        img = self.observation(s)
        dist = np.linalg.norm(block_xyz[:2] - self._goal)
        self._dist.append(dist)
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print('Bad reset, recursing.')
            return self.reset(options=options)
        return np.concatenate([img, self._goal_img]), info

    def step(self, action):
        s = _step_obs(super(FetchPushImageEnv, self).step(action))
        block_xy = _get_joint_qpos(self, 'object0:joint')[:2]
        dist = np.linalg.norm(block_xy - self._goal)
        self._dist.append(dist)
        terminated = False
        truncated = False
        is_success = float(dist < 0.05)  # Taken from the original task code.
        img = self.observation(s)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return np.concatenate([img, self._goal_img]), r, terminated, truncated, info

    def observation(self, observation):
        img = self.render()
        return img.flatten()

    def compute_reward(self, achieved_goal, goal, info):
        # just image comparison
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.reward_mode == 'positive':
            r = is_success
        else:
            assert self.reward_mode == 'negative'
            r = is_success - 1
        return r



class BackToGymWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, is_image_based: bool):
        super().__init__(env)
        if is_image_based:
            single_ospace = gym.spaces.Box(
                low=np.full((64, 64, 3), 0),
                high=np.full((64, 64, 3), 255),
                dtype=np.uint8,
            )
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
            ospace: gym.spaces.Box = env.observation_space
            single_ospace = gym.spaces.Box(
                low=np.split(ospace.low, 2)[0],
                high=np.split(ospace.high, 2)[0],
                dtype=ospace.dtype,
            )
        self.observation_space = gym.spaces.Dict(dict(
            observation=single_ospace,
            achieved_goal=single_ospace,
            desired_goal=single_ospace,
        ))

    def observation(self, observation):
        o, g = np.split(observation, 2)
        return dict(
            observation=o,
            achieved_goal=o,
            desired_goal=g,
        )

    def compute_reward(self, achieved_goal, goal, info):
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.env.reward_mode == 'positive':
            r = is_success
        else:
            assert self.env.reward_mode == 'negative'
            r = is_success - 1
        return r
