import metaworld
from metaworld import (
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
)


import inspect
from typing import Callable, Dict, List
import gymnasium as gym
from gymnasium.wrappers import (
    DtypeObservation,
    TimeLimit,
    RecordEpisodeStatistics,
    RecordVideo,
    ClipAction,
)
import numpy as np
from envs.metaworld_env_sequences import (
    GOOD_RPO_SEQS,
    RPO20_SEQ,
    RPO10_SEQ,
    RPO10_SHORT,
)


version = 3

EnvFn = Callable[[], gym.Env]


def v3_task_name(task_name):
    if task_name is None or not task_name.endswith("-v2"):
        return task_name
    return f"{task_name[:-3]}-v3"


class MetaWorldGoalConditionedWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        task_name=None,
        reward_type="sparse",
        success_threshold=0.05,
        achieved_goal="auto",
    ):
        super().__init__(env)
        self.task_name = task_name or ""
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.achieved_goal = achieved_goal

        if reward_type not in {"sparse", "dense", "success"}:
            raise ValueError(f"Unsupported goal-conditioned reward type: {reward_type}")
        if achieved_goal not in {"auto", "object", "tcp"}:
            raise ValueError(f"Unsupported achieved goal source: {achieved_goal}")

        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
            raise TypeError(
                "MetaWorldGoalConditionedWrapper expects a flat Box observation space."
            )

        observation_low = np.asarray(obs_space.low[:-3], dtype=np.float32)
        observation_high = np.asarray(obs_space.high[:-3], dtype=np.float32)
        goal_low = np.full(3, -np.inf, dtype=np.float32)
        goal_high = np.full(3, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    observation_low, observation_high, dtype=np.float32
                ),
                "achieved_goal": gym.spaces.Box(goal_low, goal_high, dtype=np.float32),
                "desired_goal": gym.spaces.Box(goal_low, goal_high, dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal_obs = self._goal_observation(obs)
        info = self._goal_info(info, goal_obs)
        return goal_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal_obs = self._goal_observation(obs)
        reward = self.compute_reward(
            goal_obs["achieved_goal"], goal_obs["desired_goal"], info
        )
        info = self._goal_info(info, goal_obs)
        return goal_obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        del info
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if self.reward_type == "dense":
            reward = -distance
        else:
            reached = distance <= self.success_threshold
            reward = reached.astype(np.float32)
            if self.reward_type == "sparse":
                reward = reward - 1.0

        if np.ndim(reward) == 0:
            return np.float32(reward)
        return reward.astype(np.float32)

    def _goal_observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return {
            "observation": np.array(obs[:-3], dtype=np.float32, copy=True),
            "achieved_goal": self._get_achieved_goal(),
            "desired_goal": self._get_desired_goal(),
        }

    def _goal_info(self, info, goal_obs):
        info = dict(info)
        distance = float(
            np.linalg.norm(goal_obs["achieved_goal"] - goal_obs["desired_goal"])
        )
        info["goal_distance"] = distance
        info["is_success"] = np.float32(distance <= self.success_threshold)
        info["achieved_goal"] = np.array(
            goal_obs["achieved_goal"], dtype=np.float32, copy=True
        )
        info["desired_goal"] = np.array(
            goal_obs["desired_goal"], dtype=np.float32, copy=True
        )
        return info

    def _get_desired_goal(self):
        env = self.unwrapped
        if hasattr(env, "_get_pos_goal"):
            return (
                np.asarray(env._get_pos_goal(), dtype=np.float32).reshape(-1)[:3].copy()
            )
        if hasattr(env, "_target_pos"):
            return np.asarray(env._target_pos, dtype=np.float32).reshape(-1)[:3].copy()
        raise AttributeError(
            "MetaWorld environment does not expose a desired goal position."
        )

    def _get_achieved_goal(self):
        env = self.unwrapped
        source = self.achieved_goal
        if source == "auto":
            source = "tcp" if self.task_name.startswith("reach") else "object"

        if source == "object" and hasattr(env, "_get_pos_objects"):
            try:
                object_pos = np.asarray(
                    env._get_pos_objects(), dtype=np.float32
                ).reshape(-1)
                if object_pos.size >= 3 and np.all(np.isfinite(object_pos[:3])):
                    return object_pos[:3].copy()
            except NotImplementedError:
                pass

        if hasattr(env, "get_endeff_pos"):
            return (
                np.asarray(env.get_endeff_pos(), dtype=np.float32)
                .reshape(-1)[:3]
                .copy()
            )
        if hasattr(env, "tcp_center"):
            return np.asarray(env.tcp_center, dtype=np.float32).reshape(-1)[:3].copy()
        raise AttributeError(
            "MetaWorld environment does not expose an achieved goal position."
        )


GOOD_ENVS = [
    "handle-press-side-v2",
    "faucet-close-v2",
    "plate-slide-v2",
    "window-open-v2",
    "reach-wall-v2",
    "button-press-v2",
    "plate-slide-side-v2",
    "handle-press-v2",
]


SMALL_SEQUENCE_ENVS = [
    "handle-press-v2",
    "plate-slide-side-v2",
    "button-press-v2",
    "plate-slide-v2",
    "handle-press-side-v2",
    "faucet-close-v2",
]


class MetaWorldSingleEnvSequence:
    def __init__(
        self,
        change_freq=1e7,
        base_task_name=None,
        env_sequence="metaworld_sequence_set1",
        goal_hidden=True,
        goal_conditioned=True,
        gc_reward_type="sparse",
        gc_success_threshold=0.05,
        gc_achieved_goal="auto",
        normalize_obs="straight",
        normalize_avg_coef=0.0001,
        normalize_rewards=True,
        reset_obs_stats=False,
        change_when_solved=False,
        freeze_rand_vec=True,
        capture_video=False,
        seed=None,
        *args,
        **kwargs,
    ):
        """
        This class is used to generate a stream of tasks in one of the metaworld envs
        Each task is generated with a new random seed. (I think you can go on forever?)
        base_task_name: name of metaworld env e.g. 'reach-v2'
        change_freq: number of steps until a task change
        env_sequence: String that specifies which env sequence to use
        normalize_obs: If True, keeps a moving average of observations and normalizes the observations
        normalize_avg_coef: To be used in the moving avg
        reset_obs_stats: If True, resets the observation normalization stats every time the task changes
        goal_hidden: If True, the agent does not observe the goal location
        goal_conditioned: If True, return a dict with observation, achieved_goal, and desired_goal
        gc_reward_type: Reward used by the goal-conditioned wrapper: sparse, dense, or success
        gc_success_threshold: Distance threshold for goal success in the goal-conditioned wrapper
        gc_achieved_goal: Source for achieved_goal: auto, object, or tcp
        freeze_rand_vec: If True, reuse one task instance across episode resets
        change_when_solved: Change to the next task whenever the agent receives above 90% success 5 times in a row.
        """

        self.base_task_name = v3_task_name(base_task_name)
        self.env_sequence = env_sequence

        if self.env_sequence is not None:
            if self.env_sequence[-2:].isnumeric():
                env_set_id = int(self.env_sequence[-2:])
            else:
                env_set_id = int(self.env_sequence[-1])
            self.env_list = [
                v3_task_name(task_name) for task_name in RPO10_SEQ[env_set_id - 1]
            ]

        self.goal_str = "-goal-hidden" if goal_hidden else "-goal-observable"
        self.metaworld_envs = (
            ALL_V3_ENVIRONMENTS_GOAL_HIDDEN
            if goal_hidden
            else ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
        )
        if self.base_task_name is not None:
            self.base_task_class = self.metaworld_envs[
                self.base_task_name + self.goal_str
            ]

        self.env = None
        self.base_seed = seed
        self.current_seed = seed
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.normalize_rewards = normalize_rewards
        self.reset_obs_stats = reset_obs_stats
        self.change_when_solved = change_when_solved
        self.goal_conditioned = goal_conditioned
        self.gc_reward_type = gc_reward_type
        self.gc_success_threshold = gc_success_threshold
        self.gc_achieved_goal = gc_achieved_goal
        self.freeze_rand_vec = bool(freeze_rand_vec)

        self.change_freq = change_freq
        self.timestep_counter = 0
        self.task_counter = 0
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 1e-4
        self.bias_correction = False

        self.eval_success_history = []
        self._change_task_next_step = False

        self.rng = np.random.RandomState(seed=self.current_seed)

        self.make_task()

    def reset(self):
        obs, info = self.env.reset()

        if self._uses_obs_normalization():
            obs = self._normalize_obs(obs)

        return obs, info

    def make_task(self):

        self.task_counter += 1

        if self.env_sequence is not None:
            if self.env_sequence[-2:].isnumeric():
                env_set_id = int(self.env_sequence[-2:])
            else:
                env_set_id = int(self.env_sequence[-1])

            self.base_task_name = self.env_list[
                (self.task_counter - 1) % len(self.env_list)
            ]

            self.base_task_class = self.metaworld_envs[
                self.base_task_name + self.goal_str
            ]

        temp_env = self._make_base_env()

        self.env = self._wrap_env(temp_env)

        self.env.action_space.seed(self.current_seed)
        self.env.observation_space.seed(self.current_seed)

        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self._obs_statistics_shape())

            if self.bias_correction:
                self.obs_var = np.zeros(self._obs_statistics_shape())
            else:
                self.obs_var = np.ones(self._obs_statistics_shape())
            self.obs_count = 1e-4

        print(f"TASK {self.task_counter}  {self.current_seed} {self.base_task_name}")
        return

    def set_task(self, task_name):
        self.base_task_name = v3_task_name(task_name)
        self.base_task_class = self.metaworld_envs[self.base_task_name + self.goal_str]

        temp_env = self._make_base_env()

        self.env = self._wrap_env(temp_env)

        self.env.action_space.seed(self.current_seed)
        self.env.observation_space.seed(self.current_seed)

        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self._obs_statistics_shape())

            if self.bias_correction:
                self.obs_var = np.zeros(self._obs_statistics_shape())
            else:
                self.obs_var = np.ones(self._obs_statistics_shape())
            self.obs_count = 1e-4

    def _make_base_env(self):
        env = self.base_task_class(seed=self.current_seed)
        env._freeze_rand_vec = self.freeze_rand_vec
        return env

    def _wrap_env(self, env, eval_mode=False):

        env = DtypeObservation(env, dtype=np.float32)
        if self.goal_conditioned:
            env = MetaWorldGoalConditionedWrapper(
                env,
                task_name=self.base_task_name,
                reward_type=self.gc_reward_type,
                success_threshold=self.gc_success_threshold,
                achieved_goal=self.gc_achieved_goal,
            )

        if not eval_mode and self.normalize_rewards and not self.goal_conditioned:
            env = gym.wrappers.TransformReward(env, lambda r: r / 500)
        env = TimeLimit(env, max_episode_steps=200)
        env = RecordEpisodeStatistics(env)
        env = SuccessCounter(env)

        return env

    def step(self, action):
        self.timestep_counter += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._uses_obs_normalization():
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)

        if self.change_when_solved:
            if self._change_task_next_step:
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
        """Runs and evaluation of the agent
        It runs on the current seed i.e. the current task"""

        test_env = self._wrap_env(self._make_base_env(), eval_mode=True)
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
                current_goal_success = current_goal_success or bool(info["is_success"])
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
                    mean_goal_distances.append(float(np.mean(current_goal_distances)))
                current_goal_success = False
                current_goal_distances = []

            obs = next_obs

            if terminated or truncated:
                obs, _ = test_env.reset(**subsequent_reset_kwargs)

        agent.train()
        eval_results["episodic_returns"] = episodic_returns
        eval_results["successes"] = test_env.pop_successes()
        eval_results["goal_successes"] = goal_successes
        eval_results["final_goal_distances"] = final_goal_distances
        eval_results["min_goal_distances"] = min_goal_distances
        eval_results["mean_goal_distances"] = mean_goal_distances

        if self.change_when_solved:
            self.eval_success_history.append(np.mean(eval_results["successes"]))
            self._change_task_next_step = True
        return eval_results

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
        return normalized_obs

    def _evaluate_action(self, agent, obs):
        if isinstance(obs, dict):
            obs_vector = obs["observation"]
            act_parameters = inspect.signature(agent.act).parameters
            expected_goal_dim = getattr(agent, "actor_critic_goal_dim", None)
            desired_goal = obs["desired_goal"]
            if "goal_obs" in act_parameters and (
                expected_goal_dim is None or desired_goal.shape[-1] == expected_goal_dim
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
            "compute_reward is only available when goal_conditioned=True."
        )

    def _update_obs_statistics(self, obs):
        """Update mean and variance statistics"""
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
            tot_count = count + 1

            new_mean = mean + delta / tot_count
            new_var = var * count / tot_count + delta**2 * count / tot_count**2
            new_count = tot_count

            self.obs_mean = new_mean
            self.obs_var = new_var
            self.obs_count = new_count

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
        if len(self.eval_success_history) >= 5:
            if np.min(self.eval_success_history[-5:]) >= 0.799:
                return True
        return False


class SuccessCounter(gym.Wrapper):
    """From Continual World's Codebase"""

    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        return obs, reward, terminated, truncated, info

    def pop_successes(self):
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    ...

    np.set_printoptions(suppress=True)
    env = MetaWorldSingleEnvSequence(
        1000,
        "reach-v2",
        seed=123,
        obs_drift_std=0.0,
        obs_noise_std=0.01,
        normalize_obs=False,
    )
    obs, _ = env.reset()

    print(obs)

    ...
