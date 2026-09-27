import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

gym.register_envs(gymnasium_robotics)

train_env = gym.make("FetchPickAndPlace-v4", reward_type="sparse")
eval_env = gym.make("FetchPickAndPlace-v4",reward_type="sparse")

# 关键：Monitor 记录 is_success
eval_env = Monitor(eval_env, info_keywords=("is_success",))

model = TD3(
    policy="MultiInputPolicy",
    env=train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.95,
    learning_rate=1e-3,
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/eval/",
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
)