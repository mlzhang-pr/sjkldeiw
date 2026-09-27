# Fetch Continual Quasimetric RL

This folder contains a Fetch-specific rewrite of the continual quasimetric training entry point.

Local files:

- `train_fetch.py`: Fetch-only training script based on the algorithm flow from `Metaworld/continual_quasimetric_main2.py`.
- `train_td3_her.py`: TD3 + HER training script following the Stable-Baselines3 TD3/HER algorithm structure.
- `fetch_env.py`: Gymnasium Robotics Fetch task sequence wrapper.
- `replay_buffer.py`: local replay buffer and collector. The collector keeps the current Fetch `desired_goal` and passes it to goal-conditioned actors during rollout.
- `replay_buffer_metric.py`: HER/future-goal replay buffer used by metric SAC and the quasimetric agent.
- `replay_buffer_td3_her.py`: SB3-style future-goal relabeling buffer for TD3 + HER.
- `agent/`: local copy of the SAC, metric SAC, quasimetric, and TD3 + HER agent modules used by the training scripts.

Run from this directory:

```bash
cd continual-quasimetric-rl/Fetch
python train_fetch.py \
  --env fetch_sequence_custom \
  --task_order reach,push,pick-and-place,slide \
  --goal_conditioned 1 \
  --method buffer \
  --gpu 0 \
  --change_freq 200000 \
  --random_steps 10000 \
  --log_backends none
```

For a quick single-task sanity check:

```bash
python train_fetch.py \
  --env fetch_reach \
  --goal_conditioned 1 \
  --method independent \
  --gpu 0 \
  --change_freq 200000 \
  --log_backends none
```

TD3 + HER baseline:

```bash
python train_td3_her.py \
  --env fetch_reach \
  --fetch_goal_format native \
  --total_timesteps 200000 \
  --learning_starts 10000 \
  --eval_freq 5000 \
  --log_backends wandb \
  --wandb_project_name continual-quasimetric-rl
```

W&B logging is enabled by default in `train_td3_her.py`; pass `--log_backends none` for CSV-only local runs.

TD3 + HER on a continual Fetch sequence:

```bash
python train_td3_her.py \
  --env fetch_sequence_custom \
  --task_order reach,push,pick-and-place,slide \
  --fetch_goal_format native \
  --total_timesteps 800000 \
  --change_freq 200000 \
  --reset_buffer_on_task_change 1
```