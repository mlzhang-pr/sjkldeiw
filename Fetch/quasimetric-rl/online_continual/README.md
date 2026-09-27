# Online Continual Fetch

This folder contains the Fetch continual-learning entrypoint adapted from
`Metaworld/continual_quasimetric_main2.py`.

Run from `Fetch/quasimetric-rl`:

```sh
./online_continual/run_fetch_continual.sh \
  --env fetch_sequence_set1 \
  --method buffer \
  --gpu 0 \
  --log_backends tensorboard
```

Use a custom Fetch task stream with:

```sh
./online_continual/run_fetch_continual.sh \
  --env fetch_sequence_custom \
  --task_order reach,push,pick-and-place,slide
```

For an easier FetchSlide curriculum stage, halve only the slide target distance
while keeping the other tasks unchanged:

```sh
./online_continual/run_fetch_continual.sh \
  --env fetch_sequence_custom \
  --task_order push,pick-and-place,slide \
  --slide_goal_scale 0.5 \
  --gc_success_threshold 0.08 \
  --max_episode_steps 100
```

`--slide_goal_scale 1.0` restores the standard FetchSlide target distribution.

## Resume from a task

Resume at a 1-based position in the original task sequence with the checkpoint
saved after the preceding task:

```sh
python -m online_continual.main_from_offline \
  --resume_run_dir online_continual/results/fetch_continual_push-pick-and-place_seed0_buffer_gc-sparse \
  --start_task 2 \
  --gpu 0
```

The source `run_config.json` supplies the original task sequence and model
configuration. By default, task 2 loads the `*_task1` student and meta
checkpoints. Use `--resume_model_name` to select another checkpoint basename,
or `--load_resume_buffer 1` to also restore a saved quasimetric replay buffer.
Resumed outputs use a `_from-taskN` suffix and do not overwrite the source run.

## FAME baseline

Run the FAME fast/meta learner baseline from `Fetch/quasimetric-rl`:

```sh
python -m online_continual.fame \
  --env fetch_sequence_custom \
  --env_sequence none \
  --task_order reach,push,pick-and-place,slide \
  --gpu 0 \
  --log_backends tensorboard
```

FAME keeps one QRL fast learner across tasks and a separate QRL meta policy. At a
task boundary it compares the fast, meta, and fresh policies, uses the meta
policy as a temporary KL regularizer when it transfers best, and integrates
the fast learner's recent trajectories into the long-term meta memory. The
second task intentionally skips meta detection because the meta policy has not
yet seen enough tasks for its first integration update.

Set `--force_meta 1` to include meta in task-2 detection and select it at every
task boundary regardless of the detection returns. The main FAME controls are
`--force_meta`, `--detection_episodes`, `--warmup_steps`, `--lambda_reg`,
`--store_traj_num`, and `--meta_update_steps`.
