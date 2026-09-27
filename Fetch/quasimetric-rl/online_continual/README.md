# Online Continual Fetch

This folder contains the Fetch CONQUEST continual-learning entrypoint.

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

Evaluate a trained run:

```sh
python -m online_continual.conquest_eval \
  --run_dir online_continual/results/cqrl/RUN_NAME \
  --checkpoint final \
  --agent_kind meta \
  --eval_seeds 0 1 2
```

Use `python -m online_continual.conquest_zeroshot --help` for out-of-distribution
goal and initial-state evaluation.
