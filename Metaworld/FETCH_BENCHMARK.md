# Fetch Goal-Conditioned Continual Benchmark

This benchmark is implemented in `envs/fetch_env.py` and follows the same task-stream interface as `envs/metaworld_env.py`.

Available task names are:

- `reach`
- `push`
- `pick-and-place`
- `slide`

Built-in sequences:

- `fetch_sequence_set1`: `reach,push,pick-and-place,slide`
- `fetch_sequence_set2`: `push,slide,reach,pick-and-place`
- `fetch_sequence_set3`: `pick-and-place,reach,slide,push`
- `fetch_sequence_set4`: `slide,pick-and-place,push,reach`
- `fetch_sequence_easy2`: `reach,push`
- `fetch_sequence_manipulation3`: `push,pick-and-place,slide`

Custom task order:

```bash
python continual_quasimetric_main2.py \
  --env fetch_sequence_custom \
  --task_order reach,push,pick-and-place,slide \
  --goal_conditioned 1 \
  --method buffer \
  --gpu 0
```

The task order can also be a text or JSON file. Text files can contain task names separated by commas, spaces, semicolons, or new lines. JSON files can be a list or an object with `task_order`, `tasks`, or `sequence`.

Observation dimensions are normalized across the selected task sequence. `FetchReach` has a native 10-D observation, while `FetchPush`, `FetchPickAndPlace`, and `FetchSlide` use 25-D observations. When a sequence mixes them, the environment pads the shorter observation with zeros so the agent, replay buffer, and quasimetric encoder all see a fixed 25-D observation.

Optional dependency:

```bash
python -m pip install --no-deps -r requirements-fetch.txt
```

Use `--no-deps` because `gymnasium-robotics==1.3.1` declares an old MuJoCo range, while this project may need a newer MuJoCo version for MetaWorld/dm-control.