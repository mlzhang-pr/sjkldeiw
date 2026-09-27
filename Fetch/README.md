# Fetch CONQUEST

The maintained Fetch implementation is `quasimetric-rl/online_continual/conquest.py`.
Its local dependencies are the Fetch environment, replay buffers, agent modules,
and the `quasimetric_rl`/`torchqmet` packages.

Run from `Fetch/quasimetric-rl`:

```bash
./online_continual/run_fetch_continual.sh \
  --env fetch_sequence_custom \
  --task_order reach,push,pick-and-place,slide \
  --gpu 0
```

Evaluate a trained run with:

```bash
python -m online_continual.conquest_eval \
  --run_dir online_continual/results/cqrl/RUN_NAME \
  --checkpoint final \
  --agent_kind meta \
  --eval_seeds 0 1 2
```