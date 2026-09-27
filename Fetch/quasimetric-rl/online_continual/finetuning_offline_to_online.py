"""Resume finetuning from a previous model and its replay buffer."""

import sys

try:
	from . import finetuning_baseline
except ImportError:
	import finetuning_baseline


def _has_option(argv, option):
	return any(
		argument == option or argument.startswith(f"{option}=")
		for argument in argv
	)


def main():
	if not _has_option(sys.argv[1:], "--start_task"):
		sys.argv.extend(["--start_task", "2"])
	return finetuning_baseline.main(carry_replay_buffer=True)


if __name__ == "__main__":
	sys.exit(main())
'''
cd /hpc2hdd/home/mzhang943/projects/continual-rl-progress/Fetch/quasimetric-rl

python -m online_continual.finetuning_offline_to_online \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --gpu 0

'''