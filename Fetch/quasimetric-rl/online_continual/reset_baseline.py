"""Independent reset baseline for online continual Fetch tasks.

A fresh student, optimizer, scheduler, and task-local replay buffer are used
for every task. No model or replay state is transferred across task boundaries.
"""

import sys

try:
	from . import finetuning_baseline
except ImportError:
	import finetuning_baseline


def main():
	return finetuning_baseline.main(reset_agent=True)


if __name__ == "__main__":
	sys.exit(main())


'''
cd Fetch/quasimetric-rl

python -m online_continual.reset_baseline \
	--env fetch_sequence_custom \
	--env_sequence none \
	--task_order slide,pick-and-place \
	--slide_goal_scale 0.795 \
	--gpu 1 \
	--seed 1 \
	--save_path online_continual/results/reset
'''
