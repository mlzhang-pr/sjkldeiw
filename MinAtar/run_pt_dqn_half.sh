#!/usr/bin/env bash

set -euo pipefail

SEQS=(0 1 2)
SEEDS=(0 1 2)
GPU=0
RESET=1
BOUNDARY=0
WANDB_PROJECT="minatar-pt-dqn-half"
WANDB_MODE="online"
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
	cat <<'EOF'
Usage: ./run_pt_dqn_half.sh [options]

Options:
  --seqs "0 1 2"        Sequence indices to run (default: 0 1 2)
  --seeds "0 1 2"       Random seeds to run (default: 0 1 2)
  --gpu ID               CUDA device index (default: 0)
  --reset 0|1            Reset the transient network at task boundaries (default: 1)
  --boundary 0|1         Use unknown or known task boundaries (default: 0)
  --wandb-project NAME   W&B project (default: minatar-pt-dqn-half)
  --wandb-mode MODE      online, offline, or disabled (default: online)
  --dry-run              Print commands without running experiments
  -h, --help             Show this help message

Environment:
  PYTHON_BIN             Python executable to use (default: python)
EOF
}

while (($# > 0)); do
	case "$1" in
		--seqs)
			[[ $# -ge 2 ]] || { echo "Missing value for --seqs" >&2; exit 2; }
			read -r -a SEQS <<< "$2"
			shift 2
			;;
		--seeds)
			[[ $# -ge 2 ]] || { echo "Missing value for --seeds" >&2; exit 2; }
			read -r -a SEEDS <<< "$2"
			shift 2
			;;
		--gpu)
			[[ $# -ge 2 ]] || { echo "Missing value for --gpu" >&2; exit 2; }
			GPU="$2"
			shift 2
			;;
		--reset)
			[[ $# -ge 2 ]] || { echo "Missing value for --reset" >&2; exit 2; }
			RESET="$2"
			shift 2
			;;
		--boundary)
			[[ $# -ge 2 ]] || { echo "Missing value for --boundary" >&2; exit 2; }
			BOUNDARY="$2"
			shift 2
			;;
		--wandb-project)
			[[ $# -ge 2 ]] || { echo "Missing value for --wandb-project" >&2; exit 2; }
			WANDB_PROJECT="$2"
			shift 2
			;;
		--wandb-mode)
			[[ $# -ge 2 ]] || { echo "Missing value for --wandb-mode" >&2; exit 2; }
			WANDB_MODE="$2"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

if ((${#SEQS[@]} == 0 || ${#SEEDS[@]} == 0)); then
	echo "--seqs and --seeds must not be empty" >&2
	exit 2
fi

if [[ "$RESET" != "0" && "$RESET" != "1" ]]; then
	echo "--reset must be 0 or 1" >&2
	exit 2
fi

if [[ "$BOUNDARY" != "0" && "$BOUNDARY" != "1" ]]; then
	echo "--boundary must be 0 or 1" >&2
	exit 2
fi

if [[ "$WANDB_MODE" != "online" && "$WANDB_MODE" != "offline" && "$WANDB_MODE" != "disabled" ]]; then
	echo "--wandb-mode must be online, offline, or disabled" >&2
	exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_experiment() {
	if ((DRY_RUN)); then
		printf '%q ' "$@"
		printf '\n'
	else
		"$@"
	fi
}

for seq in "${SEQS[@]}"; do
	for seed in "${SEEDS[@]}"; do
		echo "[$(date '+%F %T')] Starting PT-DQN-half: seq=$seq seed=$seed gpu=$GPU"
		run_experiment "$PYTHON_BIN" PT_DQN_half.py \
			--seq "$seq" \
			--seed "$seed" \
			--gpu "$GPU" \
			--reset "$RESET" \
			--boundary "$BOUNDARY" \
			--CNNhalf 1 \
			--lr1 1e-8 \
			--lr2 1e-4 \
			--decay 0.75 \
			--save \
			--save-model \
			--wandb-project "$WANDB_PROJECT" \
			--wandb-name "pt-dqn-half-seq${seq}-seed${seed}" \
			--wandb-mode "$WANDB_MODE"
	done
done

echo "[$(date '+%F %T')] All PT-DQN-half experiments completed on $(hostname)"