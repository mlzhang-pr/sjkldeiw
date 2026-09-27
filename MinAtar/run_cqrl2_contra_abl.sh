#!/usr/bin/env bash

set -euo pipefail

SEQS=(0 1 2)
SEEDS=(0 1 2)
GPU=0
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
	cat <<'EOF'
Usage: ./run_cqrl2.sh [options]

Options:
  --seqs "0 1 2"   Sequence indices to run (default: 0 1 2)
  --seeds "0 1 2"  Random seeds to run (default: 0 1 2)
  --gpu ID          CUDA device index (default: 0)
  --dry-run         Print commands without running experiments
  -h, --help        Show this help message

Environment:
  PYTHON_BIN         Python executable to use (default: python)
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
		echo "[$(date '+%F %T')] Starting CQRL2: seq=$seq seed=$seed gpu=$GPU"
		run_experiment "$PYTHON_BIN" cqrl2.py \
			--seq "$seq" \
			--seed "$seed" \
			--gpu "$GPU" \
			--save \
			--save-model \
			--output-dir  ./results/ablation \
			--wandb-project minatar-cqrl2-contra-abl \
			--wandb-name "cqrl2-seq${seq}-seed${seed}-contra-abl" \
			--qm-nce-mode backward_nce \
			--qm-transition-consistency-coef 0.0 \
			--qm-contrastive-coef 0.0 
	done
done

echo "[$(date '+%F %T')] All CQRL2 experiments completed on $(hostname)"