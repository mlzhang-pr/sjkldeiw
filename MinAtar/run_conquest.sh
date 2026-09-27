#!/usr/bin/env bash

set -euo pipefail

SEQS=(0 1 2)
SEEDS=(0 1 2 3)
GPU=0
EXTRACTION_UPDATES=2000
EXTRACTION_BATCH_SIZE=256
EXTRACTION_GOALS=32
EXTRACTION_BETA=2.0
EXTRACTION_MAX_WEIGHT=20.0
EXTRACTION_LR=3e-4
EXTRACTION_BC_COEF=0.2
EXTRACTION_BC_WARMUP_FRACTION=0.25
EVALUATION_EPISODES=30
EVALUATION_MAX_STEPS=300
EVALUATION_SEED=1000
WANDB_MODE=online
OUTPUT_DIR=./results/conquest
DRY_RUN=0
EXTRA_ARGS=()

if [[ -z "${PYTHON_BIN:-}" ]]; then
	RLL3_PYTHON="${HOME}/.conda/envs/RLL3/bin/python"
	if [[ -x "$RLL3_PYTHON" ]]; then
		PYTHON_BIN="$RLL3_PYTHON"
	else
		PYTHON_BIN=python
	fi
fi

usage() {
	cat <<'EOF'
Usage: ./run_conquest.sh [options] [-- conquest.py options]

Options:
  --seqs "0 1 2"          Sequence indices to run (default: 0 1 2)
	--seeds "0 1 2 3"       Random seeds to run (default: 0 1 2 3)
  --gpu ID                 CUDA device index (default: 0)
	--extraction-updates N   Shared-policy updates per task (default: 2000)
  --extraction-batch-size N
                           Policy extraction batch size (default: 256)
	--extraction-goals K     Goals sampled per transition (default: 32)
	--extraction-beta BETA   Advantage temperature (default: 2.0)
  --extraction-max-weight W
                           Maximum AWR weight (default: 20.0)
  --extraction-lr LR       Shared-policy learning rate (default: 3e-4)
  --extraction-bc-coef C   BC coefficient after warmup (default: 0.2)
  --extraction-bc-warmup-fraction F
                           Fraction of initial pure-BC updates (default: 0.25)
  --evaluation-episodes N  Episodes per shared-policy/task evaluation (default: 30)
  --evaluation-max-steps N Maximum steps per evaluation episode (default: 300)
  --evaluation-seed SEED   Base evaluation seed (default: 1000)
	--output-dir DIR         Result directory (default: ./results/conquest)
  --wandb-mode MODE        online, offline, or disabled (default: online)
  --dry-run                Print commands without running experiments
  -h, --help               Show this help message

Environment:
  PYTHON_BIN                Python executable (default: RLL3, then python)

Arguments after -- are forwarded to conquest.py.
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
		--extraction-updates)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-updates" >&2; exit 2; }
			EXTRACTION_UPDATES="$2"
			shift 2
			;;
		--extraction-batch-size)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-batch-size" >&2; exit 2; }
			EXTRACTION_BATCH_SIZE="$2"
			shift 2
			;;
		--extraction-goals)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-goals" >&2; exit 2; }
			EXTRACTION_GOALS="$2"
			shift 2
			;;
		--extraction-beta)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-beta" >&2; exit 2; }
			EXTRACTION_BETA="$2"
			shift 2
			;;
		--extraction-max-weight)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-max-weight" >&2; exit 2; }
			EXTRACTION_MAX_WEIGHT="$2"
			shift 2
			;;
		--extraction-lr)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-lr" >&2; exit 2; }
			EXTRACTION_LR="$2"
			shift 2
			;;
		--extraction-bc-coef)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-bc-coef" >&2; exit 2; }
			EXTRACTION_BC_COEF="$2"
			shift 2
			;;
		--extraction-bc-warmup-fraction)
			[[ $# -ge 2 ]] || { echo "Missing value for --extraction-bc-warmup-fraction" >&2; exit 2; }
			EXTRACTION_BC_WARMUP_FRACTION="$2"
			shift 2
			;;
		--evaluation-episodes)
			[[ $# -ge 2 ]] || { echo "Missing value for --evaluation-episodes" >&2; exit 2; }
			EVALUATION_EPISODES="$2"
			shift 2
			;;
		--evaluation-max-steps)
			[[ $# -ge 2 ]] || { echo "Missing value for --evaluation-max-steps" >&2; exit 2; }
			EVALUATION_MAX_STEPS="$2"
			shift 2
			;;
		--evaluation-seed)
			[[ $# -ge 2 ]] || { echo "Missing value for --evaluation-seed" >&2; exit 2; }
			EVALUATION_SEED="$2"
			shift 2
			;;
		--output-dir)
			[[ $# -ge 2 ]] || { echo "Missing value for --output-dir" >&2; exit 2; }
			OUTPUT_DIR="$2"
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
		--)
			shift
			EXTRA_ARGS=("$@")
			break
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
		echo "[$(date '+%F %T')] Starting Conquest: seq=$seq seed=$seed gpu=$GPU beta=$EXTRACTION_BETA"
		run_experiment "$PYTHON_BIN" conquest.py \
			--seq "$seq" \
			--seed "$seed" \
			--gpu "$GPU" \
			--save \
			--save-model \
			--output-dir "$OUTPUT_DIR" \
			--wandb-project minatar-conquest \
			--wandb-name "conquest-seq${seq}-seed${seed}" \
			--wandb-mode "$WANDB_MODE" \
			--qm-nce-mode backward_nce \
			--qm-transition-consistency-coef 0.0 \
			--policy-extraction-updates "$EXTRACTION_UPDATES" \
			--policy-extraction-batch-size "$EXTRACTION_BATCH_SIZE" \
			--policy-extraction-goals "$EXTRACTION_GOALS" \
			--policy-extraction-beta "$EXTRACTION_BETA" \
			--policy-extraction-max-weight "$EXTRACTION_MAX_WEIGHT" \
			--policy-extraction-lr "$EXTRACTION_LR" \
			--policy-extraction-bc-coef "$EXTRACTION_BC_COEF" \
			--policy-extraction-bc-warmup-fraction "$EXTRACTION_BC_WARMUP_FRACTION" \
			--evaluation-episodes "$EVALUATION_EPISODES" \
			--evaluation-max-steps "$EVALUATION_MAX_STEPS" \
			--evaluation-seed "$EVALUATION_SEED" \
			${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
	done
done

echo "[$(date '+%F %T')] All Conquest experiments completed on $(hostname)"