#!/bin/bash
#SBATCH --job-name=dbos_rtg
#SBATCH --output=slurm_logs/%x_%A_%a.log
#SBATCH --account=rpaleja
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G

# Train one Rescue-the-General seed per array task with
# runfiles/train_rtg_paper.py, using the paper's settings (150M steps).
# Example, six seeds of D-BOS k=1:
#
#   sbatch --array=0-5 --export=ALL,CONFIG=configs/rtg/dbos.yaml,BOS_K=1,\
#   RUN_PREFIX=rtg_kfuture_bos_k1_action,WANDB_PROJECT=my_project,SEEDS="0 1 2 3 4 5" \
#   scripts/slurm/train_rtg.sbatch
#
# Configs: configs/rtg/ppo.yaml, bbm.yaml, dbos.yaml (D-BOS, set BOS_K).
# A full run takes about two days, longer than the 24-hour training QoS:
# resubmit with RESUME_FROM set to the last periodic checkpoint to continue.

set -euo pipefail

: "${CONFIG:?set CONFIG}" "${RUN_PREFIX:?set RUN_PREFIX}" "${WANDB_PROJECT:?set WANDB_PROJECT}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4 5}"
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="${RUN_PREFIX}_s${SEED}"
SOURCE_DIR="${SOURCE_DIR:-$SLURM_SUBMIT_DIR}"
SCRATCH="${SCRATCH_ROOT:-/scratch/gilbreth/${USER}/bos_shaping/neurips/results}"
PYTHON="${PYTHON:-$HOME/.conda/envs/temp-decept/bin/python}"

export WANDB_ENTITY="${WANDB_ENTITY:-variable_decept}"
export WANDB_PROJECT
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

EXTRA=()
[ -n "${BOS_K:-}" ] && EXTRA+=(--bos-k "$BOS_K")
[ -n "${RESUME_FROM:-}" ] && EXTRA+=(--resume-from "$RESUME_FROM")

cd "$SOURCE_DIR"
echo "rtg config=$CONFIG run=$RUN_NAME project=$WANDB_ENTITY/$WANDB_PROJECT source=$SOURCE_DIR"
"$PYTHON" runfiles/train_rtg_paper.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --device cuda \
  --prediction-mode action \
  --epochs 150 \
  --parallel-envs 64 \
  --steps-per-epoch 1000000 \
  --checkpoint-name "$RUN_NAME" \
  --checkpoint-interval 10000000 \
  --eval-interval 0 \
  "${EXTRA[@]}" \
  --wandb
