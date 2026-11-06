#!/bin/bash
#SBATCH --job-name=dbos_eval
#SBATCH --output=slurm_logs/%x_%A_%a.log
#SBATCH --account=rpaleja
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# Evaluate final checkpoints over a fixed episode budget (Avalon, Coin Game).
# One array task per seed; the checkpoint is ${CKPT_PREFIX}_s${SEED}_final.pt.
# Example, 50 episodes for six PPO seeds in Avalon:
#
#   sbatch --array=0-5 --export=ALL,LABEL=ppo,\
#   CKPT_PREFIX=$SCRATCH_ROOT/checkpoints/avalon5_adapter_tuned/avalon5_ppo,\
#   WANDB_PROJECT=my_eval_project,SEEDS="42 43 44 45 46 47" scripts/slurm/eval_checkpoint.sbatch

set -euo pipefail

: "${LABEL:?set LABEL}" "${CKPT_PREFIX:?set CKPT_PREFIX}" "${WANDB_PROJECT:?set WANDB_PROJECT}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46 47}"
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}
SOURCE_DIR="${SOURCE_DIR:-$SLURM_SUBMIT_DIR}"
SCRATCH="${SCRATCH_ROOT:-/scratch/gilbreth/${USER}/bos_shaping/neurips/results}"
PYTHON="${PYTHON:-$HOME/.conda/envs/temp-decept/bin/python}"
OUT_DIR="${OUT_DIR:-$SCRATCH/evals/$WANDB_PROJECT}"

export WANDB_ENTITY="${WANDB_ENTITY:-variable_decept}"
export WANDB_PROJECT
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$OUT_DIR"

cd "$SOURCE_DIR"
"$PYTHON" runfiles/eval_avalon_checkpoint.py \
  --checkpoint "${CKPT_PREFIX}_s${SEED}_final.pt" \
  --train-seed "$SEED" \
  --eval-seed "${EVAL_SEED:-20260729}" \
  --episodes "${EPISODES:-50}" \
  --label "$LABEL" \
  --env-tag "${ENV_TAG:-avalon5}" \
  --device cuda \
  --output-dir "$OUT_DIR" \
  --wandb
