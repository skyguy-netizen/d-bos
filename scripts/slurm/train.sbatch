#!/bin/bash
#SBATCH --job-name=dbos_train
#SBATCH --output=slurm_logs/%x_%A_%a.log
#SBATCH --account=rpaleja
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# Train one seed per array task with runfiles/train.py (Avalon, Coin Game,
# belief-relay game). Example, six seeds of D-BOS k=1 in Avalon:
#
#   sbatch --array=0-5 --export=ALL,METHOD=bos,\
#   CONFIG=configs/avalon/dbos_k1_estimated.yaml,\
#   WANDB_PROJECT=my_project,SEEDS="42 43 44 45 46 47" scripts/slurm/train.sbatch
#
# METHOD is one of: bos ppo ppo_belief aa bbm lola_dice.
# Set SOURCE_DIR to a frozen snapshot (git archive) so later edits to the
# working tree cannot change queued tasks; it defaults to the submit dir.

set -euo pipefail

: "${METHOD:?set METHOD}" "${CONFIG:?set CONFIG}" "${WANDB_PROJECT:?set WANDB_PROJECT}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46 47}"
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}
SOURCE_DIR="${SOURCE_DIR:-$SLURM_SUBMIT_DIR}"
SCRATCH="${SCRATCH_ROOT:-/scratch/gilbreth/${USER}/bos_shaping/neurips/results}"
PYTHON="${PYTHON:-$HOME/.conda/envs/temp-decept/bin/python}"

export WANDB_ENTITY="${WANDB_ENTITY:-variable_decept}"
export WANDB_PROJECT
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"
export TARGET_ENV_TIMESTEPS="${TARGET_ENV_TIMESTEPS:-20000000}"
export EVAL_ENV_TIMESTEPS="${EVAL_ENV_TIMESTEPS:-200000}"
export SAVE_ENV_TIMESTEPS="${SAVE_ENV_TIMESTEPS:-5000000}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

cd "$SOURCE_DIR"
echo "method=$METHOD config=$CONFIG seed=$SEED project=$WANDB_ENTITY/$WANDB_PROJECT source=$SOURCE_DIR"
"$PYTHON" runfiles/train.py --method "$METHOD" --config "$CONFIG" --seed "$SEED" --device cuda --wandb
