#!/bin/bash
#SBATCH --job-name=act_fast_libero
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH -A cis260038p
#SBATCH --array=0-9
#SBATCH --output=logs/act_fast_%A_%a.out
#SBATCH --error=logs/act_fast_%A_%a.err

cd /ocean/projects/cis260038p/mlee12/ACT-Tokenizer

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/ACT-Tokenizer/detr:/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

# Shared args
POLICY_ARGS="--policy_class ACT --kl_weight 0.01 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg"
FAST_ARGS="--use_fast_tokens --fast_tokenizer_path ./fast_tokenizer"

###############################################################################
# STEP 0: Train FAST tokenizer on LIBERO action data (run once)
#         Skip this if you already have ./fast_tokenizer/
###############################################################################

# python tokenizer.py \
#     --dataset_path /ocean/projects/cis260038p/shared/datasets/libero/libero_90 \
#     --save_path ./fast_tokenizer \
#     --chunk_size 50 \
#     --action_dim 7

###############################################################################
# Per-task training with SLURM job array (10 tasks train in parallel)
# Usage: sbatch job_actFAST.sh libero_spatial
# Suites: libero_spatial, libero_object, libero_goal, libero_10
###############################################################################

SUITE=${1:-libero_spatial}
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================"
echo "Training ${SUITE} task ${TASK_ID} (FAST tokens, array job ${SLURM_ARRAY_JOB_ID})"
echo "========================================"

python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --task_id ${TASK_ID} \
    --ckpt_dir ./checkpoints/${SUITE}_act_fast/task_${TASK_ID} \
    $POLICY_ARGS \
    $FAST_ARGS \
    --batch_size 32 \
    --num_epochs 800 \
    --lr 5e-4
