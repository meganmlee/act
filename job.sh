#!/bin/bash
#SBATCH --job-name=act_libero
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH -A cis260038p
#SBATCH --array=0-9
#SBATCH --output=logs/act_libero_%A_%a.out
#SBATCH --error=logs/act_libero_%A_%a.err

cd /ocean/projects/cis260038p/mlee12/ACT-Tokenizer

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/ACT-Tokenizer/detr:/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

# Shared args (updated to match ACT paper hyperparams)
POLICY_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg"

###############################################################################
# Per-task training with SLURM job array (10 tasks train in parallel)
# Usage: sbatch job.sh libero_spatial
# Suites: libero_spatial, libero_object, libero_goal, libero_10
###############################################################################

SUITE=${1:-libero_spatial}
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================"
echo "Training ${SUITE} task ${TASK_ID} (array job ${SLURM_ARRAY_JOB_ID})"
echo "========================================"

python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --task_id ${TASK_ID} \
    --ckpt_dir ./checkpoints/${SUITE}_act/task_${TASK_ID} \
    $POLICY_ARGS \
    --batch_size 8 \
    --num_epochs 2000 \
    --lr 1e-5
