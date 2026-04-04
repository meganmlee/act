#!/bin/bash
#SBATCH --job-name=act_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/act_eval_%j.out
#SBATCH --error=logs/act_eval_%j.err

cd /ocean/projects/cis260038p/mlee12/ACT-Tokenizer

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/ACT-Tokenizer/detr:/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

POLICY_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg"

###############################################################################
# Evaluate all 10 per-task checkpoints and report suite-level average
# Usage: sbatch --dependency=afterok:<TRAIN_JOB_ID> eval.sh libero_spatial
###############################################################################

SUITE=${1:-libero_spatial}

python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --ckpt_dir ./checkpoints/${SUITE}_act \
    $POLICY_ARGS \
    --batch_size 8 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --eval
