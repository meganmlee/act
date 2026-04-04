#!/bin/bash
#SBATCH --job-name=act_fast_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/act_fast_eval_%j.out
#SBATCH --error=logs/act_fast_eval_%j.err

cd /ocean/projects/cis260038p/mlee12/ACT-Tokenizer

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/ACT-Tokenizer/detr:/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

POLICY_ARGS="--policy_class ACT --kl_weight 0.01 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg"
FAST_ARGS="--use_fast_tokens --fast_tokenizer_path ./fast_tokenizer"

###############################################################################
# Evaluate all 10 per-task checkpoints and report suite-level average
# Usage: sbatch --dependency=afterok:<TRAIN_JOB_ID> eval_fast.sh libero_spatial
###############################################################################

SUITE=${1:-libero_spatial}

python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --ckpt_dir ./checkpoints/${SUITE}_act_fast \
    $POLICY_ARGS \
    $FAST_ARGS \
    --batch_size 32 \
    --num_epochs 800 \
    --lr 5e-4 \
    --eval
