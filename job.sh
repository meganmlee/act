#!/bin/bash
#SBATCH --job-name=act_libero
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=48:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/act_libero_%j.out
#SBATCH --error=logs/act_libero_%j.err

module load anaconda3
conda activate aloha

cd /ocean/projects/cis260038p/mlee12/act

# Start with single task + larger batch to validate quickly
python3 imitate_episodes.py \
    --task_name libero_90 \
    --ckpt_dir ./checkpoints/libero_90_act \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 800 \
    --lr 1e-5 \
    --seed 0