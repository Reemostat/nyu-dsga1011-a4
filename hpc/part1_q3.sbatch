#!/bin/bash
#SBATCH --job-name=a4_p1_q3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

# Q3: train on augmented data, then evaluate on both test sets
#   -> part1/out_augmented_transformed.txt and part1/out_augmented_original.txt
source "$SLURM_SUBMIT_DIR/config.sh"
run_in_env a4p1 "
cd $A4_ROOT/part1
python3 main.py --train_augmented --eval_transformed
python3 main.py --eval --model_dir out_augmented
"
