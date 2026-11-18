#!/bin/bash
#SBATCH --job-name=a4_p1_q1q2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

# Q1: full fine-tune + eval        -> part1/out_original.txt   (need > 91%)
# Q2: eval on transformed test set -> part1/out_transformed.txt (want > 4 point drop)
source "$SLURM_SUBMIT_DIR/config.sh"
run_in_env a4p1 "
cd $A4_ROOT/part1
python3 main.py --train --eval
python3 main.py --eval_transformed
"
