#!/bin/bash
#SBATCH --job-name=a4_p1_q2
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

# Re-run ONLY the Q2 evaluation (e.g. after making the transformation stronger in
# part1/utils.py). Uses the already-trained model in part1/out. No retraining.
source "$SLURM_SUBMIT_DIR/config.sh"
run_in_env a4p1 "
cd $A4_ROOT/part1
python3 main.py --eval_transformed
"
