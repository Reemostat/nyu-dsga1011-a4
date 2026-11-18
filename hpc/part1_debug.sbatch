#!/bin/bash
#SBATCH --job-name=a4_p1_debug
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

# Q1 sanity check: small subset training, expect > 88% accuracy.
# Also prints 5 transformed examples so you can check the Q2 transformation.
source "$SLURM_SUBMIT_DIR/config.sh"
run_in_env a4p1 "
cd $A4_ROOT/part1
nvidia-smi
python3 main.py --train --eval --debug_train
python3 main.py --eval_transformed --debug_transformation
"
