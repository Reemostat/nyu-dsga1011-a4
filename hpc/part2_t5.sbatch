#!/bin/bash
#SBATCH --job-name=a4_p2_t5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --requeue

# Part 2: fine-tune T5-small for text-to-SQL.
# Usage:  sbatch part2_t5.sbatch                      (final model, experiment name "experiment")
#         sbatch part2_t5.sbatch lr1e-3 --learning_rate 1e-3     (an ablation: name + extra args)
# Outputs: part2/results/t5_ft_<name>_{dev,test}.sql, part2/records/t5_ft_<name>_{dev,test}.pkl
#          and the checkpoint in part2/checkpoints/ft_experiments/<name>/best
NAME=${1:-experiment}
shift || true
EXTRA_ARGS="$@"

source "$SLURM_SUBMIT_DIR/config.sh"
run_in_env a4p2 "
cd $A4_ROOT/part2
python3 train_t5.py --finetune --experiment_name $NAME $EXTRA_ARGS
python3 evaluate.py --predicted_sql results/t5_ft_${NAME}_dev.sql --predicted_records records/t5_ft_${NAME}_dev.pkl \
    --development_sql data/dev.sql --development_records records/ground_truth_dev.pkl
"
