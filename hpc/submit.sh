#!/bin/bash
# Submit a job with your account filled in from config.sh. Run from the hpc/ directory:
#   bash submit.sh part1_debug.sbatch
#   bash submit.sh part2_t5.sbatch lr1e-3 --learning_rate 1e-3
cd "$(dirname "$0")"
source ./config.sh
if [ "$SLURM_ACCOUNT_NAME" = "YOUR_DSGA1011_ACCOUNT" ]; then
    echo "Edit SLURM_ACCOUNT_NAME in hpc/config.sh first."; exit 1
fi
mkdir -p logs
sbatch --account="$SLURM_ACCOUNT_NAME" "$@"
