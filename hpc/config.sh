# Shared settings for all HPC scripts. EDIT THE FIRST LINE before submitting jobs.
# Find your account with:  sacctmgr show assoc user=$USER format=account%50
export SLURM_ACCOUNT_NAME="torch_pr_932_general"
# Leave empty on the main Torch cluster. On the burst portal use e.g. "c12m85-a100-1".
export SLURM_PARTITION_NAME=""

export A4_ROOT="/scratch/$USER/nyu-dsga1011-a4"          # where you git clone the repo
export SIF="/scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"
export OVERLAY="/scratch/$USER/a4-overlay.ext3"

# Keep all caches on /scratch (home quota is small)
export HF_HOME="/scratch/$USER/hf_cache"
export NLTK_DATA="/scratch/$USER/nltk_data"
export TMPDIR="/scratch/$USER/tmp"
mkdir -p "$HF_HOME" "$NLTK_DATA" "$TMPDIR"

# Fallback location if the Singularity overlay can't be used (see setup_env.sh)
export NATIVE_CONDA="/scratch/$USER/a4_miniconda3"
export ENV_MODE_FILE="/scratch/$USER/.a4_env_mode"

# Run a command with the given conda env activated.
#   run_in_env <env-name> "<shell command>"
# overlay mode: inside the container, overlay mounted read-only (so jobs can run in parallel)
# native mode : plain miniconda on /scratch
run_in_env() {
    local mode=$(cat "$ENV_MODE_FILE" 2>/dev/null || echo overlay)
    local body="
        export HF_HOME=$HF_HOME NLTK_DATA=$NLTK_DATA TMPDIR=$TMPDIR TOKENIZERS_PARALLELISM=false
        $2"
    if [ "$mode" = "native" ]; then
        bash -c "source $NATIVE_CONDA/etc/profile.d/conda.sh && conda activate $1 && $body"
    else
        singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
            source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate $1 && $body"
    fi
}
