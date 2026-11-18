# Shared settings for all HPC scripts. EDIT THE FIRST LINE before submitting jobs.
# Find your account with:  sacctmgr show assoc user=$USER format=account%50
export SLURM_ACCOUNT_NAME="YOUR_DSGA1011_ACCOUNT"

export A4_ROOT="/scratch/$USER/nyu-dsga1011-a4"          # where you git clone the repo
export SIF="/scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"
export OVERLAY="/scratch/$USER/a4-overlay-25GB-500K.ext3"

# Keep all caches on /scratch (home quota is small)
export HF_HOME="/scratch/$USER/hf_cache"
export NLTK_DATA="/scratch/$USER/nltk_data"
export TMPDIR="/scratch/$USER/tmp"
mkdir -p "$HF_HOME" "$NLTK_DATA" "$TMPDIR"

# Run a command inside the container with the given conda env.
#   run_in_env <env-name> "<shell command>"
# The overlay is mounted read-only so several jobs can run at the same time.
run_in_env() {
    singularity exec --bind /scratch --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate $1
        export HF_HOME=$HF_HOME NLTK_DATA=$NLTK_DATA TMPDIR=$TMPDIR TOKENIZERS_PARALLELISM=false
        $2"
}
