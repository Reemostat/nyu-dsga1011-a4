#!/bin/bash
# ONE-TIME setup: installs miniconda + two conda envs (a4p1 for Part 1, a4p2 for Part 2)
# into a Singularity overlay. Run this from a terminal on the HPC (inside an interactive
# job on the burst side), from the repo's hpc/ directory:
#     bash setup_env.sh
set -e
source "$(dirname "$0")/config.sh"

cd /scratch/$USER
find_first() { for p in "$@"; do [ -f "$p" ] && { echo "$p"; return; }; done; }
if [ ! -f "$SIF" ]; then
    SRC=$(find_first /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
                     $(ls /share/apps/images/cuda12*ubuntu22*.sif 2>/dev/null))
    [ -z "$SRC" ] && { echo "No CUDA 12 .sif found in /share/apps/images - run: ls /share/apps/images"; exit 1; }
    echo "Using image $SRC"; cp "$SRC" "$SIF"
fi
if [ ! -f "$OVERLAY" ]; then
    SRC=$(find_first /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz \
                     /share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz)
    [ -z "$SRC" ] && { echo "Overlay template not found - run: ls /share/apps/overlay-fs-ext3 /scratch/work/public/overlay-fs-ext3"; exit 1; }
    echo "Using overlay $SRC"; cp "$SRC" "$OVERLAY.gz"; gunzip "$OVERLAY.gz"
fi

# Installation needs the overlay mounted read-write
singularity exec --bind /scratch --nv --overlay "$OVERLAY:rw" "$SIF" /bin/bash -c "
set -e
export TMPDIR=$TMPDIR
if [ ! -d /ext3/miniconda3 ]; then
    cd /ext3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
fi
source /ext3/miniconda3/etc/profile.d/conda.sh
conda config --set always_yes yes
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- Part 1 env ----
conda env list | grep -q '^a4p1 ' || conda create -n a4p1 python=3.11
conda activate a4p1
pip install -r $A4_ROOT/part1/requirements.txt
python -c \"import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA'); nltk.download('punkt', download_dir='$NLTK_DATA')\"
conda deactivate

# ---- Part 2 env ----
conda env list | grep -q '^a4p2 ' || conda create -n a4p2 python=3.11
conda activate a4p2
pip install -r $A4_ROOT/part2/requirements.txt
python -c \"import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')\"
conda deactivate
"

echo "Checking GPU visibility from both envs..."
run_in_env a4p1 "python -c 'import torch; print(\"a4p1 torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'"
run_in_env a4p2 "python -c 'import torch; print(\"a4p2 torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'"
echo "Setup done."
