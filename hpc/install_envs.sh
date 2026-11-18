#!/bin/bash
# Installs miniconda into $1 and creates the two conda envs. Called by setup_env.sh
# (either inside the Singularity container or directly on the node).
#   bash install_envs.sh <conda-dir> <repo-root> <nltk-dir>
set -e
CONDA_DIR="$1"; A4_ROOT="$2"; NLTK_DATA="$3"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    rm -rf "$CONDA_DIR"
    cd "$(dirname "$CONDA_DIR")"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_installer.sh
    bash miniconda_installer.sh -b -p "$CONDA_DIR"
    rm miniconda_installer.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set always_yes yes
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- Part 1 env ----
conda env list | grep -q '^a4p1 ' || conda create -n a4p1 python=3.11
conda activate a4p1
pip install --no-cache-dir -r "$A4_ROOT/part1/requirements.txt"
python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA'); nltk.download('punkt', download_dir='$NLTK_DATA')"
conda deactivate

# ---- Part 2 env ----
conda env list | grep -q '^a4p2 ' || conda create -n a4p2 python=3.11
conda activate a4p2
pip install --no-cache-dir -r "$A4_ROOT/part2/requirements.txt"
python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')"
conda deactivate
conda clean -a -y
