#!/bin/bash
# ONE-TIME setup: installs miniconda + two conda envs (a4p1 for Part 1, a4p2 for Part 2).
# Run from the repo's hpc/ directory:   bash setup_env.sh
#
# Tries, in order:
#   1. Singularity overlay mounted read-write
#   2. Singularity overlay with --fakeroot
#   3. Plain miniconda in /scratch/$USER/a4_miniconda3 (no container)
# The chosen mode is written to $ENV_MODE_FILE and used by run_in_env in config.sh.
set -e
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/config.sh"
cd /scratch/$USER

find_first() { for p in "$@"; do [ -f "$p" ] && { echo "$p"; return; }; done; }
if [ ! -f "$SIF" ]; then
    SRC=$(find_first /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
                     $(ls /share/apps/images/cuda12*ubuntu22*.sif 2>/dev/null))
    [ -z "$SRC" ] && { echo "No CUDA 12 .sif found in /share/apps/images"; exit 1; }
    echo "Using image $SRC"; cp "$SRC" "$SIF"
fi
if [ ! -f "$OVERLAY" ]; then
    SRC=$(find_first /share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz \
                     /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz \
                     /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz)
    [ -z "$SRC" ] && { echo "Overlay template not found"; exit 1; }
    echo "Using overlay $SRC"; cp "$SRC" "$OVERLAY.gz"; gunzip "$OVERLAY.gz"
fi

MODE=""
RW_FLAGS=""
# FORCE_NATIVE=1 bash setup_env.sh   -> skip the container entirely
[ "${FORCE_NATIVE:-0}" = "1" ] && SKIP_OVERLAY=1
for flags in "" "--fakeroot"; do
    [ -n "$SKIP_OVERLAY" ] && break
    echo ">>> Testing writable overlay ${flags:-(default)}"
    if singularity exec $flags --overlay "$OVERLAY:rw" "$SIF" touch /ext3/.write_test 2>/dev/null; then
        MODE="overlay"; RW_FLAGS="$flags"; break
    fi
done

if [ "$MODE" = "overlay" ]; then
    echo ">>> Installing into overlay (flags: ${RW_FLAGS:-none})"
    singularity exec $RW_FLAGS --overlay "$OVERLAY:rw" "$SIF" \
        env TMPDIR="$TMPDIR" bash "$HPC_DIR/install_envs.sh" /ext3/miniconda3 "$A4_ROOT" "$NLTK_DATA"
    # Jobs mount the overlay read-only without fakeroot; make sure that works
    if ! singularity exec --overlay "$OVERLAY:ro" "$SIF" test -x /ext3/miniconda3/bin/conda 2>/dev/null; then
        echo ">>> Overlay not readable in job mode; switching to native install"
        MODE=""
    fi
fi

if [ "$MODE" != "overlay" ]; then
    MODE="native"
    echo ">>> Overlay not usable on this cluster; installing miniconda directly in $NATIVE_CONDA"
    TMPDIR="$TMPDIR" bash "$HPC_DIR/install_envs.sh" "$NATIVE_CONDA" "$A4_ROOT" "$NLTK_DATA"
fi

echo "$MODE" > "$ENV_MODE_FILE"
echo ">>> Environment mode: $MODE"
echo "Checking both envs..."
run_in_env a4p1 "python -c 'import torch, transformers, datasets; print(\"a4p1 torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'"
run_in_env a4p2 "python -c 'import torch, transformers; print(\"a4p2 torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'"
echo "Setup done."
