#!/bin/bash
# Conda environment for TAMU HPRC Grace.
# Override HRRRCAST_CONDA_SH if your conda/miniforge lives somewhere else, e.g.
#   export HRRRCAST_CONDA_SH=$HOME/miniforge3/etc/profile.d/conda.sh
module purge >/dev/null 2>&1 || true
HRRRCAST_CONDA_SH=${HRRRCAST_CONDA_SH:-${SCRATCH:-/scratch/user/$USER}/miniforge3/etc/profile.d/conda.sh}
if [ ! -f "$HRRRCAST_CONDA_SH" ]; then
    echo "ERROR: conda not found at $HRRRCAST_CONDA_SH (set HRRRCAST_CONDA_SH)" >&2
    exit 1
fi
source "$HRRRCAST_CONDA_SH"
conda activate ${HRRRCAST_CONDA_ENV:-hrrrcast}
