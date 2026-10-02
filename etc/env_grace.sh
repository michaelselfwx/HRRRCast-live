#!/bin/bash
# Conda environment for TAMU HPRC Grace (sourced by every jobs/grace/*.sh).
# Same pattern as a typical HPRC job script:  eval "$(conda shell.bash hook)"; conda activate <env>
# This works because Slurm passes your login PATH (which has conda on it) into the job.
# If conda isn't on PATH in the job, set HRRRCAST_CONDA_SH to your .../etc/profile.d/conda.sh.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -n "${HRRRCAST_CONDA_SH:-}" ] && [ -f "$HRRRCAST_CONDA_SH" ]; then
    source "$HRRRCAST_CONDA_SH"
else
    echo "ERROR: conda not found on PATH; set HRRRCAST_CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
    exit 1
fi
conda activate ${HRRRCAST_CONDA_ENV:-hrrrcast}
echo "Using python: $(command -v python)"
