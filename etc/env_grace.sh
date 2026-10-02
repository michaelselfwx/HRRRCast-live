#!/bin/bash
# Conda environment for TAMU HPRC Grace (sourced by every jobs/grace/*.sh).
#
# Grace's conda is the Miniconda3 module (/sw/eb/sw/Miniconda3/...). A batch job inherits the
# PATH of the shell you submitted from, so if `hrrrcast` was active there, the env's python
# comes first on PATH and the module's `conda` script breaks ("No module named 'conda'").
# Loading the module puts its own python back in front, then we activate properly so the
# env's activate.d hooks (e.g. ESMFMKFILE for xesmf) run.
#
# Overrides: HRRRCAST_CONDA_MODULE (default Miniconda3/24.11.1), HRRRCAST_CONDA_ENV (default hrrrcast)
module load ${HRRRCAST_CONDA_MODULE:-Miniconda3/24.11.1} >/dev/null 2>&1 \
    || echo "WARNING: could not load ${HRRRCAST_CONDA_MODULE:-Miniconda3/24.11.1}" >&2
CONDA_ROOT=${EBROOTMINICONDA3:-$(dirname "$(dirname "${CONDA_EXE:-/nonexistent/bin/conda}")")}
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found under $CONDA_ROOT (set HRRRCAST_CONDA_MODULE)" >&2
    exit 1
fi
conda activate ${HRRRCAST_CONDA_ENV:-hrrrcast} || { echo "ERROR: conda activate ${HRRRCAST_CONDA_ENV:-hrrrcast} failed" >&2; exit 1; }
echo "Using python: $(command -v python)"
