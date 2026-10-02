#!/bin/bash
# Conda environment for TAMU HPRC Grace (sourced by every jobs/grace/*.sh).
#
# A batch job inherits the environment of the shell you submitted from. If `hrrrcast` was
# active there, the job starts with CONDA_SHLVL/CONDA_DEFAULT_ENV saying "hrrrcast is active",
# so `conda activate hrrrcast` silently does nothing -- while loading the Miniconda3 module has
# already put the base python first on PATH. So: forget the inherited conda state, then activate.
#
# Overrides: HRRRCAST_CONDA_MODULE (default Miniconda3/24.11.1), HRRRCAST_CONDA_ENV (default hrrrcast)
HRRRCAST_CONDA_ENV=${HRRRCAST_CONDA_ENV:-hrrrcast}

# 1. drop conda state and env bin dirs inherited from the submitting shell
for v in $(env | grep -oE '^CONDA_(SHLVL|PREFIX(_[0-9]+)?|DEFAULT_ENV|PROMPT_MODIFIER)='); do unset "${v%=}"; done
PATH=$(echo "$PATH" | tr ':' '\n' | grep -vE '/envs/[^/]+/bin$' | paste -sd: -)
export PATH

# 2. conda itself (HPRC module), with its own python in front so the conda CLI works
module load ${HRRRCAST_CONDA_MODULE:-Miniconda3/24.11.1} >/dev/null 2>&1 \
    || echo "WARNING: could not load ${HRRRCAST_CONDA_MODULE:-Miniconda3/24.11.1}" >&2
CONDA_ROOT=${EBROOTMINICONDA3:-$(dirname "$(dirname "${CONDA_EXE:-/nonexistent/bin/conda}")")}
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda.sh not found under $CONDA_ROOT (set HRRRCAST_CONDA_MODULE)" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# 3. activate and check it took
conda activate "$HRRRCAST_CONDA_ENV" || { echo "ERROR: conda activate $HRRRCAST_CONDA_ENV failed" >&2; exit 1; }
case "$(command -v python)" in
    */envs/"$HRRRCAST_CONDA_ENV"/bin/python) ;;
    *) echo "ERROR: python is $(command -v python), not the $HRRRCAST_CONDA_ENV env" >&2; exit 1 ;;
esac
echo "Using python: $(command -v python)"
