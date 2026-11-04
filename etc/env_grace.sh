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
# pip-installed wheels (protobuf, tensorflow) link against libstdc++ but carry no rpath to the
# env, so the loader falls back to the older /lib64/libstdc++.so.6 (no GLIBCXX_3.4.29) and the
# import fails. Put the env's own lib/ first so its newer libstdc++ is used.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! strings "$CONDA_PREFIX/lib/libstdc++.so.6" 2>/dev/null | grep -q GLIBCXX_3.4.29; then
    echo "WARNING: $CONDA_PREFIX/lib/libstdc++.so.6 lacks GLIBCXX_3.4.29; run: conda install -n $HRRRCAST_CONDA_ENV -c conda-forge 'libstdcxx-ng>=12'" >&2
fi
# XLA (TensorFlow's GPU compiler) needs CUDA's libdevice.10.bc. TF 2.15's pip build looks for it at
# a broken path (".../libtensorflow_framework.so.2/../../nvidia/cuda_nvcc/..."), aborts with
# "Could not open input file ... libdevice.10.bc", and kills the forecast whenever XLA compiles a
# kernel (it can even with HRRRCAST_XLA=0). Point XLA at the pip CUDA compiler package directly.
NVCC_DIR=$(ls -d "$CONDA_PREFIX"/lib/python3*/site-packages/nvidia/cuda_nvcc 2>/dev/null | head -1)
if [ -n "$NVCC_DIR" ] && [ -f "$NVCC_DIR/nvvm/libdevice/libdevice.10.bc" ]; then
    case " ${XLA_FLAGS:-} " in
        *xla_gpu_cuda_data_dir*) ;;
        *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$NVCC_DIR" ;;
    esac
else
    echo "WARNING: no libdevice.10.bc under $CONDA_PREFIX (pip install nvidia-cuda-nvcc-cu12 in $HRRRCAST_CONDA_ENV); XLA compiles will abort" >&2
fi
echo "Using python: $(command -v python)"
