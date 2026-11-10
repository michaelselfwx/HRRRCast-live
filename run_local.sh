#!/bin/bash
# HRRRCast end-to-end on one desktop (WSL2 / Linux, one GPU) -- no Slurm.
# Same steps and order as submit_grace.sh; Slurm's job dependencies become plain waits.
#
# Usage:
#   ./run_local.sh INIT_TIME [INIT_TIME ...]          INIT_TIME = YYYY-MM-DDTHH
#   LEAD_HOUR=18 N_ENSEMBLES=9 ./run_local.sh 2026-10-21T12
#
# Several cases: list them (run one after another), or start each in the background --
#   nohup ./run_local.sh 2026-10-21T12 > /dev/null 2>&1 &
#   nohup ./run_local.sh 2026-10-22T12 > /dev/null 2>&1 &
# -- their download/preprocess/plot steps overlap, and the GPU forecast step waits its turn
# (a lock file), so two forecasts never share the GPU.
#
# Per case, in order:
#   1. get_ics.py + get_bcs.py      (in parallel; downloads)
#   2. make_ics.py + make_bcs.py    (in parallel)
#   3. fcst.py                      (GPU, all members in one process, one case at a time)
#   4. get_hrrr_fcst.py + compute_pmm.py   (in parallel)
#   5. derived_precip.py            (accumulations + LPMM)
#   6. plot.py                      (hrrr, members, avg, spr in one process pool)
#   7. make_viewer_index.py --local (viewer.html in DATAROOT)   8. optional cleanup
# Logs: $DATAROOT/logs/local_<YYYYMMDDHH>/<step>.log
#
# Environment knobs:
#   DATAROOT        data root (default ~/hrrrcast-data)
#   LEAD_HOUR       forecast length in hours (default 18)
#   N_ENSEMBLES     members (default 9)
#   BATCH_SIZE      members per GPU batch in fcst.py (default 1; raise if the GPU has room)
#   FCST_EXTRA      extra fcst.py args (default: the Texas --bbox used on Grace)
#   GET_BCS_EXTRA   extra get_bcs.py args, e.g. "--stitch_cycles"
#   RUNHRRR         YES (default) / NO   operational HRRR as member "hrrr"
#   RUNPLOT         YES (default) / NO
#   PLOT_LPMM       NO (default) / YES   also the stand-alone LPMM precip panels
#   BCS_WORKERS     make_bcs.py processes (default 4; each holds full-grid fields -> RAM-bound)
#   PLOT_PRODUCTS   which maps to draw (space-separated; "" = every product). Default:
#                "REFC APCP* PWAT CAPE CIN HGT TMP UGRD VGRD". A name matches a whole variable
#                (TMP = all 20 levels, APCP = 1-h + run-total/6-h/12-h), or one level (TMP_850hPa),
#                and may use wildcards ("*_500hPa" = every variable at 500 hPa). Others: summary,
#                T2M, D2M, WIND_10M, GUST, MSLMA, HLCY_0_3km, VIS, TCDC ... (cf_attributes.py).
#   PLOT_WORKERS    plot processes (default: half the CPU threads; each needs ~1-2 GB RAM)
#   LPMM_PATCH / LPMM_HALO   LPMM tile / halo in grid points (default 16 / 24)
#   START_AT        skip earlier steps of a case already partly run: fcst (inputs already
#                   preprocessed), post (forecast done: HRRR/pmm/precip/plots) or plot
#   CLEANUP         NO (default) / YES   afterwards keep only hrrrcast.*.pgrb2* files (as on Grace)
#   HRRRCAST_CONDA_ENV  conda env name (default hrrrcast)
#   HRRRCAST_PLOT_DOMAINS  default "tx hcfcd_hrrr"

set -u
PACKAGEROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAROOT=${DATAROOT:-$HOME/hrrrcast-data}
LEAD_HOUR=${LEAD_HOUR:-18}
N_ENSEMBLES=${N_ENSEMBLES:-18}
BATCH_SIZE=${BATCH_SIZE:-1}
FCST_EXTRA=${FCST_EXTRA:---bbox 25.8,36.5,-106.7,-93.5}   # texas domain, as on Grace
GET_BCS_EXTRA=${GET_BCS_EXTRA:-}
RUNHRRR=${RUNHRRR:-YES}
RUNPLOT=${RUNPLOT:-YES}
PLOT_LPMM=${PLOT_LPMM:-NO}
NCPU=$(nproc 2>/dev/null || echo 4)
PLOT_WORKERS=${PLOT_WORKERS:-$(( NCPU > 1 ? NCPU / 2 : 1 ))}
BCS_WORKERS=${BCS_WORKERS:-8}
LPMM_PATCH=${LPMM_PATCH:-16}
LPMM_HALO=${LPMM_HALO:-24}
CLEANUP=${CLEANUP:-NO}
START_AT=${START_AT:-}
case "$START_AT" in ""|fcst|post|plot) ;; *) echo "START_AT must be fcst, post or plot" >&2; exit 1;; esac
PLOT_DOMAINS=${HRRRCAST_PLOT_DOMAINS:-tx hcfcd_hrrr}
PLOT_PRODUCTS=${PLOT_PRODUCTS-REFC APCP* PWAT CAPE CIN HGT TMP UGRD VGRD}   # "" = every product
HRRRCAST_CONDA_ENV=${HRRRCAST_CONDA_ENV:-hrrrcast}

if [ $# -eq 0 ]; then
    sed -n '2,47p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

# ---- conda env (desktop: conda on PATH or in ~/miniconda3 / ~/miniforge3) ----
if [ "${CONDA_DEFAULT_ENV:-}" != "$HRRRCAST_CONDA_ENV" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    for d in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
        [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; break; }
    done
    conda activate "$HRRRCAST_CONDA_ENV" || { echo "ERROR: cannot activate conda env $HRRRCAST_CONDA_ENV" >&2; exit 1; }
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
export TF_CPP_MIN_LOG_LEVEL=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export HRRRCAST_XLA=${HRRRCAST_XLA:-0}
export HRRRCAST_PLOT_WORKERS=$PLOT_WORKERS
export HRRRCAST_BCS_WORKERS=$BCS_WORKERS
export HRRRCAST_PLOT_PRODUCTS=$PLOT_PRODUCTS   # read by plot.py; empty = every product
export LPMM_PATCH LPMM_HALO

mkdir -p "$DATAROOT/logs"
DATAROOT=$(cd "$DATAROOT" && pwd)   # absolute: each case cd's into it, so relative paths would break
GPU_LOCK="$DATAROOT/.gpu.lock"
SRC="$PACKAGEROOT/src"

run_case() {
    local INIT=$1
    if ! [[ "$INIT" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2})$ ]]; then
        echo "Skipping '$INIT': expected YYYY-MM-DDTHH" >&2; return 1
    fi
    local YMD="${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}" HH="${BASH_REMATCH[4]}"
    local LOGD="$DATAROOT/logs/local_${YMD}${HH}"
    mkdir -p "$LOGD"
    local T0=$SECONDS

    # step NAME CMD... : run, log to $LOGD/NAME.log, report time and status
    step() {
        local name=$1; shift
        local t=$SECONDS
        echo "[$INIT] $name ..."
        if "$@" > "$LOGD/$name.log" 2>&1; then
            echo "[$INIT] $name done ($(( SECONDS - t ))s)"
        else
            echo "[$INIT] $name FAILED ($(( SECONDS - t ))s) - see $LOGD/$name.log" >&2
            return 1
        fi
    }

    echo "[$INIT] start: ${LEAD_HOUR} h, ${N_ENSEMBLES} members, data in $DATAROOT, logs in $LOGD"
    cd "$DATAROOT" || return 1

    if [ -z "$START_AT" ]; then
    # 1. downloads
    step get_ics python "$SRC/get_ics.py" "$INIT" --base_dir "$DATAROOT" & local p1=$!
    step get_bcs python "$SRC/get_bcs.py" "$INIT" "$LEAD_HOUR" --base_dir "$DATAROOT" $GET_BCS_EXTRA & local p2=$!
    wait $p1 || { echo "[$INIT] stopping: no initial conditions" >&2; wait $p2; return 1; }

    # 2. preprocessing (make_ics as soon as the ICs are in; make_bcs once the BCs are)
    step make_ics python "$SRC/make_ics.py" "$PACKAGEROOT/net-diffusion/normalize-stats.nc" "$INIT" \
        --base_dir "$DATAROOT" --output_dir "$DATAROOT" & local p3=$!
    wait $p2 || { echo "[$INIT] stopping: no boundary conditions" >&2; wait $p3; return 1; }
    step make_bcs python "$SRC/make_bcs.py" "$PACKAGEROOT/net-diffusion/normalize-stats.nc" "$INIT" "$LEAD_HOUR" \
        --base_dir "$DATAROOT" --output_dir "$DATAROOT" \
        --hrrr_grid_file "${YMD}/${HH}/hrrr_${YMD}_${HH}_surface.grib2" & local p4=$!
    wait $p3 || { wait $p4; return 1; }
    wait $p4 || return 1
    fi

    if [ -z "$START_AT" ] || [ "$START_AT" == "fcst" ]; then
    # 3. forecast on the GPU (one case at a time across all running copies of this script)
    echo "[$INIT] waiting for the GPU ..."
    (
        flock 9
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
            | sed "s/^/[$INIT] GPU: /"
        step fcst python "$SRC/fcst.py" "$PACKAGEROOT/net-diffusion/model.keras" "$INIT" "$LEAD_HOUR" \
            --num_members "$N_ENSEMBLES" --members "0-$(( N_ENSEMBLES - 1 ))" --batch_size "$BATCH_SIZE" \
            --base_dir "$DATAROOT" --output_dir "$DATAROOT" $FCST_EXTRA
    ) 9> "$GPU_LOCK" || { echo "[$INIT] stopping: forecast failed" >&2; return 1; }
    fi

    if [ "$START_AT" != "plot" ]; then

    # 4. operational HRRR + ensemble mean/spread
    local ph="" pp=""
    if [ "$RUNHRRR" == "YES" ]; then
        step get_hrrr_fcst python "$SRC/get_hrrr_fcst.py" "$INIT" "$LEAD_HOUR" --base_dir "$DATAROOT" --workers 6 & ph=$!
    fi
    if (( N_ENSEMBLES >= 2 )); then
        step compute_pmm python "$SRC/compute_pmm.py" "$INIT" "$LEAD_HOUR" --forecast_dir "$DATAROOT" \
            --output_dir "$DATAROOT" --n_ensembles "$N_ENSEMBLES" --no_wait & pp=$!
    fi
    [ -n "$ph" ] && wait $ph
    [ -n "$pp" ] && wait $pp

    # 5. accumulations + LPMM (needs every member, and the HRRR if fetched)
    step derived_precip python "$SRC/derived_precip.py" "$INIT" "$LEAD_HOUR" --forecast_dir "$DATAROOT" \
        --lpmm_patch "$LPMM_PATCH" --lpmm_halo "$LPMM_HALO"
    fi

    # 6. plots: everything in one plot.py call (one shared worker pool)
    if [ "$RUNPLOT" == "YES" ]; then
        local members="0-$(( N_ENSEMBLES - 1 ))"
        (( N_ENSEMBLES >= 2 )) && members="$members avg spr"
        [ "$PLOT_LPMM" == "YES" ] && (( N_ENSEMBLES >= 2 )) && members="$members lpmm"
        [ "$RUNHRRR" == "YES" ] && members="hrrr $members"
        step plot python "$SRC/plot.py" "$INIT" "$LEAD_HOUR" --members $members --domains $PLOT_DOMAINS \
            --forecast_dir "$DATAROOT" --output_dir "$DATAROOT"
        # 7. viewer next to the plots (images load from disk)
        step viewer python "$SRC/make_viewer_index.py" --base_dir "$DATAROOT" --local
    fi

    # 8. cleanup like the Grace cleanup job: keep the GRIB2 output (and the plots)
    if [ "$CLEANUP" == "YES" ]; then
        find "$DATAROOT/$YMD/$HH" -maxdepth 1 -type f ! -name 'hrrrcast.*.pgrb2*' -delete
        echo "[$INIT] cleanup: kept hrrrcast.*.pgrb2* and plots"
    fi
    echo "[$INIT] finished in $(( (SECONDS - T0) / 60 )) min"
}

status=0
for INIT in "$@"; do
    run_case "$INIT" || status=1
done
exit $status
