#!/bin/bash
# MRMS observations for HRRRCast cases that have already run -- separate from the forecast.
# Fetches REFC / 1-h QPE (+ run-total, 6-h, 12-h sums) / rotation track / MESH at the top of every
# forecast hour (src/get_mrms.py) and plots them as member "mrms" (src/plot.py).
#
# Usage:
#   ./mrms.sh INIT [INIT ...]                 INIT = YYYY-MM-DDTHH
#   ./mrms.sh 2026-10-21T12 2026-11-02T17 2026-11-04T03
#
# Runs on Grace (one Slurm job per case, when sbatch is available) or right here on the desktop
# (MODE=local). Lead hours and domain come from each case's HRRRCast output (NetCDF or the kept
# GRIB2), so cleaned-up cases work too. Products already fetched are kept; rerun any time to fill in
# hours that were still in the future, or to add products.
#
# Environment knobs:
#   DATAROOT       data root (default $SCRATCH/hrrrcast-data if $SCRATCH is set, else ~/hrrrcast-data)
#   MODE           auto (default: grace if sbatch exists, else local) / grace / local
#   MRMS_EXTRA     extra get_mrms.py args, e.g. "--grid hrrrcast" or "--products REFC APCP" or "--overwrite"
#   PLOT           YES (default) / NO
#   PLOT_PRODUCTS  maps to draw (default: all MRMS fields), e.g. "REFC APCP*"
#   HRRRCAST_PLOT_DOMAINS  default "tx hcfcd_hrrr"
#   VIEWER         local mode: YES (default) rebuild viewer.html in DATAROOT (make_viewer_index.py --local)
#   ACCNR          Grace account;  MRMS_WALLTIME (default 01:00:00);  MRMS_CPUS (default 48)
#   DRYRUN=1       Grace: write the job scripts but don't submit them
#   HRRRCAST_CONDA_ENV  local mode: conda env (default hrrrcast)

set -u
PACKAGEROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -n "${SCRATCH:-}" ]; then DATAROOT=${DATAROOT:-$SCRATCH/hrrrcast-data}; else DATAROOT=${DATAROOT:-$HOME/hrrrcast-data}; fi
MODE=${MODE:-auto}
MRMS_EXTRA=${MRMS_EXTRA:-}
PLOT=${PLOT:-YES}
PLOT_PRODUCTS=${PLOT_PRODUCTS:-}
PLOT_DOMAINS=${HRRRCAST_PLOT_DOMAINS:-tx hcfcd_hrrr}
VIEWER=${VIEWER:-YES}
MRMS_WALLTIME=${MRMS_WALLTIME:-01:00:00}
MRMS_CPUS=${MRMS_CPUS:-48}
HRRRCAST_CONDA_ENV=${HRRRCAST_CONDA_ENV:-hrrrcast}

if [ $# -eq 0 ]; then
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
if [ "$MODE" == "auto" ]; then
    if command -v sbatch >/dev/null 2>&1; then MODE=grace; else MODE=local; fi
fi
case "$MODE" in grace|local) ;; *) echo "MODE must be auto, grace or local" >&2; exit 1;; esac
mkdir -p "$DATAROOT/logs"
DATAROOT=$(cd "$DATAROOT" && pwd)

# last forecast hour of a case, from its member NetCDF or the GRIB2 the cleanup job keeps
case_lead() {
    local dir=$1 hh=$2
    { ls "$dir"/hrrrcast_m[0-9][0-9]_f[0-9]*.nc 2>/dev/null | grep -E '_f[0-9]+\.nc$' | sed -E 's/.*_f0*([0-9]+)\.nc$/\1/'
      ls "$dir"/hrrrcast.m[0-9][0-9].t${hh}z.pgrb2.f[0-9]* 2>/dev/null | grep -E 'pgrb2\.f[0-9]+$' | sed -E 's/.*pgrb2\.f0*([0-9]+)$/\1/'
    } | sort -n | tail -1
}

if [ "$MODE" == "local" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$HRRRCAST_CONDA_ENV" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    for d in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
        [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; break; }
    done
    conda activate "$HRRRCAST_CONDA_ENV" || { echo "ERROR: cannot activate conda env $HRRRCAST_CONDA_ENV" >&2; exit 1; }
fi

status=0
for INIT in "$@"; do
    if ! [[ "$INIT" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2})$ ]]; then
        echo "Skipping '$INIT': expected YYYY-MM-DDTHH" >&2; status=1; continue
    fi
    YMD="${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}"; HH="${BASH_REMATCH[4]}"
    CASE_DIR="$DATAROOT/$YMD/$HH"
    LEAD=$(case_lead "$CASE_DIR" "$HH")
    if [ -z "$LEAD" ]; then
        echo "Skipping $INIT: no HRRRCast output (hrrrcast_mNN_fHH.nc / hrrrcast.mNN.t${HH}z.pgrb2.fHH) in $CASE_DIR" >&2
        status=1; continue
    fi

    FETCH="python $PACKAGEROOT/src/get_mrms.py $INIT $LEAD --base_dir $DATAROOT --workers 8 $MRMS_EXTRA"
    PLOTCMD="python $PACKAGEROOT/src/plot.py $INIT $LEAD --members mrms --domains $PLOT_DOMAINS --forecast_dir $DATAROOT --output_dir $DATAROOT"

    if [ "$MODE" == "local" ]; then
        echo "[$INIT] MRMS f00-f$LEAD -> $CASE_DIR"
        export HRRRCAST_PLOT_PRODUCTS="$PLOT_PRODUCTS"
        $FETCH || { echo "[$INIT] get_mrms had errors" >&2; status=1; }
        if [ "$PLOT" == "YES" ]; then
            $PLOTCMD || { echo "[$INIT] plot failed" >&2; status=1; }
        fi
        continue
    fi

    JOB="$DATAROOT/logs/job-mrms-${YMD}${HH}.sh"
    cat > "$JOB" <<JOBEOF
#!/bin/bash
#SBATCH --job-name=mrms_${YMD}${HH}
#SBATCH --output=${DATAROOT}/logs/mrms_${YMD}${HH}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${MRMS_CPUS}
#SBATCH --mem=180G
#SBATCH --time=${MRMS_WALLTIME}

source ${PACKAGEROOT}/etc/env_grace.sh
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; MRMS / map downloads may fail"
cd ${DATAROOT}
export HRRRCAST_PLOT_PRODUCTS="${PLOT_PRODUCTS}"
echo "MRMS ${INIT}: f00-f${LEAD}, plot=${PLOT}, products: \${HRRRCAST_PLOT_PRODUCTS:-all MRMS fields}"
${FETCH} || echo "WARNING: get_mrms had errors (see above); plotting what was fetched"
if [ "${PLOT}" == "YES" ]; then
    ${PLOTCMD}
fi
JOBEOF
    if [ -n "${DRYRUN:-}" ]; then
        echo "$INIT: wrote $JOB (not submitted), f00-f$LEAD"
    else
        SB_ACCOUNT=""; [ -n "${ACCNR:-}" ] && SB_ACCOUNT="--account=${ACCNR}"
        JID=$(sbatch --parsable $SB_ACCOUNT "$JOB") || { echo "sbatch failed for $INIT" >&2; status=1; continue; }
        echo "$INIT: submitted MRMS job $JID (f00-f$LEAD) -> $DATAROOT/logs/mrms_${YMD}${HH}_${JID}.out"
    fi
done

if [ "$MODE" == "local" ] && [ "$PLOT" == "YES" ] && [ "$VIEWER" == "YES" ]; then
    python "$PACKAGEROOT/src/make_viewer_index.py" --base_dir "$DATAROOT" --local || status=1
elif [ "$MODE" == "grace" ]; then
    echo "When the jobs finish: python src/make_viewer_index.py --base_dir $DATAROOT  (then publish the viewer)"
fi
exit $status
