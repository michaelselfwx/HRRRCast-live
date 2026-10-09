#!/bin/bash
# Re-make the derived products and plots for cases that were already forecast (Grace / Slurm).
#
# Usage:
#   ./replot_grace.sh INIT [INIT ...]          INIT = YYYY-MM-DDTHH (the HRRRCast init time)
#   ./replot_grace.sh 2025-06-12T12 2026-10-03T16 2026-10-06T16
#
# One CPU job per case, which runs in order:
#   1. get_hrrr_fcst.py   operational HRRR of the same cycle (skips hours already downloaded)
#   2. derived_precip.py  run-total / 6 h / 12 h precip and LPMM
#   3. plot.py            hrrr, lpmm, m00..m(N-1) on the tx and hcfcd domains
#   4. make_viewer_index.py
# The lead hours and member count are read from the hrrrcast_mNN_fHH.nc files in each case.
#
# Environment knobs (same names as submit_grace.sh where they overlap):
#   DATAROOT           data root (default $SCRATCH/hrrrcast-data)
#   ACCNR              HPRC account number
#   RUNHRRR            YES (default) / NO   fetch + plot the operational HRRR
#   HRRRCAST_PLOT_DOMAINS   default "tx hcfcd"
#   PLOT_PRODUCTS      default: REFC APCP* CAPE T2M WIND_10M MSLMA HLCY_0_3km HGT_500hPa
#   REMOVE_PMM_PLOTS   NO (default) / YES   delete old avg_/spr_ plot folders so they leave the viewer
#   REPLOT_WALLTIME    default 02:00:00
#   DRYRUN=1           write the job scripts but don't submit them

PACKAGEROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAROOT=${DATAROOT:-$SCRATCH/hrrrcast-data}
RUNHRRR=${RUNHRRR:-YES}
PLOT_DOMAINS=${HRRRCAST_PLOT_DOMAINS:-tx hcfcd}
PLOT_PRODUCTS=${PLOT_PRODUCTS:-REFC APCP* CAPE T2M WIND_10M MSLMA HLCY_0_3km HGT_500hPa}
REMOVE_PMM_PLOTS=${REMOVE_PMM_PLOTS:-NO}
REPLOT_WALLTIME=${REPLOT_WALLTIME:-02:00:00}

if [ $# -eq 0 ]; then
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
# quote each product for the job script; no wildcard expansion here (APCP* is for plot.py)
set -f; PRODUCT_ARGS=""; for p in $PLOT_PRODUCTS; do PRODUCT_ARGS+="\"$p\" "; done; set +f
SB_ACCOUNT=""
[ -n "${ACCNR:-}" ] && SB_ACCOUNT="--account=${ACCNR}"
mkdir -p "$DATAROOT/logs"

for INIT in "$@"; do
    if ! [[ "$INIT" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2})$ ]]; then
        echo "Skipping '$INIT': expected YYYY-MM-DDTHH" >&2; continue
    fi
    YMD="${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}"; HH="${BASH_REMATCH[4]}"
    CASE_DIR="$DATAROOT/$YMD/$HH"

    # members = hrrrcast_mNN_f01.nc present; lead = last hour of m00 (or the first member found)
    MEMBERS=$(ls "$CASE_DIR"/hrrrcast_m[0-9][0-9]_f01.nc 2>/dev/null | sed -E 's/.*_m([0-9]+)_f01\.nc/\1/' | sort -n)
    if [ -z "$MEMBERS" ]; then
        echo "Skipping $INIT: no hrrrcast_mNN_f01.nc in $CASE_DIR (NetCDF deleted by cleanup?)" >&2; continue
    fi
    N=$(echo "$MEMBERS" | wc -l)
    FIRST=$(echo "$MEMBERS" | head -1)
    LEAD=$(ls "$CASE_DIR"/hrrrcast_m${FIRST}_f[0-9][0-9]*.nc 2>/dev/null | grep -E '_f[0-9]+\.nc$' \
           | sed -E 's/.*_f([0-9]+)\.nc/\1/' | sort -n | tail -1 | sed 's/^0*//')
    MEMLIST=$(echo "$MEMBERS" | sed 's/^0*\([0-9]\)/\1/' | paste -sd' ')
    PLOT_MEMBERS="$MEMLIST"
    (( N >= 2 )) && PLOT_MEMBERS="lpmm $PLOT_MEMBERS"
    [ "$RUNHRRR" == "YES" ] && PLOT_MEMBERS="hrrr $PLOT_MEMBERS"

    JOB="$DATAROOT/logs/job-replot-${YMD}${HH}.sh"
    cat > "$JOB" <<JOBEOF
#!/bin/bash
#SBATCH --job-name=replot_${YMD}${HH}
#SBATCH --output=${DATAROOT}/logs/replot_${YMD}${HH}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=${REPLOT_WALLTIME}

source ${PACKAGEROOT}/etc/env_grace.sh
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; downloads may fail"
cd ${DATAROOT}
echo "Replot ${INIT}: ${N} members (${MEMLIST}), f01-f${LEAD}, plotting: ${PLOT_MEMBERS}; domains: ${PLOT_DOMAINS}"

if [ "${RUNHRRR}" == "YES" ]; then
    python ${PACKAGEROOT}/src/get_hrrr_fcst.py ${INIT} ${LEAD} --base_dir ${DATAROOT} --workers 6 \\
        || echo "WARNING: get_hrrr_fcst failed; continuing without (some) HRRR hours"
fi

python ${PACKAGEROOT}/src/derived_precip.py ${INIT} ${LEAD} --forecast_dir ${DATAROOT} \\
    || echo "WARNING: derived_precip failed; accumulated precip / LPMM plots will be missing"

if [ "${REMOVE_PMM_PLOTS}" == "YES" ]; then
    find ${CASE_DIR} -maxdepth 2 -type d \\( -name 'avg_lead*' -o -name 'spr_lead*' \\) -prune -exec rm -rf {} +
    echo "Removed old avg_/spr_ plot folders"
fi

python ${PACKAGEROOT}/src/plot.py ${INIT} ${LEAD} --members ${PLOT_MEMBERS} --domains ${PLOT_DOMAINS} \\
    --products ${PRODUCT_ARGS}--forecast_dir ${DATAROOT} --output_dir ${DATAROOT}

python ${PACKAGEROOT}/src/make_viewer_index.py --base_dir ${DATAROOT}
JOBEOF

    if [ -n "${DRYRUN:-}" ]; then
        echo "$INIT: wrote $JOB (not submitted): $N members, f01-f$LEAD, plots: $PLOT_MEMBERS"
    else
        JID=$(sbatch --parsable $SB_ACCOUNT "$JOB") || { echo "sbatch failed for $INIT" >&2; continue; }
        echo "$INIT: submitted job $JID ($N members, f01-f$LEAD, plots: $PLOT_MEMBERS) -> $DATAROOT/logs/replot_${YMD}${HH}_${JID}.out"
    fi
done
