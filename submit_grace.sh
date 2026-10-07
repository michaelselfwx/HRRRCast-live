#!/bin/bash
# HRRRCast end-to-end submission for TAMU HPRC Grace (Slurm).
# Same pipeline and arguments as submit_all.sh, using jobs/grace/*.sh and etc/env_grace.sh.
#
# Usage:
#   ./submit_grace.sh INIT_TIME LEAD_HOUR [N_ENSEMBLES] [N_GPUS] [PACKAGEROOT] [DATAROOT] [RUNPLOT] [unused] [RUNCLEANUP]
# Example (3 members on 3 A100s, 12 h, data in scratch):
#   ACCNR=123456 ./submit_grace.sh 2025-07-03T18 12 3 3 $PWD $SCRATCH/hrrrcast
#
# Environment knobs:
#   ACCNR        HPRC project/account number (see `myproject`); default account if unset
#   GRACE_GPU    a100 (40 GB, default) or a40 (48 GB)
#   FCST_EXTRA   extra fcst.py args, e.g. "--bbox 25.8,36.5,-106.7,-93.5"
#   GET_BCS_EXTRA extra get_bcs.py args, e.g. "--stitch_cycles" for pre-2021 cases
#   FCST_WALLTIME / PMM_WALLTIME / ...  override any walltime below
#   HRRRCAST_CONDA_SH / HRRRCAST_CONDA_ENV  where conda lives / env name (etc/env_grace.sh)

[ -n "${DEBUG:-}" ] && set -x   # DEBUG=1 ./submit_grace.sh ... for a full trace

INIT_TIME=${1:-"2024-07-17T23"}
LEAD_HOUR=${2:-18}
N_ENSEMBLES=${3:-1}
N_GPUS=${4:-1}
PACKAGEROOT=${5:-`pwd`}
DATAROOT=${6:-$SCRATCH/hrrrcast-data}
RUNPLOT=${7:-"YES"}
RUNCLEANUP=${9:-"NO"}

GRACE_GPU=${GRACE_GPU:-a100}
case "$GRACE_GPU" in a100|a40) ;; *) echo "GRACE_GPU must be a100 or a40" >&2; exit 1;; esac
FCST_EXTRA=${FCST_EXTRA:---bbox 25.8,36.5,-106.7,-93.5} # texas domain
GET_BCS_EXTRA=${GET_BCS_EXTRA:-}

SBATCH_ACCOUNT_OPT=""
if [ -n "${ACCNR:-}" ]; then SBATCH_ACCOUNT_OPT="--account=${ACCNR}"; fi

# Grace nodes have 48 cores: make_bcs/plot use two process per lead hour, capped at 48
CPU_TASKS=$(( LEAD_HOUR < 48 ? LEAD_HOUR : 48 ))
(( CPU_TASKS < 1 )) && CPU_TASKS=2

# wall clock limits (A100/A40 are slower than the H100s the defaults were tuned for)
hr=$(echo "$INIT_TIME" | grep -oP '\d{2}$')
if [[ "$hr" =~ ^(00|06|12|18)$ ]]; then
    FCST_WALLTIME=${FCST_WALLTIME:-"06:00:00"}
    PMM_WALLTIME=${PMM_WALLTIME:-"06:30:00"}
    GET_BCS_WALLTIME=${GET_BCS_WALLTIME:-"01:00:00"}
    MAKE_BCS_WALLTIME=${MAKE_BCS_WALLTIME:-"01:30:00"}
else
    FCST_WALLTIME=${FCST_WALLTIME:-"03:00:00"}
    PMM_WALLTIME=${PMM_WALLTIME:-"03:30:00"}
    GET_BCS_WALLTIME=${GET_BCS_WALLTIME:-"00:30:00"}
    MAKE_BCS_WALLTIME=${MAKE_BCS_WALLTIME:-"01:00:00"}
fi
GET_ICS_WALLTIME=${GET_ICS_WALLTIME:-"00:20:00"}
MAKE_ICS_WALLTIME=${MAKE_ICS_WALLTIME:-"00:20:00"}
PLOT_WALLTIME=${PLOT_WALLTIME:-"01:00:00"}
DERIVED_WALLTIME=${DERIVED_WALLTIME:-"00:30:00"}
LPMM_PATCH=${LPMM_PATCH:-16}   # LPMM patch / halo size in grid points (3 km)
LPMM_HALO=${LPMM_HALO:-24}

PMM_POLL_SECONDS="60"
PMM_MIN_AGE_SECONDS="90"
PMM_TIMEOUT_SECONDS="600"
NETCDF2GRIB_SECTION3=
WGRIB2=${WGRIB2:-"wgrib2"}   # from the conda env

JOBDIR=$PACKAGEROOT/jobs/grace

submit_with_check() {
    local jobid
    jobid=$(eval "$@")
    if [[ $? -ne 0 || -z "$jobid" ]]; then
        echo "Failed to submit job: $*" >&2
        exit 1
    fi
    echo "$jobid"
}
sb() { submit_with_check sbatch --parsable ${SBATCH_ACCOUNT_OPT} "$@"; }

if grep -q $'\r' $PACKAGEROOT/atparse.bash $JOBDIR/*.sh 2>/dev/null; then
    echo "ERROR: Windows (CRLF) line endings in atparse.bash or jobs/grace/*.sh; run: dos2unix atparse.bash jobs/grace/*.sh etc/env_grace.sh" >&2
    exit 1
fi
source $PACKAGEROOT/atparse.bash
mkdir -p $DATAROOT/logs
cd $DATAROOT
echo "PACKAGEROOT=$PACKAGEROOT,DATAROOT=$DATAROOT,GRACE_GPU=$GRACE_GPU"

atparse < $JOBDIR/job-get-ics.sh > logs/job-get-ics.sh
jobid1=$(sb logs/job-get-ics.sh) || exit 1; echo "Submitted get_ics: $jobid1"

atparse < $JOBDIR/job-get-bcs.sh > logs/job-get-bcs.sh
jobid2=$(sb logs/job-get-bcs.sh) || exit 1; echo "Submitted get_bcs: $jobid2"

atparse < $JOBDIR/job-make-ics.sh > logs/job-make-ics.sh
jobid3=$(sb --dependency=afterok:$jobid1 logs/job-make-ics.sh) || exit 1; echo "Submitted make_ics: $jobid3"

atparse < $JOBDIR/job-make-bcs.sh > logs/job-make-bcs.sh
jobid4=$(sb --dependency=afterok:$jobid2 logs/job-make-bcs.sh) || exit 1; echo "Submitted make_bcs: $jobid4"

atparse < $JOBDIR/job-fcst.sh > logs/job-fcst.sh
ARRAY_SPEC="0-$((N_GPUS-1))"
jobid5=$(sb --dependency=afterok:$jobid3:$jobid4 --array=$ARRAY_SPEC logs/job-fcst.sh) || exit 1
echo "Submitted forecast array: $jobid5"
last_jobid=$jobid5

# derived precip (run total, 6 h, 12 h, LPMM); needs every member finished
atparse < $JOBDIR/job-derived-precip.sh > logs/job-derived-precip.sh
jobidD=$(sb --dependency=afterok:$jobid5 logs/job-derived-precip.sh) || exit 1; echo "Submitted derived_precip: $jobidD"
last_jobid=$jobidD

if [ "$RUNPLOT" == "YES" ]; then
    atparse < $JOBDIR/job-plot.sh > logs/job-plot.sh
    jobid6=$(sb --dependency=afterok:$jobid5:$jobidD --array=$ARRAY_SPEC logs/job-plot.sh) || exit 1
    echo "Submitted plot array: $jobid6"
    last_jobid=$jobid6
fi

if [ $N_ENSEMBLES -ge 2 ]; then
    atparse < $JOBDIR/job-compute-pmm.sh > logs/job-compute-pmm.sh
    jobid7=$(sb --dependency=after:$jobid5 logs/job-compute-pmm.sh) || exit 1; echo "Submitted compute_pmm: $jobid7"
    last_jobid=$jobid7
    if [ "$RUNPLOT" == "YES" ]; then
        atparse < $JOBDIR/job-plot.sh > logs/job-plot-pmm.sh
        jobid8=$(sb --dependency=afterok:$jobid7:$jobidD logs/job-plot-pmm.sh) || exit 1; echo "Submitted PMM plot: $jobid8"
        last_jobid=$jobid8
    fi
fi

if [ "$RUNCLEANUP" == "YES" ]; then
    atparse < $JOBDIR/job-cleanup.sh > logs/job-cleanup.sh
    jobidc=$(sb --dependency=afterany:$last_jobid logs/job-cleanup.sh) || exit 1; echo "Submitted cleanup: $jobidc"
fi
