#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=@[CPU_TASKS]
#SBATCH --time=@[PLOT_WALLTIME]
#SBATCH --mem=360G

# set vars
INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
N_ENSEMBLES=@[N_ENSEMBLES]
N_GPUS=@[N_GPUS]

# conda
source ${PACKAGEROOT}/etc/env_grace.sh
# cartopy downloads coastline/border shapefiles on first use -> needs internet
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; cartopy map downloads may fail"

if [ -v SLURM_ARRAY_TASK_ID ]; then
    # job array task -> compute member range for this task
    TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

    # Compute start-end range for this task using block distribution
    chunk=$(( N_ENSEMBLES / N_GPUS ))
    rem=$(( N_ENSEMBLES % N_GPUS ))
    extra=0
    if (( TASK_ID < rem )); then extra=1; fi
    if (( TASK_ID < rem )); then
        start=$(( TASK_ID * chunk + TASK_ID ))
    else
        start=$(( TASK_ID * chunk + rem ))
    fi
    end=$(( start + chunk + extra - 1 ))

    if (( start > end )); then
        echo "No members assigned to plot array task ${TASK_ID} (N_ENSEMBLES=${N_ENSEMBLES}, N_GPUS=${N_GPUS}). Exiting."
        exit 0
    fi
    MEMBER_RANGE="${start}-${end}"
else
    MEMBER_RANGE="avg spr"
fi

echo "In plot, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}, TASK_ID=${TASK_ID}, MEMBER_RANGE=${MEMBER_RANGE}"

python3 ${PACKAGEROOT}/src/plot.py ${INIT_TIME} ${LEAD_HOUR} --members ${MEMBER_RANGE} --products REFC APCP CAPE T2M WIND_10M MSLMA HLCY_0_3km HGT_500hPa --forecast_dir ${DATAROOT} --output_dir ${DATAROOT}

# refresh the HTML viewer (viewer.html + viewer_manifest.js in DATAROOT)
python3 ${PACKAGEROOT}/src/make_viewer_index.py --base_dir ${DATAROOT} || true
