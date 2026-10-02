#!/bin/bash
#SBATCH --job-name=make_bcs
#SBATCH --output=logs/make_bcs_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=@[CPU_TASKS]
#SBATCH --mem=360G
#SBATCH --time=@[MAKE_BCS_WALLTIME]
#SBATCH --exclusive

INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

DATE=${INIT_TIME%%T*}
DATE=${DATE//-/}
HOUR=${INIT_TIME#*T}

source ${PACKAGEROOT}/etc/env_grace.sh

echo "In make_bcs, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}"
python3 ${PACKAGEROOT}/src/make_bcs.py ${PACKAGEROOT}/net-diffusion/normalize-stats.nc ${INIT_TIME} ${LEAD_HOUR} --base_dir ${DATAROOT} --output_dir ${DATAROOT} --hrrr_grid_file "${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2"
