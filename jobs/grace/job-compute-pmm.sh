#!/bin/bash
#SBATCH --job-name=compute_pmm
#SBATCH --output=logs/compute_pmm_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=@[PMM_WALLTIME]

INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
LEAD_HOUR=@[LEAD_HOUR]
N_ENSEMBLES=@[N_ENSEMBLES]

export PMM_POLL_SECONDS=@[PMM_POLL_SECONDS]
export PMM_MIN_AGE_SECONDS=@[PMM_MIN_AGE_SECONDS]
export PMM_TIMEOUT_SECONDS=@[PMM_TIMEOUT_SECONDS]
export NETCDF2GRIB_SECTION3=@[NETCDF2GRIB_SECTION3]
export WGRIB2=@[WGRIB2]

source ${PACKAGEROOT}/etc/env_grace.sh

echo "In compute_pmm, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}, n_ensembles=${N_ENSEMBLES}"
python ${PACKAGEROOT}/src/compute_pmm.py ${INIT_TIME} ${LEAD_HOUR} --forecast_dir ${DATAROOT} --output_dir ${DATAROOT} --n_ensembles ${N_ENSEMBLES}
