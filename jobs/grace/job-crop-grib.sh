#!/bin/bash
#SBATCH --job-name=crop_grib
#SBATCH --output=logs/crop_grib_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=@[CROP_WALLTIME]

# Cut every hrrrcast.<mem>.tHHz.pgrb2.fHH of this case (members, hrrr, avg, spr, lpmm) down to the
# CROP_DOMAINS boxes: YYYYMMDD/HH/<domain>/hrrrcast.<mem>.tHHz.pgrb2.fHH (src/crop_grib.py).
# Runs once every GRIB2 writer (fcst, get_hrrr_fcst, compute_pmm, derived_precip) has finished.

INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
CROP_DOMAINS="@[CROP_DOMAINS]"
export WGRIB2=@[WGRIB2]   # crop_grib.py uses wgrib2 if it runs, else the eccodes Python package

source ${PACKAGEROOT}/etc/env_grace.sh

echo "In crop_grib, init_time=${INIT_TIME}, domains=${CROP_DOMAINS}"
python ${PACKAGEROOT}/src/crop_grib.py ${INIT_TIME} --forecast_dir ${DATAROOT} --domains ${CROP_DOMAINS} --workers 8
