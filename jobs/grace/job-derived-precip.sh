#!/bin/bash
#SBATCH --job-name=derived_precip
#SBATCH --output=logs/derived_precip_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=@[DERIVED_WALLTIME]

INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
LEAD_HOUR=@[LEAD_HOUR]

source ${PACKAGEROOT}/etc/env_grace.sh

echo "In derived_precip, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}"
python ${PACKAGEROOT}/src/derived_precip.py ${INIT_TIME} ${LEAD_HOUR} --forecast_dir ${DATAROOT} --lpmm_patch @[LPMM_PATCH] --lpmm_halo @[LPMM_HALO]
