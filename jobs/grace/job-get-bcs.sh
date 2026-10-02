#!/bin/bash
#SBATCH --job-name=get_bcs
#SBATCH --output=logs/get_bcs_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=@[GET_BCS_WALLTIME]

INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

source ${PACKAGEROOT}/etc/env_grace.sh
# HPRC compute nodes reach the internet through a proxy module
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; downloads may fail on compute nodes"

echo "In get_bcs, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}"
python3 ${PACKAGEROOT}/src/get_bcs.py ${INIT_TIME} ${LEAD_HOUR} --base_dir ${DATAROOT} @[GET_BCS_EXTRA]
