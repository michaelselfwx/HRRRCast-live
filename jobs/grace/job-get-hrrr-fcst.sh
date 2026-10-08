#!/bin/bash
#SBATCH --job-name=get_hrrr_fcst
#SBATCH --output=logs/get_hrrr_fcst_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=@[HRRR_WALLTIME]

# Operational HRRR forecast from the same cycle, written as hrrrcast_hrrr_fHH.nc on the
# HRRRCast (cropped) grid so it can be plotted / viewed as member "hrrr".
INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
LEAD_HOUR=@[LEAD_HOUR]

source ${PACKAGEROOT}/etc/env_grace.sh
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; HRRR download may fail"

echo "In get_hrrr_fcst, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}"
python ${PACKAGEROOT}/src/get_hrrr_fcst.py ${INIT_TIME} ${LEAD_HOUR} --base_dir ${DATAROOT} --workers 6
