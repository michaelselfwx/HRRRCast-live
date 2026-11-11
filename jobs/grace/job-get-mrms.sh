#!/bin/bash
#SBATCH --job-name=get_mrms
#SBATCH --output=logs/get_mrms_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=@[MRMS_WALLTIME]

# Observed MRMS composite reflectivity at the top of every forecast hour, on this case's domain,
# written as hrrrcast_mrms_fHH.nc so it plots / shows in the viewer as member "mrms".
# Hours that are still in the future are skipped (rerun later, e.g. replot_grace.sh with ONLY_PLOT=mrms).
INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
LEAD_HOUR=@[LEAD_HOUR]

source ${PACKAGEROOT}/etc/env_grace.sh
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; MRMS download may fail"

echo "In get_mrms, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}"
python ${PACKAGEROOT}/src/get_mrms.py ${INIT_TIME} ${LEAD_HOUR} --base_dir ${DATAROOT} --workers 6 @[MRMS_EXTRA]
