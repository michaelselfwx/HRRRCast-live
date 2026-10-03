#!/bin/bash
#SBATCH --job-name=fcst
#SBATCH --output=logs/fcst_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:@[GRACE_GPU]:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=@[FCST_WALLTIME]
# Grace GPU nodes: 2 GPUs, 48 cores, 384 GB each. Asking for half a node lets two
# forecast tasks share one node. a100 = 40 GB, a40 = 48 GB: both fit the full CONUS grid.

INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
N_ENSEMBLES=@[N_ENSEMBLES]
N_GPUS=@[N_GPUS]
FCST_EXTRA="@[FCST_EXTRA]"

export NETCDF2GRIB_SECTION3=@[NETCDF2GRIB_SECTION3]
export WGRIB2=@[WGRIB2]
export TF_CPP_MIN_LOG_LEVEL=1
export TF_GPU_ALLOCATOR=cuda_malloc_async

source ${PACKAGEROOT}/etc/env_grace.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# job array task -> member range (block distribution, same as jobs/job-fcst.sh)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
chunk=$(( N_ENSEMBLES / N_GPUS ))
rem=$(( N_ENSEMBLES % N_GPUS ))
if (( TASK_ID < rem )); then
    start=$(( TASK_ID * chunk + TASK_ID )); extra=1
else
    start=$(( TASK_ID * chunk + rem )); extra=0
fi
end=$(( start + chunk + extra - 1 ))
if (( start > end )); then
    echo "No members assigned to array task ${TASK_ID}. Exiting."
    exit 0
fi
MEMBER_RANGE="${start}-${end}"

echo "In fcst, INIT_TIME=${INIT_TIME}, LEAD_HOUR=${LEAD_HOUR}, TASK_ID=${TASK_ID}, MEMBER_RANGE=${MEMBER_RANGE}"
python ${PACKAGEROOT}/src/fcst.py $PACKAGEROOT/net-diffusion/model.keras ${INIT_TIME} ${LEAD_HOUR} \
    --num_members ${N_ENSEMBLES} --members ${MEMBER_RANGE} --base_dir ${DATAROOT} --output_dir ${DATAROOT} ${FCST_EXTRA}
