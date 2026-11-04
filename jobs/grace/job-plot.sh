#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=logs/plot_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=@[PLOT_CPUS]
#SBATCH --time=@[PLOT_WALLTIME]
#SBATCH --mem=360G

# Plots PLOT_MEMBERS (e.g. "0-17", "hrrr", "avg") for every lead hour. Submitted as a job array
# of PLOT_SHARDS tasks: task k plots every PLOT_SHARDS-th member-hour starting at k (plot.py
# --shard k/N), so N nodes share the work. Each task uses all PLOT_CPUS cores of its node.

INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
PLOT_MEMBERS="@[PLOT_MEMBERS]"
PLOT_SHARDS=@[PLOT_SHARDS]
# products to draw, read by plot.py (empty = everything); wildcards stay literal in the quotes
export HRRRCAST_PLOT_PRODUCTS="@[PLOT_PRODUCTS]"

source ${PACKAGEROOT}/etc/env_grace.sh
# cartopy downloads coastline/border shapefiles on first use -> needs internet
module load WebProxy 2>/dev/null || echo "WebProxy module not loaded; cartopy map downloads may fail"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "In plot, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}, members=${PLOT_MEMBERS}, products=${HRRRCAST_PLOT_PRODUCTS:-all}, shard=${TASK_ID}/${PLOT_SHARDS}, cpus=${SLURM_CPUS_PER_TASK:-?}"

# plot domains: tx = full forecast grid, hcfcd_hrrr = Harris County box on HRRR corners (YYYYMMDD/HH/hcfcd_hrrr/);
# hcfcd_href (HREF corners) can be added with HRRRCAST_PLOT_DOMAINS="tx hcfcd_hrrr hcfcd_href"
PLOT_DOMAINS=${HRRRCAST_PLOT_DOMAINS:-tx hcfcd_hrrr}
python3 ${PACKAGEROOT}/src/plot.py ${INIT_TIME} ${LEAD_HOUR} --members ${PLOT_MEMBERS} --shard ${TASK_ID}/${PLOT_SHARDS} \
    --domains ${PLOT_DOMAINS} --forecast_dir ${DATAROOT} --output_dir ${DATAROOT}
