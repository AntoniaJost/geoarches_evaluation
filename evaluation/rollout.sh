#!/bin/sh
#SBATCH --job-name=AW-M-0-aimip-stats-forcings_surface-interpolgt
#SBATCH --account=bk1450
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err


. ~/.bashrc
lenv  # activate weather env
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# First use right branch of geoarches repository
cdga
git checkout feature/forcings

# Switch back to geoarches_evaluation repository
cdge 
cd evaluation

######### ArchesWeatherDet #########
name="AW-M-1-aimip-w_forcings-interpolgt"
aimip_name="AW-M-avg-aimip-w_forcings-interpolgt"
echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_CPUS_PER_TASK} CPUs per task."
echo "Using ${SLURM_GPUS_PER_NODE} GPUs."
srun --cpu-bind=none --mem-bind=none --mem=0  --cpus-per-task=8 python3 rollout.py \
    ++model_name=${name} \
    ++aimip.aimip_name=${aimip_name} \
    '++aimip.continue_rollout=False' \
    '++aimip.member="avg"' \
    '++aimip.sst_scenario="0"' \
    '++start_timestamp="1980-01-01T12:00"' \
    '++end_timestamp="2000-12-31T12:00"' \
    '++aimip.ablate_forcings=False' \
    '++aimip.replace_land_grid_from_forcings=False' \

######### ArchesWeatherGen #########
#name="archesgen-m-gc-sst_sic-weight_01"
#mkdir -p logs/${name}/${SLURM_JOB_ID}
#echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_CPUS_PER_TASK} CPUs per task."
#echo "Using ${SLURM_GPUS_PER_NODE} GPUs."
#srun --cpu-bind=none --mem-bind=none --cpus-per-task=8 python3 rollout.py \
#    '++model_name="archesgen-m-gc-sst_sic-weight_01"' \
#    '++aimip.name="archesgen-sst_sic"' \
#    '++aimip.member=0' \
#    '++start_timestamp="1980-01-01T12:00"' \
#    '++end_timestamp="2100-01-01T12:00"' \
#    > ${SLURM_SUBMIT_DIR}/logs/${name}/${SLURM_JOB_ID}/output.txt


