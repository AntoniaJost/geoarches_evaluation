#!/bin/bash
#SBATCH --job-name=cmor_m0v3s
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

source /work/bk1450/a270220/aimip/bin/activate

set -euo pipefail

# ADJUST THESE 5 VARS ACCORDING TO YOUR DATA & STRUCTURE
MODEL_TAG="archesgen-m-gc-sst_sic-weight_01_member00_v2"
ENSEMBLE="r0i1p1f1"
TAG="archesgen-sst_sic_aimip_${ENSEMBLE}_gn"
INPUT_DIR="/work/bk1450/b383170/eval/archesgen-m-gc-sst_sic-weight_01/1980-01-01T12:00_2100-01-01T12:00/daily/member_00"
TIMESPAN="1980-2066" # timespan of your input files

export RUN_DIR="/work/bk1450/a270220/cmorised_awm/${MODEL_TAG}"
mkdir -p "${RUN_DIR}/logs" \
         "${RUN_DIR}/1_means/daily_means" \
         "${RUN_DIR}/1_means/monthly_means" \
         "${RUN_DIR}/2_renamed" \
         "${RUN_DIR}/3_split" \
         "${RUN_DIR}/4_cmorisation"

# Create a per-run config to pass RUN_DIR, ENSEMBLE and MODEL_TAG
export ENSEMBLE
export MODEL_TAG
envsubst '${RUN_DIR} ${ENSEMBLE} ${MODEL_TAG}' < config.yaml > "${RUN_DIR}/config.yaml"
echo "Wrote per-run config: ${RUN_DIR}/config.yaml"

# DO NOT CHANGE THESE! 
MERGED_FILE="${RUN_DIR}/1_means/daily_means/daily_${TAG}_${TIMESPAN}.nc"
MONTHLY_MEAN_FILE="${RUN_DIR}/1_means/monthly_means/monthly_${TAG}_${TIMESPAN}.nc"

# Merge all .nc files if file doesn't already exist
if [ ! -s "${MERGED_FILE}" ]; then
  echo "Merging daily files into: ${MERGED_FILE}"
  cdo mergetime "${INPUT_DIR}/day_${TAG}_*.nc" "${MERGED_FILE}"
else
  echo "Skipping mergetime: ${MERGED_FILE} already exists."
fi

# Compute monthly means if file doesn't already exist
if [ ! -s "${MONTHLY_MEAN_FILE}" ]; then
  echo "Computing monthly mean into: ${MONTHLY_MEAN_FILE}"
  cdo monmean "${MERGED_FILE}" "${MONTHLY_MEAN_FILE}"
else
  echo "Skipping monmean: ${MONTHLY_MEAN_FILE} already exists."
fi

python src/pipeline.py --config "${RUN_DIR}/config.yaml"
echo "Done. Outputs under: ${RUN_DIR}"

# In case you want to move your output to a different location
# bash move.sh