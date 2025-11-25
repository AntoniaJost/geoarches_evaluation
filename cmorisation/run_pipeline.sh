#!/bin/bash
#SBATCH --job-name=cmor_renu1.1
#SBATCH --time=06:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

source /work/bk1450/a270220/aimip/bin/activate

set -euo pipefail

# ADJUST THESE 5 VARS ACCORDING TO YOUR DATA & STRUCTURE
MODEL_TAG="AWM-1_renu" #"AW-M-0-google-forcings-maskedLoss" #"AW-M-1-aimip-stats-forcings_surface" # part of folder path of input data
NAME="ArchesWeather" #"ArchesWeather"
ENSEMBLE="r1i1p1f1"

TAG="AWM-1_renu_${ENSEMBLE}_gn" #"AW-M-0-google-forcings-maskedLoss_${ENSEMBLE}_gn" #"AW-M-1-aimip-stats-forcings_surface_aimip_${ENSEMBLE}_gn" # part of the input filename
INPUT_DIR="/home/b/b383170/repositories/geoarches_evaluation/data/rollouts/AWM-1_renu/1978-10-01T00:00/sst_0/daily/member_0" 
# "/home/b/b383170/repositories/scripts/evalstore/AW-M-0-google-forcings-maskedLoss/1978-10-01T00:00/sst_0/daily/member_01"
# "/work/bk1450/a270220/evalstore/AW-M-1-aimip-stats-forcings_surface/1980-01-01T12:00_2018-12-31T12:00/daily/member_0"
TIMESPAN="1978-2025" # timespan of your input files
ZG_TO_500="false" # decide if zg (geopotential height) shall be reduced to only contain 500hPa

export RUN_DIR="/work/bk1450/a270220/cmorised_awm/${MODEL_TAG}"
# "/work/bk1450/a270220/cmorised_awm/${MODEL_TAG}"

mkdir -p "${RUN_DIR}/logs" \
         "${RUN_DIR}/1_means/daily_means" \
         "${RUN_DIR}/1_means/monthly_means" \
         "${RUN_DIR}/2_renamed" \
         "${RUN_DIR}/3_split" \
         "${RUN_DIR}/4_cmorisation"

# Create a per-run config to pass RUN_DIR, ENSEMBLE and MODEL_TAG
export ENSEMBLE MODEL_TAG ZG_TO_500 NAME
envsubst '${RUN_DIR} ${ENSEMBLE} ${MODEL_TAG} ${ZG_TO_500} ${NAME}' < config.yaml > "${RUN_DIR}/config.yaml"
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