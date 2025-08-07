#!/bin/bash
#SBATCH --job-name=cmor_pipeline
#SBATCH --output=logs/pipeline.log
#SBATCH --error=logs/pipeline.log
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450

source /work/bk1450/a270220/aimip/bin/activate

mkdir -p logs 
mkdir -p data/1_means/daily_means data/1_means/monthly_means 
mkdir -p data/2_renamed data/3_split data/4_cmorisation

# ADJUST THESE 4 VARS ACCORDING TO YOUR DATA & STRUCTURE
TAG="AWM-sst-w005-custom_aimip_r0i1p1f1_gn"
INPUT_DIR="/work/bk1450/b383170/eval/archesweather-m-seed0-sst-weight_005-ftar/1980-01-01T12:00_2020-01-01T12:00/daily"
BASE_OUT="/work/bk1450/a270220/repos"
TIMESPAN="2000-2035" # timespan of your input files

# DO NOT CHANGE THESE! 
MERGED_FILE="${BASE_OUT}/geoarches_evaluation/cmorisation/data/1_means/daily_means/daily_${TAG}_${TIMESPAN}.nc"
MONTHLY_MEAN_FILE="${BASE_OUT}/geoarches_evaluation/cmorisation/data/1_means/monthly_means/monthly_${TAG}_${TIMESPAN}.nc"

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

python src/pipeline.py --config config.yaml

# In case you want to move your output to a different location
# bash move.sh