#!/bin/bash
#SBATCH --job-name=cmor_0k_m1-1978
#SBATCH --time=06:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24G
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --dependency=afterany:22210134  # replace with actual job ID if needed

module load cdo/2.5.3-gcc-11.2.0 
source ~/repositories/geoenv/bin/activate

export ACCOUNT="b383170"

set -euo pipefail

# ADJUST THESE 5 VARS ACCORDING TO YOUR DATA & STRUCTURE
MODEL_TAG="ArchesWeatherGen" # part of folder path of input data
NAME="ArchesWeatherGen" #"ArchesWeather"
MEMBER=1  # member number of the ensemble you want to cmorise
ENSEMBLE="r${MEMBER}i1p1f1"
SST="0k"

TAG="${MODEL_TAG}_${ENSEMBLE}_gn" # part of the input filename
INPUT_DIR="/home/b/b383170/repositories/geoarches_evaluation/cmorisation/rollouts/${MODEL_TAG}/sst_${SST}/daily/member_${MEMBER}" # path to your input files
TIMESPAN="1978-2025" # timespan of your input files
TIMESPAN_DAILY='["1978-10-01", "1979-12-31"]' # timespan for which daily data is wanted. has to be of format ["start date", "end date"], cannot span multiple time frames, needs to be run twice. ["1978-10-01", "1979-12-31"],
#TIMESPAN_DAILY='["2024-01-01", "2024-12-31"]' # second run for 2024
ZG_TO_500="true" # decide if zg (geopotential height) shall be reduced to only contain 500hPa

export RUN_DIR="/home/b/b383170/repositories/geoarches_evaluation/cmorisation/rollouts/cmorised_aimip/${MODEL_TAG}/sst_${SST}/member_${MEMBER}"

mkdir -p "${RUN_DIR}/logs" \
         "${RUN_DIR}/1_means/daily_means" \
         "${RUN_DIR}/1_means/monthly_means" \
         "${RUN_DIR}/2_renamed" \
         "${RUN_DIR}/3_split" \
         "${RUN_DIR}/4_cmorisation"

# iterate over all daily timespans
for (( i=0; i<${#TIMESPANS_DAILY[@]}; i++ )); do # (https://www.gnu.org/software/bash/manual/bash.html) search for ${#parameter}
  TIMESPAN_DAILY="${TIMESPANS_DAILY[$i]}"
  printf "Iteration %d: TIMESPAN_DAILY=%s\n" "$i" "$TIMESPAN_DAILY"

  export ENSEMBLE MODEL_TAG ZG_TO_500 NAME TIMESPAN_DAILY LEVELS AIMIP RUN_DIR REPO_DIR LOG_DIR
  envsubst '${RUN_DIR} ${REPO_DIR} ${LOG_DIR} ${ENSEMBLE} ${MODEL_TAG} ${ZG_TO_500} ${NAME} ${TIMESPAN_DAILY} ${LEVELS} ${AIMIP}' < config.yaml > "${RUN_DIR}/config.yaml" # Create a per-run config to pass RUN_DIR, ENSEMBLE and MODEL_TAG

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

python3 src/pipeline.py --config "${RUN_DIR}/config.yaml"

echo "Done. Outputs under: ${RUN_DIR}"
