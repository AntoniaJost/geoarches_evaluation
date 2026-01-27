#!/bin/bash
#SBATCH --job-name=FR19.1-cmor
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

source /work/bk1450/a270220/aimip/bin/activate

set -euo pipefail

### ----------------------------
# IDEALLY, ALL YOU NEED TO TOUCH IS WITHIN THIS BLOCK. ADJUST ACCORDING TO YOUR DATA & STRUCTURE
MODEL_TAG="AWGen-unforced-forced-residuals" # part of folder path of input data
NAME="ArchesWeatherGen" #"ArchesWeather", "ArchesWeatherGen"
MEMBER="1" # typically 1 to 5
ENSEMBLE="r${MEMBER}i1p1f1"
TAG="AWGen-unforced-forced-residuals_${ENSEMBLE}_gn" # part of the input filename, without "day_"
INPUT_DIR="/work/bk1450/b383170/rollouts/AWGen-unforced-forced-residuals/1978-10-01/member_${MEMBER}"
TIMESPAN="1978-2014" # timespan of your input files
TIMESPANS_DAILY=(
  '["1978-10-01 ","2024-12-31"]'
  # '["2013-01-01","2014-12-31"]'
) # timespan(s) for which daily data is wanted. Has to be of format ["start date", "end date"]. if aimip=true timespan will be overwritten by aimip requirements
AIMIP="false" # "true" or "false"
export RUN_DIR="/work/bk1450/a270220/cmorised_awm/${MODEL_TAG}" # working directory where all output will be stores
### ----------------------------

if [[ "${AIMIP,,}" == "true" ]]; then # spaces are important, don't change!!
# DO NOT TOUCH THESE, they are the aimip requirements
  TIMESPANS_DAILY=(
    '["1978-10-01", "1979-12-31"]'
    '["2024-01-01", "2024-12-31"]'
  )
  ZG_TO_500="true"
  LEVELS="[1000, 850, 700, 500, 250, 100, 50]"
else
  ZG_TO_500="false" # decide if zg (geopotential height) shall be reduced to only contain 500hPa
  LEVELS="[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]"
fi

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

  export ENSEMBLE MODEL_TAG ZG_TO_500 NAME TIMESPAN_DAILY LEVELS AIMIP
  envsubst '${RUN_DIR} ${ENSEMBLE} ${MODEL_TAG} ${ZG_TO_500} ${NAME} ${TIMESPAN_DAILY} ${LEVELS} ${AIMIP}' < config.yaml > "${RUN_DIR}/config.yaml" # Create a per-run config to pass RUN_DIR, ENSEMBLE and MODEL_TAG

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
done

echo "ALL DONE."
