#!/bin/bash
#SBATCH --job-name=AWM11-aimip_default-unforced
#SBATCH --time=08:00:00
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

### ----------------------------
# IDEALLY, ALL YOU NEED TO TOUCH IS WITHIN THIS BLOCK. ADJUST ACCORDING TO YOUR DATA & STRUCTURE
#MODEL_TAG="era5_1x1_averaged" # part of folder path of input data
#NAME="era5" #"ArchesWeather", "ArchesWeatherGen"
#MEMBER="average" # typically 1 to 5
#ENSEMBLE="ERA5"
#SCENARIO="0k"
#TAG="${NAME}" # part of the input filename, without "day_"
#MODEL_FILE_NAME="${TAG}*" # part of the input filename, with "day_" at the beginning and without ".nc" at the end
#INPUT_DIR="/work/bk1450/b383170/era5/${MODEL_TAG}"
#TIMESPAN="1978-2025" # timespan of your input files
#TIMESPANS_DAILY=(
#  '["1978-10-01 ","2025-01-01"]'
#  # '["2013-01-01","2014-12-31"]'
#) # timespan(s) for which daily data is wanted. Has to be of format ["start date", "end date"]. if aimip=true timespan will be overwritten by aimip requirements
#AIMIP="false" # "true" or "false"
#RUN_DIR="/work/bk1450/b383170/era5/era5_1x1_averaged_cmor" # working directory where all output will be stores
#REPO_DIR="/work/bk1450/b383170/repositories/geoarches_evaluation/cmorisation" # directory of the repository (important: with "cmorisation" folder!)
#LOG_DIR="/work/bk1450/b383170/era5/era5_1x1_averaged_cmor/logs" # where you want your logs
### ----------------------------

# IDEALLY, ALL YOU NEED TO TOUCH IS WITHIN THIS BLOCK. ADJUST ACCORDING TO YOUR DATA & STRUCTURE
MODEL_TAG="AWM11-aimip_default-unforced" # part of folder path of input data
NAME="AWM11-aimip_default-unforced" #"ArchesWeather", "ArchesWeatherGen"
MEMBER="4" # typically 1 to 5
ENSEMBLE="r${MEMBER}i1p1f1"
SCENARIO="0k"
INIT_TIME="1978-10-01" # timespan of your input files
TAG="${NAME}_${ENSEMBLE}_gn" # part of the input filename, without "day_"
MODEL_FILE_NAME="day_${TAG}*" # part of the input filename, with "day_" at the beginning and without ".nc" at the end
INPUT_DIR="/work/bk1450/b383170/rollouts/${MODEL_TAG}/sst_${SCENARIO}/${INIT_TIME}/member_${MEMBER}"
TIMESPAN="1978-2025" # timespan of your input files
TIMESPANS_DAILY=(
  '["1978-10-01 ","2025-01-01"]'
  # '["2013-01-01","2014-12-31"]'
) # timespan(s) for which daily data is wanted. Has to be of format ["start date", "end date"]. if aimip=true timespan will be overwritten by aimip requirements
AIMIP="False" # "true" or "false"
RUN_DIR="/work/bk1450/b383170/rollouts/cmorised_aimip/${MODEL_TAG}/${SCENARIO}/${INIT_TIME}/mem${MEMBER}" # working directory where all output will be stores
REPO_DIR="/work/bk1450/b383170/repositories/geoarches_evaluation/cmorisation" # directory of the repository (important: with "cmorisation" folder!)
LOG_DIR="/work/bk1450/b383170/rollouts/cmorised_aimip/${MODEL_TAG}/${SCENARIO}/${INIT_TIME}/mem${MEMBER}/logs" # where you want your logs

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

  export ENSEMBLE MODEL_TAG ZG_TO_500 NAME TIMESPAN_DAILY LEVELS AIMIP RUN_DIR REPO_DIR LOG_DIR
  envsubst '${RUN_DIR} ${REPO_DIR} ${LOG_DIR} ${ENSEMBLE} ${MODEL_TAG} ${ZG_TO_500} ${NAME} ${TIMESPAN_DAILY} ${LEVELS} ${AIMIP}' < config.yaml > "${RUN_DIR}/config.yaml" # Create a per-run config to pass RUN_DIR, ENSEMBLE and MODEL_TAG

  echo "Wrote per-run config: ${RUN_DIR}/config.yaml"

  # DO NOT CHANGE THESE! 
  MERGED_FILE="${RUN_DIR}/1_means/daily_means/daily_${TAG}_${TIMESPAN}.nc"
  MONTHLY_MEAN_FILE="${RUN_DIR}/1_means/monthly_means/monthly_${TAG}_${TIMESPAN}.nc"

  # Merge all .nc files if file doesn't already exist
  if [ ! -s "${MERGED_FILE}" ]; then
    echo "Merging daily files into: ${MERGED_FILE}"
    cdo mergetime "${INPUT_DIR}/${MODEL_FILE_NAME}*.nc" "${MERGED_FILE}"
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
done

python3 src/pipeline.py --config "${RUN_DIR}/config.yaml"
echo "Done. Outputs under: ${RUN_DIR}"
