#!/bin/bash
#SBATCH --job-name=cmor_pipeline
#SBATCH --output=logs/pipeline.log
#SBATCH --error=logs/pipeline.log
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450

source /work/bk1450/a270220/aimip/bin/activate

mkdir -p logs
mkdir -p intermediate

INPUT_DIR="/work/bk1450/b383170/eval/archesweather-m-seed0-sst-weight_005-ftar/1980-01-01T12:00_2020-01-01T12:00/daily"
MERGED_FILE="/work/bk1450/a270220/repos/cmorisation_v2/data/1_means/daily_means/daily_AWM-sst-w005-custom_aimip_r0i1p1f1_gn_1980-2019.nc"
MONTHLY_MEAN_FILE="/work/bk1450/a270220/repos/cmorisation_v2/data/1_means/monthly_means/monthly_AWM-sst-w005-custom_aimip_r0i1p1f1_gn_1980-2019.nc"

# Merge all .nc files
cdo mergetime ${INPUT_DIR}/day_AWM-sst-w005-custom_aimip_r0i1p1f1_gn_*.nc ${MERGED_FILE}

# Compute monthly means
cdo monmean ${MERGED_FILE} ${MONTHLY_MEAN_FILE}

python src/pipeline.py --config config.yaml