#!/bin/bash
#SBATCH --job-name=cmor_pipeline
#SBATCH --output=logs/pipeline.log
#SBATCH --error=logs/pipeline.log
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450

module load python/3.11.5
source /work/bk1450/a270220/aimip/bin/activate

mkdir -p logs
python src/pipeline.py --config config.yaml