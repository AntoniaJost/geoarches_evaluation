#!/bin/sh
#SBATCH --job-name=eval
#SBATCH --account=bk1450
#SBATCH --qos=normal
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err


. ~/.bashrc
lenv  # activate weather env


########## CONFIG PARAMS ##########
srun --cpu-bind=none --mem-bind=none --mem=0  --cpus-per-task=8 python3 eval.py



