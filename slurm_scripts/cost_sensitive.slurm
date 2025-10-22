#!/bin/bash
#SBATCH --job-name=cost_sens
#SBATCH -c 1
#SBATCH -t 48:00:00
#SBATCH -o slurm_scripts/output/cost_sens_%j_%a.out
#SBATCH -e slurm_scripts/output/cost_sens_%j_%a.err
#SBATCH --mem=100G
#SBATCH --array=0-50
#SBATCH --partition=compsci

# RUN THIS FILE FROM THE BASE DIRECTORY

# Activate virtual environment
source ven1/bin/activate

# run baseline 
python -u -m experiments.run_cost_sensitive $SLURM_ARRAY_TASK_ID
