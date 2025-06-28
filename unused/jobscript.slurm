#!/bin/bash -l
#SBATCH --job-name=utm
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=1G
module load gurobi/11.0.0/binary

# define some environment variables
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/env

# activate the conda environment
module purge
conda activate "$ENV_PREFIX"
srun python3 simulator.py
