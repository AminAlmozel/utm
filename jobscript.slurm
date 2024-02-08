#!/bin/bash -l
#SBATCH --job-name=myfirstjob
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --constraint=v100
module load cuda
./my_application [blah]
srun –n 2 –c 4 ./my_multithreaded_app
