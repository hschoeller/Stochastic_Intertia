#!/bin/bash

#SBATCH --job-name=CdV
#SBATCH --partition=main
#SBATCH --array=1-100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./CdV_%a.out
#SBATCH --error=./CdV_%a.err

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    gfortran -cpp -o my_executable -D SUB_SIGMA=0.0 -D SUB_NOISE_TYPE='"w"' \
    -D SUB_R=0.0 params.f90 coeffs.f90 utils.f90 \
    barotropic6d.f90 barotropic_model.f90 -llapack -lblas
fi

sleep 5

# Run the executable with the provided parameters
./my_executable $SLURM_ARRAY_TASK_ID