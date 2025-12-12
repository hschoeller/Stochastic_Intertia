#!/bin/bash

#SBATCH --job-name=CdV
#SBATCH --partition=main
#SBATCH --array=0-101
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/CdV_%a.out
#SBATCH --error=./logs/CdV_%a.err

# The following is for compiling and running different sigmas
sigmas=($(python3 - <<'EOF'
import numpy as np
vals = np.linspace(0, 0.05, 101)
print(" ".join(map(str, vals)))
EOF
))
sigma=${sigmas[$SLURM_ARRAY_TASK_ID]}
# sigma=0.0
safe_sigma=$(echo "$sigma" | sed 's/\./p/g')
exe_name="exe/exe_${safe_sigma}"

# Create a job-specific module directory
moddir="mods/mod_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$moddir"

# Compile with chosen sigma, putting .mod files into $moddir
gfortran -cpp -J"$moddir" -o "$exe_name" \
    -D SUB_SIGMA=${sigma} \
    -D SUB_NOISE_TYPE='"w"' \
    -D SUB_R=0.0 \
    params.f90 coeffs.f90 utils.f90 \
    barotropic6d.f90 barotropic_model.f90 -llapack -lblas

# Run executable
./"$exe_name" 1

# The following is for compiling and running a single sigma multiple instances

# if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
#     moddir="mods/mod_${SLURM_ARRAY_TASK_ID}"
#     gfortran -cpp -J"$moddir" -o my_executable -D SUB_SIGMA=0.01 -D SUB_NOISE_TYPE='"w"' \
#     -D SUB_R=0.0 params.f90 coeffs.f90 utils.f90 \
#     barotropic6d.f90 barotropic_model.f90 -llapack -lblas
# fi

# sleep 5

# # Run the executable with the provided parameters
# ./my_executable $SLURM_ARRAY_TASK_ID