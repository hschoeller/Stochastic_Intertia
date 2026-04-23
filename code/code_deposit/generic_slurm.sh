#!/bin/bash
#SBATCH --job-name=sigma
#SBATCH --output=cdv_%j.out
#SBATCH --error=cdv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive           # allocate entire node (all CPUs + all memory)
#SBATCH --partition=main

ENV_NAME="CdV"

# activate environment (mambaforge/conda). adjust to your cluster's setup.
source "${HOME}/mambaforge/etc/profile.d/conda.sh" 2>/dev/null || true
mamba activate "$ENV_NAME" || conda activate "$ENV_NAME" || { echo "Failed to activate env $ENV_NAME"; exit 4; }

# Make sure Python numeric libraries use all allocated threads
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_ON_NODE:-1}
export PYTHONUNBUFFERED=1
# Run your script
# python ./subsample_analyze.py
python ./sigma_analyze.py
