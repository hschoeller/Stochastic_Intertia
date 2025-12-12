#!/bin/bash

#SBATCH --job-name=reinsert_to_sigma
#SBATCH --output=reinsert_to.%A_%a.out
#SBATCH --error=reinsert_to.%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-100

ENV_NAME="CdV"

# activate environment (mambaforge/conda). adjust to your cluster's setup.
source "${HOME}/mambaforge/etc/profile.d/conda.sh" 2>/dev/null || true
mamba activate "$ENV_NAME" || conda activate "$ENV_NAME" || { echo "Failed to activate env $ENV_NAME"; exit 4; }

PY="compute_reinsertion.py"   # adjust path to script
OUT_BASE="reinsert_TO"        # base name for partials; final pickle will be reinsert_TO_3d.pkl

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

IDX=${SLURM_ARRAY_TASK_ID}

python "${PY}" \
  --dt 0.05 \
  --n-steps 999 \
  --save-every 1 \
  --rng-seed 42 \
  --sigma-index ${IDX} \
  --out "${OUT_BASE}" \
  --threads ${SLURM_CPUS_PER_TASK}

# After the array finishes, merge partials into the final pickle:
# Submit the array with:
#   JOBID=$(sbatch --parsable run_reinsertion.slurm.sh)
# Then submit the merge job which depends on the array:
#   sbatch --dependency=afterok:${JOBID} --wrap="python ${PY} --merge --out ${OUT_BASE}"
#
# Or run the merge from a login node once all array jobs complete:
#   python ${PY} --merge --out ${OUT_BASE}
