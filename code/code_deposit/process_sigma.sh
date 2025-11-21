#!/usr/bin/env bash
#SBATCH --job-name=calc_esc
#SBATCH --output=logs/esc_%a.out
#SBATCH --error=logs/esc_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=main
#SBATCH --array=1-501
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


ENV_NAME="CdV"
SCRIPT="process_sigma.py"
SUBDIR="cdv_model/data/"
PATTERN="dataOro20_sigma0p*.bin"

mapfile -t files < <(printf '%s\n' "$SUBDIR"/$PATTERN 2>/dev/null | sort)
idx=$((SLURM_ARRAY_TASK_ID - 1))
INPUT="${files[$idx]}"
echo "Task ${SLURM_ARRAY_TASK_ID} processing: ${INPUT}"

source "${HOME}/mambaforge/etc/profile.d/conda.sh" 2>/dev/null || true
mamba activate "$ENV_NAME" || conda activate "$ENV_NAME" || { echo "Failed to activate env $ENV_NAME"; exit 4; }

# SCRIPT="process_TO.py"
# INPUT=${SLURM_ARRAY_TASK_ID}

python "$SCRIPT" "$INPUT"