#!/bin/bash
#SBATCH --job-name=ifd-reprod
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; load your conda module before submitting this job." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate ifd

bash run_exps.sh
python src_try/evaluate.py
