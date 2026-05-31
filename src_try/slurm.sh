#!/bin/bash
#SBATCH --job-name=ifd-bloodmnist-ab
#SBATCH --partition=gpu-long
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-80

# BloodMNIST A/B: n_classes = 8 (correct -> should match paper Table 1) vs
# 10 (historical bug -> expect divergence, esp. OOD). For each class count we
# train IFD, L2D-Pop (QC & QI), and Multi-Expert L2D, on both archetypes and
# several seeds, then score against paper Table 1 (Blood Cells) with eval_auto.py.
#
# Submit FROM THE REPO ROOT:   sbatch src_try/slurm.sh
#
# SEEDS: 3 seeds is a quick directional pass that fits in 24h. For the paper's
# 10-cohort comparison set SEEDS="1 2 3 4 5 6 7 8 9 10" and raise --time (or run
# the two phases as separate jobs) — ~160 trainings won't fit one 24h slot.

set -uo pipefail

SEEDS="1 2 3"          # <-- bump to "1 2 3 4 5 6 7 8 9 10" for the full Table-1 comparison

echo "Job started at: $(date) on $(hostname)  (SLURM_JOB_ID=${SLURM_JOB_ID:-NA})"

# repo root = SLURM submit dir, or (for a plain `bash src_try/slurm.sh`) the script's parent
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f src_try/main.py ] || { echo "ERROR: run from repo root (src_try/main.py not found in $(pwd))"; exit 1; }
mkdir -p logs

# ----- conda -----
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate ifd || { echo "ERROR: conda env 'ifd' not found"; exit 1; }
echo "Python: $(which python) | env: ${CONDA_DEFAULT_ENV:-?}"

# one training run; a single failure logs a warning but never aborts the sweep
train () {
  echo ">> main.py $*"
  python src_try/main.py "$@" || echo "!! WARN: training failed for: $*"
}

run_phase () {                       # args: <n_classes> <experiment-prefix>
  local NC="$1" PREFIX="$2"
  export BLOOD_N_CLASSES="$NC"
  echo ""
  echo "############ PHASE: BloodMNIST n_classes=${NC}  (prefix=${PREFIX}) ############"

  # Expert-prediction cache and prototypicality classifier are keyed by dataset
  # name only, so they MUST be cleared between class counts or the second phase
  # would reuse the first phase's experts. (Safe: they regenerate on first run.)
  rm -rf expert_predictions/blood_mnist
  rm -f  best_expert_clf_blood_mnist.pth

  for SEED in $SEEDS; do
    for ARCH in realistic_specialist variable_specialist; do   # Stable / Variable
      # IFD (ours): paper-faithful loss = LCB alpha=1, no auxiliary classification loss
      train --model ifd       --dataset blood_mnist --mode ID --seed "$SEED" \
            --expert_archetypes "$ARCH" --lcb --lcb_alpha 1 --early_stopping 50 \
            --experiment_name "${PREFIX}_ifd"
      # L2D-Pop (QC): attentive / query-conditioned
      train --model l2d-pop   --with_attn True  --dataset blood_mnist --mode ID --seed "$SEED" \
            --expert_archetypes "$ARCH" --early_stopping 50 \
            --experiment_name "${PREFIX}_l2dpop_qc"
      # L2D-Pop (QI): Deep-Sets mean / query-independent
      train --model l2d-pop   --with_attn False --dataset blood_mnist --mode ID --seed "$SEED" \
            --expert_archetypes "$ARCH" --early_stopping 50 \
            --experiment_name "${PREFIX}_l2dpop_qi"
      # Multi-Expert L2D (ID-only baseline)
      train --model l2d-multi --dataset blood_mnist --mode ID --seed "$SEED" \
            --expert_archetypes "$ARCH" --early_stopping 50 \
            --experiment_name "${PREFIX}_l2dmulti"
    done
  done

  echo "==== scoring phase ${PREFIX} (n_classes=${NC}) vs paper Table 1 ===="
  python src_try/eval_auto.py --prefix "${PREFIX}" --dataset blood_mnist \
      | tee "logs/results_${PREFIX}.txt"
  unset BLOOD_N_CLASSES
}

run_phase 8  blood_n8       # 8 first  (correct / paper)
run_phase 10 blood_n10      # then 10  (historical bug)

echo ""
echo "Done at $(date)."
echo "Compare:  logs/results_blood_n8.txt   vs   logs/results_blood_n10.txt"
