#!/usr/bin/env bash
#SBATCH --job-name=csbench_multi
#SBATCH --partition=gpu-a30
#SBATCH --account=cellseek
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Change these if needed when submitting:
#   sbatch --partition=<partition> --gres=gpu:1 sbatch_run.sh
#   TRACKASTRA_ENV=cellseek-trackastra ULTRACK_ENV=cellseek-ultrack CELLTRACTR_ENV=cellseek-trackastra sbatch sbatch_run.sh
#
# Defaults:
# - TRACKASTRA_ENV falls back to DEFAULT_ENV
# - ULTRACK_ENV falls back to DEFAULT_ENV
# - CELLTRACTR_ENV falls back to TRACKASTRA_ENV (delegate tracker is Trackastra)

DEFAULT_ENV="${DEFAULT_ENV:-cellseek-ultrack}"
TRACKASTRA_ENV="${TRACKASTRA_ENV:-${DEFAULT_ENV}}"
ULTRACK_ENV="${ULTRACK_ENV:-${DEFAULT_ENV}}"
CELLTRACTR_ENV="${CELLTRACTR_ENV:-${TRACKASTRA_ENV}}"

# SLURM executes a copied script under /var/spool; use submit dir for relative paths.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"

if [[ ! -d "configs/experiments" ]]; then
  echo "[$(date)] ERROR: configs/experiments not found under ${SCRIPT_DIR}" >&2
  echo "Submit from repo root, or pass absolute --config-dir in this script." >&2
  exit 2
fi

echo "[$(date)] starting benchmark batch on host $(hostname)"
echo "[$(date)] repo: ${SCRIPT_DIR}"
echo "[$(date)] envs: trackastra=${TRACKASTRA_ENV}, ultrack=${ULTRACK_ENV}, celltractr=${CELLTRACTR_ENV}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

run_exp() {
  local exp_cfg="$1"
  local conda_env="$2"
  echo "[$(date)] >>> running ${exp_cfg}"
  echo "[$(date)]     env=${conda_env}"
  # Activate env per task to avoid cross-task state pollution.
  conda deactivate >/dev/null 2>&1 || true
  conda activate "${conda_env}"
  python -V
  which csbench
  csbench --config-dir configs --benchmark-config "${exp_cfg}"
  echo "[$(date)] <<< done ${exp_cfg}"
}

# ctc_bf_c2dl_hsc: Trackastra, Ultrack, Cell-TRACTR
run_exp "experiments/trackastra_ctc_bf_c2dl_hsc_tracking.yaml" "${TRACKASTRA_ENV}"
run_exp "experiments/ultrack_ctc_bf_c2dl_hsc_tracking.yaml" "${ULTRACK_ENV}"


# ctc_bf_c2dl_musc: Ultrack, Cell-TRACTR
run_exp "experiments/ultrack_ctc_bf_c2dl_musc_tracking.yaml" "${ULTRACK_ENV}"

echo "[$(date)] all requested experiments finished."
