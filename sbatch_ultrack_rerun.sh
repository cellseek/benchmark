#!/usr/bin/env bash
#SBATCH --job-name=csbench_ultrack
#SBATCH --partition=gpu-l20
#SBATCH --account=cellseek
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# Re-run Ultrack only (HSC + MuSC) after ArrayMap read-only fix in ultrack_adapter.py.
# Submit from cellseek_benchmark repo root:
#   sbatch sbatch_ultrack_rerun.sh
#   sbatch --array=0-1 sbatch_ultrack_rerun.sh   # parallel (one dataset per task)

ULTRACK_ENV="${ULTRACK_ENV:-cellseek-ultrack}"
CONFIG_DIR="${CONFIG_DIR:-configs}"

EXPERIMENTS=(
  experiments/ultrack_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/ultrack_ctc_bf_c2dl_musc_tracking.yaml
)

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

run_one() {
  local exp_cfg="$1"
  echo "[$(date)] >>> ${exp_cfg} (env=${ULTRACK_ENV})"
  conda deactivate >/dev/null 2>&1 || true
  conda activate "${ULTRACK_ENV}"
  python -V
  python -c "import cellseek_benchmark; print(cellseek_benchmark.__file__)"
  python -m cellseek_benchmark.test \
    --config-dir "${CONFIG_DIR}" \
    --benchmark-config "${exp_cfg}"
  echo "[$(date)] <<< done ${exp_cfg}"
}

echo "[$(date)] host=$(hostname) repo=${SCRIPT_DIR} env=${ULTRACK_ENV}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx < 0 || idx >= ${#EXPERIMENTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${idx} out of range 0..$((${#EXPERIMENTS[@]} - 1))" >&2
    exit 2
  fi
  run_one "${EXPERIMENTS[$idx]}"
else
  for exp_cfg in "${EXPERIMENTS[@]}"; do
    run_one "${exp_cfg}"
  done
fi

echo "[$(date)] ultrack rerun complete."
