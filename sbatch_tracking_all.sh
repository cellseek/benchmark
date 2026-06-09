#!/usr/bin/env bash
#SBATCH --job-name=csbench_trk_all
# Default gpu-l20 (~48GB VRAM). SAM3 video tracking OOMs on gpu-a30 (24GB).
# Override: sbatch --partition=gpu-a30 sbatch_tracking_all.sh
#SBATCH --partition=gpu-l20
#SBATCH --account=cellseek
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# Many sub-sequences per split (man_track_filtered.json); raise if QOS allows:
#   sbatch --time=7-00:00:00 sbatch_tracking_all.sh
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# Re-run all CTC **tracking** models on BF-C2DL-HSC and BF-C2DL-MuSC (JSON sub-sequences).
# Submit from cellseek_benchmark repo root:
#
#   sbatch sbatch_tracking_all.sh
#     → runs 12 experiments sequentially in one job
#
#   sbatch --array=0-13 sbatch_tracking_all.sh
#     → one experiment per array task (recommended; 14 jobs)
#
#   sbatch --array=0-5 sbatch_tracking_all.sh
#     → HSC main-board only (indices 0–5)
#
# Per-model conda env overrides (create envs as in README / requirements-extra-tracking.txt):
#   CELLSEEK_ENV=cellseek-benchmark \
#   TRACKASTRA_ENV=cellseek-ultrack \
#   ULTRACK_ENV=cellseek-ultrack \
#   sbatch --array=0-11 sbatch_tracking_all.sh
#
# Continue after a failed experiment (sequential mode only):
#   CONTINUE_ON_ERROR=1 sbatch sbatch_tracking_all.sh

CONFIG_DIR="${CONFIG_DIR:-configs}"

CELLSEEK_ENV="${CELLSEEK_ENV:-cellseek-benchmark}"
TRACKASTRA_ENV="${TRACKASTRA_ENV:-${CELLSEEK_ENV}}"
ULTRACK_ENV="${ULTRACK_ENV:-cellseek-ultrack}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# 0–5 HSC / 6–11 MuSC main board; 12–13 linker sub-board (shared CellSAM + Trackastra).
EXPERIMENTS=(
  experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/microsam_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/sam3_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/trackastra_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/ultrack_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/celltractr_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/cellseek_ctc_bf_c2dl_musc_tracking.yaml
  experiments/microsam_ctc_bf_c2dl_musc_tracking.yaml
  experiments/sam3_ctc_bf_c2dl_musc_tracking.yaml
  experiments/trackastra_ctc_bf_c2dl_musc_tracking.yaml
  experiments/ultrack_ctc_bf_c2dl_musc_tracking.yaml
  experiments/celltractr_ctc_bf_c2dl_musc_tracking.yaml
  experiments/trackastra_shared_cellseek_ctc_bf_c2dl_hsc_tracking.yaml
  experiments/trackastra_shared_cellseek_ctc_bf_c2dl_musc_tracking.yaml
)

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "${CONFIG_DIR}/experiments" ]]; then
  echo "[$(date)] ERROR: ${CONFIG_DIR}/experiments not found under ${SCRIPT_DIR}" >&2
  exit 2
fi

conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "${conda_sh}" ]]; then
  echo "[$(date)] ERROR: conda not found at ${conda_sh}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${conda_sh}"

env_for_config() {
  local cfg="$1"
  case "${cfg}" in
    *ultrack*)
      echo "${ULTRACK_ENV}"
      ;;
    *shared_cellseek*)
      # CellSAM + Trackastra linking; needs cellseek + trackastra in env.
      echo "${TRACKASTRA_ENV}"
      ;;
    *trackastra*|*celltractr*)
      echo "${TRACKASTRA_ENV}"
      ;;
    *)
      echo "${CELLSEEK_ENV}"
      ;;
  esac
}

run_one() {
  local exp_cfg="$1"
  local conda_env
  conda_env="$(env_for_config "${exp_cfg}")"
  echo "[$(date)] >>> ${exp_cfg} (env=${conda_env})"
  conda deactivate >/dev/null 2>&1 || true
  conda activate "${conda_env}"
  python -V
  python -c "import cellseek_benchmark; print('cellseek_benchmark:', cellseek_benchmark.__file__)"
  if command -v csbench >/dev/null 2>&1; then
    csbench --config-dir "${CONFIG_DIR}" --benchmark-config "${exp_cfg}"
  else
    python -m cellseek_benchmark.test \
      --config-dir "${CONFIG_DIR}" \
      --benchmark-config "${exp_cfg}"
  fi
  echo "[$(date)] <<< done ${exp_cfg}"
}

echo "[$(date)] host=$(hostname) repo=${SCRIPT_DIR}"
echo "[$(date)] envs: cellseek/sam3/microsam=${CELLSEEK_ENV} trackastra/celltractr=${TRACKASTRA_ENV} ultrack=${ULTRACK_ENV}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx < 0 || idx >= ${#EXPERIMENTS[@]} )); then
    echo "[$(date)] ERROR: SLURM_ARRAY_TASK_ID=${idx} out of range 0..$((${#EXPERIMENTS[@]} - 1))" >&2
    exit 2
  fi
  run_one "${EXPERIMENTS[$idx]}"
else
  failed=0
  for exp_cfg in "${EXPERIMENTS[@]}"; do
    if ! run_one "${exp_cfg}"; then
      failed=$((failed + 1))
      echo "[$(date)] FAILED ${exp_cfg}" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    fi
  done
  if (( failed > 0 )); then
    echo "[$(date)] finished with ${failed} failure(s)" >&2
    exit 1
  fi
fi

echo "[$(date)] tracking batch complete."
