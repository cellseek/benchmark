#!/usr/bin/env bash
#SBATCH --job-name=csbench_omnipose
#SBATCH --partition=gpu-a30
#SBATCH --account=cellseek
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# QOS often caps walltime; shorten if submit fails (e.g. QOSMaxWallDurationPerJobLimit):
#   sbatch --time=12:00:00 sbatch.sh
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Omnipose segmentation via cellseek-benchmark (``python -m cellseek_benchmark.test``;
# does not require ``csbench`` on PATH if the repo is not pip-installed in this env).
# Submit from repo root:
#   sbatch sbatch.sh
#
# Defaults: SEG/man_seg GT on BF-C2DL-MuSC. Override:
#   sbatch --partition=gpu-l20 sbatch.sh

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-experiments/omnipose_ctc_bf_c2dl_musc_segmentation.yaml}"
CONDA_ENV="${CONDA_ENV:-omnipose_env}"
CONFIG_DIR="${CONFIG_DIR:-configs}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"
OMNIPOSE_SRC="${OMNIPOSE_SRC:-/home/fzhaoai/omnipose/src}"
export PYTHONPATH="${SCRIPT_DIR}:${OMNIPOSE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${CONFIG_DIR}/experiments" ]]; then
  echo "[$(date)] ERROR: ${CONFIG_DIR}/experiments not found under ${SCRIPT_DIR}" >&2
  exit 2
fi

echo "[$(date)] host=$(hostname) repo=${SCRIPT_DIR}"
echo "[$(date)] BENCHMARK_CONFIG=${BENCHMARK_CONFIG} CONDA_ENV=${CONDA_ENV}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true
conda activate "${CONDA_ENV}"

python -V
python -c "import cellseek_benchmark; print('cellseek_benchmark:', cellseek_benchmark.__file__)"
python -c "
from omnipose.models import OmniModel
print('omnipose OmniModel import ok')
"

python -m cellseek_benchmark.test \
  --config-dir "${CONFIG_DIR}" \
  --benchmark-config "${BENCHMARK_CONFIG}"

echo "[$(date)] done ${BENCHMARK_CONFIG}"
