#!/usr/bin/env bash
#SBATCH --job-name=csbench_seg
#SBATCH --partition=gpu-a30
#SBATCH --account=cellseek
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Default walltime must fit your QOS (many sites cap at 24h). Raise only if policy allows:
#   sbatch --time=3-00:00:00 sbatch_segmentation.sh
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# CTC segmentation via cellseek-benchmark — default GT is SEG/man_seg (BF-C2DL-MuSC, sparse frames).
# Submit from cellseek_benchmark repo root:
#   sbatch sbatch_segmentation.sh
# Omnipose + SEG: use ``sbatch.sh`` in this directory.
#
# Override partition / time / experiment / conda env:
#   sbatch --partition=gpu-l20 --time=2-00:00:00 sbatch_segmentation.sh
#   BENCHMARK_CONFIG=experiments/omnipose_ctc_bf_c2dl_hsc_segmentation.yaml CONDA_ENV=omnipose_env sbatch sbatch_segmentation.sh
#
# BENCHMARK_CONFIG: YAML under configs/ (relative to --config-dir).
# CONDA_ENV: env where ``csbench`` + model deps (cellsam / omnipose) are installed.

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-experiments/cellsam_seg_ctc_bf_c2dl_musc_segmentation.yaml}"
CONDA_ENV="${CONDA_ENV:-cellseek-benchmark}"
CONFIG_DIR="${CONFIG_DIR:-configs}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

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

python -m cellseek_benchmark.test \
  --config-dir "${CONFIG_DIR}" \
  --benchmark-config "${BENCHMARK_CONFIG}"

echo "[$(date)] done ${BENCHMARK_CONFIG}"
