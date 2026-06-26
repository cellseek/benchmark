#!/usr/bin/env bash
# Cell-TRACTR assets + deps for the unified ``benchmark`` conda env.
set -euo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${CELLTRACTR_REPO:-/home/fzhaoai/cellseek/Cell-TRACTR}"
CKPT_DIR="${BENCH}/checkpoints/celltractr"
ZENODO="https://zenodo.org/api/records/14509424/files"

mkdir -p "${CKPT_DIR}"

if [[ ! -d "${REPO}/src/trackformer" ]]; then
  echo "Cell-TRACTR repo not found at ${REPO}"
  echo "Clone it outside benchmark, e.g.: git clone https://gitlab.com/dunloplab/Cell-TRACTR.git /home/fzhaoai/cellseek/Cell-TRACTR"
  exit 1
fi

download() {
  local name="$1"
  local dest="${CKPT_DIR}/${name}"
  if [[ -f "${dest}" ]]; then
    echo "exists: ${dest}"
    return 0
  fi
  echo "downloading ${name} …"
  curl -L --fail --retry 3 --output "${dest}" "${ZENODO}/${name}/content"
}

download checkpoint_deepcell.pth
download checkpoint_moma.pth

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "benchmark" ]]; then
  echo "Warning: activate the benchmark env first: conda activate benchmark"
fi

echo "Installing Cell-TRACTR Python deps into current env …"
pip install -q sacred ffmpeg-python fvcore ninja matplotlib

if [[ ! -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
  echo "Installing conda compilers into benchmark env …"
  conda install -y -c conda-forge 'gcc_linux-64=13.*' 'gxx_linux-64=13.*'
fi
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="${CXX}"

if ! command -v nvcc >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then
    module load cuda/12.4.0-uhdfj7w || module load cuda || true
  fi
fi
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi

echo "Checking CUDA/PyTorch before compiling ops …"
python - <<PY
import os
import shutil
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
print("nvcc:", shutil.which("nvcc"))
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("CC:", os.environ.get("CC"))
print("CXX:", os.environ.get("CXX"))
PY

if python - <<PY
import torch
import MultiScaleDeformableAttention  # noqa: F401
PY
then
  echo "MultiScaleDeformableAttention already importable; skipping compile."
else
  echo "Compiling MultiScaleDeformableAttention (requires CUDA toolkit + GPU driver) …"
  pip install --no-build-isolation -e "${REPO}/src/trackformer/models/ops"
fi

python - <<PY
import torch
import MultiScaleDeformableAttention  # noqa: F401
print("OK:", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "Done. Repo: ${REPO}"
echo "Checkpoints: ${CKPT_DIR}"
