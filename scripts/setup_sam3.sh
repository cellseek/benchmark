#!/usr/bin/env bash
# SAM3 package + assets for the unified ``benchmark`` conda env.
set -euo pipefail

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "benchmark" ]]; then
  echo "Warning: activate the benchmark env first: conda activate benchmark"
fi

echo "Installing SAM3 into current env …"
python -m pip install -q sam3==0.1.4

ASSET_DIR="$(python - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "assets")
PY
)"
BPE="${ASSET_DIR}/bpe_simple_vocab_16e6.txt.gz"

mkdir -p "${ASSET_DIR}"
if [[ ! -f "${BPE}" ]]; then
  echo "Downloading SAM3 BPE tokenizer asset …"
  curl -L --fail --retry 3 \
    --output "${BPE}" \
    https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
else
  echo "exists: ${BPE}"
fi

python - <<PY
from pathlib import Path
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
bpe = Path("${BPE}")
assert bpe.is_file(), bpe
print("SAM3 OK:", bpe)
PY
