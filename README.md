# CellSeek CTC benchmark

Headless evaluation of the CellSeek segment-then-link pipeline on [Cell Tracking Challenge](http://celltrackingchallenge.net/) datasets. This folder is a sibling of `gui/` and `paper/` under the CellSeek project root.

## Setup

1. Copy CTC training datasets to `data/ctc/` (see [`data/ctc/README.md`](data/ctc/README.md)).
2. Use the same Python environment as the GUI (`pip install -r ../gui/requirements.txt`). The benchmark imports segmentation and tracking from `../gui/`.

## Run

From this directory (`cellseek/benchmark/`):

```bash
# Full evaluation on all five paper-aligned datasets
python benchmark_ctc.py

# Quick smoke test (first 5 frames)
python benchmark_ctc.py --datasets PhC-C2DL-PSC --max-frames 5

# Ablation modes: full, seg-only, link-gt
python benchmark_ctc.py --all-modes

# Re-score saved predictions without re-running models
python benchmark_ctc.py --skip-inference
```

Results are written to `results/summary.json` and per-dataset predicted masks under `results/<dataset>_01_<mode>/pred_TRA/`.

## Metrics

| Metric | Description |
|--------|-------------|
| SEG | Mean matched instance IoU per frame |
| DET | Detection F1 at 50% overlap |
| TRA | Simplified track-consistency proxy (verify with official CTC eval for publication) |
| CTC | Cell-count accuracy score |

DIV is omitted because CellSeek uses Trackastra `greedy_nodiv` linking.

## Literature comparison

Published baseline scores for the paper comparison table are stored in [`literature_baselines.json`](literature_baselines.json). After running the benchmark, copy CellSeek scores from `results/summary.json` into the paper Results tables.

## Supervised evaluation

Use [`results/supervised_template.json`](results/supervised_template.json) to record one GUI supervised run (auto vs corrected TRA, time, frames corrected).
