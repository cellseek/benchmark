# CellSeek Benchmark

`benchmark` is the CellSeek evaluation package under `cellseek/benchmark/` for segmentation and tracking on open microscopy datasets.

It currently supports:

- Segmentation metrics: `Precision`, `Recall`, `F1` (instance-level IoU matching)
- Tracking metrics: **Cell-HOTA**, `DetA`, `AssA` (HOTA-style on centroid tracks; see `docs/TRACKING_METRICS.md`)

---

## 1) What this benchmark includes

### Datasets (configured)

- `cellpose` (segmentation)
- `BBBC038` (segmentation; configure local folder path)
- Every **`ctc_*`** entry (Cell Tracking Challenge layout): **tracking metrics only** — no segmentation pass on these datasets (standard TRA/SEG inputs).

### Model adapters

- `cellseek` (enabled): default **`tracking_propagation: trackastra`** — Cellpose-SAM per frame + **Trackastra** linking (same as `gui/`). **cpsam** weights: see §3 Checkpoints.
- `cellpose` (enabled)
- `sam3` (enabled, native-or-fallback mode)
- `microsam` (enabled, checkpoint-aware native-or-fallback mode)
- `trackastra` (included in unified env `benchmark`)
- `ultrack` (optional): global linking from a label timelapse built via the same generic mask generator as fallbacks; install **`ultrack`** and a solver backend per [Ultrack docs](https://ultrack.readthedocs.io/). Tune **`linking_max_distance`**, **`solver_name`**, **`time_limit`**.
- `celltractr` (optional, **native**): real Cell-TRACTR Trackformer in the same `benchmark` env — see [`docs/CELLTRACTR.md`](docs/CELLTRACTR.md).

### Output artifacts

Runs produce:

- per-task JSON files in `outputs/<dataset>/<model>/...`
- consolidated JSON report: `outputs/report.json`
- flat CSV summary table: `outputs/summary.csv`

---

## 2) Repository structure

```text
benchmark/
  README.md
  environment.yml       # conda env "benchmark" (recommended)
  requirements-benchmark.txt
  test.py
  configs/
  benchmark/
    test.py
    ...
```

---

## 3) Environment setup

**One conda env for the whole benchmark:** `benchmark`

It covers **cellseek**, **cellpose**, **sam3**, **microsam**, **trackastra**, **ultrack**, **celltractr**, and **omnipose**.

**Python:** 3.11 in `environment.yml` (3.10–3.13 supported). **GPU:** PyTorch + CUDA 12.4 via conda — change `pytorch-cuda` in `environment.yml` if your driver needs another version.

### Recommended: conda env `benchmark`

```bash
cd /home/fzhaoai/cellseek/benchmark

conda env create -f environment.yml
conda activate benchmark
pip install -e .
```

Update an existing env after dependency changes:

```bash
conda env update -f environment.yml --prune
conda activate benchmark
pip install -e .
```

### Verify

```bash
conda activate benchmark
cd /home/fzhaoai/cellseek/benchmark

python -c "import benchmark; print('benchmark ok')"
python -c "import cellpose, trackastra, ultrack; print('stack ok')"
python test.py --help
```

Smoke test:

```bash
python test.py --config-dir configs \
  --benchmark-config experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml
```

### Requirements

| File | Purpose |
|------|---------|
| `environment.yml` | Creates conda env **`benchmark`** (recommended) |
| `requirements-benchmark.txt` | All pip deps for that env |

Pip-only fallback (no conda):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-benchmark.txt
pip install -e .
```

SAM3 is installed in the unified env; set `checkpoint_path` in `configs/models.yaml`.

### Checkpoints and weights

Place files under **`cellseek/checkpoints/`** (paths in `configs/models.yaml` are relative to `benchmark/`):

| File | Used by | Required? |
|------|---------|-----------|
| `microsam_vit_b_lm.pt` | `microsam` | Yes, for microSAM runs |
| `cpsam` | `cellseek`, `cellpose`, `cellsam_seg` | Optional — Cellpose can auto-download; or set `CELLSEEK_CPSAM_PATH` |
| `sam3.pt` | `sam3` | Optional — default config may point at Hugging Face cache instead |

Resolution order for **cpsam** (`benchmark/cellpose_sam.py`): `CELLSEEK_CPSAM_PATH` → `cellseek/checkpoints/cpsam` → `benchmark/weights/cpsam` → `gui/weights/cpsam` → Cellpose download.

Do **not** keep a separate `sam_vit_b_01ec64.pth` for benchmark — microSAM loads the fine-tuned `microsam_vit_b_lm.pt` directly.

### Troubleshooting

- **`No module named 'segment_anything'`** — `pip uninstall segment-anything -y && pip install segment-anything`
- **`No module named 'cellpose'`** — `conda activate benchmark` and `pip install -r requirements-benchmark.txt`
- **CUDA mismatch** — edit `pytorch-cuda=12.4` in `environment.yml`, then `conda env update -f environment.yml --prune`
- **CPU-only debug** — set `use_gpu: false` in `configs/models.yaml` (slow)
- **sam3 runs** — verify `import sam3` in the `benchmark` env and set `checkpoint_path` in `configs/models.yaml`

---

## 4) Configure datasets and models

Edit:

- `configs/datasets.yaml` (catalog of dataset entries; you choose **one** key per run)
- `configs/models.yaml` (catalog of model entries; you choose **one** key per run)
- `configs/metrics.yaml`
- **`configs/experiments/*.yaml`** — one file per run: `experiment_id`, `dataset`, `model`, `tasks`, optional `output_parent` / `run_name` (results go to `<output_parent>/<run_name>_<YYYYMMDD_HHMMSS>/`, default parent `experiments`), plus optional `datasets_config` / `models_config` / `metrics_config` (filenames relative to `configs/`)

Minimum required updates for your machine:

- **`configs/datasets.yaml`** — set CTC root (`/project/cellseek/cell_tracking` by default), `cellpose` / `BBBC038` paths
- **`configs/models.yaml`** — enable/disable models; checkpoint paths (see §3)
- **`configs/experiments/*.yaml`** — one run = one dataset × one model × tasks

Checkpoint paths already wired in `models.yaml`:

- **microsam:** `../checkpoints/microsam_vit_b_lm.pt`
- **sam3:** Hugging Face cache (or `../checkpoints/sam3.pt` if you copy it locally)
- **cellseek / cellpose:** `checkpoint_path: null` — auto-resolve **cpsam** (see §3)

---

## 5) Running the benchmark

Each `test.py` invocation evaluates **one** dataset and **one** model. Pick one experiment YAML under `configs/experiments/`, or use CLI overrides for a one-off run.

### Run an experiment file

Default preset is CellSeek × CTC (`ctc_bf_c2dl_hsc`) × tracking:

```bash
cd /home/fzhaoai/cellseek/benchmark
python test.py --config-dir configs
```

Equivalent:

```bash
python test.py --config-dir configs
```

Equivalent explicit path:

```bash
python test.py --config-dir configs \
  --benchmark-config experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml
```

Copy `configs/experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml` to a new name and edit `dataset`, `model`, `tasks`, and optionally `experiment_id` / `output_parent` for additional experiments (each run creates a new timestamped folder).

### Override dataset, model, or tasks without editing YAML

```bash
python test.py --config-dir configs \
  --benchmark-config experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml \
  --dataset cellpose \
  --model cellseek \
  --tasks both
```

`--tasks` must be one of: `segmentation`, `tracking`, `both`.

### Common focused runs

CTC segmentation (SEG GT, Omnipose):

```bash
python test.py --config-dir configs \
  --benchmark-config experiments/omnipose_ctc_bf_c2dl_hsc_segmentation.yaml
```

CTC segmentation (SEG GT, CellSAM):

```bash
python test.py --config-dir configs \
  --benchmark-config experiments/cellsam_seg_ctc_bf_c2dl_hsc_segmentation.yaml
```

Tracking only (CTC):

```bash
python test.py --config-dir configs --tasks tracking
```

### Smaller on-disk subset for debugging

Use `scripts/build_benchmark_subset.py` to mirror a tiny slice of each dataset under a folder (for example `benchmark_subset_10/`). Copy `configs/` (including `experiments/`) to a new directory, point `datasets.yaml` roots at that subset, then run `python test.py --config-dir <that_directory> --benchmark-config experiments/<your_experiment>.yaml`.

---

## 6) Evaluation pipeline details

### Segmentation pipeline

For each image:

1. collect GT and predicted instance IDs (exclude `0`)
2. compute IoU matrix for all GT/pred instance pairs
3. run Hungarian one-to-one assignment on `1 - IoU`
4. keep matched pairs with `IoU >= threshold` (default `0.5`)
5. compute:
   - `TP`: matched pairs
   - `FP`: unmatched predicted instances
   - `FN`: unmatched GT instances
   - `Precision = TP / (TP + FP)`
   - `Recall = TP / (TP + FN)`
   - `F1 = 2PR / (P + R)`

Configured in `configs/metrics.yaml` under:

- `segmentation.iou_thresholds`

### Tracking pipeline (CTC sub-sequences)

CTC tracking uses **`man_track_filtered.json`**: each JSON entry is one benchmark sequence (frame range + `Valid_IDs`). Summary metrics use **macro mean per sub-sequence** (not frame-weighted).

For each sub-sequence:

1. model produces point tracks `(frame, track_id, x, y)` (and optionally masks)
2. optional: drop predicted tracks shorter than `tracking.pred_min_track_frames` (**same rule for all models**)
3. frame-wise Hungarian matching on centroids (`match_radius_px` global, default 20 px)
4. `DetA`, `AssA`, `Cell-HOTA` (= \(\sqrt{\text{DetA}\cdot\text{AssA}}\)); optional **mask F1** on SEG GT frames
5. `tracking.json` includes **`stratified`** buckets by segment length and `constant_cell_count`

**Leaderboards:**

- **End-to-end (main):** each model’s default pipeline (`cellseek` = Cellpose-SAM + Trackastra; `sam3` = video propagate; `microsam` = per-frame SAM + linking; `trackastra`/`ultrack` = Otsu fallback masks + linker unless configured otherwise).
- **Linker sub-board (optional):** `trackastra_shared_cellseek` / `ultrack_shared_cellseek` use **CellSAM masks** then compare linking only.

**cellseek tracking defaults:** `tracking_propagation: trackastra`, seed search in the first *k* frames, Trackastra `greedy_nodiv` linking between consecutive segmented frames. No GT seeding.

### Joint segmentation + tracking (datasets that support both tasks)

On datasets that still run segmentation **and** tracking (not CTC), if `tasks` is `both`, the dataset adapter sets `joint_segmentation_with_tracking`, and the model defines **`predict_tracks_returning_masks`**, the benchmark may run **one inference pass per sequence** for both metrics. **CTC** does not use joint mode (`joint_segmentation_with_tracking` is false); use separate experiment YAMLs for `tasks: segmentation` (SEG GT) vs `tasks: tracking` (TRA).

Fallback behavior (to keep runs robust across all models/datasets):

- If a model native backend is disabled or unavailable, adapters use a deterministic fallback:
  - segmentation: Otsu threshold + connected components
  - tracking: nearest-neighbor centroid linking
- This guarantees that benchmark inference can run end-to-end on all configured models/datasets.
- For research-grade comparisons, enable native modes and provide valid checkpoints/exports.

Configured in `configs/metrics.yaml` under `tracking.*` (`match_radius_px`, `pred_min_track_frames`, `mask_metrics_on_tracking`, `oom_fallback_max_frames`).

---

## 7) Results and where to find them

After a run:

- global report: `outputs/report.json`
- summary table: `outputs/summary.csv`
- per dataset/model:
  - segmentation: `outputs/<dataset>/<model>/segmentation/summary.json`
  - tracking: `outputs/<dataset>/<model>/tracking/tracking.json`

---

## 8) Notes specific to your current datasets

- `cellpose` dataset is directly usable for segmentation benchmarking.
- **CTC** (`ctc_*` keys): **`tasks: tracking`** uses TRA sub-sequences from **`man_track_filtered.json`**; **`tasks: segmentation`** uses sparse **`SEG/man_seg`**. **`cellseek`** tracking: Cellpose-SAM + Trackastra (`tracking_propagation: trackastra`). **`trackastra`/`ultrack`** default to Otsu **`fallback`** masks (weak on BF phase contrast); use **`trackastra_shared_cellseek`** for a fair linker comparison.
- `BBBC038` adapter is ready; update glob patterns/path to your local BBBC038 structure.

---

## 9) Extending the benchmark

### Add a new model

Implement `ModelAdapter` in `benchmark/models/base.py`:

- `predict_mask(image) -> instance_mask`
- `predict_tracks(frames) -> DataFrame(frame, track_id, x, y)`

Register it in `benchmark/registry.py`.

### Add a new dataset

Implement `DatasetAdapter` in `benchmark/datasets/base.py`:

- `iter_segmentation() -> SegSample`
- `iter_tracking() -> TrackSequence`

Register in `registry.py` and config in `configs/datasets.yaml`.

