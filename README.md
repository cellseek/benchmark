# CellSeek Benchmark

`cellseek_benchmark` is a unified benchmark repository under `cellseek/` for evaluating segmentation and tracking models on open microscopy datasets.

It currently supports:

- Segmentation metrics: `Precision`, `Recall`, `F1` (instance-level IoU matching)
- Tracking metrics: `TA`, `HOTA`, `DetA`, `AssA` (track association aware)

---

## 1) What this benchmark includes

### Datasets (configured)

- `cellpose` (segmentation)
- `BBBC038` (segmentation; configure local folder path)
- Every **`ctc_*`** entry (Cell Tracking Challenge layout): **tracking metrics only** — no segmentation pass on these datasets (standard TRA/SEG inputs).

### Model adapters

- `cellseek` (enabled): default **`tracking_propagation: cutie`** uses CellSAM on the first frame and **`cellsam.MaskTracker`** for propagation. Place weights under **`cellsam/weights/`** (`cpsam`, `cutie-base-mega.pth`) or set **`checkpoint_path`** / **`cutie_weights_path`**. Use **`cellsam`** only for the legacy ablation (CellSAM on every frame).
- `cellpose` (enabled)
- `sam3` (enabled, native-or-fallback mode)
- `microsam` (enabled, checkpoint-aware native-or-fallback mode)
- `trackastra` (optional): transformer linking from a timelapse + per-frame instance masks; install **`requirements-extra-tracking.txt`** (`pip install trackastra`). Configure **`pretrained`** / **`track_mode`** / **`device`** in `configs/models.yaml`.
- `ultrack` (optional): global linking from a label timelapse built via the same generic mask generator as fallbacks; install **`ultrack`** and a solver backend per [Ultrack docs](https://ultrack.readthedocs.io/). Tune **`linking_max_distance`**, **`solver_name`**, **`time_limit`**.
- `celltractr` (optional): **does not call** the Cell-TRACTR Trackformer repo inside `csbench`. It **`delegate_tracker: trackastra`** by default so you keep the same `csbench` CLI while using Trackastra linking; run native Cell-TRACTR separately via `models/Cell-TRACTR/src/pipeline.py`.

### Output artifacts

Runs produce:

- per-task JSON files in `outputs/<dataset>/<model>/...`
- consolidated JSON report: `outputs/report.json`
- flat CSV summary table: `outputs/summary.csv`

---

## 2) Repository structure

```text
cellseek_benchmark/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    datasets.yaml
    models.yaml
    metrics.yaml
    experiments/
      cellseek_ctc_bf_c2dl_hsc_tracking.yaml
      …
  cellseek_benchmark/
    cli.py
    test.py
    registry.py
    datasets/
    models/
    metrics/
```

---

## 3) Environment setup (full instructions)

You can use either Conda or Python venv.

**Python version:** the benchmark package targets **CPython 3.10+** and is routinely usable on **3.13** (same `requires-python` as `pyproject.toml`). Heavy optional stacks (PyTorch, CellPose, SAM, Fiji) must still support your chosen interpreter—install those in the same environment you use for `csbench`.

### Option A: Conda (recommended)

```bash
cd /home/fzhaoai/cellseek/cellseek_benchmark
conda create -n cellseek-benchmark python=3.13 -y
conda activate cellseek-benchmark
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-cellseek-tracking.txt
# Optional Trackastra / Ultrack models: pip install -r requirements-extra-tracking.txt
pip install -e .
```

The second requirements file installs **`cellsam`** and the CUTIE backend (`pip install -e ../cellsam` and `pip install -e ../cutie`). Place **`cpsam`** and **`cutie-base-mega.pth`** under **`cellseek/cellsam/weights/`** (the GUI downloads them there on first launch), or set **`CELLSEEK_CPSAM_PATH`** / **`CELLSEEK_CUTIE_PATH`**.

Verify install:

```bash
python -c "import cellseek_benchmark; print('ok')"
csbench --help
```

### Option B: Python venv

```bash
cd /home/fzhaoai/cellseek/cellseek_benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-cellseek-tracking.txt
# Optional Trackastra / Ultrack models: pip install -r requirements-extra-tracking.txt
pip install -e .
```

Verify:

```bash
python -c "import cellseek_benchmark; print('ok')"
csbench --help
```

---

## 4) Configure datasets and models

Edit:

- `configs/datasets.yaml` (catalog of dataset entries; you choose **one** key per run)
- `configs/models.yaml` (catalog of model entries; you choose **one** key per run)
- `configs/metrics.yaml`
- **`configs/experiments/*.yaml`** — one file per run: `experiment_id`, `dataset`, `model`, `tasks`, optional `output_parent` / `run_name` (results go to `<output_parent>/<run_name>_<YYYYMMDD_HHMMSS>/`, default parent `experiments`), plus optional `datasets_config` / `models_config` / `metrics_config` (filenames relative to `configs/`)

Minimum required updates for your machine:

- set `BBBC038` root path if present
- default `datasets.yaml` uses **`/project/cellseek/cell_tracking`** for **CTC** entries (challenge folders such as `BF-C2DL-HSC` directly under that root); adjust `cellpose` paths if needed
- enable/disable models in `configs/models.yaml`
- for `cellseek` native mode, CPSAM weights are resolved via **`cellsam.resolve_cpsam_path`**. If `cellseek.checkpoint_path` is **null**, searches **`CELLSEEK_CPSAM_PATH`**, **`cellsam/weights/cpsam`**, or **`./weights/cpsam`**. For **tracking on CTC**, also place propagation weights (`cutie-base-mega.pth`) beside CPSAM or set **`cutie_weights_path`** / **`CELLSEEK_CUTIE_PATH`**.
- for `microsam` native mode, set a valid fine-tuned `checkpoint_path` (`checkpoint.pth`)
- **CTC segmentation** uses sparse **`SEG/man_seg`** GT (`ctc_bf_c2dl_hsc`, etc.). **CTC tracking** uses TRA centroids via `tasks: tracking`.

---

## 5) Running the benchmark

Each invocation evaluates **one** dataset and **one** model (see `cellseek_benchmark/test.py`). **Pick exactly one experiment YAML** under `configs/experiments/` (or add your own).

### Run an experiment file

Default preset is CellSeek × CTC (`ctc_bf_c2dl_hsc`) × tracking:

```bash
cd /home/fzhaoai/cellseek/cellseek_benchmark
csbench --config-dir configs
```

Equivalent explicit path:

```bash
csbench --config-dir configs \
  --benchmark-config experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml
```

Copy `configs/experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml` to a new name and edit `dataset`, `model`, `tasks`, and optionally `experiment_id` / `output_parent` for additional experiments (each run creates a new timestamped folder).

### Override dataset, model, or tasks without editing YAML

```bash
csbench --config-dir configs \
  --benchmark-config experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml \
  --dataset cellpose \
  --model cellseek \
  --tasks both
```

`--tasks` must be one of: `segmentation`, `tracking`, `both`.

### Common focused runs

CTC segmentation (SEG GT, Omnipose):

```bash
python -m cellseek_benchmark.test --config-dir configs \
  --benchmark-config experiments/omnipose_ctc_bf_c2dl_hsc_segmentation.yaml
```

CTC segmentation (SEG GT, CellSAM):

```bash
python -m cellseek_benchmark.test --config-dir configs \
  --benchmark-config experiments/cellsam_seg_ctc_bf_c2dl_hsc_segmentation.yaml
```

Slurm: `sbatch sbatch.sh` (Omnipose) or `sbatch sbatch_segmentation.sh` (CellSAM).

Tracking only (CTC):

```bash
csbench --config-dir configs --tasks tracking
```

### Smaller on-disk subset for debugging

Use `scripts/build_benchmark_subset.py` to mirror a tiny slice of each dataset under a folder (for example `benchmark_subset_10/`). Copy `configs/` (including `experiments/`) to a new directory, point `datasets.yaml` roots at that subset, then run `csbench --config-dir <that_directory> --benchmark-config experiments/<your_experiment>.yaml`.

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
3. frame-wise Hungarian matching (`match_radius_px` is **global**, do not tune per model)
4. `DetA`, `AssA`, `HOTA`; optional **mask F1** on frames with sparse SEG GT
5. `tracking.json` includes **`stratified`** buckets by segment length and `constant_cell_count`

**Leaderboards:**

- **End-to-end (main):** each model’s default pipeline (`cellseek` = CellSAM + CUTIE; `sam3` = video propagate; `microsam` = per-frame SAM + linking; `trackastra`/`ultrack` = Otsu fallback masks + linker unless configured otherwise).
- **Linker sub-board (optional):** `trackastra_shared_cellseek` / `ultrack_shared_cellseek` use **CellSAM masks** then compare linking only.

**cellseek tracking defaults:** `tracking_propagation: cutie`, seed search in the first *k* frames, fresh CUTIE state per sub-sequence, optional early stop on consecutive empty masks. No GT seeding; no per-model `match_radius_px` for HOTA.

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
- **CTC** (`ctc_*` keys): **`tasks: tracking`** uses TRA sub-sequences from **`man_track_filtered.json`**; **`tasks: segmentation`** uses sparse **`SEG/man_seg`**. **`cellseek`** tracking: CellSAM seed + CUTIE (`tracking_propagation: cutie`). **`trackastra`/`ultrack`** default to Otsu **`fallback`** masks (weak on BF phase contrast); use **`trackastra_shared_cellseek`** for a fair linker comparison.
- `BBBC038` adapter is ready; update glob patterns/path to your local BBBC038 structure.

---

## 9) Extending the benchmark

### Add a new model

Implement `ModelAdapter` in `cellseek_benchmark/models/base.py`:

- `predict_mask(image) -> instance_mask`
- `predict_tracks(frames) -> DataFrame(frame, track_id, x, y)`

Register it in `cellseek_benchmark/registry.py`.

### Add a new dataset

Implement `DatasetAdapter` in `cellseek_benchmark/datasets/base.py`:

- `iter_segmentation() -> SegSample`
- `iter_tracking() -> TrackSequence`

Register in `registry.py` and config in `configs/datasets.yaml`.

