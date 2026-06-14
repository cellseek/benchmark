# Cell Tracking Challenge (CTC) data layout

Place downloaded CTC training datasets here. Each dataset follows the standard CTC folder structure.

## Expected directory layout

```
data/ctc/
├── PhC-C2DL-PSC/
│   ├── 01/
│   │   ├── t000.tif
│   │   ├── t001.tif
│   │   └── ...
│   └── 01_GT/
│       ├── SEG/
│       │   ├── man_seg000.tif
│       │   └── ...
│       └── TRA/
│           ├── man_track000.tif
│           └── ...
├── PhC-C2DL-U373/
├── Fluo-C2DL-MSC/
├── DIC-C2DH-HeLa/
└── Fluo-N2DH-GOWT1/
```

## Download sources

Training datasets (with ground truth) are available from the Cell Tracking Challenge:

- http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip
- http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-U373.zip
- http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip
- http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip
- http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

Extract each zip so the dataset name folder (e.g. `PhC-C2DL-PSC/`) sits directly under `data/ctc/`.

## Sequence selection

By default the benchmark script evaluates sequence `01` from each dataset. Override with `--sequence 02` etc.

## Running the benchmark

From the `benchmark/` directory (with venv activated and GPU recommended):

```bash
python benchmark_ctc.py --datasets PhC-C2DL-PSC Fluo-C2DL-MSC
python benchmark_ctc.py --all-modes
```

Outputs:

- `results/<dataset>_01_<mode>/pred_TRA/` — predicted masks in CTC TRA format
- `results/summary.json` — SEG, DET, TRA, CTC metrics per dataset

## Supervised evaluation (optional)

To record a supervised GUI run for the paper:

1. Open CellSeek GUI and load a CTC sequence (import images from `01/`).
2. Run frame-by-frame with normal corrections.
3. Export masks or note final TRA from re-running benchmark on exported masks.
4. Record in `results/supervised_template.json`:

```json
{
  "dataset": "PhC-C2DL-PSC",
  "sequence": "01",
  "auto_tra": null,
  "supervised_tra": null,
  "time_minutes": null,
  "frames_corrected": null,
  "total_frames": null
}
```

Fill values after completing the run.
