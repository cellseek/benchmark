#!/usr/bin/env python3
"""
Headless CellSeek benchmark on Cell Tracking Challenge datasets.

Runs the production segment-then-link pipeline (Cellpose-SAM + Trackastra)
and evaluates against ground-truth masks when available.

Usage (from cellseek/benchmark/):
    python benchmark_ctc.py
    python benchmark_ctc.py --datasets PhC-C2DL-PSC --max-frames 10
    python benchmark_ctc.py --all-modes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_BENCHMARK_ROOT = Path(__file__).resolve().parent
_CELLSEEK_ROOT = _BENCHMARK_ROOT.parent
_GUI_ROOT = _CELLSEEK_ROOT / "gui"

if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from ctc_metrics import evaluate_sequence  # noqa: E402
from utils.cellsam_segment import (
    prepare_rgb_for_segmentation,  # noqa: E402
    segment_rgb,
)
from utils.trackastra_tracking import link_masks_with_trackastra  # noqa: E402

DEFAULT_DATASETS = [
    "PhC-C2DL-PSC",
    "PhC-C2DL-U373",
    "Fluo-C2DL-MSC",
    "DIC-C2DH-HeLa",
    "Fluo-N2DH-GOWT1",
]

DEFAULT_DATA_DIR = _BENCHMARK_ROOT / "data" / "ctc"
RESULTS_DIR = _BENCHMARK_ROOT / "results"


def _load_tif(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def discover_frames(seq_dir: Path) -> List[Path]:
    frames = sorted(seq_dir.glob("t*.tif"))
    if not frames:
        frames = sorted(seq_dir.glob("t*.png"))
    return frames


def discover_gt_tra(gt_dir: Path, n_frames: int) -> List[Path]:
    tra_dir = gt_dir / "TRA"
    seg_dir = gt_dir / "SEG"
    paths = []
    for i in range(n_frames):
        for folder, prefix in ((tra_dir, "man_track"), (seg_dir, "man_seg")):
            p = folder / f"{prefix}{i:03d}.tif"
            if p.exists():
                paths.append(p)
                break
        else:
            paths.append(Path())
    return paths


def run_pipeline(
    images: List[np.ndarray],
    gt_masks: Optional[List[np.ndarray]] = None,
    mode: str = "full",
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Modes:
      full     — segment-then-link (production CellSeek auto)
      seg-only — Cellpose-SAM per frame, no identity linking
      link-gt  — Trackastra linking on GT segmentations (linker ceiling)
    """
    masks: List[np.ndarray] = []
    prev_image: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None

    for idx, rgb in enumerate(images):
        t0 = time.time()
        if mode == "link-gt":
            assert gt_masks is not None
            raw = gt_masks[idx].astype(np.uint16)
        elif mode == "seg-only" or idx == 0:
            raw = segment_rgb(rgb)
        else:
            assert prev_image is not None and prev_mask is not None
            raw = segment_rgb(rgb)

        if mode == "seg-only" or (mode == "full" and idx == 0):
            mask = raw
        elif mode == "link-gt" and idx == 0:
            mask = raw
        elif mode in ("full", "link-gt"):
            assert prev_image is not None and prev_mask is not None
            mask = link_masks_with_trackastra(prev_image, prev_mask, rgb, raw)
        else:
            mask = raw

        masks.append(mask.astype(np.uint16))
        prev_image = rgb
        prev_mask = mask
        if verbose:
            elapsed = time.time() - t0
            n_cells = len([i for i in np.unique(mask) if i > 0])
            print(f"  frame {idx:03d}: {n_cells} cells ({elapsed:.1f}s) [{mode}]")

    return masks


def save_ctc_masks(masks: List[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        cv2.imwrite(str(out_dir / f"man_track{i:03d}.tif"), mask)


def evaluate_dataset(
    data_dir: Path,
    dataset: str,
    sequence: str = "01",
    max_frames: Optional[int] = None,
    skip_inference: bool = False,
    mode: str = "full",
) -> Dict:
    seq_dir = data_dir / dataset / sequence
    gt_dir = data_dir / dataset / f"{sequence}_GT"

    if not seq_dir.is_dir():
        return {
            "dataset": dataset,
            "sequence": sequence,
            "status": "missing_sequence",
            "path": str(seq_dir),
        }

    frame_paths = discover_frames(seq_dir)
    if not frame_paths:
        return {
            "dataset": dataset,
            "sequence": sequence,
            "status": "no_frames",
            "path": str(seq_dir),
        }

    if max_frames:
        frame_paths = frame_paths[:max_frames]

    images = [_to_rgb(_load_tif(p)) for p in frame_paths]
    images = [prepare_rgb_for_segmentation(img) for img in images]

    out_dir = RESULTS_DIR / f"{dataset}_{sequence}_{mode}"
    pred_dir = out_dir / "pred_TRA"

    gt_masks_for_pipeline: Optional[List[np.ndarray]] = None
    if gt_dir.is_dir():
        gt_paths = discover_gt_tra(gt_dir, len(frame_paths))
        gt_masks_for_pipeline = []
        for p in gt_paths:
            if p.exists():
                gt_masks_for_pipeline.append(_load_tif(p))
            else:
                gt_masks_for_pipeline.append(np.zeros((1, 1), dtype=np.uint16))

    if skip_inference and pred_dir.is_dir() and list(pred_dir.glob("man_track*.tif")):
        pred_masks = [_load_tif(p) for p in sorted(pred_dir.glob("man_track*.tif"))]
    else:
        print(f"\n=== {dataset}/{sequence} ({len(images)} frames, mode={mode}) ===")
        pred_masks = run_pipeline(images, gt_masks=gt_masks_for_pipeline, mode=mode)
        save_ctc_masks(pred_masks, pred_dir)

    result: Dict = {
        "dataset": dataset,
        "sequence": sequence,
        "mode": mode,
        "status": "ok",
        "num_frames": len(pred_masks),
        "pred_dir": str(pred_dir),
    }

    if gt_dir.is_dir() and gt_masks_for_pipeline is not None:
        gt_masks = gt_masks_for_pipeline[: len(pred_masks)]
        metrics = evaluate_sequence(pred_masks, gt_masks)
        result.update(metrics)
        print(
            f"  SEG={metrics['SEG']:.3f}  DET={metrics['DET']:.3f}  "
            f"TRA={metrics['TRA']:.3f}  CTC={metrics['CTC']:.3f}"
        )
    else:
        result["status"] = "no_ground_truth"
        print(f"  (no GT at {gt_dir})")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CellSeek CTC benchmark")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing CTC datasets (default: benchmark/data/ctc)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset folder names to evaluate",
    )
    parser.add_argument("--sequence", default="01", help="Sequence id (default 01)")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit frames per sequence"
    )
    parser.add_argument(
        "--mode",
        choices=("full", "seg-only", "link-gt"),
        default="full",
        help="Pipeline mode: full (default), seg-only, or link-gt",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run full, seg-only, and link-gt ablations",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Reuse previously saved predictions in results/",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    modes = ("full", "seg-only", "link-gt") if args.all_modes else (args.mode,)

    for dataset in args.datasets:
        for mode in modes:
            result = evaluate_dataset(
                args.data_dir,
                dataset,
                sequence=args.sequence,
                max_frames=args.max_frames,
                skip_inference=args.skip_inference,
                mode=mode,
            )
            all_results.append(result)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nWrote {summary_path}")
    ok = [r for r in all_results if r.get("status") == "ok" and "SEG" in r]
    if ok:
        for metric in ("SEG", "DET", "TRA", "CTC"):
            vals = [r[metric] for r in ok]
            print(f"Mean {metric}: {np.mean(vals):.3f} ({len(vals)} datasets)")
    else:
        print(
            "No datasets with ground truth evaluated. "
            f"Copy CTC data to {DEFAULT_DATA_DIR} (see data/ctc/README.md)."
        )


if __name__ == "__main__":
    main()
