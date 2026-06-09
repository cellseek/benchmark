#!/usr/bin/env python3
"""
Compare-mode tracking run for CellSeek:
- frame 0 mask is taken from GT directly (no frame-0 segmentation inference)
- subsequent frames are tracked from that seed

This is useful for isolating propagation/linking behavior from first-frame
segmentation quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import logging

import numpy as np

from cellseek_benchmark.io_utils import dump_json, ensure_dir, load_yaml
from cellseek_benchmark.metrics.tracking import eval_tracking_hota
from cellseek_benchmark.registry import DATASET_REGISTRY, MODEL_REGISTRY


def _resolve_single_run(bench_cfg: dict, dataset: str | None, model: str | None) -> tuple[str, str]:
    ds = (dataset or bench_cfg.get("dataset") or "").strip()
    md = (model or bench_cfg.get("model") or "").strip()
    if not ds:
        raise ValueError("Experiment config must define dataset, or pass --dataset.")
    if not md:
        raise ValueError("Experiment config must define model, or pass --model.")
    return ds, md


def _load_all_gt_masks(seq, n_frames: int) -> list[np.ndarray | None]:
    """
    Preload all GT masks aligned to frames (None for frames without GT masks).
    """
    if seq.gt_masks is None:
        raise ValueError(
            f"Sequence {seq.seq_id!r} has no gt_masks list; cannot run GT-seeded compare mode."
        )
    out: list[np.ndarray | None] = [None] * n_frames
    for i in range(min(n_frames, len(seq.gt_masks))):
        m = seq.gt_masks[i]
        if m is not None:
            out[i] = np.asarray(m).astype(np.int32, copy=False)
    return out


def _find_seed_from_loaded_gt(seq_id: str, gt_masks: list[np.ndarray | None]) -> tuple[int, np.ndarray]:
    for i, m in enumerate(gt_masks):
        if m is not None:
            return i, m
    raise ValueError(
        f"Sequence {seq_id!r} has no GT mask on any frame; cannot run GT-seeded compare mode."
    )


def _run_cellseek_gt_seed_sequence(model_adapter, seq):
    if not getattr(model_adapter, "native", False):
        raise RuntimeError("Expected native CellSeek adapter instance.")

    frames = seq.frames
    if not frames:
        return []

    loaded_gt_masks = _load_all_gt_masks(seq, len(frames))
    seed_idx, seed_mask = _find_seed_from_loaded_gt(seq.seq_id, loaded_gt_masks)
    pred_masks: list[np.ndarray] = [
        np.zeros(frames[i].shape[:2], dtype=np.int32) for i in range(len(frames))
    ]
    pred_masks[seed_idx] = seed_mask.copy()

    # Match CellSeek adapter's propagation choice.
    if getattr(model_adapter, "_tracking_propagation", "cellsam") == "cutie":
        logging.getLogger("cutie.inference.inference_core").setLevel(logging.ERROR)
        tracker = model_adapter._new_mask_tracker()
        prev_rgb = model_adapter._to_cutie_rgb_uint8(frames[seed_idx])
        prev_lab = seed_mask.astype(np.int32, copy=False)
        for t in range(seed_idx + 1, len(frames)):
            gt_t = loaded_gt_masks[t]
            if gt_t is not None:
                pred_masks[t] = gt_t.copy()
                prev_rgb = model_adapter._to_cutie_rgb_uint8(frames[t])
                prev_lab = gt_t.astype(np.int32, copy=False)
                continue

            # No object to propagate: skip tracker call to avoid CUTIE warning spam.
            if int(np.max(prev_lab)) <= 0:
                pred_masks[t] = np.zeros(frames[t].shape[:2], dtype=np.int32)
                prev_rgb = model_adapter._to_cutie_rgb_uint8(frames[t])
                continue

            rgb_u8 = model_adapter._to_cutie_rgb_uint8(frames[t])
            cu = tracker.track(prev_rgb, prev_lab, rgb_u8)
            lab = np.asarray(cu).astype(np.int32, copy=False)
            pred_masks[t] = lab.copy()
            prev_rgb = rgb_u8
            prev_lab = lab
    else:
        for t in range(seed_idx + 1, len(frames)):
            # Non-CUTIE mode: if GT mask exists, use it directly; otherwise predict.
            if loaded_gt_masks[t] is not None:
                lab = loaded_gt_masks[t]
            else:
                lab = model_adapter.predict_mask(frames[t])
            pred_masks[t] = np.asarray(lab).astype(np.int32, copy=False)

    return pred_masks


def main():
    p = argparse.ArgumentParser(
        description="CellSeek compare run: use GT mask on frame 0, then track."
    )
    p.add_argument("--config-dir", type=Path, default=Path("configs"))
    p.add_argument(
        "--benchmark-config",
        type=str,
        default="experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml",
    )
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--match-radius-px", type=float, default=20.0)
    p.add_argument("--output-parent", type=Path, default=Path("experiments"))
    p.add_argument(
        "--max-tracking-sequences",
        type=int,
        default=None,
        help="Optional cap on number of tracking sequences to load from dataset.",
    )
    p.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=None,
        help="Optional cap on frames per sequence to speed up debug/compare runs.",
    )
    args = p.parse_args()

    config_dir = args.config_dir.resolve()
    bc = Path(args.benchmark_config)
    bench_path = bc.resolve() if bc.is_absolute() else (config_dir / bc).resolve()
    bench_cfg = load_yaml(bench_path)
    datasets_cfg = load_yaml(config_dir / Path(bench_cfg.get("datasets_config", "datasets.yaml")).name)["datasets"]
    models_cfg = load_yaml(config_dir / Path(bench_cfg.get("models_config", "models.yaml")).name)["models"]

    dataset_name, model_name = _resolve_single_run(bench_cfg, args.dataset, args.model)
    if model_name != "cellseek":
        raise ValueError(
            f"compare_test.py is for CellSeek only; got model={model_name!r}. "
            "Pass --model cellseek or set model: cellseek in experiment YAML."
        )

    if dataset_name not in datasets_cfg:
        raise ValueError(f"Unknown dataset {dataset_name!r}")
    if model_name not in models_cfg:
        raise ValueError(f"Unknown model {model_name!r}")

    ds_type = datasets_cfg[dataset_name]["type"]
    md_type = models_cfg[model_name]["type"]
    ds_cfg = dict(datasets_cfg[dataset_name])
    if args.max_tracking_sequences is not None and int(args.max_tracking_sequences) > 0:
        ds_cfg["max_tracking_sequences"] = int(args.max_tracking_sequences)
    if args.max_frames_per_sequence is not None and int(args.max_frames_per_sequence) > 0:
        ds_cfg["max_frames_per_sequence"] = int(args.max_frames_per_sequence)

    dataset_adapter = DATASET_REGISTRY[ds_type](ds_cfg)
    model_adapter = MODEL_REGISTRY[md_type](models_cfg[model_name])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_parent / f"compare_gtseed_{dataset_name}_{stamp}"
    out_dir = out_root / dataset_name / model_name / "tracking"
    ensure_dir(out_dir)

    per_seq = []
    for seq in dataset_adapter.iter_tracking():
        if seq.gt_tracks is None or len(seq.frames) == 0:
            continue
        try:
            pred_masks = _run_cellseek_gt_seed_sequence(model_adapter, seq)
            pred_tracks = model_adapter._tracks_df_from_masks(pred_masks)
            scores = eval_tracking_hota(
                seq.gt_tracks, pred_tracks, match_radius_px=float(args.match_radius_px)
            )
            per_seq.append({"seq_id": seq.seq_id, **scores})
            print(
                f"compare_test: {seq.seq_id} HOTA={scores['HOTA']:.4f} "
                f"DetA={scores['DetA']:.4f} AssA={scores['AssA']:.4f}",
                flush=True,
            )
        except Exception as e:
            per_seq.append({"seq_id": seq.seq_id, "error": str(e)})
            print(f"compare_test: {seq.seq_id} ERROR: {e}", flush=True)

    valid = [r for r in per_seq if "HOTA" in r]
    if valid:
        summary = {
            "HOTA": float(sum(r["HOTA"] for r in valid) / len(valid)),
            "DetA": float(sum(r["DetA"] for r in valid) / len(valid)),
            "AssA": float(sum(r["AssA"] for r in valid) / len(valid)),
            "TA": float(sum(r["TA"] for r in valid) / len(valid)),
            "n_sequences_total": len(per_seq),
            "n_sequences_scored": len(valid),
            "n_sequences_error": len(per_seq) - len(valid),
        }
    else:
        summary = {
            "HOTA": None,
            "DetA": None,
            "AssA": None,
            "TA": None,
            "n_sequences_total": len(per_seq),
            "n_sequences_scored": 0,
            "n_sequences_error": len(per_seq),
        }

    dump_json({"summary": summary, "per_sequence": per_seq}, out_dir / "tracking.json")
    dump_json(
        {
            "mode": "gt_seed_frame0",
            "dataset": dataset_name,
            "model": model_name,
            "benchmark_config": str(bench_path),
            "match_radius_px": float(args.match_radius_px),
        },
        out_root / "run_meta.json",
    )
    print(f"compare_test: wrote {out_dir / 'tracking.json'}", flush=True)


if __name__ == "__main__":
    main()

