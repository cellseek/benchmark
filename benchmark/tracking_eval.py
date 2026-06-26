"""Tracking pass helpers: fair inference, macro summaries, optional mask F1."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .metrics.segmentation import prf1_iou
from .metrics.tracking import eval_tracking_hota
from .schemas import TrackSequence


def tracking_options(metrics_cfg: dict) -> dict:
    trk = metrics_cfg.get("tracking") or {}
    oom = trk.get("oom_fallback_max_frames")
    return {
        "match_radius_px": float(trk.get("match_radius_px", 20.0)),
        "pred_min_track_frames": int(trk.get("pred_min_track_frames", 1)),
        "mask_metrics_on_tracking": bool(trk.get("mask_metrics_on_tracking", True)),
        "oom_fallback_max_frames": int(oom) if oom is not None else None,
        "frame_length_buckets": trk.get("frame_length_buckets")
        or [
            {"name": "len_le_100", "max_frames": 100},
            {"name": "len_101_300", "min_frames": 101, "max_frames": 300},
            {"name": "len_gt_300", "min_frames": 301},
        ],
    }


def frame_length_bucket(n_frames: int, buckets: list[dict]) -> str:
    n = int(n_frames)
    for spec in buckets:
        lo = int(spec.get("min_frames", 0))
        hi = spec.get("max_frames")
        if hi is not None and n <= int(hi) and n >= lo:
            return str(spec["name"])
        if hi is None and n >= lo:
            return str(spec["name"])
    return "other"


def macro_metric_mean(
    rows: list[dict],
    keys: tuple[str, ...] = ("Cell-HOTA", "DetA", "AssA"),
) -> dict:
    out: dict = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] == r[k]]
        out[k] = float(sum(vals) / len(vals)) if vals else None
    return out


def mask_f1_for_sequence(
    seq: TrackSequence,
    pred_masks: list[np.ndarray],
    *,
    iou_threshold: float,
) -> dict | None:
    if not pred_masks or not seq.gt_masks or len(pred_masks) != len(seq.frames):
        return None
    agg = {"tp": 0, "fp": 0, "fn": 0, "n_frames": 0}
    for i, gm in enumerate(seq.gt_masks):
        if gm is None:
            continue
        pm = pred_masks[i]
        if pm is None:
            continue
        m = prf1_iou(gm, pm, iou_threshold=iou_threshold)
        agg["tp"] += m["tp"]
        agg["fp"] += m["fp"]
        agg["fn"] += m["fn"]
        agg["n_frames"] += 1
    if agg["n_frames"] == 0:
        return None
    p = agg["tp"] / (agg["tp"] + agg["fp"] + 1e-8)
    r = agg["tp"] / (agg["tp"] + agg["fn"] + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {
        "mask_precision": float(p),
        "mask_recall": float(r),
        "mask_f1": float(f1),
        "mask_scored_frames": int(agg["n_frames"]),
    }


def build_tracking_summary(
    per_seq: list[dict],
    *,
    buckets: list[dict],
) -> dict:
    valid_rows = [
        r
        for r in per_seq
        if "Cell-HOTA" in r
        and "DetA" in r
        and "AssA" in r
        and r.get("Cell-HOTA") == r.get("Cell-HOTA")
    ]
    error_rows = [r for r in per_seq if "error" in r]
    if valid_rows:
        summary = macro_metric_mean(valid_rows)
    else:
        summary = {"Cell-HOTA": None, "DetA": None, "AssA": None}
    summary["aggregation"] = "macro_mean_per_sequence"
    summary["n_sequences_total"] = len(per_seq)
    summary["n_sequences_scored"] = len(valid_rows)
    summary["n_sequences_error"] = len(error_rows)

    mask_f1_vals = [
        float(r["mask_f1"])
        for r in valid_rows
        if "mask_f1" in r and r["mask_f1"] == r["mask_f1"]
    ]
    if mask_f1_vals:
        summary["mask_f1_macro_mean"] = float(sum(mask_f1_vals) / len(mask_f1_vals))

    strata: dict[str, list[dict]] = defaultdict(list)
    for r in valid_rows:
        n = int(r.get("n_frames", 0))
        strata[frame_length_bucket(n, buckets)].append(r)
        cc = r.get("constant_cell_count")
        if cc is not None:
            strata[f"cells_{int(cc)}"].append(r)
    summary["stratified"] = {
        name: {**macro_metric_mean(rows), "n_sequences": len(rows)}
        for name, rows in sorted(strata.items())
    }
    return summary


def infer_tracks_for_sequence(
    model_adapter,
    seq: TrackSequence,
    *,
    oom_fallback_max_frames: int | None,
) -> tuple[pd.DataFrame, list[np.ndarray] | None, TrackSequence]:
    seq_for_infer = seq
    pred_masks: list[np.ndarray] | None = None

    def _run(s: TrackSequence) -> pd.DataFrame:
        nonlocal pred_masks
        if hasattr(model_adapter, "predict_tracks_sequence"):
            return model_adapter.predict_tracks_sequence(s)
        if hasattr(model_adapter, "predict_tracks_returning_masks"):
            df, pred_masks = model_adapter.predict_tracks_returning_masks(s.frames)
            return df
        return model_adapter.predict_tracks(s.frames)

    try:
        pred_tracks = _run(seq_for_infer)
    except RuntimeError as e:
        msg = str(e).lower()
        is_oom = "out of memory" in msg and "cuda" in msg
        if not is_oom or oom_fallback_max_frames is None:
            raise
        max_frames = int(oom_fallback_max_frames)
        if len(seq.frames) <= max_frames:
            raise
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gt = seq.gt_tracks
        gt_small = gt[gt["frame"] < max_frames].copy() if gt is not None else None
        gm_small = list(seq.gt_masks[:max_frames]) if seq.gt_masks is not None else None
        meta = {**(seq.meta or {}), "oom_fallback_max_frames": max_frames}
        seq_for_infer = TrackSequence(
            seq_id=seq.seq_id,
            frames=seq.frames[:max_frames],
            gt_masks=gm_small,
            gt_tra_masks=list(seq.gt_tra_masks[:max_frames]) if seq.gt_tra_masks else None,
            gt_tracks=gt_small,
            meta=meta,
        )
        pred_masks = None
        pred_tracks = _run(seq_for_infer)
    else:
        return pred_tracks, pred_masks, seq_for_infer

    if not isinstance(pred_tracks, pd.DataFrame):
        pred_tracks = pd.DataFrame(pred_tracks)
    return pred_tracks, pred_masks, seq_for_infer


def score_sequence(
    seq: TrackSequence,
    seq_infer: TrackSequence,
    pred_tracks: pd.DataFrame,
    pred_masks: list[np.ndarray] | None,
    *,
    match_radius_px: float,
    pred_min_track_frames: int,
    mask_metrics_on_tracking: bool,
    iou_threshold: float,
    model_adapter,
) -> dict:
    meta = seq.meta or {}
    row: dict = {
        "seq_id": seq.seq_id,
        "n_frames": len(seq_infer.frames),
        "constant_cell_count": meta.get("constant_cell_count"),
        "start_frame": meta.get("start_frame"),
        "end_frame": meta.get("end_frame"),
    }
    if callable(getattr(model_adapter, "get_last_tracking_stats", None)):
        row["inference_stats"] = model_adapter.get_last_tracking_stats()
    scores = eval_tracking_hota(
        seq_infer.gt_tracks,
        pred_tracks,
        match_radius_px=match_radius_px,
        pred_min_track_frames=pred_min_track_frames,
    )
    row.update(scores)
    if mask_metrics_on_tracking and pred_masks is not None:
        mf = mask_f1_for_sequence(seq_infer, pred_masks, iou_threshold=iou_threshold)
        if mf:
            row.update(mf)
    return row
