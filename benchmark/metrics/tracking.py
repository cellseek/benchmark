"""HOTA-style tracking metrics on point tracks (DetA / AssA / Cell-HOTA)."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_TRACKING_KEYS = ("Cell-HOTA", "DetA", "AssA")


def _validate_columns(df: pd.DataFrame, name: str):
    required = {"frame", "track_id", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _frame_match(gt_f: pd.DataFrame, pr_f: pd.DataFrame, match_radius_px: float):
    """Hungarian matching with Euclidean distance threshold."""
    if len(gt_f) == 0 or len(pr_f) == 0:
        return []

    gxy = gt_f[["x", "y"]].to_numpy(dtype=float)
    pxy = pr_f[["x", "y"]].to_numpy(dtype=float)
    diff = gxy[:, None, :] - pxy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))

    big = 1e9
    cost = dist.copy()
    cost[cost > match_radius_px] = big

    gi, pi = linear_sum_assignment(cost)
    matched = []
    for g_idx, p_idx in zip(gi, pi):
        if cost[g_idx, p_idx] < big:
            g_tid = int(gt_f.iloc[g_idx]["track_id"])
            p_tid = int(pr_f.iloc[p_idx]["track_id"])
            matched.append((g_tid, p_tid))
    return matched


def filter_ephemeral_pred_tracks(
    pred_tracks: pd.DataFrame | None,
    min_frames: int,
) -> pd.DataFrame:
    """
    Drop predicted tracks that appear in fewer than ``min_frames`` frames.

    Applied uniformly to all models when ``metrics.tracking.pred_min_track_frames`` > 1.
    """
    if pred_tracks is None or len(pred_tracks) == 0 or min_frames <= 1:
        if pred_tracks is None:
            return pd.DataFrame(columns=["frame", "track_id", "x", "y"])
        return pred_tracks.copy()
    counts = pred_tracks.groupby("track_id")["frame"].nunique()
    keep = set(int(tid) for tid, c in counts.items() if int(c) >= int(min_frames))
    if not keep:
        return pred_tracks.iloc[0:0].copy()
    out = pred_tracks[pred_tracks["track_id"].isin(keep)].copy()
    return out


def eval_tracking_hota(
    gt_tracks: pd.DataFrame,
    pred_tracks: pd.DataFrame,
    match_radius_px: float = 20.0,
    *,
    pred_min_track_frames: int = 1,
) -> dict:
    """
    Compute DetA / AssA / Cell-HOTA from point tracks.

    Cell-HOTA = sqrt(DetA * AssA), the standard HOTA combination on centroid tracks
    (frame-wise Hungarian matching with ``match_radius_px``).

    This is the benchmark's historical scoring method; it is not the division-aware
    Cell-HOTA metric from O'Connor & Dunlop (2025).
    """
    empty = {k: np.nan for k in _TRACKING_KEYS}
    if gt_tracks is None or len(gt_tracks) == 0:
        return empty

    if pred_tracks is None:
        pred_tracks = pd.DataFrame(columns=["frame", "track_id", "x", "y"])
    pred_tracks = filter_ephemeral_pred_tracks(pred_tracks, int(pred_min_track_frames))

    _validate_columns(gt_tracks, "gt_tracks")
    _validate_columns(pred_tracks, "pred_tracks")

    gt_tracks = gt_tracks.copy()
    pred_tracks = pred_tracks.copy()
    gt_tracks["frame"] = gt_tracks["frame"].astype(int)
    pred_tracks["frame"] = pred_tracks["frame"].astype(int)

    frames = sorted(set(gt_tracks["frame"].tolist()) | set(pred_tracks["frame"].tolist()))

    tp = fp = fn = 0
    matched_pairs = []
    pair_count: dict = {}
    gt_match_count: dict = {}
    pr_match_count: dict = {}

    for fr in frames:
        gt_f = gt_tracks[gt_tracks["frame"] == fr]
        pr_f = pred_tracks[pred_tracks["frame"] == fr]
        matches = _frame_match(gt_f, pr_f, match_radius_px=match_radius_px)
        matched_pairs.extend(matches)

        frame_tp = len(matches)
        tp += frame_tp
        fp += len(pr_f) - frame_tp
        fn += len(gt_f) - frame_tp

        for g, p in matches:
            pair_count[(g, p)] = pair_count.get((g, p), 0) + 1
            gt_match_count[g] = gt_match_count.get(g, 0) + 1
            pr_match_count[p] = pr_match_count.get(p, 0) + 1

    deta = tp / max(1, (tp + fp + fn))

    if tp == 0:
        assa = 0.0
    else:
        assoc_sum = 0.0
        for g, p in matched_pairs:
            m_gp = pair_count[(g, p)]
            m_g = gt_match_count[g]
            m_p = pr_match_count[p]
            assoc = m_gp / max(1, (m_g + m_p - m_gp))
            assoc_sum += assoc
        assa = assoc_sum / tp

    cell_hota = float(np.sqrt(max(0.0, deta) * max(0.0, assa)))

    return {
        "Cell-HOTA": cell_hota,
        "DetA": float(deta),
        "AssA": float(assa),
    }
