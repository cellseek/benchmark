import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _validate_columns(df: pd.DataFrame, name: str):
    required = {"frame", "track_id", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _frame_match(gt_f: pd.DataFrame, pr_f: pd.DataFrame, match_radius_px: float):
    """
    Hungarian matching with Euclidean distance threshold.
    Returns list of matched (gt_track_id, pred_track_id).
    """
    if len(gt_f) == 0 or len(pr_f) == 0:
        return []

    gxy = gt_f[["x", "y"]].to_numpy(dtype=float)
    pxy = pr_f[["x", "y"]].to_numpy(dtype=float)

    # pairwise euclidean distances
    diff = gxy[:, None, :] - pxy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))

    # Large penalty for invalid pairs so Hungarian avoids them
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


def eval_tracking_hota(
    gt_tracks: pd.DataFrame,
    pred_tracks: pd.DataFrame,
    match_radius_px: float = 20.0,
) -> dict:
    """
    Compute DetA / AssA / HOTA from point tracks.

    This is a direct implementation of HOTA-style decomposition using:
    - frame-wise Hungarian detection matching (distance threshold)
    - association Jaccard consistency over matched track pairs

    Expected canonical columns:
      - frame
      - track_id
      - x
      - y
    """
    if gt_tracks is None or len(gt_tracks) == 0:
        return {"HOTA": np.nan, "DetA": np.nan, "AssA": np.nan, "TA": np.nan}

    if pred_tracks is None:
        pred_tracks = pd.DataFrame(columns=["frame", "track_id", "x", "y"])

    _validate_columns(gt_tracks, "gt_tracks")
    _validate_columns(pred_tracks, "pred_tracks")

    gt_tracks = gt_tracks.copy()
    pred_tracks = pred_tracks.copy()
    gt_tracks["frame"] = gt_tracks["frame"].astype(int)
    pred_tracks["frame"] = pred_tracks["frame"].astype(int)

    frames = sorted(set(gt_tracks["frame"].tolist()) | set(pred_tracks["frame"].tolist()))

    tp = 0
    fp = 0
    fn = 0

    # per-frame matched (g,p) track ids
    matched_pairs = []
    # counts for association decomposition
    pair_count = {}  # (g,p) -> matched times
    gt_match_count = {}  # g -> matched times
    pr_match_count = {}  # p -> matched times

    for fr in frames:
        gt_f = gt_tracks[gt_tracks["frame"] == fr]
        pr_f = pred_tracks[pred_tracks["frame"] == fr]
        matches = _frame_match(gt_f, pr_f, match_radius_px=match_radius_px)
        matched_pairs.extend(matches)

        frame_tp = len(matches)
        frame_fp = len(pr_f) - frame_tp
        frame_fn = len(gt_f) - frame_tp

        tp += frame_tp
        fp += frame_fp
        fn += frame_fn

        for g, p in matches:
            pair_count[(g, p)] = pair_count.get((g, p), 0) + 1
            gt_match_count[g] = gt_match_count.get(g, 0) + 1
            pr_match_count[p] = pr_match_count.get(p, 0) + 1

    deta = tp / max(1, (tp + fp + fn))

    # Association: average Jaccard over all TP matches
    # For pair (g,p): assoc = m_gp / (m_g + m_p - m_gp)
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

    hota = float(np.sqrt(max(0.0, deta) * max(0.0, assa)))
    # Tracking Accuracy alias for convenience
    ta = hota

    return {"HOTA": float(hota), "DetA": float(deta), "AssA": float(assa), "TA": float(ta)}
