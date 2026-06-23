"""Utilities for converting masks and centroids to benchmark track tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .tracking_schema import TRACK_COLUMNS, empty_tracks_df

Point = tuple[float, float]


def points_from_label_mask(mask: np.ndarray) -> list[Point]:
    """Return instance centroids from one label mask, ignoring background label 0."""

    lab = np.asarray(mask)
    points: list[Point] = []
    for oid in np.unique(lab):
        oid_i = int(oid)
        if oid_i <= 0:
            continue
        ys, xs = np.nonzero(lab == oid_i)
        if len(xs) == 0:
            continue
        points.append((float(xs.mean()), float(ys.mean())))
    return points


def centroid_link_hungarian(
    points_by_frame: Sequence[Sequence[Point]],
    *,
    match_radius_px: float,
    keep_previous_on_empty_frame: bool = False,
) -> pd.DataFrame:
    """Link per-frame centroids with the existing Hungarian nearest-neighbour rule."""

    rows: list[dict] = []
    next_tid = 1
    prev: dict[int, Point] = {}
    for t, cur_pts_raw in enumerate(points_by_frame):
        cur_pts = list(cur_pts_raw)
        if keep_previous_on_empty_frame and not cur_pts:
            continue
        if not prev:
            mapping = {
                i: tid
                for i, tid in enumerate(range(next_tid, next_tid + len(cur_pts)))
            }
            next_tid += len(cur_pts)
        else:
            prev_ids = list(prev.keys())
            prev_xy = np.array([prev[tid] for tid in prev_ids], dtype=float)
            cur_xy = (
                np.array(cur_pts, dtype=float)
                if cur_pts
                else np.zeros((0, 2), dtype=float)
            )
            mapping: dict[int, int] = {}
            if len(cur_xy) > 0 and len(prev_xy) > 0:
                d = np.linalg.norm(prev_xy[:, None, :] - cur_xy[None, :, :], axis=2)
                r, c = linear_sum_assignment(d)
                matched_cur = set()
                for rr, cc in zip(r, c):
                    if d[rr, cc] <= match_radius_px:
                        tid = prev_ids[rr]
                        mapping[int(cc)] = int(tid)
                        matched_cur.add(int(cc))
                for i in range(len(cur_xy)):
                    if i not in matched_cur:
                        mapping[i] = next_tid
                        next_tid += 1
            else:
                for i in range(len(cur_pts)):
                    mapping[i] = next_tid
                    next_tid += 1

        new_prev: dict[int, Point] = {}
        for i, (x, y) in enumerate(cur_pts):
            tid = int(mapping[i])
            rows.append({"frame": int(t), "track_id": tid, "x": x, "y": y})
            new_prev[tid] = (x, y)
        prev = new_prev

    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def tracks_from_label_masks_hungarian(
    masks: Sequence[np.ndarray],
    *,
    match_radius_px: float,
    keep_previous_on_empty_frame: bool = False,
) -> pd.DataFrame:
    """Link instance centroids from independent per-frame label masks."""

    return centroid_link_hungarian(
        [points_from_label_mask(mask) for mask in masks],
        match_radius_px=match_radius_px,
        keep_previous_on_empty_frame=keep_previous_on_empty_frame,
    )


def labels_to_centroid_tracks(masks: Iterable[np.ndarray] | np.ndarray) -> pd.DataFrame:
    """Convert tracked label masks to centroids, treating label values as track IDs."""

    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        iterable = [masks[t] for t in range(masks.shape[0])]
    else:
        iterable = list(masks)

    rows: list[dict] = []
    for t, lab_raw in enumerate(iterable):
        lab = np.asarray(lab_raw)
        for tid in np.unique(lab):
            tid_i = int(tid)
            if tid_i <= 0:
                continue
            ys, xs = np.where(lab == tid_i)
            if len(xs) == 0:
                continue
            rows.append(
                {
                    "frame": int(t),
                    "track_id": tid_i,
                    "x": float(xs.mean()),
                    "y": float(ys.mean()),
                }
            )
    if not rows:
        return empty_tracks_df()
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)
