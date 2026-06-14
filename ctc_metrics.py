"""
CTC-style evaluation metrics for predicted vs ground-truth label masks.

Implements SEG (average matched IoU), DET (detection F1 at overlap threshold),
and a simplified TRA proxy based on per-frame ID matching consistency.
For publication, verify against the official CTC evaluation binaries when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FrameMetrics:
    seg: float
    det_precision: float
    det_recall: float
    det_f1: float
    matched: int
    gt_count: int
    pred_count: int


def _instance_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, overlap: float = 0.5) -> Tuple[float, int, int, int]:
    """Return mean IoU of matched instances, match count, gt count, pred count."""
    gt_ids = [int(i) for i in np.unique(gt_mask) if i > 0]
    pred_ids = [int(i) for i in np.unique(pred_mask) if i > 0]

    if not gt_ids and not pred_ids:
        return 1.0, 0, 0, 0
    if not gt_ids or not pred_ids:
        return 0.0, 0, len(gt_ids), len(pred_ids)

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for gi, gid in enumerate(gt_ids):
        gt_bin = gt_mask == gid
        for pi, pid in enumerate(pred_ids):
            pred_bin = pred_mask == pid
            inter = np.logical_and(gt_bin, pred_bin).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_bin, pred_bin).sum()
            iou_matrix[gi, pi] = inter / union

    if iou_matrix.size == 0:
        return 0.0, 0, len(gt_ids), len(pred_ids)

    cost = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    ious = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= overlap:
            ious.append(iou_matrix[r, c])

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou, len(ious), len(gt_ids), len(pred_ids)


def frame_metrics(pred: np.ndarray, gt: np.ndarray, overlap: float = 0.5) -> FrameMetrics:
    seg, matched, gt_count, pred_count = _instance_iou(pred, gt, overlap)
    precision = matched / pred_count if pred_count else (1.0 if gt_count == 0 else 0.0)
    recall = matched / gt_count if gt_count else (1.0 if pred_count == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return FrameMetrics(seg=seg, det_precision=precision, det_recall=recall, det_f1=f1,
                        matched=matched, gt_count=gt_count, pred_count=pred_count)


def _label_centroids(mask: np.ndarray) -> Dict[int, Tuple[float, float]]:
    centroids: Dict[int, Tuple[float, float]] = {}
    for label in np.unique(mask):
        label = int(label)
        if label == 0:
            continue
        ys, xs = np.where(mask == label)
        centroids[label] = (float(xs.mean()), float(ys.mean()))
    return centroids


def tra_proxy(
    pred_sequence: List[np.ndarray],
    gt_sequence: List[np.ndarray],
    max_distance: float = 30.0,
) -> float:
    """
    Simplified TRA proxy: fraction of GT-to-pred centroid links consistent across
    consecutive frames (greedy nearest-neighbour within max_distance pixels).
    Not identical to official CTC TRA/AOGM; use official eval for final numbers.
    """
    if len(pred_sequence) != len(gt_sequence) or not pred_sequence:
        return 0.0

    prev_map: Dict[int, int] = {}
    consistent = 0
    total = 0

    for pred, gt in zip(pred_sequence, gt_sequence):
        gt_cent = _label_centroids(gt)
        pred_cent = _label_centroids(pred)
        if not gt_cent:
            prev_map = {}
            continue

        current_map: Dict[int, int] = {}
        for gt_id, (gx, gy) in gt_cent.items():
            best_pid = None
            best_dist = max_distance
            for pid, (px, py) in pred_cent.items():
                d = np.hypot(gx - px, gy - py)
                if d < best_dist:
                    best_dist = d
                    best_pid = pid
            if best_pid is not None:
                current_map[gt_id] = best_pid
                total += 1
                if gt_id in prev_map and prev_map[gt_id] == best_pid:
                    consistent += 1

        prev_map = current_map

    return consistent / total if total else 0.0


def ctc_count_score(pred_sequence: List[np.ndarray], gt_sequence: List[np.ndarray]) -> float:
    """1 minus mean absolute relative cell-count error per frame."""
    errors = []
    for pred, gt in zip(pred_sequence, gt_sequence):
        gt_n = len([i for i in np.unique(gt) if i > 0])
        pred_n = len([i for i in np.unique(pred) if i > 0])
        if gt_n == 0 and pred_n == 0:
            errors.append(0.0)
        elif gt_n == 0:
            errors.append(1.0)
        else:
            errors.append(abs(pred_n - gt_n) / gt_n)
    return 1.0 - float(np.mean(errors))


def evaluate_sequence(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    overlap: float = 0.5,
) -> Dict[str, float]:
    if len(pred_masks) != len(gt_masks):
        raise ValueError(f"Frame count mismatch: {len(pred_masks)} pred vs {len(gt_masks)} gt")

    seg_scores = []
    det_f1_scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        fm = frame_metrics(pred, gt, overlap)
        seg_scores.append(fm.seg)
        det_f1_scores.append(fm.det_f1)

    return {
        "SEG": float(np.mean(seg_scores)),
        "DET": float(np.mean(det_f1_scores)),
        "TRA": tra_proxy(pred_masks, gt_masks),
        "CTC": ctc_count_score(pred_masks, gt_masks),
        "num_frames": len(pred_masks),
    }
