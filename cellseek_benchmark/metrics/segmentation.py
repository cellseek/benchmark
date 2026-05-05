import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(gt: np.ndarray, pred: np.ndarray):
    gt_ids = np.array([i for i in np.unique(gt) if i > 0], dtype=np.int32)
    pr_ids = np.array([i for i in np.unique(pred) if i > 0], dtype=np.int32)
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return gt_ids, pr_ids, np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)

    iou = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    for gi, g in enumerate(gt_ids):
        gm = gt == g
        for pi, p in enumerate(pr_ids):
            pm = pred == p
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            union = np.logical_or(gm, pm).sum()
            iou[gi, pi] = inter / max(union, 1)
    return gt_ids, pr_ids, iou


def prf1_iou(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5):
    gt_ids, pr_ids, iou = _iou_matrix(gt, pred)
    tp = 0
    if iou.size:
        cost = 1.0 - iou
        g_idx, p_idx = linear_sum_assignment(cost)
        tp = int(sum(iou[g, p] >= iou_threshold for g, p in zip(g_idx, p_idx)))

    fp = int(len(pr_ids) - tp)
    fn = int(len(gt_ids) - tp)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
