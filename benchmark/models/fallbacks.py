import cv2
import numpy as np
import pandas as pd


def fallback_instance_mask(image: np.ndarray) -> np.ndarray:
    """
    Generic unsupervised instance mask fallback:
    Otsu threshold + connected components.
    """
    if image.ndim == 3:
        if image.shape[-1] == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image[..., 0].astype(np.float32)
    else:
        gray = image.astype(np.float32)

    if gray.dtype != np.uint8:
        gmin, gmax = float(gray.min()), float(gray.max())
        if gmax > gmin:
            gray = ((gray - gmin) / (gmax - gmin) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(bw)
    if n <= 1:
        return np.zeros_like(gray, dtype=np.int32)
    return labels.astype(np.int32)


def fallback_track_from_masks(mask_fn, frames: list[np.ndarray]) -> pd.DataFrame:
    """
    Minimal nearest-neighbor centroid tracking from per-frame instance masks.
    """
    rows = []
    next_tid = 1
    prev_centroids = []
    prev_tids = []
    for t, frame in enumerate(frames):
        m = mask_fn(frame)
        ids = [i for i in np.unique(m) if i > 0]
        centroids = []
        for i in ids:
            ys, xs = np.where(m == i)
            if len(xs) == 0:
                continue
            centroids.append((float(xs.mean()), float(ys.mean())))

        if t == 0:
            tids = list(range(next_tid, next_tid + len(centroids)))
            next_tid += len(centroids)
        else:
            tids = []
            for c in centroids:
                if not prev_centroids:
                    tids.append(next_tid)
                    next_tid += 1
                    continue
                dists = [
                    ((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2, tid)
                    for p, tid in zip(prev_centroids, prev_tids)
                ]
                best_dist, best_tid = min(dists, key=lambda x: x[0])
                if best_dist < 400.0:
                    tids.append(best_tid)
                else:
                    tids.append(next_tid)
                    next_tid += 1

        for tid, (x, y) in zip(tids, centroids):
            rows.append({"frame": t, "track_id": int(tid), "x": x, "y": y})

        prev_centroids = centroids
        prev_tids = tids

    return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"])
