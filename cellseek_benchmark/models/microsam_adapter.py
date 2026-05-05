import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import torch

from .base import ModelAdapter


def _float_to_uint8(image: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Convert microscopy float/integer images to uint8 for SAM.

    Global min–max is fragile on 16-bit / high-dynamic-range TIFFs (outliers
    wash out contrast), which often yields an empty ``SamAutomaticMaskGenerator``
    result and benchmark scores of zero. Default to percentile stretch.
    """
    mode = str(cfg.get("float_to_uint8", "percentile")).lower()
    x = np.asarray(image, dtype=np.float32)
    if mode == "minmax":
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return np.zeros(x.shape, dtype=np.uint8)
        return np.clip((x - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    p_lo = float(cfg.get("uint8_percentile_low", 1.0))
    p_hi = float(cfg.get("uint8_percentile_high", 99.5))
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo + 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


class MicroSAMAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        checkpoint_path = cfg.get("checkpoint_path")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if self.checkpoint_path is None:
            raise ValueError(
                "microSAM requires `checkpoint_path` "
                "(fine-tuned SAM checkpoint, e.g. checkpoint.pth)."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"microSAM checkpoint not found: {self.checkpoint_path}"
            )
        try:
            from segment_anything import SamAutomaticMaskGenerator
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as e:
            raise RuntimeError(
                "Failed to import segment_anything stack."
            ) from e

        self.model_type = str(cfg.get("model_type", "vit_b"))
        # If device is not set explicitly, use CUDA when available.
        self.device = cfg.get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        # Nearest-neighbour linking between frames; CTC BF sequences can move faster
        # than ~1 px/frame — default 80 px avoids spurious track breaks.
        self.match_radius_px = float(cfg.get("match_radius_px", 80.0))
        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)
        mask_generator_kwargs = {}
        for key in (
            "points_per_side",
            "points_per_batch",
            "pred_iou_thresh",
            "stability_score_thresh",
            "stability_score_offset",
            "box_nms_thresh",
            "crop_n_layers",
            "crop_nms_thresh",
            "crop_overlap_ratio",
            "crop_n_points_downscale_factor",
            "min_mask_region_area",
        ):
            if key in cfg and cfg[key] is not None:
                mask_generator_kwargs[key] = cfg[key]

        # segment_anything defaults (pred_iou_thresh=0.88, stability_score_thresh=0.95)
        # are tuned for natural photos; on CTC microscopy they often yield an
        # empty mask list, so segmentation and tracking-from-centroids both read as 0.
        # Apply gentler defaults unless ``models.yaml`` already set these keys above.
        for k, v in (("pred_iou_thresh", 0.35), ("stability_score_thresh", 0.65)):
            if k not in mask_generator_kwargs:
                mask_generator_kwargs[k] = v

        self._mask_generator = SamAutomaticMaskGenerator(
            self._predictor.model, **mask_generator_kwargs
        )
        tmf = cfg.get("tracking_max_frames")
        self._tracking_max_frames = int(tmf) if tmf is not None and int(tmf) > 0 else None

    def _prepare_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[-1] > 3:
            image = image[..., :3]

        if image.dtype == np.uint8:
            if image.ndim == 2:
                rgb = np.stack([image, image, image], axis=-1)
            else:
                rgb = image
        else:
            if image.ndim == 2:
                u8 = _float_to_uint8(image, self.cfg)
                rgb = np.stack([u8, u8, u8], axis=-1)
            else:
                chans = [_float_to_uint8(image[..., c], self.cfg) for c in range(image.shape[2])]
                rgb = np.stack(chans, axis=-1)
                if rgb.shape[2] < 3:
                    pad = 3 - rgb.shape[2]
                    rgb = np.concatenate([rgb] + [rgb[..., -1:]] * pad, axis=-1)
                elif rgb.shape[2] > 3:
                    rgb = rgb[:, :, :3]
        return rgb

    def _sorted_generated_masks_rgb(self, rgb: np.ndarray) -> list:
        masks = self._mask_generator.generate(rgb)
        if not masks:
            return []
        return sorted(masks, key=lambda m: float(m.get("area", 0.0)), reverse=True)

    def _filtered_centroids_from_preds(
        self, preds: list, rgb_hw: tuple[int, int]
    ) -> list[tuple[float, float]]:
        """
        Centroids for tracking only. Filters SAM masks by pixel area so huge
        border / background proposals (common on BF phase-contrast) do not drown
        the true cell detections in downstream linking and HOTA association.
        """
        h, w = rgb_hw
        img_area = float(h * w)
        min_a = int(self.cfg.get("tracking_min_mask_area", 50))
        max_a_raw = self.cfg.get("tracking_max_mask_area")
        if max_a_raw is not None:
            max_a = int(max_a_raw)
        else:
            max_a = int(min(8000, 0.006 * img_area))
        cand: list[tuple[float, float, int]] = []
        for pred in preds:
            area = int(pred.get("area", 0))
            if area < min_a or area > max_a:
                continue
            seg = np.asarray(pred["segmentation"], dtype=bool)
            ys, xs = np.nonzero(seg)
            if len(xs) == 0:
                continue
            cand.append((float(xs.mean()), float(ys.mean()), area))
        cand.sort(key=lambda t: -t[2])
        max_inst = self.cfg.get("tracking_max_instances_per_frame")
        if max_inst is not None:
            cand = cand[: int(max_inst)]
        return [(x, y) for x, y, _ in cand]

    def _mask_centroids_for_tracking(self, image: np.ndarray) -> list[tuple[float, float]]:
        rgb = self._prepare_rgb(image)
        preds = self._sorted_generated_masks_rgb(rgb)
        if not preds:
            return []
        return self._filtered_centroids_from_preds(preds, rgb.shape[:2])

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        rgb = self._prepare_rgb(image)
        preds = self._sorted_generated_masks_rgb(rgb)
        if not preds:
            return np.zeros(rgb.shape[:2], dtype=np.int32)
        seg = np.zeros(rgb.shape[:2], dtype=np.int32)
        for i, pred in enumerate(preds, start=1):
            seg[np.asarray(pred["segmentation"], dtype=bool)] = i
        return seg

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        if len(frames) == 0:
            return (
                pd.DataFrame(columns=["frame", "track_id", "x", "y"]),
                [],
            )
        if self._tracking_max_frames is not None:
            frames = frames[: self._tracking_max_frames]
        rows: list[dict] = []
        pred_masks: list[np.ndarray] = []
        next_tid = 1
        prev: dict[int, tuple[float, float]] = {}
        for t, frame in enumerate(frames):
            rgb = self._prepare_rgb(frame)
            preds = self._sorted_generated_masks_rgb(rgb)
            seg = np.zeros(rgb.shape[:2], dtype=np.int32)
            for i, pred in enumerate(preds, start=1):
                seg[np.asarray(pred["segmentation"], dtype=bool)] = i
            pred_masks.append(seg.copy())

            cur_pts = (
                self._filtered_centroids_from_preds(preds, rgb.shape[:2]) if preds else []
            )
            if not cur_pts:
                continue

            if not prev:
                mapping = {i: tid for i, tid in enumerate(range(next_tid, next_tid + len(cur_pts)))}
                next_tid += len(cur_pts)
            else:
                prev_ids = list(prev.keys())
                prev_xy = np.array([prev[tid] for tid in prev_ids], dtype=float)
                cur_xy = np.array(cur_pts, dtype=float)
                mapping: dict[int, int] = {}
                if len(cur_xy) > 0 and len(prev_xy) > 0:
                    d = np.linalg.norm(prev_xy[:, None, :] - cur_xy[None, :, :], axis=2)
                    r, c = linear_sum_assignment(d)
                    matched_cur = set()
                    for rr, cc in zip(r, c):
                        if d[rr, cc] <= self.match_radius_px:
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

            new_prev: dict[int, tuple[float, float]] = {}
            for i, (x, y) in enumerate(cur_pts):
                tid = int(mapping[i])
                rows.append({"frame": int(t), "track_id": tid, "x": x, "y": y})
                new_prev[tid] = (x, y)
            prev = new_prev

        return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"]), pred_masks

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        df, _ = self.predict_tracks_returning_masks(frames)
        return df
