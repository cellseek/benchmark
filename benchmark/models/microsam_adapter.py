import numpy as np
import pandas as pd
import torch

from ..image_prep import ensure_rgb_uint8_percentile
from ..io_utils import resolve_checkpoint_path
from ..mask_tracks import centroid_link_hungarian
from ..tracking_schema import empty_tracks_df
from .base import ModelAdapter


class MicroSAMAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        checkpoint_path = resolve_checkpoint_path(cfg.get("checkpoint_path"))
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is None:
            raise ValueError(
                "microSAM requires `checkpoint_path` "
                "(fine-tuned SAM checkpoint, e.g. ../checkpoints/microsam_vit_b_lm.pt)."
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
        return ensure_rgb_uint8_percentile(image, self.cfg)

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
            return empty_tracks_df(), []
        if self._tracking_max_frames is not None:
            frames = frames[: self._tracking_max_frames]
        points_by_frame: list[list[tuple[float, float]]] = []
        pred_masks: list[np.ndarray] = []
        for frame in frames:
            rgb = self._prepare_rgb(frame)
            preds = self._sorted_generated_masks_rgb(rgb)
            seg = np.zeros(rgb.shape[:2], dtype=np.int32)
            for i, pred in enumerate(preds, start=1):
                seg[np.asarray(pred["segmentation"], dtype=bool)] = i
            pred_masks.append(seg.copy())

            cur_pts = (
                self._filtered_centroids_from_preds(preds, rgb.shape[:2]) if preds else []
            )
            points_by_frame.append(cur_pts)

        df = centroid_link_hungarian(
            points_by_frame,
            match_radius_px=self.match_radius_px,
            keep_previous_on_empty_frame=True,
        )
        return df, pred_masks
