"""Trackastra: transformer linking given per-frame instance masks + raw images."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..image_prep import to_gray_float
from ..mask_tracks import labels_to_centroid_tracks
from .base import ModelAdapter
from .shared_segmentation import resolve_shared_segmentation_mask_fn


def _quiet_tqdm(*args, **kwargs):
    """Disable Trackastra internal progress bars during benchmark runs."""
    kwargs.setdefault("disable", True)
    return tqdm(*args, **kwargs)


class TrackastraAdapter(ModelAdapter):
    """
    Uses pretrained Trackastra association on instance masks built from each frame.

    Segmentation-only benchmarks use the same generic ``fallback_instance_mask`` as other
    lightweight adapters unless you swap preprocessing later.
    """

    def __init__(self, cfg: dict):
        try:
            from trackastra.model import Trackastra
        except ImportError as e:
            raise ImportError(
                "The trackastra package is required for model type 'trackastra'. "
                "Install with: pip install trackastra"
            ) from e

        self.cfg = cfg
        self.native = True
        name = str(cfg.get("pretrained", cfg.get("pretrained_name", "general_2d")))
        device = cfg.get("device", "automatic")
        batch_size = cfg.get("batch_size")
        bs_kw = {} if batch_size is None else {"batch_size": int(batch_size)}
        self._verbose = bool(cfg.get("verbose", False))
        self._track_mode = str(cfg.get("track_mode", "greedy"))
        print(
            f"cellseek-benchmark: loading Trackastra pretrained={name!r} device={device!r} …",
            flush=True,
        )
        self._model = Trackastra.from_pretrained(name, device=device, **bs_kw)
        self._mask_fn = resolve_shared_segmentation_mask_fn(cfg)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return self._mask_fn(image)

    def _volume_from_frames(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not frames:
            raise ValueError("TrackastraAdapter.predict_tracks: empty frames.")
        imgs = np.stack([to_gray_float(f) for f in frames], axis=0)
        masks = np.stack([self.predict_mask(f).astype(np.int32) for f in frames], axis=0)
        if imgs.shape != masks.shape:
            raise RuntimeError(
                f"TrackastraAdapter: image volume {imgs.shape} != mask volume {masks.shape}"
            )
        return imgs, masks

    def _track_impl(
        self, frames: list[np.ndarray], *, return_masks: bool
    ) -> tuple[pd.DataFrame, list[np.ndarray] | None]:
        if len(frames) < 2:
            masks = np.stack(
                [self.predict_mask(f).astype(np.int32, copy=True) for f in frames], axis=0
            )
            df = labels_to_centroid_tracks(masks)
            if return_masks:
                return df, [masks[t].copy() for t in range(masks.shape[0])]
            return df, None

        imgs, masks = self._volume_from_frames(frames)
        prog_cls = tqdm if self._verbose else _quiet_tqdm
        bs = self.cfg.get("batch_size")
        track_kw: dict = {
            "mode": self._track_mode,
            "normalize_imgs": True,
            "progbar_class": prog_cls,
            "n_workers": int(self.cfg.get("n_workers", 0)),
            # Pairwise temporal radius for candidate edges (delta_t=1 ≈ consecutive frames only).
            "delta_t": int(self.cfg.get("delta_t", 1)),
        }
        if bs is not None:
            track_kw["batch_size"] = int(bs)
        md = self.cfg.get("max_distance")
        if md is not None:
            track_kw["max_distance"] = int(md)
        mn = self.cfg.get("max_neighbors")
        if mn is not None:
            track_kw["max_neighbors"] = int(mn)
        ud = self.cfg.get("use_distance")
        if ud is not None:
            track_kw["use_distance"] = bool(ud)
        _graph, masks_tracked = self._model.track(imgs, masks, **track_kw)

        df = labels_to_centroid_tracks(np.asarray(masks_tracked))
        if return_masks:
            mt = np.asarray(masks_tracked)
            mask_list = [mt[t].copy() for t in range(mt.shape[0])]
            return df, mask_list
        return df, None

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        df, _ = self._track_impl(frames, return_masks=False)
        return df

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        df, masks = self._track_impl(frames, return_masks=True)
        assert masks is not None
        return df, masks
