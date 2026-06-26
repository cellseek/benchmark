"""Native Cell-TRACTR adapter (Trackformer pipeline — not Trackastra)."""

from __future__ import annotations

import numpy as np

from ..celltractr_native import CelltractrRuntime
from .base import ModelAdapter


class CelltractrAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.native = True
        self._runtime = CelltractrRuntime(self.cfg)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        _, masks = self._runtime.infer_sequence([image], seq_id="mask")
        if not masks:
            h, w = np.asarray(image).shape[:2]
            return np.zeros((h, w), dtype=np.int32)
        return masks[0].astype(np.int32, copy=False)

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ):
        return self._runtime.infer_sequence(frames, seq_id="01")
