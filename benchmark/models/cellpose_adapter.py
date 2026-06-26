import numpy as np
import pandas as pd

from .base import ModelAdapter


class CellposeAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        self.model = None
        from cellpose import models

        self.model = models.CellposeModel(
            gpu=True, pretrained_model=cfg.get("pretrained_model", "cpsam")
        )

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Cellpose adapter is not initialized.")
        masks, _, _ = self.model.eval(image, diameter=None)
        return masks.astype(np.int32)

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        raise NotImplementedError(
            "Cellpose native tracking is not implemented in this benchmark adapter."
        )
