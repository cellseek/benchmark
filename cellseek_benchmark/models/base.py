from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class ModelAdapter(ABC):
    @abstractmethod
    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        raise NotImplementedError

    # Optional: ``predict_tracks_returning_masks(frames) -> (DataFrame, list[ndarray])``
    # enables a single-pass seg+track benchmark on CTC-style datasets.
