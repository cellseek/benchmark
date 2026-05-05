from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SegSample:
    sample_id: str
    image: np.ndarray
    gt_mask: np.ndarray | None
    meta: dict


@dataclass
class TrackSequence:
    seq_id: str
    frames: list[np.ndarray]
    #: Per-frame GT instance masks aligned with ``frames`` (e.g. CTC SEG). ``None``
    #: entries skip segmentation scoring for that frame. Outer ``None`` means not provided.
    gt_masks: list[np.ndarray | None] | None
    gt_tracks: pd.DataFrame | None
    meta: dict
