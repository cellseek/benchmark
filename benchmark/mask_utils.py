"""Small mask-shape helpers shared by SAM-family adapters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image


def squeeze_mask_2d(mask: np.ndarray) -> np.ndarray:
    """Normalize common model mask shapes to 2D, preserving existing fallback rules."""

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            mask_np = mask_np.max(axis=0)
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got {mask_np.shape}")
    return mask_np


def resize_bool_mask_nearest(mask_bool: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to `(H, W)` with nearest-neighbour sampling."""

    h0, w0 = out_hw
    mask_2d = squeeze_mask_2d(mask_bool).astype(bool, copy=False)
    mh, mw = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    if mh == h0 and mw == w0:
        return mask_2d
    pil_m = Image.fromarray(mask_2d.astype(np.uint8) * 255)
    resized = pil_m.resize((w0, h0), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def paint_bool_masks(
    masks: Iterable[np.ndarray],
    out_hw: tuple[int, int],
    *,
    start_id: int = 1,
) -> np.ndarray:
    """Paint boolean instance masks into one int32 label image using last-wins overlap."""

    labeled = np.zeros(out_hw, dtype=np.int32)
    for instance_id, mask in enumerate(masks, start=start_id):
        labeled[squeeze_mask_2d(mask).astype(bool)] = int(instance_id)
    return labeled
