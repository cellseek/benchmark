"""Explicit image preparation helpers used by model adapters."""

from __future__ import annotations

import numpy as np
from PIL import Image


def float_to_uint8(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert a numeric image to uint8 using the configured MicroSAM policy."""

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


def ensure_rgb_uint8_percentile(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Prepare images for MicroSAM with its existing percentile/min-max policy."""

    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    if image.dtype == np.uint8:
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        return image

    if image.ndim == 2:
        u8 = float_to_uint8(image, cfg)
        return np.stack([u8, u8, u8], axis=-1)

    chans = [float_to_uint8(image[..., c], cfg) for c in range(image.shape[2])]
    rgb = np.stack(chans, axis=-1)
    if rgb.shape[2] < 3:
        pad = 3 - rgb.shape[2]
        rgb = np.concatenate([rgb] + [rgb[..., -1:]] * pad, axis=-1)
    elif rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    return rgb


def to_pil_rgb_minmax(image: np.ndarray) -> Image.Image:
    """Prepare an image for SAM3 using its existing global min-max conversion."""

    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for SAM3: {image.shape}")

    if rgb.dtype != np.uint8:
        rgb_min = float(np.min(rgb))
        rgb_max = float(np.max(rgb))
        if rgb_max > rgb_min:
            rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255.0).astype(np.uint8)
        else:
            rgb = np.zeros_like(rgb, dtype=np.uint8)
    return Image.fromarray(rgb)


def pil_resize_max_long_edge(pil: Image.Image, max_long_edge: int) -> Image.Image:
    """Resize a PIL image, preserving aspect ratio, using the existing SAM3 policy."""

    if max_long_edge <= 0:
        return pil
    w, h = pil.size
    m = max(w, h)
    if m <= max_long_edge:
        return pil
    scale = max_long_edge / m
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return pil.resize((nw, nh), Image.Resampling.LANCZOS)


def to_gray_float(frame: np.ndarray) -> np.ndarray:
    """Convert a 2D/RGB-like frame to the float grayscale volume Trackastra expects."""

    a = np.asarray(frame, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] >= 3:
            return np.mean(a[..., :3], axis=-1)
        return a[..., 0]
    raise ValueError(f"Unsupported frame shape {a.shape}")


def to_uint16_ctc(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to the uint16 TIFF format expected by Cell-TRACTR."""

    img = np.asarray(frame)
    if img.ndim == 3:
        if img.shape[-1] >= 3:
            img = np.mean(img[..., :3], axis=-1)
        else:
            img = img[..., 0]
    if img.dtype in (np.float32, np.float64):
        mx = float(np.max(img)) if img.size else 0.0
        if mx <= 1.0 + 1e-6:
            img = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
        else:
            img = np.clip(img, 0, 65535).astype(np.uint16)
    elif img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    elif img.dtype != np.uint16:
        img = img.astype(np.uint16)
    return img
