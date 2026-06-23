"""Cellpose-SAM (cpsam) helpers aligned with ``gui/utils/cellsam_segment.py``."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_model: Any | None = None
_model_lock = threading.Lock()


def resolve_cpsam_path(
    *,
    checkpoint_path: str | Path | None = None,
    search_from: Path | None = None,
) -> str:
    """Resolve CPSAM weights: explicit path, env, local weights/, GUI weights/, or ``cpsam``."""
    if checkpoint_path:
        p = Path(checkpoint_path).expanduser()
        if p.is_file():
            return os.fspath(p)
        raise FileNotFoundError(f"CellSeek checkpoint not found: {p}")

    env = os.environ.get("CELLSEEK_CPSAM_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return os.fspath(p)
        raise FileNotFoundError(f"CELLSEEK_CPSAM_PATH not found: {p}")

    here = Path(__file__).resolve().parent
    roots = [here.parent, here.parent.parent]
    if search_from is not None:
        roots.insert(0, search_from.resolve())

    seen: set[Path] = set()
    for root in roots:
        for rel in ("weights/cpsam", "gui/weights/cpsam", "checkpoints/cpsam"):
            candidate = (root / rel).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return os.fspath(candidate)

    return "cpsam"


def _load_cellpose_models():
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from cellpose import models as cp_models
    except ImportError as e:
        raise ImportError(
            "Cellpose is required for CellSeek. Install with:\n"
            "  conda env create -f environment.yml && conda activate benchmark"
        ) from e

    logging.getLogger("cellpose").setLevel(logging.WARNING)
    return cp_models


def get_cellpose_model(*, gpu: bool = True, checkpoint_path: str | None = None) -> Any:
    global _model
    with _model_lock:
        if _model is None:
            cp_models = _load_cellpose_models()
            _model = cp_models.CellposeModel(
                gpu=gpu,
                pretrained_model=checkpoint_path or resolve_cpsam_path(),
            )
        return _model


def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    channels = image.shape[-1]
    if channels == 1:
        return np.concatenate([image] * 3, axis=-1)
    if channels == 2:
        pad = np.zeros((*image.shape[:2], 1), dtype=image.dtype)
        return np.concatenate([image, pad], axis=-1)
    if channels == 4:
        return image[..., :3]
    return image[..., :3]


def segment_instance_mask(
    image: np.ndarray,
    *,
    model: Any | None = None,
    gpu: bool = True,
    checkpoint_path: str | None = None,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """Run Cellpose-SAM; returns int32 instance labels at input resolution."""
    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"Expected 2D/3D image array, got shape={img.shape}")

    h, w = int(img.shape[0]), int(img.shape[1])
    rgb = _ensure_three_channels(img)
    seg_model = model or get_cellpose_model(gpu=gpu, checkpoint_path=checkpoint_path)

    masks, _, _ = seg_model.eval(
        rgb,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True,
        channel_axis=-1,
    )
    masks = np.asarray(masks, dtype=np.int32)
    while masks.ndim > 2:
        masks = masks.squeeze(axis=0)

    if masks.shape[:2] != (h, w):
        masks = cv2.resize(masks, (w, h), interpolation=cv2.INTER_NEAREST)
    return masks.astype(np.int32, copy=False)
