"""Optional shared per-frame segmentation for linker models (Trackastra / Ultrack)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .fallbacks import fallback_instance_mask

if TYPE_CHECKING:
    from .cellseek_adapter import CellSeekAdapter


def resolve_shared_segmentation_mask_fn(cfg: dict):
    """
    Return ``mask_fn(image) -> label array`` for per-frame instance masks.

    ``shared_segmentation_backend``:
      - ``fallback`` (default): Otsu + connected components
      - ``cellseek``: CellSAM via :class:`CellSeekAdapter` (lazy init)
    """
    backend = str(cfg.get("shared_segmentation_backend", "fallback")).strip().lower()
    if backend in ("fallback", "otsu", "generic"):
        return fallback_instance_mask

    if backend == "cellseek":
        holder: dict = {"adapter": None}

        def _mask(image: np.ndarray) -> np.ndarray:
            if holder["adapter"] is None:
                from .cellseek_adapter import CellSeekAdapter

                sub = dict(cfg.get("shared_cellseek_config") or {})
                for key in ("checkpoint_path", "cutie_weights_path", "use_gpu", "use_bfloat16"):
                    if key not in sub and key in cfg:
                        sub[key] = cfg[key]
                print(
                    "cellseek-benchmark: shared_segmentation_backend=cellseek "
                    "(CellSAM masks for linker model).",
                    flush=True,
                )
                holder["adapter"] = CellSeekAdapter(sub)
            return holder["adapter"].predict_mask(image)

        return _mask

    raise ValueError(
        f"Unknown shared_segmentation_backend {backend!r}; use 'fallback' or 'cellseek'."
    )
