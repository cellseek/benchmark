"""Ultrack: hypothesis segmentation + global linking from label stacks (approximated via fallback masks)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .base import ModelAdapter
from .fallbacks import fallback_instance_mask


class UltrackAdapter(ModelAdapter):
    """
    Runs Ultrack's ``track(labels=…)`` pipeline on a time stack built from generic
    per-frame instance masks (Otsu + connected components).

    Requires optional solver backend (Gurobi or CBC via python-mip); see Ultrack docs.
    """

    def __init__(self, cfg: dict):
        try:
            import ultrack  # noqa: F401
            from ultrack.config import MainConfig
        except ImportError as e:
            raise ImportError(
                "The ultrack package is required for model type 'ultrack'. "
                "Install with: pip install ultrack"
            ) from e

        self.cfg = cfg
        self.native = True
        self._MainConfig = MainConfig
        print("cellseek-benchmark: UltrackAdapter configured (solver via Ultrack defaults).", flush=True)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return fallback_instance_mask(image)

    def _make_main_config(self, working_dir: Path):
        linking_md = float(self.cfg.get("linking_max_distance", 25.0))
        max_neighbors = int(self.cfg.get("max_neighbors", 10))
        solver = self.cfg.get("solver_name", "")
        if solver is None:
            solver = ""
        time_limit = int(self.cfg.get("time_limit", 600))
        raw = {
            "data": {
                "working_dir": str(working_dir.resolve()),
                "database_file_name": str(self.cfg.get("database_file_name", "ultrack_bench.db")),
            },
            "linking": {
                "max_distance": linking_md,
                "max_neighbors": max_neighbors,
            },
            "tracking": {
                "solver_name": str(solver),
                "time_limit": time_limit,
                "window_size": self.cfg.get("window_size"),
            },
        }
        if raw["tracking"]["window_size"] is None:
            raw["tracking"].pop("window_size")
        return self._MainConfig.model_validate(raw)

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        if not frames:
            raise ValueError("UltrackAdapter.predict_tracks: empty frames.")

        try:
            from ultrack.core.main import track as ultrack_track
            from ultrack.core.export.tracks_layer import to_tracks_layer
        except ImportError as e:
            raise ImportError("ultrack core modules unavailable.") from e

        labels = np.stack([self.predict_mask(f).astype(np.int32) for f in frames], axis=0)
        sigma = self.cfg.get("labels_to_contours_sigma", 1.0)
        sigma_f = float(sigma) if sigma is not None else None

        with tempfile.TemporaryDirectory(prefix="cellseek_ultrack_") as tmp:
            wd = Path(tmp)
            cfg = self._make_main_config(wd)
            ultrack_track(
                cfg,
                labels=labels,
                sigma=sigma_f,
                overwrite="all",
            )
            df, _ = to_tracks_layer(cfg, include_parents=False, include_node_ids=False)

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["frame", "track_id", "x", "y"])

        df = df.rename(columns={"t": "frame"})
        df["frame"] = df["frame"].astype(int)
        df["track_id"] = df["track_id"].astype(int)
        return df[["frame", "track_id", "x", "y"]]
