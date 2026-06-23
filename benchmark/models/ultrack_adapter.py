"""Ultrack: hypothesis segmentation + global linking from label stacks (approximated via fallback masks)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .base import ModelAdapter
from .shared_segmentation import resolve_shared_segmentation_mask_fn


def _patch_ultrack_mip_solver_arraymap_readonly() -> None:
    """
    ultrack 0.7.x + scikit-image >= 0.26: ``ArrayMap`` indexing is read-only and
    ``MIPSolver.add_edges`` raises ``ValueError: buffer source array is read-only``.
    Use a writable numpy lookup table instead (same semantics).
    """
    try:
        from ultrack.core.solve.solver import mip_solver
    except ImportError:
        return
    if getattr(mip_solver, "_cellseek_arraymap_patch", False):
        return

    _orig_add_nodes = mip_solver.MIPSolver.add_nodes

    def add_nodes_patched(self, indices, is_first_t, is_last_t, is_border=False, nodes_prob=None):
        _orig_add_nodes(self, indices, is_first_t, is_last_t, is_border, nodes_prob)
        if self._backward_map is None or len(self._backward_map) == 0:
            self._forward_map = np.zeros(0, dtype=np.int64)
            return
        indices = np.asarray(self._backward_map, dtype=np.int64)
        max_idx = int(indices.max())
        fwd = np.full(max_idx + 1, -1, dtype=np.int64)
        fwd[indices] = np.arange(len(indices), dtype=np.int64)
        self._forward_map = fwd

    def add_edges_patched(self, sources, targets, weights):
        if self._edges is not None:
            raise ValueError("Edges have already been added.")
        mip_solver.assert_same_length(sources=sources, targets=targets, weights=weights)
        weights = self._config.apply_link_function(np.asarray(weights, dtype=float))
        src = np.asarray(sources, dtype=np.int64)
        tgt = np.asarray(targets, dtype=np.int64)
        sources = self._forward_map[src]
        targets = self._forward_map[tgt]
        if np.any(sources < 0) or np.any(targets < 0):
            raise ValueError("Ultrack MIP: edge references unknown node id.")
        self._edges = self._model.add_var_tensor(
            (len(weights),), name="edges", var_type=mip_solver.mip.BINARY
        )
        self._edges_df = mip_solver.pd.DataFrame(
            np.asarray([sources, targets]).T, columns=["sources", "targets"]
        )
        self._model.objective += mip_solver.mip.xsum(weights * self._edges)

    mip_solver.MIPSolver.add_nodes = add_nodes_patched
    mip_solver.MIPSolver.add_edges = add_edges_patched
    mip_solver._cellseek_arraymap_patch = True


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
        self._mask_fn = resolve_shared_segmentation_mask_fn(cfg)
        _patch_ultrack_mip_solver_arraymap_readonly()
        print("cellseek-benchmark: UltrackAdapter configured (solver via Ultrack defaults).", flush=True)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return self._mask_fn(image)

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

        # Ultrack and its backends may request a writable contiguous buffer.
        # Ensure labels are C-contiguous and writeable (some upstream arrays can be read-only views).
        labels = np.stack(
            [self.predict_mask(f).astype(np.int32, copy=True) for f in frames], axis=0
        )
        labels = np.array(labels, dtype=np.int32, order="C", copy=True)
        sigma = self.cfg.get("labels_to_contours_sigma", 1.0)
        sigma_f = float(sigma) if sigma is not None else None

        with tempfile.TemporaryDirectory(prefix="cellseek_ultrack_") as tmp:
            wd = Path(tmp)
            cfg = self._make_main_config(wd)
            try:
                ultrack_track(
                    cfg,
                    labels=labels,
                    sigma=sigma_f,
                    overwrite="all",
                )
            except ValueError as e:
                msg = str(e)
                if any(
                    s in msg
                    for s in (
                        "NO_SOLUTION_FOUND",
                        "must be optimized before returning solution",
                        "Infeasible solution found",
                    )
                ):
                    print(
                        "cellseek-benchmark: Ultrack ILP had no feasible solution for "
                        f"this sub-sequence ({len(frames)} frames); returning empty tracks.",
                        flush=True,
                    )
                    return pd.DataFrame(columns=["frame", "track_id", "x", "y"])
                raise
            df, _ = to_tracks_layer(cfg, include_parents=False, include_node_ids=False)

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["frame", "track_id", "x", "y"])

        df = df.rename(columns={"t": "frame"})
        df["frame"] = df["frame"].astype(int)
        df["track_id"] = df["track_id"].astype(int)
        return df[["frame", "track_id", "x", "y"]]
