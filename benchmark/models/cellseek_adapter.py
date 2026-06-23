"""CellSeek adapter: Cellpose-SAM segmentation + optional Trackastra linking (GUI-aligned)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ..cellpose_sam import get_cellpose_model, resolve_cpsam_path, segment_instance_mask
from ..mask_tracks import tracks_from_label_masks_hungarian
from ..tracking_schema import empty_tracks_df
from ..trackastra_linking import get_trackastra_model, track_sequence_with_trackastra
from .base import ModelAdapter


class CellSeekAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        self.match_radius_px = float(cfg.get("match_radius_px", 30.0))
        self._last_tracking_stats: dict = {}

        use_gpu = bool(cfg.get("use_gpu", True))
        ckpt = resolve_cpsam_path(
            checkpoint_path=cfg.get("checkpoint_path"),
            search_from=Path(__file__),
        )
        print(
            f"cellseek-benchmark: CellSeek use_gpu={use_gpu}, checkpoint={ckpt}",
            flush=True,
        )
        self._checkpoint_path = ckpt
        self._use_gpu = use_gpu
        self._model = get_cellpose_model(gpu=use_gpu, checkpoint_path=ckpt)
        print(
            f"cellseek-benchmark: Cellpose-SAM ready (gpu={use_gpu}).",
            flush=True,
        )

        tp = str(cfg.get("tracking_propagation", "trackastra")).strip().lower()
        if tp in ("cutie", "gui", "cellsam_then_cutie"):
            print(
                "cellseek-benchmark: tracking_propagation=cutie is deprecated; using trackastra.",
                flush=True,
            )
            tp = "trackastra"
        if tp in ("trackastra", "cellsam_then_trackastra"):
            self._tracking_propagation = "trackastra"
        elif tp in ("cellsam", "cellsam_per_frame", "per_frame"):
            self._tracking_propagation = "cellsam"
        else:
            raise ValueError(
                f"Unknown cellseek.tracking_propagation {tp!r}; "
                "use 'trackastra' or 'cellsam'."
            )

        self._tracking_seed_search_frames = int(cfg.get("tracking_seed_search_frames", 5))
        self._trackastra_mode = str(cfg.get("trackastra_mode", "greedy_nodiv"))
        self._trackastra_pretrained = str(cfg.get("trackastra_pretrained", "general_2d"))
        self._trackastra_device = str(cfg.get("trackastra_device", "automatic"))
        self._trackastra_model = None
        if self._tracking_propagation == "trackastra":
            print(
                "cellseek-benchmark: tracking_propagation=trackastra "
                "(CellSAM segment + Trackastra link, same as GUI).",
                flush=True,
            )

    def get_last_tracking_stats(self) -> dict:
        return dict(self._last_tracking_stats)

    def _segment(self, image: np.ndarray) -> np.ndarray:
        diam = self.cfg.get("diameter")
        return segment_instance_mask(
            image,
            model=self._model,
            gpu=self._use_gpu,
            checkpoint_path=self._checkpoint_path,
            diameter=float(diam) if diam is not None else None,
            flow_threshold=float(self.cfg.get("flow_threshold", 0.4)),
            cellprob_threshold=float(self.cfg.get("cellprob_threshold", 0.0)),
        )

    def _tracks_df_from_masks(self, pred_masks: list[np.ndarray]) -> pd.DataFrame:
        return tracks_from_label_masks_hungarian(
            pred_masks, match_radius_px=self.match_radius_px
        )

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return self._segment(image)

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        if len(frames) == 0:
            return empty_tracks_df(), []

        if self._tracking_propagation == "trackastra":
            if self._trackastra_model is None:
                self._trackastra_model = get_trackastra_model(
                    pretrained=self._trackastra_pretrained,
                    device=self._trackastra_device,
                )
            pred_masks, stats = track_sequence_with_trackastra(
                frames,
                self._segment,
                seed_search_frames=self._tracking_seed_search_frames,
                track_mode=self._trackastra_mode,
                model=self._trackastra_model,
            )
            self._last_tracking_stats = stats
            if stats.get("used_per_frame_cellsam_fallback"):
                print(
                    "cellseek-benchmark: no seed mask in first "
                    f"{stats.get('seed_search_frames')} frames — per-frame CellSAM fallback.",
                    flush=True,
                )
        else:
            self._last_tracking_stats = {"propagation": "cellsam"}
            pred_masks = [self._segment(f).copy() for f in frames]

        df = self._tracks_df_from_masks(pred_masks)
        return df, pred_masks

