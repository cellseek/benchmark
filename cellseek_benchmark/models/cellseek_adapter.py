import logging
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from cellsam import MaskTracker, resolve_cpsam_path, resolve_cutie_weights_path, to_tracker_rgb_uint8

from .base import ModelAdapter


class CellSeekAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        self.model = None
        self.checkpoint_path: Path | None = None
        self.match_radius_px = float(cfg.get("match_radius_px", 30.0))
        ckpt = resolve_cpsam_path(
            checkpoint_path=cfg.get("checkpoint_path"),
            search_from=Path(__file__),
        )
        self.checkpoint_path = ckpt
        import torch
        from cellsam import CellSAM

        use_gpu = bool(cfg.get("use_gpu", True))
        cuda_ok = torch.cuda.is_available()
        print(
            f"cellseek-benchmark: CellSAM use_gpu={use_gpu}, "
            f"torch.cuda.is_available()={cuda_ok}",
            flush=True,
        )
        if not use_gpu:
            print(
                "cellseek-benchmark: CellSAM will use CPU (use_gpu: false in config).",
                flush=True,
            )
        if use_gpu and not cuda_ok:
            raise RuntimeError(
                "CellSeek requires CUDA, but torch.cuda.is_available() is False."
            )

        print(
            "cellseek-benchmark: loading CellSAM weights (large file; "
            "no output until torch finishes reading) ...",
            flush=True,
        )
        print(f"cellseek-benchmark: CellSAM checkpoint: {ckpt}", flush=True)
        self.model = CellSAM(
            gpu=use_gpu,
            model_path=str(ckpt),
            use_bfloat16=bool(cfg.get("use_bfloat16", True)),
        )
        print(
            f"cellseek-benchmark: CellSAM ready (device={self.model.device}).",
            flush=True,
        )

        tp = str(cfg.get("tracking_propagation", "cellsam")).strip().lower()
        if tp in ("cellsam", "cellsam_per_frame", "per_frame"):
            self._tracking_propagation = "cellsam"
        elif tp in ("cutie", "gui", "cellsam_then_cutie"):
            self._tracking_propagation = "cutie"
        else:
            raise ValueError(
                f"Unknown cellseek.tracking_propagation {tp!r}; "
                "use 'cellsam' (segment every frame) or 'cutie' (CellSAM seed + propagation)."
            )
        self._cutie_weights_resolved: Path | None = None
        self._last_tracking_stats: dict = {}
        self._tracking_seed_search_frames = int(cfg.get("tracking_seed_search_frames", 5))
        stop_raw = cfg.get("tracking_cutie_consecutive_empty_stop", 10)
        self._tracking_cutie_consecutive_empty_stop = (
            int(stop_raw) if stop_raw is not None and int(stop_raw) > 0 else 0
        )
        if self._tracking_propagation == "cutie":
            print(
                "cellseek-benchmark: tracking_propagation=cutie "
                "(CellSAM initial mask + mask propagation).",
                flush=True,
            )
            self._cutie_weights_resolved = resolve_cutie_weights_path(
                cutie_weights_path=cfg.get("cutie_weights_path"),
                cellsam_checkpoint=ckpt,
                search_from=Path(__file__),
            )
            print(
                f"cellseek-benchmark: propagation checkpoint: {self._cutie_weights_resolved}",
                flush=True,
            )

    def _new_mask_tracker(self) -> MaskTracker:
        assert self._cutie_weights_resolved is not None
        return MaskTracker(weights_path=str(self._cutie_weights_resolved))

    def get_last_tracking_stats(self) -> dict:
        return dict(self._last_tracking_stats)

    def _find_seed_frame_mask(
        self, frames: list[np.ndarray]
    ) -> tuple[int | None, np.ndarray | None]:
        k = min(self._tracking_seed_search_frames, len(frames))
        for t in range(k):
            lab = self.predict_mask(frames[t])
            if int(np.max(lab)) > 0:
                return t, lab.astype(np.int32, copy=False)
        return None, None

    def _empty_mask_like(self, frame: np.ndarray) -> np.ndarray:
        return np.zeros(frame.shape[:2], dtype=np.int32)

    def _track_cutie_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        logging.getLogger("cutie.inference.inference_core").setLevel(logging.ERROR)
        stats = {
            "propagation": "cutie",
            "seed_frame": None,
            "seed_search_frames": self._tracking_seed_search_frames,
            "used_per_frame_cellsam_fallback": False,
            "cutie_stopped_early": False,
        }
        seed_t, lab0 = self._find_seed_frame_mask(frames)
        if seed_t is None or lab0 is None:
            stats["used_per_frame_cellsam_fallback"] = True
            stats["reason"] = "no_nonempty_mask_in_seed_search"
            self._last_tracking_stats = stats
            return [self.predict_mask(f).copy() for f in frames]

        stats["seed_frame"] = int(seed_t)
        pred_masks: list[np.ndarray] = [
            self._empty_mask_like(frames[i]) for i in range(len(frames))
        ]

        for t in range(seed_t):
            pred_masks[t] = self.predict_mask(frames[t]).astype(np.int32, copy=False)

        tracker = self._new_mask_tracker()
        pred_masks[seed_t] = lab0.copy()
        prev_rgb = to_tracker_rgb_uint8(frames[seed_t])
        prev_lab = lab0.astype(np.int32, copy=False)
        consecutive_empty = 0
        stop_at = self._tracking_cutie_consecutive_empty_stop

        for t in range(seed_t + 1, len(frames)):
            rgb_u8 = to_tracker_rgb_uint8(frames[t])
            cu = tracker.track(prev_rgb, prev_lab, rgb_u8)
            lab = cu.astype(np.int32)
            pred_masks[t] = lab.copy()
            if int(np.max(lab)) <= 0:
                consecutive_empty += 1
                if stop_at > 0 and consecutive_empty >= stop_at:
                    stats["cutie_stopped_early"] = True
                    stats["cutie_stopped_at_frame"] = int(t)
                    for rest in range(t + 1, len(frames)):
                        pred_masks[rest] = prev_lab.copy()
                    break
            else:
                consecutive_empty = 0
            prev_rgb = rgb_u8
            prev_lab = lab

        self._last_tracking_stats = stats
        return pred_masks

    @staticmethod
    def _to_cutie_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for compare scripts and tests."""
        return to_tracker_rgb_uint8(image)

    def _tracks_df_from_masks(
        self, pred_masks: list[np.ndarray]
    ) -> pd.DataFrame:
        rows: list[dict] = []
        next_tid = 1
        prev: dict[int, tuple[float, float]] = {}
        for t, lab in enumerate(pred_masks):
            cur_pts: list[tuple[float, float]] = []
            for oid in np.unique(lab):
                if oid <= 0:
                    continue
                ys, xs = np.nonzero(lab == oid)
                if len(xs) == 0:
                    continue
                cur_pts.append((float(xs.mean()), float(ys.mean())))

            if not prev:
                mapping = {
                    i: tid
                    for i, tid in enumerate(range(next_tid, next_tid + len(cur_pts)))
                }
                next_tid += len(cur_pts)
            else:
                prev_ids = list(prev.keys())
                prev_xy = np.array([prev[tid] for tid in prev_ids], dtype=float)
                cur_xy = (
                    np.array(cur_pts, dtype=float)
                    if cur_pts
                    else np.zeros((0, 2), dtype=float)
                )
                mapping: dict[int, int] = {}
                if len(cur_xy) > 0 and len(prev_xy) > 0:
                    d = np.linalg.norm(prev_xy[:, None, :] - cur_xy[None, :, :], axis=2)
                    r, c = linear_sum_assignment(d)
                    matched_cur = set()
                    for rr, cc in zip(r, c):
                        if d[rr, cc] <= self.match_radius_px:
                            tid = prev_ids[rr]
                            mapping[int(cc)] = int(tid)
                            matched_cur.add(int(cc))
                    for i in range(len(cur_xy)):
                        if i not in matched_cur:
                            mapping[i] = next_tid
                            next_tid += 1
                else:
                    for i in range(len(cur_pts)):
                        mapping[i] = next_tid
                        next_tid += 1

            new_prev: dict[int, tuple[float, float]] = {}
            for i, (x, y) in enumerate(cur_pts):
                tid = int(mapping[i])
                rows.append({"frame": int(t), "track_id": tid, "x": x, "y": y})
                new_prev[tid] = (x, y)
            prev = new_prev

        return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"])

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CellSeek adapter is not initialized.")
        img = np.asarray(image)
        if img.ndim < 2:
            raise ValueError(f"Expected 2D/3D image array, got shape={img.shape}")

        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] > 4:
                img = img[:, :, :3]

        diam = self.cfg.get("diameter", None)
        if diam is not None:
            diam = float(diam)
        flow_threshold = float(self.cfg.get("flow_threshold", 0.4))
        cellprob_threshold = float(self.cfg.get("cellprob_threshold", 0.0))

        masks, _, _ = self.model.segment(
            img,
            diameter=diam,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize=True,
        )
        masks = np.asarray(masks)
        while masks.ndim > 2:
            masks = masks.squeeze(axis=0)

        h, w = int(img.shape[0]), int(img.shape[1])
        if masks.shape[:2] != (h, w):
            masks = cv2.resize(
                masks.astype(np.int32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        return masks.astype(np.int32)

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        if len(frames) == 0:
            return (
                pd.DataFrame(columns=["frame", "track_id", "x", "y"]),
                [],
            )
        pred_masks: list[np.ndarray] = []

        if self._tracking_propagation == "cutie":
            pred_masks = self._track_cutie_sequence(frames)
            st = self._last_tracking_stats
            if st.get("used_per_frame_cellsam_fallback"):
                print(
                    "cellseek-benchmark: no seed mask in first "
                    f"{st.get('seed_search_frames')} frames — per-frame CellSAM fallback.",
                    flush=True,
                )
        else:
            self._last_tracking_stats = {"propagation": "cellsam"}
            for frame in frames:
                lab = self.predict_mask(frame)
                pred_masks.append(lab.copy())

        df = self._tracks_df_from_masks(pred_masks)
        return df, pred_masks

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        df, _ = self.predict_tracks_returning_masks(frames)
        return df
