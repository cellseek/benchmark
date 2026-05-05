import logging
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from .base import ModelAdapter


def _resolve_cellseek_cpsam_path(cfg: dict) -> Path:
    """
    Match the main CellSeek repo / GUI weight layout.

    - `gui/main.py` downloads the CPSAM file to ``gui/weights/cpsam`` (no extension).
    - `gui/workers/cellsam_worker.py` uses ``CellSAM(gpu=True)``, i.e. CellSAM's
      default ``model_path="weights/cpsam"`` (relative to process cwd).
    - `cellsam.model.CellSAM` loads if ``os.path.exists(model_path)``.

    Resolution:
    1. If ``checkpoint_path`` is set in config (non-empty): that path only.
    2. Else ``CELLSEEK_CPSAM_PATH`` environment variable.
    3. Else walk ancestors of this package for ``gui/weights/cpsam`` (mono-repo checkout).
    4. Else ``./weights/cpsam`` under the current working directory.
    """
    explicit = cfg.get("checkpoint_path")
    if explicit not in (None, ""):
        p = Path(str(explicit)).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"cellseek checkpoint_path does not exist: {p}\n"
                "Unset checkpoint_path (null) to auto-detect gui/weights/cpsam like the GUI, "
                "or set CELLSEEK_CPSAM_PATH."
            )
        return p

    env = os.environ.get("CELLSEEK_CPSAM_PATH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    for anc in here.parents:
        candidate = anc / "gui" / "weights" / "cpsam"
        if candidate.is_file():
            return candidate

    cwd_candidate = Path.cwd() / "weights" / "cpsam"
    if cwd_candidate.is_file():
        return cwd_candidate

    raise FileNotFoundError(
        "CellSeek CPSAM weights not found. The GUI stores them at "
        "<cellseek repo>/gui/weights/cpsam (see gui/main.py check_and_download_weights). "
        "Options: run the GUI once to download, set configs/models.yaml cellseek.checkpoint_path, "
        "set CELLSEEK_CPSAM_PATH, or place weights/cpsam in the directory you run csbench from."
    )


_CUTIE_WEIGHTS_NAME = "cutie-base-mega.pth"


def _resolve_cutie_weights_path(cfg: dict, cellsam_checkpoint: Path) -> Path:
    """
    Match GUI layout: ``gui/main.py`` downloads CUTIE to ``gui/weights/cutie-base-mega.pth``.

    Resolution:
    1. Non-empty ``cutie_weights_path`` in config (must exist).
    2. ``CELLSEEK_CUTIE_PATH`` environment variable.
    3. **Beside CellSAM**: ``<parent of checkpoint_path>/cutie-base-mega.pth`` (same folder as ``cpsam``).
    4. Walk ancestors of this package for ``gui/weights/<file>`` then ``cutie/weights/<file>``.
    5. ``./weights/<file>`` under the current working directory.
    """
    explicit = cfg.get("cutie_weights_path")
    if explicit not in (None, ""):
        p = Path(str(explicit)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"cellseek cutie_weights_path does not exist: {p}\n"
                "Unset cutie_weights_path to auto-detect gui/weights like the GUI, "
                "or set CELLSEEK_CUTIE_PATH."
            )
        return p

    env = os.environ.get("CELLSEEK_CUTIE_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    sam_resolved = cellsam_checkpoint.expanduser().resolve()
    sibling = sam_resolved.parent / _CUTIE_WEIGHTS_NAME
    if sibling.is_file():
        return sibling

    here = Path(__file__).resolve()
    for anc in here.parents:
        for rel in (
            Path("gui") / "weights" / _CUTIE_WEIGHTS_NAME,
            Path("cutie") / "weights" / _CUTIE_WEIGHTS_NAME,
        ):
            candidate = anc / rel
            if candidate.is_file():
                return candidate

    cwd_candidate = Path.cwd() / "weights" / _CUTIE_WEIGHTS_NAME
    if cwd_candidate.is_file():
        return cwd_candidate

    hf_cutie = (
        "https://huggingface.co/LogicNg/cellseek/resolve/main/cutie-base-mega.pth"
    )
    raise FileNotFoundError(
        "CUTIE weights not found. Expected file next to your CellSAM checkpoint:\n"
        f"  {sibling}\n"
        "(GUI downloads both under gui/weights/; see gui/main.py.)\n"
        "Also searched ancestor directories for gui/weights/ and cutie/weights/, plus "
        f"./weights/{_CUTIE_WEIGHTS_NAME}.\n"
        "Fix: place the checkpoint at the path above, for example:\n"
        f"  wget -O {sibling} {hf_cutie}\n"
        "Or set cellseek.cutie_weights_path / CELLSEEK_CUTIE_PATH."
    )


class CellSeekAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        self.model = None
        self.checkpoint_path: Path | None = None
        self.match_radius_px = float(cfg.get("match_radius_px", 30.0))
        ckpt = _resolve_cellseek_cpsam_path(cfg)
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
        # bfloat16 is faster but can be less stable; set use_bfloat16: false in models.yaml if masks look empty.
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
                "use 'cellsam' (segment every frame) or 'cutie' (CellSAM on frame 0, CUTIE onward)."
            )
        self._cutie_weights_resolved: Path | None = None
        self._cutie_tracker = None
        if self._tracking_propagation == "cutie":
            print(
                "cellseek-benchmark: tracking_propagation=cutie "
                "(CellSAM initial mask + CUTIE propagation; matches GUI pipeline).",
                flush=True,
            )
            self._cutie_weights_resolved = _resolve_cutie_weights_path(cfg, ckpt)
            print(
                f"cellseek-benchmark: CUTIE checkpoint: {self._cutie_weights_resolved}",
                flush=True,
            )

    def _get_cutie_tracker(self):
        if self._cutie_tracker is None:
            try:
                from cutie.cutie_tracker import CutieTracker
            except ImportError as e:
                raise RuntimeError(
                    "cellseek.tracking_propagation=cutie requires the `cutie` package. "
                    "From cellseek_benchmark/: pip install -r requirements-cellseek-tracking.txt "
                    "(editable install of sibling ../cutie)."
                ) from e
            assert self._cutie_weights_resolved is not None
            self._cutie_tracker = CutieTracker(
                weights_path=str(self._cutie_weights_resolved)
            )
        return self._cutie_tracker

    @staticmethod
    def _to_cutie_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """CUTIE expects uint8 HWC RGB (same convention as tensor /255 in CutieTracker)."""
        img = np.asarray(image)
        if img.ndim == 2:
            x = img.astype(np.float32, copy=False)
            rgb = np.stack([x, x, x], axis=-1)
        elif img.ndim == 3:
            if img.shape[2] >= 3:
                rgb = img[..., :3].astype(np.float32, copy=False)
            else:
                x = img[..., 0].astype(np.float32, copy=False)
                rgb = np.stack([x, x, x], axis=-1)
        else:
            raise ValueError(f"Unsupported image shape for CUTIE: {img.shape}")

        if rgb.dtype == np.uint8:
            return np.clip(rgb, 0, 255).astype(np.uint8)

        p_lo, p_hi = np.percentile(rgb, [1.0, 99.5])
        if p_hi > p_lo + 1e-6:
            u = np.clip((rgb - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
            return u.astype(np.uint8)
        # Flat percentiles: min–max so CUTIE never gets an all-black uint8 frame by mistake
        cmin, cmax = float(np.min(rgb)), float(np.max(rgb))
        if cmax > cmin + 1e-6:
            u = np.clip((rgb - cmin) / (cmax - cmin) * 255.0, 0, 255)
            return u.astype(np.uint8)
        return np.zeros(rgb.shape, dtype=np.uint8)

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

        # CellSAM in this setup expects RGB or grayscale-like inputs.
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] > 4:
                img = img[:, :, :3]

        # Same as ``gui/workers/cellsam_worker.py``: full-resolution ``segment`` call,
        # no resize (tiling inside CellSAM handles large frames).
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
            # CUTIE (cutie.inference.inference_core) logs WARN on every frame when the
            # previous mask has no object IDs or memory is empty — one CellSAM miss on
            # frame 0 can produce thousands of duplicate lines. Raise log level; we still
            # surface the science issue via an explicit fallback below.
            logging.getLogger("cutie.inference.inference_core").setLevel(logging.ERROR)

            lab0 = self.predict_mask(frames[0])
            if int(np.max(lab0)) <= 0:
                print(
                    "cellseek-benchmark: frame 0 CellSAM mask is empty — CUTIE cannot seed "
                    "objects; using per-frame CellSAM for this sequence.",
                    flush=True,
                )
                pred_masks = []
                for frame in frames:
                    pred_masks.append(self.predict_mask(frame).copy())
            else:
                tracker = self._get_cutie_tracker()
                prev_rgb = self._to_cutie_rgb_uint8(frames[0])
                prev_lab = lab0.astype(np.int32, copy=False)
                pred_masks = [lab0.copy()]
                for t in range(1, len(frames)):
                    rgb_u8 = self._to_cutie_rgb_uint8(frames[t])
                    cu = tracker.track(prev_rgb, prev_lab, rgb_u8)
                    lab = cu.astype(np.int32)
                    pred_masks.append(lab.copy())
                    prev_rgb = rgb_u8
                    prev_lab = lab
        else:
            for frame in frames:
                lab = self.predict_mask(frame)
                pred_masks.append(lab.copy())

        df = self._tracks_df_from_masks(pred_masks)
        return df, pred_masks

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        df, _ = self.predict_tracks_returning_masks(frames)
        return df
