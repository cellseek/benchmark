"""Native Cell-TRACTR inference (Trackformer ``pipeline`` class).

Uses the upstream Cell-TRACTR repo (configured by ``celltractr_repo``) from the
unified ``benchmark`` conda env. See ``docs/CELLTRACTR.md``.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from argparse import Namespace
from ast import literal_eval
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from .image_prep import to_uint16_ctc
from .mask_tracks import labels_to_centroid_tracks
from .tracking_schema import empty_tracks_df

PRESETS: dict[str, dict[str, str]] = {
    "deepcell": {
        "dataset": "DynamicNuclearNet-tracking-v1_0",
        "checkpoint_name": "checkpoint_deepcell.pth",
    },
    "moma": {
        "dataset": "moma",
        "checkpoint_name": "checkpoint_moma.pth",
    },
}


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_celltractr_repo(cfg: dict) -> Path:
    raw = cfg.get("celltractr_repo") or os.environ.get("CELLTRACTR_REPO")
    if raw:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = (_bench_root() / p).resolve()
    else:
        p = Path("/home/fzhaoai/cellseek/Cell-TRACTR").resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"celltractr: Cell-TRACTR repo not found at {p}. "
            "Clone https://gitlab.com/dunloplab/Cell-TRACTR outside benchmark "
            "or set celltractr_repo in configs/models.yaml."
        )
    return p


def resolve_checkpoint_path(cfg: dict, preset: str) -> Path:
    raw = cfg.get("checkpoint_path") or os.environ.get("CELLTRACTR_CHECKPOINT")
    if raw:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = (_bench_root() / p).resolve()
    else:
        spec = PRESETS[preset]
        p = (_bench_root() / "checkpoints" / "celltractr" / spec["checkpoint_name"]).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"celltractr: checkpoint not found at {p}. "
            "Run: bash scripts/setup_celltractr.sh"
        )
    return p


def load_preset_yaml(repo: Path, preset: str) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(
            f"celltractr: unknown preset {preset!r}; use one of {sorted(PRESETS)}"
        )
    yaml_path = repo / "cfgs" / f"train_{PRESETS[preset]['dataset']}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"celltractr: preset config missing at {yaml_path}")
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    ts = cfg.get("target_size")
    if isinstance(ts, str):
        cfg["target_size"] = literal_eval(ts)
    return cfg


def _ensure_trackformer_import(repo: Path) -> None:
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "celltractr: PyTorch is required. Activate the benchmark conda env: "
            "conda activate benchmark"
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "celltractr: CUDA GPU is required (MultiScaleDeformableAttention). "
            "Run on a GPU node and verify torch.cuda.is_available()."
        )
    try:
        import MultiScaleDeformableAttention  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "celltractr: MultiScaleDeformableAttention not installed in the "
            "benchmark env. Run: bash scripts/setup_celltractr.sh"
        ) from e


def _frame_to_uint16(frame: np.ndarray) -> np.ndarray:
    return to_uint16_ctc(frame)


def write_frames_as_ctc_tifs(frames: list[np.ndarray], seq_dir: Path) -> list[Path]:
    seq_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, frame in enumerate(frames):
        fp = seq_dir / f"t{i:04d}.tif"
        cv2.imwrite(str(fp), _frame_to_uint16(frame))
        paths.append(fp)
    return paths


def read_mask_tifs(
    output_dir: Path,
    n_frames: int,
    *,
    fallback_shape: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for i in range(n_frames):
        fp = output_dir / f"mask{i:03d}.tif"
        if not fp.is_file():
            if fallback_shape is None:
                raise FileNotFoundError(
                    f"celltractr: expected output mask missing: {fp}"
                )
            lab = np.zeros(fallback_shape, dtype=np.int32)
            masks.append(lab)
            continue
        lab = cv2.imread(str(fp), cv2.IMREAD_ANYDEPTH)
        if lab is None:
            if fallback_shape is None:
                raise RuntimeError(f"celltractr: failed to read mask {fp}")
            lab = np.zeros(fallback_shape, dtype=np.int32)
        masks.append(lab.astype(np.int32, copy=False))
    return masks


def masks_to_track_df(masks: list[np.ndarray]) -> pd.DataFrame:
    return labels_to_centroid_tracks(masks)


def build_args(repo: Path, cfg: dict, preset: str) -> Namespace:
    from trackformer.util.misc import nested_dict_to_namespace

    yaml_cfg = load_preset_yaml(repo, preset)
    args = nested_dict_to_namespace(yaml_cfg)
    args.device = str(cfg.get("device", "cuda"))
    args.resume = str(resolve_checkpoint_path(cfg, preset))
    args.eval_only = True
    args.hooks = False
    args.avg_attn_weight_maps = False
    args.use_img_for_mask = bool(cfg.get("use_img_for_mask", True))
    if getattr(args, "dataset", None) != "moma":
        args.display_decoder_aux = False
    return args


def load_model_and_args(repo: Path, cfg: dict, preset: str):
    _ensure_trackformer_import(repo)
    import torch
    from trackformer.models import build_model
    from trackformer.util import misc as utils

    args = build_args(repo, cfg, preset)
    if bool(cfg.get("distributed", False)):
        utils.init_distributed_mode(args)
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        if args.device == "cuda":
            args.gpu = int(cfg.get("gpu", 0))
            torch.cuda.set_device(args.gpu)
    device = torch.device(args.device)
    seed = int(getattr(args, "seed", 42)) + utils.get_rank()
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion = build_model(args)
    model.to(device)
    model.train_model = False
    model.eval()
    criterion.eval_only = True
    model = utils.load_model(model, args)
    return model, args


def run_pipeline_on_fps(
    model,
    args: Namespace,
    fps: list[Path],
    output_ctc_root: Path,
) -> Path:
    from trackformer.engine import pipeline as PipelineClass

    _patch_celltractr_pipeline(PipelineClass)
    output_ctc_root.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_ctc_root
    pipe = PipelineClass(model, fps, args)
    pipe.write_video = False
    pipe.forward()
    return pipe.output_dir


def _patch_celltractr_pipeline(PipelineClass) -> None:
    """Make upstream Cell-TRACTR inference robust to empty per-frame detections.

    The released Trackformer pipeline assumes every predicted cell has a
    non-empty mask. On BEAS frames, the model can emit detections whose masks are
    empty after post-processing; upstream then either indexes stale query arrays
    or writes an uninitialized CTC label mask. Keep the patch local to the
    benchmark wrapper instead of editing the external Cell-TRACTR checkout.
    """

    target_cls = _unwrap_celltractr_pipeline_class(PipelineClass)

    if getattr(target_cls, "_cellseek_empty_mask_patch", False):
        return

    from skimage.measure import label

    def get_track_object_div_indices(self, pred_logits):
        n_queries, _ = pred_logits.shape
        self.num_TQs = n_queries - self.num_queries
        keep = pred_logits[:, 0] > self.threshold
        keep_div = pred_logits[:, 1] > self.threshold
        keep_div[-self.num_queries :] = False
        keep_div[~keep] = False

        # Track query indices address ``prevcells``. If prior frames produced no
        # valid cells, or fewer cells than query slots, upstream can index past
        # ``prevcells``. Treat those invalid positives as false detections.
        prev_count = len(getattr(self, "cells", []))
        if self.num_TQs > prev_count:
            keep[prev_count : self.num_TQs] = False
            keep_div[prev_count : self.num_TQs] = False

        all_indices = keep.nonzero()[0]
        track_indices = np.array([idx for idx in all_indices if idx < self.num_TQs], dtype=int)
        object_indices = np.array([idx for idx in all_indices if idx >= self.num_TQs], dtype=int)
        div_indices = keep_div.nonzero()[0]
        return all_indices, track_indices, object_indices, div_indices

    def post_process_masks(self, masks, boxes):
        masks_np = masks.detach().cpu().numpy()
        if masks_np.shape[0] == 0:
            h, w = self.img_size[1], self.img_size[0]
            return np.zeros((0, h, w), dtype=bool), boxes[:0]

        masks_np = cv2.resize(np.transpose(masks_np, (1, 2, 0)), self.img_size)
        masks_np = masks_np[:, :, None] if masks_np.ndim == 2 else masks_np
        masks_np = np.transpose(masks_np, (-1, 0, 1))

        masks_filt = np.zeros(masks_np.shape, dtype=masks_np.dtype)
        argmax = np.argmax(masks_np, axis=0)
        for m in range(masks_np.shape[0]):
            masks_filt[m, argmax == m] = masks_np[m, argmax == m]

        masks_bool = masks_filt > self.mask_threshold
        n = min(len(self.all_indices), len(self.cells), masks_bool.shape[0], boxes.shape[0])
        if n == 0:
            self.cells = self.cells[:0]
            self.all_indices = self.all_indices[:0]
            self.track_indices = self.track_indices[:0]
            self.object_indices = self.object_indices[:0]
            self.div_indices = self.div_indices[:0]
            self.div_track = self.div_track[:0]
            return masks_bool[:0], boxes[:0]

        self.all_indices = self.all_indices[:n]
        self.cells = self.cells[:n]
        self.div_track = self.div_track[:n]
        masks_bool = masks_bool[:n]
        boxes = boxes[:n]

        keep = np.zeros(n, dtype=bool)
        for m, mask in enumerate(masks_bool):
            if mask.sum() == 0:
                continue
            label_mask = label(mask)
            labels = np.unique(label_mask)
            labels = labels[labels != 0]
            if len(labels) == 0:
                continue
            areas = np.array([(label_mask == lab).sum() for lab in labels])
            lab = labels[int(np.argmax(areas))]
            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[label_mask == lab] = True
            masks_bool[m] = new_mask
            keep[m] = True

        removed_track_queries = int(((self.all_indices < self.num_TQs) & (~keep)).sum())
        self.all_indices = self.all_indices[keep]
        self.cells = self.cells[keep]
        self.div_track = self.div_track[keep]
        masks_bool = masks_bool[keep]
        boxes = boxes[keep]

        kept_divs = set(self.div_track[self.div_track != -1].tolist())
        for div in list(kept_divs):
            if int(np.sum(self.div_track == div)) < 2:
                self.div_track[self.div_track == div] = -1

        self.track_indices = self.all_indices[self.all_indices < self.num_TQs]
        self.object_indices = self.all_indices[self.all_indices >= self.num_TQs]
        self.div_indices = np.array(
            sorted({int(v) for v in self.div_track if int(v) != -1}),
            dtype=int,
        )
        self.num_TQs -= removed_track_queries

        return masks_bool, boxes

    def save_ctc(self, ctc_data, masks):
        h, w = masks.shape[-2:] if masks.ndim >= 2 else self.img.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.uint16)

        n = min(len(self.cells), masks.shape[0] if masks.ndim >= 3 else 0)
        for m, cell in enumerate(self.cells[:n]):
            if masks[m].sum() > 0:
                label_mask[masks[m] > 0] = int(cell)

        max_cellnb = ctc_data.shape[0]
        ctc_cells_new = np.copy(self.cells)
        mask_copy = np.copy(label_mask)
        prevcells = getattr(self, "prevcells", np.zeros((0,), dtype=np.uint16))
        prev_ctc_cells = getattr(self, "ctc_cells", np.zeros((0,), dtype=np.uint16))

        if prevcells is not None and len(prevcells) > 0:
            for c, cell in enumerate(prevcells):
                if cell in self.cells and c < len(prev_ctc_cells):
                    ctc_cell = int(prev_ctc_cells[c])
                    cell_div = self.div_track[self.cells == cell]
                    if len(cell_div) > 0 and int(cell_div[0]) != -1:
                        div_ind = int(cell_div[0])
                        div_cells = self.cells[self.div_track == div_ind]
                        if len(div_cells) >= 2:
                            max_cellnb += 1
                            new_cell_1 = np.array([max_cellnb, self.i, self.i, ctc_cell])[None]
                            ctc_cells_new[self.cells == div_cells[0]] = max_cellnb
                            label_mask[mask_copy == div_cells[0]] = max_cellnb
                            max_cellnb += 1
                            new_cell_2 = np.array([max_cellnb, self.i, self.i, ctc_cell])[None]
                            ctc_data = np.concatenate((ctc_data, new_cell_1, new_cell_2), axis=0)
                            ctc_cells_new[self.cells == div_cells[1]] = max_cellnb
                            label_mask[mask_copy == div_cells[1]] = max_cellnb
                    elif 0 < ctc_cell <= ctc_data.shape[0]:
                        ctc_data[ctc_cell - 1, 2] = self.i
                        ctc_cells_new[self.cells == cell] = ctc_cell
                        label_mask[mask_copy == cell] = ctc_cell

        for c, cell in enumerate(self.cells):
            div_value = int(self.div_track[c]) if c < len(self.div_track) else -1
            is_new = prevcells is None or len(prevcells) == 0 or cell not in prevcells
            if is_new and div_value == -1:
                max_cellnb += 1
                new_cell = np.array([max_cellnb, self.i, self.i, 0])
                ctc_data = np.concatenate((ctc_data, new_cell[None]), axis=0)
                ctc_cells_new[self.cells == cell] = max_cellnb
                label_mask[mask_copy == cell] = max_cellnb

        self.ctc_cells = ctc_cells_new
        cv2.imwrite(str(self.output_dir / f"mask{self.i:03d}.tif"), label_mask)
        return ctc_data

    target_cls.get_track_object_div_indices = get_track_object_div_indices
    target_cls.post_process_masks = post_process_masks
    target_cls.save_ctc = save_ctc
    target_cls._cellseek_empty_mask_patch = True


def _unwrap_celltractr_pipeline_class(obj):
    """Return the real pipeline class hidden by upstream ``@torch.no_grad``."""

    seen = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, type):
            return cur
        wrapped = getattr(cur, "__wrapped__", None)
        if wrapped is not None:
            stack.append(wrapped)
        closure = getattr(cur, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    stack.append(cell.cell_contents)
                except ValueError:
                    pass
    return obj


def infer_sequence_inprocess(
    model,
    args: Namespace,
    frames: list[np.ndarray],
    *,
    seq_id: str = "01",
    work_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    if not frames:
        return empty_tracks_df(), []
    cleanup = work_dir is None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="celltractr_")
        work_dir = Path(tmp.name)
    else:
        tmp = None
    try:
        input_root = work_dir / "input"
        output_root = work_dir / "CTC"
        seq_dir = input_root / seq_id
        fps = write_frames_as_ctc_tifs(frames, seq_dir)
        out_dir = run_pipeline_on_fps(model, args, fps, output_root)
        masks = read_mask_tifs(out_dir, len(frames), fallback_shape=frames[0].shape[:2])
        return masks_to_track_df(masks), masks
    finally:
        if cleanup and tmp is not None:
            tmp.cleanup()


class CelltractrRuntime:
    """Load Cell-TRACTR once in the benchmark env; run native pipeline per sequence."""

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.preset = str(cfg.get("preset", "deepcell")).strip().lower()
        self.repo = resolve_celltractr_repo(cfg)
        self._model = None
        self._args = None
        print(
            f"cellseek-benchmark: loading native Cell-TRACTR "
            f"preset={self.preset!r} repo={self.repo} …",
            flush=True,
        )
        self._model, self._args = load_model_and_args(
            self.repo, self.cfg, self.preset
        )
        print("cellseek-benchmark: native Cell-TRACTR ready.", flush=True)

    def infer_sequence(
        self,
        frames: list[np.ndarray],
        *,
        seq_id: str = "01",
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        assert self._model is not None and self._args is not None
        return infer_sequence_inprocess(
            self._model, self._args, frames, seq_id=seq_id
        )
