"""
CTC-style cell tracking datasets (shared layout across challenges).

Per sequence / split:
  - Raw frames: ``<split>/t<time>.tif`` (time is zero-padded; width varies by set).
  - Segmentation benchmark GT: ``<split>_GT/SEG/man_seg*.tif`` (sparse instance masks).
  - Tracking GT: ``<split>_GT/TRA/man_track<time>.tif`` (per-frame label masks).
  - Tracking **sequences** are defined by ``<split>_GT/TRA/man_track_filtered.json``
    (``Sequences[]`` with ``Start_Frame``, ``End_Frame``, ``Valid_IDs``). Each entry
    becomes one ``TrackSequence`` (not one sequence per entire split ``01``/``02``).

``gt_tracks.frame`` is **local** to the subsequence (0 .. len(frames)-1), aligned
with ``frames[i]``. Raw ``t*.tif`` must still cover ``0 .. T-1`` without gaps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .base import DatasetAdapter
from ..schemas import SegSample, TrackSequence

_RAW_TIF = re.compile(r"^t(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_MAN_SEG = re.compile(r"^man_seg(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_MAN_TRACK = re.compile(r"^man_track(\d+)\.(?:tif|tiff)$", re.IGNORECASE)


def _path_segment(x) -> str:
    """
    YAML parses unquoted 01/02 as integers; CTC layout uses zero-padded dirs (01, 02).
    """
    if isinstance(x, int):
        return f"{x:02d}"
    s = str(x)
    if s.isdigit():
        return s.zfill(2)
    return s


def _to_float_image(arr: np.ndarray) -> np.ndarray:
    """
    Squeeze common singleton dimensions from microscopy TIFFs.

    Typical CTC raw: ``(H, W)``, ``(H, W, C)``, ``(1, H, W)``, or ``(H, W, 1)``.
    """
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        while x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
    return x.astype(np.float32, copy=False)


def _read_ctc_image(path: Path) -> np.ndarray:
    return _to_float_image(tifffile.imread(str(path)))


def _sorted_raw_frame_paths(raw_dir: Path) -> list[Path] | None:
    """
    Return raw frame paths ordered by CTC time index, or None if indices are not
    exactly ``0 .. T-1`` (required so ``frames[i]`` aligns with ``man_track`` index ``i``).
    """
    pairs: list[tuple[int, Path]] = []
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        m = _RAW_TIF.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    t_vals = [t for t, _ in pairs]
    t_max = t_vals[-1]
    if t_vals[0] != 0 or len(t_vals) != t_max + 1:
        return None
    if any(t_vals[i] != i for i in range(len(t_vals))):
        return None
    return [p for _, p in pairs]


def _sorted_seg_paths(seg_dir: Path) -> list[Path]:
    out: list[tuple[int, Path]] = []
    for p in seg_dir.iterdir():
        if not p.is_file():
            continue
        m = _MAN_SEG.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return [p for _, p in sorted(out, key=lambda x: x[0])]


def _sorted_tra_mask_paths(tra_dir: Path) -> list[Path]:
    """TRA label masks only (excludes ``man_track.txt`` and other files)."""
    out: list[tuple[int, Path]] = []
    for p in tra_dir.iterdir():
        if not p.is_file():
            continue
        m = _MAN_TRACK.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return [p for _, p in sorted(out, key=lambda x: x[0])]


def _seg_time_str(seg_path: Path) -> str | None:
    m = _MAN_SEG.match(seg_path.name)
    return m.group(1) if m else None


def _seg_time_int(seg_path: Path) -> int | None:
    m = _MAN_SEG.match(seg_path.name)
    return int(m.group(1)) if m else None


def _raw_frame_idx_str(frame_path: Path) -> str | None:
    m = _RAW_TIF.match(frame_path.name)
    return m.group(1) if m else None


def _tra_time_int(tra_path: Path) -> int | None:
    m = _MAN_TRACK.match(tra_path.name)
    return int(m.group(1)) if m else None


def _resolve_raw_for_seg_time(raw_dir: Path, time_digits: str) -> Path | None:
    """``man_seg`` time suffix must match a raw ``t<same digits>.tif`` name."""
    for ext in (".tif", ".tiff"):
        cand = raw_dir / f"t{time_digits}{ext}"
        if cand.is_file():
            return cand
    return None


def _load_tra_sequence_specs(tra_dir: Path, json_name: str) -> list[dict] | None:
    """Load ``Sequences`` from ``man_track_filtered.json``, or None if missing."""
    path = tra_dir / json_name
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    seqs = data.get("Sequences")
    if not isinstance(seqs, list):
        raise ValueError(f"{path}: expected top-level 'Sequences' list, got {type(seqs).__name__}")
    return seqs


def _gt_tracks_from_tra_window(
    tra_mask_by_t: dict[int, np.ndarray],
    t_start: int,
    t_end: int,
    valid_ids: set[int] | None,
) -> pd.DataFrame:
    """
  Build point tracks for ``t_start .. t_end`` inclusive.

  ``frame`` in the output is local (0 at ``t_start``). Only track IDs in
  ``valid_ids`` are kept when that set is provided.
    """
    gt_rows: list[dict] = []
    for local_f, global_t in enumerate(range(t_start, t_end + 1)):
        lab = tra_mask_by_t.get(global_t)
        if lab is None:
            continue
        for tid in np.unique(lab):
            tid = int(tid)
            if tid <= 0:
                continue
            if valid_ids is not None and tid not in valid_ids:
                continue
            ys, xs = np.nonzero(lab == tid)
            if len(xs) == 0:
                continue
            gt_rows.append(
                {
                    "frame": local_f,
                    "track_id": tid,
                    "x": float(xs.mean()),
                    "y": float(ys.mean()),
                }
            )
    return pd.DataFrame(gt_rows, columns=["frame", "track_id", "x", "y"])


class CTCCellTrackingAdapter(DatasetAdapter):
    """
    Adapter for CTC-style cell tracking datasets under one root.

    Expected per dataset directory:
      <dataset>/<split>/t*.tif
      <dataset>/<split>_GT/SEG/man_seg*.tif
      <dataset>/<split>_GT/TRA/man_track*.tif
      <dataset>/<split>_GT/TRA/man_track_filtered.json  (tracking sub-sequences)
    """

    joint_segmentation_with_tracking = False

    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        ds_names = cfg.get("datasets")
        if ds_names:
            self.dataset_names = [str(x) for x in ds_names]
        else:
            self.dataset_names = sorted(
                [p.name for p in self.root.iterdir() if p.is_dir()]
            )
        self.splits = [_path_segment(s) for s in cfg.get("splits", ["01", "02"])]
        mts = cfg.get("max_tracking_sequences")
        self.max_tracking_sequences = int(mts) if mts is not None else None
        mfs = cfg.get("max_frames_per_sequence")
        self.max_frames_per_sequence = int(mfs) if mfs is not None else None
        mss = cfg.get("max_segmentation_samples")
        self.max_segmentation_samples = int(mss) if mss is not None else None
        self.tracking_sequence_json = str(
            cfg.get("tracking_sequence_json", "man_track_filtered.json")
        )

    def _iter_dataset_split(self):
        for ds in self.dataset_names:
            ds_dir = self.root / ds
            for split in self.splits:
                raw_dir = ds_dir / split
                gt_dir = ds_dir / f"{split}_GT"
                if raw_dir.exists() and gt_dir.exists():
                    yield ds, split, raw_dir, gt_dir

    def iter_segmentation(self):
        yielded = 0
        for ds, split, raw_dir, gt_dir in self._iter_dataset_split():
            seg_dir = gt_dir / "SEG"
            if not seg_dir.exists():
                continue
            for seg_fp in _sorted_seg_paths(seg_dir):
                idx_str = _seg_time_str(seg_fp)
                if idx_str is None:
                    continue
                img_fp = _resolve_raw_for_seg_time(raw_dir, idx_str)
                if img_fp is None:
                    continue
                img = _read_ctc_image(img_fp)
                gt = tifffile.imread(str(seg_fp)).astype("int32")
                yield SegSample(
                    sample_id=f"{ds}/{split}/{idx_str}",
                    image=img,
                    gt_mask=gt,
                    meta={"dataset": ds, "split": split, "seg_path": str(seg_fp)},
                )
                yielded += 1
                if (
                    self.max_segmentation_samples is not None
                    and yielded >= self.max_segmentation_samples
                ):
                    return

    def _iter_tracking_specs(self, tra_dir: Path) -> list[dict]:
        specs = _load_tra_sequence_specs(tra_dir, self.tracking_sequence_json)
        if specs is not None:
            return specs
        print(
            f"cellseek-benchmark: {tra_dir / self.tracking_sequence_json} not found; "
            f"falling back to one sequence per split (full frame range).",
            flush=True,
        )
        return []

    def _yield_track_sequence(
        self,
        *,
        ds: str,
        split: str,
        seq_spec: dict,
        seq_index: int,
        frame_paths: list[Path],
        tra_mask_by_t: dict[int, np.ndarray],
        tra_dir: Path,
        seg_mask_by_t: dict[int, np.ndarray],
        t_start: int,
        t_end: int,
        seq_id: str | None = None,
    ):
        max_t = len(frame_paths) - 1
        if t_start > max_t:
            return
        t_end = min(int(t_end), max_t)
        if t_end < t_start:
            return

        sub_paths = frame_paths[t_start : t_end + 1]
        if self.max_frames_per_sequence is not None:
            sub_paths = sub_paths[: self.max_frames_per_sequence]
        if not sub_paths:
            return
        t_end_eff = t_start + len(sub_paths) - 1

        valid_raw = seq_spec.get("Valid_IDs")
        valid_ids = None if valid_raw is None else {int(x) for x in valid_raw}

        frames = [_read_ctc_image(fp) for fp in sub_paths]
        gt_tracks = _gt_tracks_from_tra_window(
            tra_mask_by_t, t_start, t_end_eff, valid_ids
        )

        gt_masks: list[np.ndarray | None] | None = None
        frame_seg_sample_ids: list[str] = []
        if seg_mask_by_t:
            gt_masks = []
            for fp in sub_paths:
                idx_str = _raw_frame_idx_str(fp)
                global_t = int(idx_str) if idx_str is not None else -1
                gt_masks.append(seg_mask_by_t.get(global_t))
                frame_seg_sample_ids.append(
                    f"{ds}/{split}/{idx_str}" if idx_str is not None else f"{ds}/{split}/?"
                )

        gt_tra_masks: list[np.ndarray] = []
        for fp in sub_paths:
            idx_str = _raw_frame_idx_str(fp)
            global_t = int(idx_str) if idx_str is not None else -1
            tra = tra_mask_by_t.get(global_t)
            if tra is None:
                h, w = frames[len(gt_tra_masks)].shape[:2]
                gt_tra_masks.append(np.zeros((h, w), dtype=np.int32))
            else:
                m = tra.astype(np.int32, copy=True)
                if valid_ids is not None:
                    keep = np.isin(m, sorted(valid_ids | {0}))
                    m = np.where(keep, m, 0)
                gt_tra_masks.append(m)

        if seq_id is None:
            seq_id = f"{ds}_{split}_seq{seq_index:03d}_f{t_start:04d}-{t_end_eff:04d}"
        cc_raw = seq_spec.get("Constant_Cell_Count")
        meta = {
            "dataset": ds,
            "split": split,
            "sequence_index": seq_index,
            "start_frame": t_start,
            "end_frame": t_end_eff,
            "n_frames": len(frames),
            "constant_cell_count": int(cc_raw) if cc_raw is not None else None,
            "valid_ids": sorted(valid_ids) if valid_ids is not None else None,
            "tracking_json": self.tracking_sequence_json,
            "man_track_path": str(tra_dir / "man_track.txt"),
            "tra_global_t_start": int(t_start),
        }
        if frame_seg_sample_ids:
            meta["frame_seg_sample_ids"] = frame_seg_sample_ids

        yield TrackSequence(
            seq_id=seq_id,
            frames=frames,
            gt_masks=gt_masks,
            gt_tra_masks=gt_tra_masks,
            gt_tracks=gt_tracks,
            meta=meta,
        )

    def iter_tracking(self):
        emitted = 0
        for ds, split, raw_dir, gt_dir in self._iter_dataset_split():
            tra_dir = gt_dir / "TRA"
            if not tra_dir.exists():
                continue
            frame_paths = _sorted_raw_frame_paths(raw_dir)
            if frame_paths is None:
                continue
            if not frame_paths:
                continue

            tra_mask_by_t: dict[int, np.ndarray] = {}
            for tra_fp in _sorted_tra_mask_paths(tra_dir):
                ti = _tra_time_int(tra_fp)
                if ti is not None:
                    tra_mask_by_t[ti] = tifffile.imread(str(tra_fp)).astype(np.int32)

            seg_dir = gt_dir / "SEG"
            seg_mask_by_t: dict[int, np.ndarray] = {}
            if seg_dir.is_dir():
                for seg_fp in _sorted_seg_paths(seg_dir):
                    ti = _seg_time_int(seg_fp)
                    if ti is not None:
                        seg_mask_by_t[ti] = tifffile.imread(str(seg_fp)).astype(np.int32)

            specs = self._iter_tracking_specs(tra_dir)
            if specs:
                print(
                    f"cellseek-benchmark: {ds}/{split} — {len(specs)} tracking sub-sequences "
                    f"from {self.tracking_sequence_json}",
                    flush=True,
                )
                for seq_index, spec in enumerate(specs):
                    yield from self._yield_track_sequence(
                        ds=ds,
                        split=split,
                        seq_spec=spec,
                        seq_index=seq_index,
                        frame_paths=frame_paths,
                        tra_mask_by_t=tra_mask_by_t,
                        tra_dir=tra_dir,
                        seg_mask_by_t=seg_mask_by_t,
                        t_start=int(spec["Start_Frame"]),
                        t_end=int(spec["End_Frame"]),
                    )
                    emitted += 1
                    if (
                        self.max_tracking_sequences is not None
                        and emitted >= self.max_tracking_sequences
                    ):
                        return
            else:
                max_t = len(frame_paths) - 1
                yield from self._yield_track_sequence(
                    ds=ds,
                    split=split,
                    seq_spec={"Start_Frame": 0, "End_Frame": max_t, "Valid_IDs": None},
                    seq_index=0,
                    frame_paths=frame_paths,
                    tra_mask_by_t=tra_mask_by_t,
                    tra_dir=tra_dir,
                    seg_mask_by_t=seg_mask_by_t,
                    t_start=0,
                    t_end=max_t,
                    seq_id=f"{ds}_{split}",
                )
                emitted += 1
                if (
                    self.max_tracking_sequences is not None
                    and emitted >= self.max_tracking_sequences
                ):
                    return
