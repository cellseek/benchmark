#!/usr/bin/env python3
"""
Build a small on-disk benchmark mirror: same layout as full datasets, subsampled.

- cellpose / BBBC038: random K images (K=10 by default).
- CTC dataset entries (``ctc_*`` keys): split ``01`` only (one sequence), first K timepoints (default 10)
    so tracking stays consistent with the adapter (contiguous frame indices from 0).

Run from the benchmark repo root (directory that contains ``configs/``):

  python scripts/build_benchmark_subset.py

Then copy ``configs/`` (including ``experiments/``) to a folder (for example
``configs_subset/``), set dataset roots in ``datasets.yaml`` to this subset directory,
and run ``csbench --config-dir`` that folder with ``--benchmark-config experiments/<name>.yaml``.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import yaml

_RAW_FRAME = re.compile(r"^t(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_MAN_TRACK = re.compile(r"^man_track(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_MAN_SEG = re.compile(r"^man_seg(\d+)\.(?:tif|tiff)$", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_source_datasets(repo: Path) -> dict:
    path = repo / "configs" / "datasets.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"]


def _tif_frame_index(tif_path: Path) -> int | None:
    m = _RAW_FRAME.match(tif_path.name)
    return int(m.group(1)) if m else None


def copy_ctc_time_range(
    src_dataset_dir: Path,
    dest_dataset_dir: Path,
    *,
    split: str,
    max_frames: int,
) -> bool:
    raw = src_dataset_dir / split
    gt = src_dataset_dir / f"{split}_GT"
    tra = gt / "TRA"
    seg = gt / "SEG"
    if not raw.is_dir() or not tra.is_dir():
        return False

    dest_raw = dest_dataset_dir / split
    dest_gt = dest_dataset_dir / f"{split}_GT"
    dest_tra = dest_gt / "TRA"
    dest_seg = dest_gt / "SEG"
    dest_raw.mkdir(parents=True, exist_ok=True)
    dest_tra.mkdir(parents=True, exist_ok=True)

    tifs = [p for p in raw.iterdir() if p.is_file() and _RAW_FRAME.match(p.name)]
    tifs.sort(key=lambda p: int(_RAW_FRAME.match(p.name).group(1)))  # type: ignore[union-attr]
    if not tifs:
        return False
    chosen = tifs[: min(max_frames, len(tifs))]
    frame_indices = [_tif_frame_index(p) for p in chosen]
    if any(i is None for i in frame_indices):
        return False
    idx_ints = [int(i) for i in frame_indices]  # type: ignore[arg-type]
    if idx_ints[0] != 0 or idx_ints != list(range(idx_ints[-1] + 1)):
        return False
    indices = set(idx_ints)

    for p in chosen:
        shutil.copy2(p, dest_raw / p.name)

    for tra_fp in sorted(tra.iterdir(), key=lambda p: p.name):
        if not tra_fp.is_file():
            continue
        m = _MAN_TRACK.match(tra_fp.name)
        if not m:
            continue
        fi = int(m.group(1))
        if fi in indices:
            shutil.copy2(tra_fp, dest_tra / tra_fp.name)

    if seg.is_dir():
        dest_seg.mkdir(parents=True, exist_ok=True)
        for seg_fp in sorted(seg.iterdir(), key=lambda p: p.name):
            if not seg_fp.is_file():
                continue
            m = _MAN_SEG.match(seg_fp.name)
            if not m:
                continue
            fi = int(m.group(1))
            if fi in indices:
                shutil.copy2(seg_fp, dest_seg / seg_fp.name)

    return True


def subset_cellpose(src_cfg: dict, dest_root: Path, k: int, rng: random.Random) -> None:
    root = Path(src_cfg["root"])
    splits = src_cfg.get("splits", ["train", "test"])
    img_suf = src_cfg.get("image_suffix", "_img.png")
    mask_suf = src_cfg.get("mask_suffix", "_masks.png")

    pool: list[tuple[str, Path, Path]] = []
    for sp in splits:
        sd = root / sp
        if not sd.is_dir():
            continue
        for img_fp in sorted(sd.glob(f"*{img_suf}")):
            stem = img_fp.name.replace(img_suf, "")
            mask_fp = sd / f"{stem}{mask_suf}"
            if mask_fp.is_file():
                pool.append((sp, img_fp, mask_fp))

    if not pool:
        print("cellpose: no paired samples found; skipping", file=sys.stderr)
        return

    pick = rng.sample(pool, k=min(k, len(pool)))
    out = dest_root / "cellpose"
    for sp, img_fp, mask_fp in pick:
        d = out / sp
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_fp, d / img_fp.name)
        shutil.copy2(mask_fp, d / mask_fp.name)
    print(f"cellpose: copied {len(pick)} image/mask pairs -> {out}")


def subset_bbbc038(src_cfg: dict, dest_root: Path, k: int, rng: random.Random) -> None:
    root = Path(src_cfg["root"])
    sample_dirs = sorted({p.parent.parent for p in root.glob("**/images/*.png")})
    if not sample_dirs:
        print("bbbc038: no **/images/*.png layout found; skipping", file=sys.stderr)
        return
    pick = rng.sample(sample_dirs, k=min(k, len(sample_dirs)))
    out = dest_root / "bbbc038"
    for sdir in pick:
        rel = sdir.relative_to(root)
        target = out / rel
        shutil.copytree(sdir, target, dirs_exist_ok=True)
    print(f"bbbc038: copied {len(pick)} sample folders -> {out}")


def subset_ctc_block(
    key: str,
    src_cfg: dict,
    dest_root: Path,
    max_frames: int,
) -> None:
    src_root = Path(src_cfg["root"])
    names = src_cfg.get("datasets")
    if not names:
        print(f"{key}: no datasets list; skipping", file=sys.stderr)
        return
    ds_name = str(names[0])
    src_ds = src_root / ds_name
    if not src_ds.is_dir():
        print(f"{key}: missing {src_ds}; skipping", file=sys.stderr)
        return

    dest_base = dest_root / key
    dest_ds = dest_base / ds_name
    dest_ds.mkdir(parents=True, exist_ok=True)
    # Mini CTC: split 01 only, first ``max_frames`` frames (see module docstring).
    if copy_ctc_time_range(src_ds, dest_ds, split="01", max_frames=max_frames):
        print(f"{key}: copied {ds_name} split 01 (<= {max_frames} frames) -> {dest_base}")
    else:
        print(f"{key}: could not copy CTC data from {src_ds}; skipping", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/benchmark_subset_10)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random images for cellpose and BBBC038",
    )
    parser.add_argument(
        "--ctc-frames",
        type=int,
        default=10,
        help="Number of consecutive frames (from t0000…) per CTC challenge",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = _repo_root()
    dest = (args.dest or (repo / "benchmark_subset_10")).resolve()
    rng = random.Random(args.seed)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    src_all = _load_source_datasets(repo)

    subset_cellpose(src_all["cellpose"], dest, args.count, rng)
    subset_bbbc038(src_all["bbbc038"], dest, args.count, rng)

    for key, cfg in src_all.items():
        if not key.startswith("ctc_"):
            continue
        if cfg.get("type") != "ctc_cell_tracking":
            continue
        subset_ctc_block(key, cfg, dest, args.ctc_frames)

    print(f"\nDone. Subset root: {dest}")
    print("Update dataset roots in a copy of configs/ to point here, then run:")
    print(
        "  csbench --config-dir <your_configs_dir> "
        "--benchmark-config experiments/<your_experiment>.yaml"
    )


if __name__ == "__main__":
    main()
