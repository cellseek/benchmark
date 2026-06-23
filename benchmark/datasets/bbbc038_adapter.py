from pathlib import Path
import cv2
import numpy as np
import tifffile

from .base import DatasetAdapter
from ..schemas import SegSample


class BBBC038Adapter(DatasetAdapter):
    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        self.image_glob = cfg.get("image_glob", "**/*img*.tif")
        self.mask_glob = cfg.get("mask_glob", "**/*mask*.tif")
        m = cfg.get("max_segmentation_samples")
        self._max_segmentation_samples = int(m) if m is not None else None

    def _mask_index(self):
        idx = {}
        for fp in self.root.glob(self.mask_glob):
            key = fp.stem.replace("mask", "").replace("_", "")
            idx[key] = fp
        return idx

    @staticmethod
    def _load_image(fp: Path) -> np.ndarray:
        if fp.suffix.lower() in {".tif", ".tiff"}:
            return tifffile.imread(str(fp)).astype(np.float32)
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {fp}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    @staticmethod
    def _load_mask(fp: Path) -> np.ndarray:
        if fp.suffix.lower() in {".tif", ".tiff"}:
            gt = tifffile.imread(str(fp))
        else:
            gt = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if gt is None:
            raise RuntimeError(f"Failed to read mask: {fp}")
        gt = gt.astype(np.int32)
        if gt.ndim > 2:
            gt = np.squeeze(gt)
        return gt

    def _iter_png_instance_layout(self):
        """
        Support BBBC038 style:
          <sample_id>/images/<sample_id>.png
          <sample_id>/masks/<instance_id>.png
        """
        image_files = sorted(self.root.glob("**/images/*.png"))
        if self._max_segmentation_samples is not None:
            image_files = image_files[: self._max_segmentation_samples]

        for img_fp in image_files:
            sample_dir = img_fp.parent.parent
            mask_dir = sample_dir / "masks"
            if not mask_dir.exists():
                continue

            img = self._load_image(img_fp)
            instance_masks = sorted(mask_dir.glob("*.png"))
            if not instance_masks:
                continue

            gt = np.zeros(img.shape[:2], dtype=np.int32)
            inst_id = 1
            for mfp in instance_masks:
                m = self._load_mask(mfp)
                if m.shape[:2] != gt.shape[:2]:
                    continue
                fg = m > 0
                if fg.any():
                    gt[fg] = inst_id
                    inst_id += 1

            yield SegSample(
                sample_id=sample_dir.name,
                image=img,
                gt_mask=gt,
                meta={"img_path": str(img_fp), "mask_dir": str(mask_dir)},
            )

    def iter_segmentation(self):
        if not self.root.exists():
            return

        idx = self._mask_index()
        image_files = sorted(self.root.glob(self.image_glob))
        if self._max_segmentation_samples is not None:
            image_files = image_files[: self._max_segmentation_samples]
        # Path 1: legacy paired image/mask files via configured globs.
        if image_files:
            for img_fp in image_files:
                key = img_fp.stem.replace("img", "").replace("_", "")
                gt_fp = idx.get(key)

                img = self._load_image(img_fp)
                gt = None
                if gt_fp is not None and gt_fp.exists():
                    gt = self._load_mask(gt_fp)

                yield SegSample(
                    sample_id=img_fp.stem,
                    image=img,
                    gt_mask=gt,
                    meta={"img_path": str(img_fp), "mask_path": str(gt_fp) if gt_fp else None},
                )
            return

        # Path 2: BBBC038-like PNG directory layout with per-instance masks.
        yield from self._iter_png_instance_layout()

    def iter_tracking(self):
        return iter([])
