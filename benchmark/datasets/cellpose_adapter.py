from pathlib import Path
import cv2
import numpy as np

from .base import DatasetAdapter
from ..schemas import SegSample


class CellposeAdapter(DatasetAdapter):
    def __init__(self, cfg: dict):
        self.root = Path(cfg["root"])
        self.splits = cfg.get("splits", ["train", "test"])
        self.image_suffix = cfg.get("image_suffix", "_img.png")
        self.mask_suffix = cfg.get("mask_suffix", "_masks.png")
        m = cfg.get("max_segmentation_samples")
        self._max_segmentation_samples = int(m) if m is not None else None

    def iter_segmentation(self):
        yielded = 0
        for split in self.splits:
            split_dir = self.root / split
            if not split_dir.exists():
                continue
            for img_fp in sorted(split_dir.glob(f"*{self.image_suffix}")):
                stem = img_fp.name.replace(self.image_suffix, "")
                mask_fp = split_dir / f"{stem}{self.mask_suffix}"

                img = cv2.imread(str(img_fp), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                gt = None
                if mask_fp.exists():
                    gt = cv2.imread(str(mask_fp), cv2.IMREAD_UNCHANGED)
                    if gt is not None:
                        gt = gt.astype(np.int32)

                yield SegSample(
                    sample_id=f"{split}/{stem}",
                    image=img.astype(np.float32),
                    gt_mask=gt,
                    meta={"img_path": str(img_fp), "mask_path": str(mask_fp)},
                )
                yielded += 1
                if (
                    self._max_segmentation_samples is not None
                    and yielded >= self._max_segmentation_samples
                ):
                    return

    def iter_tracking(self):
        return iter([])
