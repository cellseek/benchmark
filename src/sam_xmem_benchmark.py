#!/usr/bin/env python3
"""
SAM + XMem Benchmark Framework

This module implements a comprehensive benchmarking system for cell segmentation
and tracking using SAM for segmentation and XMem for tracking on the Cell Tracking Challenge dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional imports
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    raise ImportError("NumPy is required but not available")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback tqdm
    def tqdm(iterable, desc=""):
        return iterable


# Add SAM and XMem paths
sam_path = Path(__file__).parent.parent.parent / "sam"
xmem_path = Path(__file__).parent.parent.parent / "xmem"
sys.path.append(str(sam_path))
sys.path.append(str(xmem_path))

# Import SAM and XMem
try:
    from sam.model import CellSAM

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn("SAM not available. Please check SAM installation.")

try:
    import torch
    from xmem import XMem

    XMEM_AVAILABLE = True
except ImportError:
    XMEM_AVAILABLE = False
    warnings.warn("XMem not available. Please check XMem installation.")

# Image processing
try:
    from skimage import io, measure, morphology, segmentation

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available. Using opencv for image I/O.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration parameters
DEFAULT_SAM_PARAMS = {
    "diameter": 30,  # Approximate cell diameter in pixels
    "channels": [0, 0],  # [cytoplasm, nucleus] channels (0,0 for grayscale)
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_size": 15,  # Minimum cell size in pixels
}

DEFAULT_XMEM_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,  # Use automatic mixed precision
    "mem_every": 5,  # Memory frames interval
}

DEFAULT_BENCHMARK_SETTINGS = {
    "max_sequences_per_dataset": None,  # None means all sequences
    "save_intermediate_results": True,
    "generate_visualizations": False,
}


def load_image(image_path: Path) -> np.ndarray:
    """Load image using available backend."""
    if SKIMAGE_AVAILABLE:
        return io.imread(str(image_path))
    elif CV2_AVAILABLE:
        return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    else:
        raise ImportError(
            "No image loading backend available. Install scikit-image or opencv-python."
        )


def save_image(image: np.ndarray, image_path: Path):
    """Save image using available backend."""
    if SKIMAGE_AVAILABLE:
        io.imsave(str(image_path), image)
    elif CV2_AVAILABLE:
        cv2.imwrite(str(image_path), image)
    else:
        raise ImportError(
            "No image saving backend available. Install scikit-image or opencv-python."
        )


@dataclass
class DatasetInfo:
    """Information about a Cell Tracking Challenge dataset."""

    name: str
    path: Path
    microscopy_type: str  # BF, DIC, Fluo, PhC
    dimensionality: str  # 2D, 3D
    cell_type: str  # HSC, MuSC, HeLa, etc.
    sequences: List[str]  # ['01', '02']
    has_ground_truth: bool

    @property
    def full_name(self) -> str:
        return f"{self.microscopy_type}-{self.dimensionality}-{self.cell_type}"


@dataclass
class SegmentationMetrics:
    """Segmentation evaluation metrics."""

    iou: float
    dice: float
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    detection_rate: float
    false_positive_rate: float
    seg_score: float  # Official CTC segmentation score

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TrackingMetrics:
    """Tracking evaluation metrics."""

    tra_score: float  # Official CTC tracking accuracy
    det_score: float  # Official CTC detection accuracy
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision
    idf1: float  # Identity F1 Score
    mt_ratio: float  # Mostly Tracked trajectories ratio
    ml_ratio: float  # Mostly Lost trajectories ratio
    id_switches: int  # Number of identity switches
    fragmentation: int  # Number of trajectory fragmentations

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics."""

    segmentation_time: float  # seconds per frame for segmentation
    tracking_time: float  # seconds per frame for tracking
    total_time: float  # total processing time
    memory_usage: float  # GB peak memory
    fps: float  # frames per second

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for SAM + XMem on a dataset."""

    dataset_name: str
    sequence: str
    segmentation_metrics: Optional[SegmentationMetrics]
    tracking_metrics: Optional[TrackingMetrics]
    performance_metrics: PerformanceMetrics
    timestamp: str
    sam_parameters: Dict[str, Any]
    xmem_parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.segmentation_metrics:
            result["segmentation_metrics"] = self.segmentation_metrics.to_dict()
        if self.tracking_metrics:
            result["tracking_metrics"] = self.tracking_metrics.to_dict()
        result["performance_metrics"] = self.performance_metrics.to_dict()
        return result


class CTCDatasetLoader:
    """Loader for Cell Tracking Challenge datasets."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.datasets_info = self._discover_datasets()

    def _discover_datasets(self) -> Dict[str, DatasetInfo]:
        """Discover all available datasets in the data directory."""
        datasets = {}

        for split in ["train", "test"]:
            split_path = self.data_root / split
            if not split_path.exists():
                continue

            for dataset_dir in split_path.iterdir():
                if not dataset_dir.is_dir():
                    continue

                # Parse dataset name (e.g., "BF-C2DL-HSC")
                parts = dataset_dir.name.split("-")
                if len(parts) != 3:
                    continue

                microscopy_type, dimensionality, cell_type = parts

                # Find sequences
                sequences = []
                for seq_dir in dataset_dir.iterdir():
                    if seq_dir.is_dir() and seq_dir.name.isdigit():
                        sequences.append(seq_dir.name)

                sequences.sort()

                dataset_key = f"{split}_{dataset_dir.name}"
                datasets[dataset_key] = DatasetInfo(
                    name=dataset_dir.name,
                    path=dataset_dir,
                    microscopy_type=microscopy_type,
                    dimensionality=dimensionality,
                    cell_type=cell_type,
                    sequences=sequences,
                    has_ground_truth=(split == "train"),
                )

        return datasets

    def get_dataset_info(
        self, dataset_name: str, split: str = "train"
    ) -> Optional[DatasetInfo]:
        """Get information about a specific dataset."""
        key = f"{split}_{dataset_name}"
        return self.datasets_info.get(key)

    def list_datasets(self, split: Optional[str] = None) -> List[str]:
        """List available datasets."""
        if split:
            return [
                name.split("_", 1)[1]
                for name in self.datasets_info.keys()
                if name.startswith(f"{split}_")
            ]
        return list(set(name.split("_", 1)[1] for name in self.datasets_info.keys()))

    def load_sequence_images(
        self, dataset_name: str, sequence: str, split: str = "train"
    ) -> List[np.ndarray]:
        """Load all images from a sequence."""
        dataset_info = self.get_dataset_info(dataset_name, split)
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_name} not found in {split} split")

        sequence_path = dataset_info.path / sequence
        if not sequence_path.exists():
            raise ValueError(f"Sequence {sequence} not found in {dataset_name}")

        # Find all TIFF files
        image_files = sorted(sequence_path.glob("t*.tif"))
        images = []

        for img_file in image_files:
            img = load_image(img_file)
            images.append(img)

        return images

    def load_ground_truth_tracking(
        self, dataset_name: str, sequence: str
    ) -> Dict[int, np.ndarray]:
        """Load ground truth tracking masks."""
        dataset_info = self.get_dataset_info(dataset_name, "train")
        if not dataset_info or not dataset_info.has_ground_truth:
            raise ValueError(f"No ground truth available for {dataset_name}")

        gt_tra_path = dataset_info.path / f"{sequence}_GT" / "TRA"
        if not gt_tra_path.exists():
            return {}

        # Load tracking masks
        track_files = sorted(gt_tra_path.glob("man_track*.tif"))
        tracking_masks = {}

        for track_file in track_files:
            if track_file.name == "man_track.txt":
                continue
            # Extract frame number from filename
            frame_num = int(track_file.stem.split("track")[1])
            mask = load_image(track_file)
            tracking_masks[frame_num] = mask

        return tracking_masks

    def load_lineage_info(self, dataset_name: str, sequence: str) -> List[Dict]:
        """Load lineage information from man_track.txt."""
        dataset_info = self.get_dataset_info(dataset_name, "train")
        if not dataset_info or not dataset_info.has_ground_truth:
            raise ValueError(f"No ground truth available for {dataset_name}")

        gt_tra_path = dataset_info.path / f"{sequence}_GT" / "TRA"
        lineage_file = gt_tra_path / "man_track.txt"
        lineages = []

        if lineage_file.exists():
            with open(lineage_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        lineages.append(
                            {
                                "track_id": int(parts[0]),
                                "start_frame": int(parts[1]),
                                "end_frame": int(parts[2]),
                                "parent_id": int(parts[3]) if parts[3] != "0" else None,
                            }
                        )

        return lineages


class SAMXMemBenchmark:
    """Main benchmark class for SAM + XMem pipeline."""

    def __init__(
        self,
        data_root: Path,
        results_dir: Path,
        sam_model_path: Optional[Path] = None,
        xmem_model_path: Optional[Path] = None,
    ):
        self.data_root = Path(data_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.loader = CTCDatasetLoader(data_root)

        # Initialize models
        self.sam_model = None
        self.xmem_model = None

        if SAM_AVAILABLE and sam_model_path:
            self.sam_model = self._initialize_sam(sam_model_path)

        if XMEM_AVAILABLE and xmem_model_path:
            self.xmem_model = self._initialize_xmem(xmem_model_path)

    def _initialize_sam(self, model_path: Path) -> "CellSAM":
        """Initialize SAM model."""
        logger.info(f"Loading SAM model from {model_path}")
        try:
            model = CellSAM()
            # Load model weights if needed
            return model
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            return None

    def _initialize_xmem(self, model_path: Path) -> "XMem":
        """Initialize XMem model."""
        logger.info(f"Loading XMem model from {model_path}")
        try:
            # Create args object for XMem
            class Args:
                def __init__(self):
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

            args = Args()
            model = XMem(str(model_path), args)
            return model
        except Exception as e:
            logger.error(f"Failed to load XMem model: {e}")
            return None

    def run_sam_segmentation(
        self, images: List[np.ndarray], **sam_params
    ) -> List[np.ndarray]:
        """Run SAM segmentation on a sequence of images."""
        if not self.sam_model:
            raise ValueError("SAM model not initialized")

        segmentation_masks = []

        for i, image in enumerate(tqdm(images, desc="SAM Segmentation")):
            try:
                # Run SAM segmentation
                mask, _ = self.sam_model.eval(image, **sam_params)
                segmentation_masks.append(mask)
            except Exception as e:
                logger.error(f"SAM segmentation failed for frame {i}: {e}")
                # Use empty mask as fallback
                segmentation_masks.append(np.zeros_like(image))

        return segmentation_masks

    def run_xmem_tracking(
        self, images: List[np.ndarray], initial_mask: np.ndarray, **xmem_params
    ) -> List[np.ndarray]:
        """Run XMem tracking given initial mask."""
        if not self.xmem_model:
            raise ValueError("XMem model not initialized")

        try:
            # Run XMem tracking
            masks, _, _ = self.xmem_model.generator(images, initial_mask)
            return masks
        except Exception as e:
            logger.error(f"XMem tracking failed: {e}")
            # Return empty masks as fallback
            return [np.zeros_like(img) for img in images]

    def evaluate_segmentation(
        self, pred_masks: List[np.ndarray], gt_masks: Dict[int, np.ndarray]
    ) -> SegmentationMetrics:
        """Evaluate segmentation performance."""
        ious = []
        dices = []
        precisions = []
        recalls = []

        for frame_idx, pred_mask in enumerate(pred_masks):
            if frame_idx in gt_masks:
                gt_mask = gt_masks[frame_idx]

                # Compute IoU
                intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
                union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
                iou = intersection / union if union > 0 else 0.0
                ious.append(iou)

                # Compute Dice
                dice = (
                    2 * intersection / (np.sum(pred_mask > 0) + np.sum(gt_mask > 0))
                    if (np.sum(pred_mask > 0) + np.sum(gt_mask > 0)) > 0
                    else 0.0
                )
                dices.append(dice)

                # Compute precision and recall
                tp = intersection
                fp = np.sum(pred_mask > 0) - tp
                fn = np.sum(gt_mask > 0) - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                precisions.append(precision)
                recalls.append(recall)

        # Compute aggregate metrics
        mean_iou = np.mean(ious) if ious else 0.0
        mean_dice = np.mean(dices) if dices else 0.0
        mean_precision = np.mean(precisions) if precisions else 0.0
        mean_recall = np.mean(recalls) if recalls else 0.0
        f1_score = (
            2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            if (mean_precision + mean_recall) > 0
            else 0.0
        )

        return SegmentationMetrics(
            iou=mean_iou,
            dice=mean_dice,
            precision=mean_precision,
            recall=mean_recall,
            f1_score=f1_score,
            average_precision=mean_precision,  # Simplified
            detection_rate=mean_recall,
            false_positive_rate=1.0 - mean_precision if mean_precision > 0 else 1.0,
            seg_score=mean_dice,  # Use Dice as proxy for CTC SEG score
        )

    def evaluate_tracking(
        self, pred_masks: List[np.ndarray], gt_masks: Dict[int, np.ndarray]
    ) -> TrackingMetrics:
        """Evaluate tracking performance."""
        # Simplified tracking evaluation
        # In a full implementation, you would compute proper MOTA, MOTP, etc.

        # For now, compute simple overlap-based metrics
        overlaps = []

        for frame_idx, pred_mask in enumerate(pred_masks):
            if frame_idx in gt_masks:
                gt_mask = gt_masks[frame_idx]

                # Compute overlap
                intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
                union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
                overlap = intersection / union if union > 0 else 0.0
                overlaps.append(overlap)

        mean_overlap = np.mean(overlaps) if overlaps else 0.0

        return TrackingMetrics(
            tra_score=mean_overlap,  # Simplified
            det_score=mean_overlap,  # Simplified
            mota=mean_overlap,
            motp=mean_overlap,
            idf1=mean_overlap,
            mt_ratio=mean_overlap,
            ml_ratio=1.0 - mean_overlap,
            id_switches=0,  # Would need proper tracking evaluation
            fragmentation=0,  # Would need proper tracking evaluation
        )

    def run_benchmark(
        self,
        dataset_name: str,
        sequence: str,
        sam_params: Dict = None,
        xmem_params: Dict = None,
    ) -> BenchmarkResult:
        """Run complete benchmark on a dataset sequence."""
        # Use default parameters if none provided
        if sam_params is None:
            sam_params = DEFAULT_SAM_PARAMS.copy()
        else:
            # Merge with defaults
            merged_sam_params = DEFAULT_SAM_PARAMS.copy()
            merged_sam_params.update(sam_params)
            sam_params = merged_sam_params

        if xmem_params is None:
            xmem_params = DEFAULT_XMEM_PARAMS.copy()
        else:
            # Merge with defaults
            merged_xmem_params = DEFAULT_XMEM_PARAMS.copy()
            merged_xmem_params.update(xmem_params)
            xmem_params = merged_xmem_params

        logger.info(f"Running benchmark on {dataset_name} sequence {sequence}")

        # Load data
        images = self.loader.load_sequence_images(dataset_name, sequence)
        gt_tracking_masks = self.loader.load_ground_truth_tracking(
            dataset_name, sequence
        )

        if not images:
            raise ValueError(f"No images found for {dataset_name} sequence {sequence}")

        # Get initial mask from ground truth (first frame)
        initial_mask = gt_tracking_masks.get(0, np.zeros_like(images[0]))

        # Timing
        start_time = time.time()

        # Run SAM segmentation
        seg_start = time.time()
        sam_masks = self.run_sam_segmentation(images, **sam_params)
        seg_time = time.time() - seg_start

        # Run XMem tracking with ground truth initial mask
        track_start = time.time()
        xmem_masks = self.run_xmem_tracking(images, initial_mask, **xmem_params)
        track_time = time.time() - track_start

        total_time = time.time() - start_time

        # Evaluate results
        seg_metrics = self.evaluate_segmentation(sam_masks, gt_tracking_masks)
        track_metrics = self.evaluate_tracking(xmem_masks, gt_tracking_masks)

        # Performance metrics
        perf_metrics = PerformanceMetrics(
            segmentation_time=seg_time / len(images),
            tracking_time=track_time / len(images),
            total_time=total_time,
            memory_usage=0.0,  # Would need proper memory profiling
            fps=len(images) / total_time,
        )

        # Create result
        result = BenchmarkResult(
            dataset_name=dataset_name,
            sequence=sequence,
            segmentation_metrics=seg_metrics,
            tracking_metrics=track_metrics,
            performance_metrics=perf_metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            sam_parameters=sam_params,
            xmem_parameters=xmem_params,
        )

        return result

    def save_results(
        self, results: List[BenchmarkResult], filename: str = "benchmark_results.json"
    ):
        """Save benchmark results to file."""
        results_file = self.results_dir / filename

        results_data = [result.to_dict() for result in results]

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def load_results(self, filename: str = "benchmark_results.json") -> List[Dict]:
        """Load benchmark results from file."""
        results_file = self.results_dir / filename

        if results_file.exists():
            with open(results_file, "r") as f:
                return json.load(f)
        return []

    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a summary report from benchmark results."""
        if not results:
            return "No results to report."

        report = ["# SAM + XMem Benchmark Report\n"]

        # Summary statistics
        seg_scores = [
            r.segmentation_metrics.dice for r in results if r.segmentation_metrics
        ]
        track_scores = [
            r.tracking_metrics.tra_score for r in results if r.tracking_metrics
        ]
        fps_scores = [r.performance_metrics.fps for r in results]

        report.append(f"## Summary ({len(results)} sequences)\n")
        report.append(
            f"- Average Segmentation Dice: {np.mean(seg_scores):.3f} ± {np.std(seg_scores):.3f}"
        )
        report.append(
            f"- Average Tracking Score: {np.mean(track_scores):.3f} ± {np.std(track_scores):.3f}"
        )
        report.append(
            f"- Average FPS: {np.mean(fps_scores):.2f} ± {np.std(fps_scores):.2f}\n"
        )

        # Per-dataset results
        report.append("## Per-Dataset Results\n")
        for result in results:
            report.append(f"### {result.dataset_name} - Sequence {result.sequence}")
            if result.segmentation_metrics:
                report.append(
                    f"- Segmentation Dice: {result.segmentation_metrics.dice:.3f}"
                )
                report.append(
                    f"- Segmentation IoU: {result.segmentation_metrics.iou:.3f}"
                )
            if result.tracking_metrics:
                report.append(
                    f"- Tracking Score: {result.tracking_metrics.tra_score:.3f}"
                )
            report.append(f"- FPS: {result.performance_metrics.fps:.2f}")
            report.append("")

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    data_root = Path("data/cell_tracking_challenge")
    results_dir = Path("benchmark_results")

    # Model paths (adjust these to your actual model locations)
    sam_model_path = Path("../sam/checkpoints/sam_model.pth")  # Adjust path
    xmem_model_path = Path("../xmem/checkpoints/XMem-s012.pth")

    # Initialize benchmark
    benchmark = SAMXMemBenchmark(
        data_root=data_root,
        results_dir=results_dir,
        sam_model_path=sam_model_path,
        xmem_model_path=xmem_model_path,
    )

    # Get available datasets
    datasets = benchmark.loader.list_datasets("train")
    logger.info(f"Available datasets: {datasets}")

    # Run benchmark on first dataset
    if datasets:
        dataset_name = datasets[0]
        sequences = benchmark.loader.get_dataset_info(dataset_name).sequences

        results = []
        for sequence in sequences[:1]:  # Test on first sequence only
            try:
                result = benchmark.run_benchmark(dataset_name, sequence)
                results.append(result)
                logger.info(f"Completed {dataset_name} sequence {sequence}")
            except Exception as e:
                logger.error(
                    f"Failed to run benchmark on {dataset_name} sequence {sequence}: {e}"
                )

        # Save and report results
        benchmark.save_results(results)
        report = benchmark.generate_report(results)

        print("\n" + report)

        # Save report
        with open(results_dir / "benchmark_report.md", "w") as f:
            f.write(report)
