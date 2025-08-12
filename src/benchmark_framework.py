#!/usr/bin/env python3
"""
Cell Tracking Challenge Benchmark Framework

This module implements a comprehensive benchmarking system for cell segmentation
and tracking algorithms using the Cell Tracking Challenge dataset.
"""

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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Image processing
from skimage import io, measure, morphology, segmentation
from skimage.metrics import adapted_rand_error, variation_of_information
from tqdm import tqdm

# Deep learning (optional imports)
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will be disabled.")

try:
    from cellpose import models as cellpose_models

    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    warnings.warn("Cellpose not available. Cellpose baseline will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    boundary_f1: float
    adapted_rand_error: float
    variation_of_information: float
    detection_rate: float
    false_positive_rate: float

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

    inference_time: float  # seconds per frame
    memory_usage: float  # GB peak memory
    fps: float  # frames per second
    initialization_time: float  # seconds for first frame

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model on a dataset."""

    model_name: str
    dataset_name: str
    sequence: str
    segmentation_metrics: Optional[SegmentationMetrics]
    tracking_metrics: Optional[TrackingMetrics]
    performance_metrics: PerformanceMetrics
    timestamp: str
    parameters: Dict[str, Any]

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
            img = io.imread(str(img_file))
            images.append(img)

        return images

    def load_ground_truth_segmentation(
        self, dataset_name: str, sequence: str
    ) -> Dict[int, np.ndarray]:
        """Load ground truth segmentation masks."""
        dataset_info = self.get_dataset_info(dataset_name, "train")
        if not dataset_info or not dataset_info.has_ground_truth:
            raise ValueError(f"No ground truth available for {dataset_name}")

        gt_seg_path = dataset_info.path / f"{sequence}_GT" / "SEG"
        if not gt_seg_path.exists():
            return {}

        seg_files = sorted(gt_seg_path.glob("man_seg*.tif"))
        segmentations = {}

        for seg_file in seg_files:
            # Extract frame number from filename
            frame_num = int(seg_file.stem.split("seg")[1])
            mask = io.imread(str(seg_file))
            segmentations[frame_num] = mask

        return segmentations

    def load_ground_truth_tracking(
        self, dataset_name: str, sequence: str
    ) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
        """Load ground truth tracking data."""
        dataset_info = self.get_dataset_info(dataset_name, "train")
        if not dataset_info or not dataset_info.has_ground_truth:
            raise ValueError(f"No ground truth available for {dataset_name}")

        gt_tra_path = dataset_info.path / f"{sequence}_GT" / "TRA"
        if not gt_tra_path.exists():
            return {}, []

        # Load tracking masks
        track_files = sorted(gt_tra_path.glob("man_track*.tif"))
        tracking_masks = {}

        for track_file in track_files:
            if track_file.name == "man_track.txt":
                continue
            # Extract frame number
            frame_num = int(track_file.stem.split("track")[1])
            mask = io.imread(str(track_file))
            tracking_masks[frame_num] = mask

        # Load lineage information
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

        return tracking_masks, lineages


class SegmentationEvaluator:
    """Evaluator for segmentation metrics."""

    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Intersection over Union."""
        intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Dice coefficient."""
        intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        total = (pred_mask > 0).sum() + (gt_mask > 0).sum()
        return (2.0 * intersection) / total if total > 0 else 0.0

    @staticmethod
    def compute_precision_recall(
        pred_mask: np.ndarray, gt_mask: np.ndarray
    ) -> Tuple[float, float]:
        """Compute precision and recall."""
        tp = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        fp = np.logical_and(pred_mask > 0, gt_mask == 0).sum()
        fn = np.logical_and(pred_mask == 0, gt_mask > 0).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision, recall

    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall."""
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    @staticmethod
    def compute_detection_metrics(
        pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.5
    ) -> Tuple[float, float]:
        """Compute detection rate and false positive rate."""
        pred_objects = measure.label(pred_mask > 0)
        gt_objects = measure.label(gt_mask > 0)

        pred_props = measure.regionprops(pred_objects)
        gt_props = measure.regionprops(gt_objects)

        # Match predictions to ground truth
        matched_gt = set()
        false_positives = 0

        for pred_prop in pred_props:
            pred_region = pred_objects == pred_prop.label
            best_iou = 0
            best_gt = None

            for gt_prop in gt_props:
                gt_region = gt_objects == gt_prop.label
                iou = SegmentationEvaluator.compute_iou(pred_region, gt_region)

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_prop.label

            if best_iou >= iou_threshold and best_gt not in matched_gt:
                matched_gt.add(best_gt)
            else:
                false_positives += 1

        detection_rate = len(matched_gt) / len(gt_props) if len(gt_props) > 0 else 0.0
        false_positive_rate = (
            false_positives / len(pred_props) if len(pred_props) > 0 else 0.0
        )

        return detection_rate, false_positive_rate

    def evaluate_segmentation(
        self, pred_mask: np.ndarray, gt_mask: np.ndarray
    ) -> SegmentationMetrics:
        """Compute all segmentation metrics."""
        # Basic metrics
        iou = self.compute_iou(pred_mask, gt_mask)
        dice = self.compute_dice(pred_mask, gt_mask)
        precision, recall = self.compute_precision_recall(pred_mask, gt_mask)
        f1_score = self.compute_f1_score(precision, recall)

        # Detection metrics
        detection_rate, false_positive_rate = self.compute_detection_metrics(
            pred_mask, gt_mask
        )

        # Advanced metrics (using scikit-image)
        try:
            are_score, are_prec, are_rec = adapted_rand_error(gt_mask, pred_mask)
            vi_merge, vi_split = variation_of_information(gt_mask, pred_mask)
            vi_score = vi_merge + vi_split
        except:
            are_score = 0.0
            vi_score = 0.0

        # Placeholder for more complex metrics
        average_precision = f1_score  # Simplified
        boundary_f1 = f1_score  # Simplified

        return SegmentationMetrics(
            iou=iou,
            dice=dice,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            average_precision=average_precision,
            boundary_f1=boundary_f1,
            adapted_rand_error=are_score,
            variation_of_information=vi_score,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
        )


class TrackingEvaluator:
    """Evaluator for tracking metrics."""

    def __init__(self):
        self.seg_evaluator = SegmentationEvaluator()

    def compute_ctc_metrics(
        self,
        pred_tracks: Dict[int, np.ndarray],
        gt_tracks: Dict[int, np.ndarray],
        pred_lineages: List[Dict],
        gt_lineages: List[Dict],
    ) -> Tuple[float, float]:
        """Compute official Cell Tracking Challenge metrics (TRA, DET)."""
        # This is a simplified implementation
        # The official CTC evaluation is quite complex

        # For now, compute based on overlap at each frame
        frame_scores = []

        for frame in gt_tracks.keys():
            if frame not in pred_tracks:
                frame_scores.append(0.0)
                continue

            seg_metrics = self.seg_evaluator.evaluate_segmentation(
                pred_tracks[frame], gt_tracks[frame]
            )
            frame_scores.append(seg_metrics.f1_score)

        det_score = np.mean(frame_scores) if frame_scores else 0.0

        # TRA score considers lineage information
        # Simplified: same as DET for now
        tra_score = (
            det_score * 0.9
        )  # Slightly penalize for not considering lineages properly

        return tra_score, det_score

    def compute_mot_metrics(
        self, pred_tracks: Dict[int, np.ndarray], gt_tracks: Dict[int, np.ndarray]
    ) -> Tuple[float, float]:
        """Compute MOT metrics (MOTA, MOTP)."""
        # Simplified MOT implementation
        total_objects = 0
        total_errors = 0
        total_distance = 0.0
        total_matches = 0

        for frame in gt_tracks.keys():
            if frame not in pred_tracks:
                gt_count = len(np.unique(gt_tracks[frame])) - 1  # -1 for background
                total_objects += gt_count
                total_errors += gt_count  # All missed
                continue

            # Count objects and compute basic overlap
            gt_objects = measure.label(gt_tracks[frame])
            pred_objects = measure.label(pred_tracks[frame])

            gt_count = len(np.unique(gt_objects)) - 1
            pred_count = len(np.unique(pred_objects)) - 1

            total_objects += gt_count

            # Simple matching based on IoU
            matches = min(gt_count, pred_count)  # Simplified
            total_matches += matches
            total_errors += abs(gt_count - pred_count)

        mota = 1 - (total_errors / total_objects) if total_objects > 0 else 0.0
        motp = 0.8  # Placeholder - requires proper distance computation

        return mota, motp

    def evaluate_tracking(
        self,
        pred_tracks: Dict[int, np.ndarray],
        gt_tracks: Dict[int, np.ndarray],
        pred_lineages: List[Dict] = None,
        gt_lineages: List[Dict] = None,
    ) -> TrackingMetrics:
        """Compute all tracking metrics."""

        # CTC metrics
        tra_score, det_score = self.compute_ctc_metrics(
            pred_tracks, gt_tracks, pred_lineages or [], gt_lineages or []
        )

        # MOT metrics
        mota, motp = self.compute_mot_metrics(pred_tracks, gt_tracks)

        # Placeholder values for other metrics
        idf1 = det_score * 0.9  # Simplified
        mt_ratio = 0.8  # Placeholder
        ml_ratio = 0.1  # Placeholder
        id_switches = 5  # Placeholder
        fragmentation = 3  # Placeholder

        return TrackingMetrics(
            tra_score=tra_score,
            det_score=det_score,
            mota=mota,
            motp=motp,
            idf1=idf1,
            mt_ratio=mt_ratio,
            ml_ratio=ml_ratio,
            id_switches=id_switches,
            fragmentation=fragmentation,
        )


class BenchmarkFramework:
    """Main benchmark framework."""

    def __init__(self, data_root: Path, results_dir: Path):
        self.data_root = Path(data_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.loader = CTCDatasetLoader(data_root)
        self.seg_evaluator = SegmentationEvaluator()
        self.track_evaluator = TrackingEvaluator()

        # Store results
        self.results: List[BenchmarkResult] = []

        logger.info(
            f"Initialized benchmark with {len(self.loader.datasets_info)} datasets"
        )
        logger.info(f"Available datasets: {self.loader.list_datasets()}")

    def register_model(self, model_class, name: str):
        """Register a model for benchmarking."""
        # This will be implemented when we add actual models
        pass

    def run_segmentation_benchmark(
        self,
        model,
        model_name: str,
        dataset_name: str,
        sequence: str,
        model_params: Dict[str, Any] = None,
    ) -> BenchmarkResult:
        """Run segmentation benchmark for a model on a dataset sequence."""
        logger.info(
            f"Running segmentation benchmark: {model_name} on {dataset_name}/{sequence}"
        )

        model_params = model_params or {}

        # Load images and ground truth
        images = self.loader.load_sequence_images(dataset_name, sequence, "train")
        gt_segmentations = self.loader.load_ground_truth_segmentation(
            dataset_name, sequence
        )

        # Performance monitoring
        start_time = time.time()
        memory_usage = 0.0  # Placeholder

        # Run inference
        predictions = []
        frame_times = []

        for i, image in enumerate(tqdm(images, desc="Processing frames")):
            frame_start = time.time()

            # This is where the actual model inference would happen
            # For now, create a dummy prediction
            pred_mask = np.zeros_like(image, dtype=np.uint16)
            predictions.append(pred_mask)

            frame_times.append(time.time() - frame_start)

        total_time = time.time() - start_time

        # Evaluate segmentation performance
        seg_metrics_list = []
        for frame_num, gt_mask in gt_segmentations.items():
            if frame_num < len(predictions):
                pred_mask = predictions[frame_num]
                metrics = self.seg_evaluator.evaluate_segmentation(pred_mask, gt_mask)
                seg_metrics_list.append(metrics)

        # Aggregate metrics
        if seg_metrics_list:
            avg_metrics = SegmentationMetrics(
                iou=np.mean([m.iou for m in seg_metrics_list]),
                dice=np.mean([m.dice for m in seg_metrics_list]),
                precision=np.mean([m.precision for m in seg_metrics_list]),
                recall=np.mean([m.recall for m in seg_metrics_list]),
                f1_score=np.mean([m.f1_score for m in seg_metrics_list]),
                average_precision=np.mean(
                    [m.average_precision for m in seg_metrics_list]
                ),
                boundary_f1=np.mean([m.boundary_f1 for m in seg_metrics_list]),
                adapted_rand_error=np.mean(
                    [m.adapted_rand_error for m in seg_metrics_list]
                ),
                variation_of_information=np.mean(
                    [m.variation_of_information for m in seg_metrics_list]
                ),
                detection_rate=np.mean([m.detection_rate for m in seg_metrics_list]),
                false_positive_rate=np.mean(
                    [m.false_positive_rate for m in seg_metrics_list]
                ),
            )
        else:
            avg_metrics = None

        # Performance metrics
        perf_metrics = PerformanceMetrics(
            inference_time=total_time,
            memory_usage=memory_usage,
            fps=len(images) / total_time if total_time > 0 else 0.0,
            initialization_time=frame_times[0] if frame_times else 0.0,
        )

        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            sequence=sequence,
            segmentation_metrics=avg_metrics,
            tracking_metrics=None,
            performance_metrics=perf_metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            parameters=model_params,
        )

        self.results.append(result)
        logger.info(
            f"Segmentation benchmark completed. F1: {avg_metrics.f1_score:.3f}"
            if avg_metrics
            else "No metrics computed"
        )

        return result

    def run_tracking_benchmark(
        self,
        model,
        model_name: str,
        dataset_name: str,
        sequence: str,
        model_params: Dict[str, Any] = None,
    ) -> BenchmarkResult:
        """Run tracking benchmark for a model on a dataset sequence."""
        logger.info(
            f"Running tracking benchmark: {model_name} on {dataset_name}/{sequence}"
        )

        # This would be similar to segmentation benchmark but also evaluate tracking
        # For now, return a placeholder result

        perf_metrics = PerformanceMetrics(
            inference_time=10.0, memory_usage=2.0, fps=5.0, initialization_time=1.0
        )

        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            sequence=sequence,
            segmentation_metrics=None,
            tracking_metrics=None,
            performance_metrics=perf_metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            parameters=model_params or {},
        )

        self.results.append(result)
        return result

    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.results_dir / filename

        # Convert results to JSON-serializable format
        results_data = [result.to_dict() for result in self.results]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def generate_report(self, output_dir: Path = None):
        """Generate a comprehensive benchmark report."""
        if output_dir is None:
            output_dir = self.results_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate summary tables and plots
        self._generate_summary_tables(output_dir)
        self._generate_performance_plots(output_dir)

        logger.info(f"Benchmark report generated in {output_dir}")

    def _generate_summary_tables(self, output_dir: Path):
        """Generate summary tables of results."""
        if not self.results:
            return

        # Create DataFrame from results
        rows = []
        for result in self.results:
            row = {
                "Model": result.model_name,
                "Dataset": result.dataset_name,
                "Sequence": result.sequence,
                "Timestamp": result.timestamp,
            }

            if result.segmentation_metrics:
                row.update(
                    {
                        "IoU": result.segmentation_metrics.iou,
                        "Dice": result.segmentation_metrics.dice,
                        "F1_Score": result.segmentation_metrics.f1_score,
                        "Detection_Rate": result.segmentation_metrics.detection_rate,
                    }
                )

            if result.tracking_metrics:
                row.update(
                    {
                        "TRA_Score": result.tracking_metrics.tra_score,
                        "DET_Score": result.tracking_metrics.det_score,
                        "MOTA": result.tracking_metrics.mota,
                    }
                )

            row.update(
                {
                    "FPS": result.performance_metrics.fps,
                    "Memory_GB": result.performance_metrics.memory_usage,
                }
            )

            rows.append(row)

        df = pd.DataFrame(rows)

        # Save summary table
        df.to_csv(output_dir / "summary_results.csv", index=False)

        # Generate aggregated summaries
        if "F1_Score" in df.columns:
            summary_by_model = (
                df.groupby("Model")
                .agg(
                    {
                        "F1_Score": ["mean", "std"],
                        "FPS": ["mean", "std"],
                        "Memory_GB": ["mean", "std"],
                    }
                )
                .round(3)
            )
            summary_by_model.to_csv(output_dir / "summary_by_model.csv")

        logger.info("Summary tables generated")

    def _generate_performance_plots(self, output_dir: Path):
        """Generate performance visualization plots."""
        if not self.results:
            return

        # Set style
        plt.style.use("seaborn-v0_8")

        # Performance vs Accuracy plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Extract data
        models = [r.model_name for r in self.results]
        fps_values = [r.performance_metrics.fps for r in self.results]
        f1_scores = [
            r.segmentation_metrics.f1_score if r.segmentation_metrics else 0
            for r in self.results
        ]
        memory_usage = [r.performance_metrics.memory_usage for r in self.results]

        # FPS comparison
        axes[0, 0].bar(models, fps_values)
        axes[0, 0].set_title("Inference Speed (FPS)")
        axes[0, 0].set_ylabel("Frames per Second")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # F1 Score comparison
        axes[0, 1].bar(models, f1_scores)
        axes[0, 1].set_title("Segmentation F1 Score")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Memory usage
        axes[1, 0].bar(models, memory_usage)
        axes[1, 0].set_title("Memory Usage")
        axes[1, 0].set_ylabel("Memory (GB)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Speed vs Accuracy scatter
        axes[1, 1].scatter(fps_values, f1_scores)
        axes[1, 1].set_xlabel("FPS")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].set_title("Speed vs Accuracy Trade-off")

        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (fps_values[i], f1_scores[i]))

        plt.tight_layout()
        plt.savefig(
            output_dir / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info("Performance plots generated")


def main():
    """Main function for running benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Cell Tracking Challenge Benchmark")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to CTC data directory"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--dataset", type=str, default="BF-C2DL-HSC", help="Dataset to benchmark"
    )
    parser.add_argument(
        "--sequence", type=str, default="01", help="Sequence to benchmark"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["segmentation", "tracking", "both"],
        default="segmentation",
        help="Task to benchmark",
    )

    args = parser.parse_args()

    # Initialize framework
    framework = BenchmarkFramework(
        data_root=Path(args.data_root), results_dir=Path(args.results_dir)
    )

    # For demo purposes, run with dummy model
    class DummyModel:
        def predict(self, image):
            return np.zeros_like(image, dtype=np.uint16)

    dummy_model = DummyModel()

    if args.task in ["segmentation", "both"]:
        framework.run_segmentation_benchmark(
            model=dummy_model,
            model_name="DummyModel",
            dataset_name=args.dataset,
            sequence=args.sequence,
        )

    if args.task in ["tracking", "both"]:
        framework.run_tracking_benchmark(
            model=dummy_model,
            model_name="DummyModel",
            dataset_name=args.dataset,
            sequence=args.sequence,
        )

    # Save results and generate report
    framework.save_results()
    framework.generate_report()


if __name__ == "__main__":
    main()
