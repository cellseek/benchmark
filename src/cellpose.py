#!/usr/bin/env python3
"""
Cellpose model implementation for Cell Tracking Challenge benchmark.

This module provides a wrapper around the Cellpose model for integration
with the benchmark framework.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from cellpose import io as cellpose_io
    from cellpose import models

    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    warnings.warn("Cellpose not available. Install with: pip install cellpose")

import cv2
from skimage import measure, morphology

logger = logging.getLogger(__name__)


class CellposeModel:
    """Cellpose model wrapper for benchmarking."""

    def __init__(
        self,
        model_type: str = "cyto",
        gpu: bool = False,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
    ):
        """
        Initialize Cellpose model.

        Args:
            model_type: Type of Cellpose model ('cyto', 'nuclei', 'cyto2')
            gpu: Whether to use GPU acceleration
            diameter: Expected cell diameter in pixels (None for auto-detection)
            flow_threshold: Flow error threshold for segmentation
            cellprob_threshold: Cell probability threshold
            min_size: Minimum size of objects in pixels
        """
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose not available. Install with: pip install cellpose"
            )

        self.model_type = model_type
        self.gpu = gpu
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size

        # Initialize model
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)

        logger.info(f"Initialized Cellpose model: {model_type}, GPU: {gpu}")

    def predict_single_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation for a single frame.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            Segmentation mask with unique labels for each cell
        """
        # Ensure image is in the right format
        if len(image.shape) == 3 and image.shape[2] > 1:
            # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Run cellpose
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[0, 0],  # Grayscale
        )

        # Post-process: remove small objects
        if self.min_size > 0:
            masks = morphology.remove_small_objects(masks, min_size=self.min_size)

        return masks.astype(np.uint16)

    def predict_sequence(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict segmentation for a sequence of images.

        Args:
            images: List of input images

        Returns:
            List of segmentation masks
        """
        masks = []
        for image in images:
            mask = self.predict_single_frame(image)
            masks.append(mask)

        return masks

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters for reporting."""
        return {
            "model_type": self.model_type,
            "gpu": self.gpu,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "min_size": self.min_size,
        }


class CellposeTracker:
    """Simple tracking implementation using Cellpose segmentation."""

    def __init__(
        self,
        cellpose_model: CellposeModel,
        max_distance: float = 50.0,
        min_overlap: float = 0.1,
    ):
        """
        Initialize tracker.

        Args:
            cellpose_model: Initialized Cellpose model
            max_distance: Maximum distance for linking objects between frames
            min_overlap: Minimum overlap ratio for linking
        """
        self.segmentation_model = cellpose_model
        self.max_distance = max_distance
        self.min_overlap = min_overlap
        self.next_track_id = 1

    def track_sequence(
        self, images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Track objects through a sequence of images.

        Args:
            images: List of input images

        Returns:
            Tuple of (tracking_masks, lineage_info)
        """
        # Get segmentation for all frames
        segmentation_masks = self.segmentation_model.predict_sequence(images)

        # Initialize tracking
        tracking_masks = []
        lineages = []
        previous_objects = {}

        for frame_idx, seg_mask in enumerate(segmentation_masks):
            current_objects = self._extract_objects(seg_mask)
            tracking_mask = np.zeros_like(seg_mask, dtype=np.uint16)

            if frame_idx == 0:
                # First frame: assign new IDs
                for obj_id, obj_info in current_objects.items():
                    track_id = self.next_track_id
                    self.next_track_id += 1

                    tracking_mask[seg_mask == obj_id] = track_id

                    lineages.append(
                        {
                            "track_id": track_id,
                            "start_frame": frame_idx,
                            "end_frame": frame_idx,  # Will be updated
                            "parent_id": None,
                        }
                    )

                    obj_info["track_id"] = track_id

                previous_objects = current_objects
            else:
                # Link objects to previous frame
                assignments = self._link_objects(previous_objects, current_objects)

                for obj_id, obj_info in current_objects.items():
                    if obj_id in assignments:
                        # Continue existing track
                        track_id = assignments[obj_id]
                        tracking_mask[seg_mask == obj_id] = track_id

                        # Update lineage end frame
                        for lineage in lineages:
                            if lineage["track_id"] == track_id:
                                lineage["end_frame"] = frame_idx
                                break
                    else:
                        # New track (division or new cell)
                        track_id = self.next_track_id
                        self.next_track_id += 1

                        tracking_mask[seg_mask == obj_id] = track_id

                        lineages.append(
                            {
                                "track_id": track_id,
                                "start_frame": frame_idx,
                                "end_frame": frame_idx,
                                "parent_id": None,  # Could implement division detection
                            }
                        )

                    obj_info["track_id"] = track_id

                previous_objects = current_objects

            tracking_masks.append(tracking_mask)

        return tracking_masks, lineages

    def _extract_objects(self, mask: np.ndarray) -> Dict[int, Dict]:
        """Extract object properties from segmentation mask."""
        props = measure.regionprops(mask)
        objects = {}

        for prop in props:
            objects[prop.label] = {
                "centroid": prop.centroid,
                "area": prop.area,
                "bbox": prop.bbox,
            }

        return objects

    def _link_objects(
        self, prev_objects: Dict[int, Dict], curr_objects: Dict[int, Dict]
    ) -> Dict[int, int]:
        """Link objects between frames based on distance and overlap."""
        assignments = {}
        used_tracks = set()

        # Simple nearest neighbor assignment
        for curr_id, curr_obj in curr_objects.items():
            best_distance = float("inf")
            best_track_id = None

            for prev_id, prev_obj in prev_objects.items():
                if prev_obj["track_id"] in used_tracks:
                    continue

                # Compute centroid distance
                distance = np.sqrt(
                    (curr_obj["centroid"][0] - prev_obj["centroid"][0]) ** 2
                    + (curr_obj["centroid"][1] - prev_obj["centroid"][1]) ** 2
                )

                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_track_id = prev_obj["track_id"]

            if best_track_id is not None:
                assignments[curr_id] = best_track_id
                used_tracks.add(best_track_id)

        return assignments


def create_cellpose_models() -> Dict[str, CellposeModel]:
    """Create different Cellpose model configurations for benchmarking."""
    models = {}

    if CELLPOSE_AVAILABLE:
        # Standard configurations
        models["cellpose_cyto"] = CellposeModel(
            model_type="cyto",
            diameter=None,  # Auto-detect
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        models["cellpose_nuclei"] = CellposeModel(
            model_type="nuclei",
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        models["cellpose_cyto2"] = CellposeModel(
            model_type="cyto2",
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        # High precision configuration
        models["cellpose_cyto_high_precision"] = CellposeModel(
            model_type="cyto",
            diameter=None,
            flow_threshold=0.2,  # Lower threshold for higher precision
            cellprob_threshold=0.2,
            min_size=25,
        )

        # Fast configuration
        models["cellpose_cyto_fast"] = CellposeModel(
            model_type="cyto",
            diameter=30,  # Fixed diameter for speed
            flow_threshold=0.6,  # Higher threshold for speed
            cellprob_threshold=-1.0,
            min_size=10,
        )

    return models


if __name__ == "__main__":
    # Test the Cellpose model
    if CELLPOSE_AVAILABLE:
        model = CellposeModel()

        # Create dummy image for testing
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Test segmentation
        result = model.predict_single_frame(test_image)
        print(f"Segmentation result shape: {result.shape}")
        print(f"Number of objects detected: {len(np.unique(result)) - 1}")

        # Test tracking
        tracker = CellposeTracker(model)
        test_images = [test_image, test_image]  # Simple test
        tracking_results, lineages = tracker.track_sequence(test_images)
        print(f"Tracking completed. Lineages: {len(lineages)}")
    else:
        print("Cellpose not available. Please install with: pip install cellpose")
