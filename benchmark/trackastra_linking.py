"""Trackastra linking aligned with ``gui/utils/trackastra_tracking.py``."""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

_TRACKASTRA_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()


class _QuietProgbar:
    def __init__(self, iterable=None, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])

    def update(self, _n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, *_args, **_kwargs) -> None:
        pass


def get_trackastra_model(
    *,
    pretrained: str = "general_2d",
    device: str = "automatic",
    batch_size: int | None = None,
):
    global _TRACKASTRA_MODEL
    with _MODEL_LOCK:
        if _TRACKASTRA_MODEL is None:
            from trackastra.model import Trackastra

            kw = {} if batch_size is None else {"batch_size": int(batch_size)}
            _TRACKASTRA_MODEL = Trackastra.from_pretrained(pretrained, device=device, **kw)
        return _TRACKASTRA_MODEL


def rgb_to_trackastra_image(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        plane = rgb
    elif rgb.shape[2] == 1:
        plane = rgb[:, :, 0]
    else:
        plane = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if plane.dtype == np.uint16:
        return plane
    if plane.max() <= 255:
        return (plane.astype(np.uint32) * 257).astype(np.uint16)
    return plane.astype(np.uint16)


def _label_map_from_graph(graph, frame_index: int = 1) -> dict[int, int]:
    prev_time = frame_index - 1
    mapping: dict[int, int] = {}
    for start, end in graph.edges:
        start_data = graph.nodes[start]
        end_data = graph.nodes[end]
        if int(start_data["time"]) == prev_time and int(end_data["time"]) == frame_index:
            mapping[int(end_data["label"])] = int(start_data["label"])
        elif int(end_data["time"]) == prev_time and int(start_data["time"]) == frame_index:
            mapping[int(start_data["label"])] = int(end_data["label"])
    return mapping


def link_masks_with_trackastra(
    previous_image: np.ndarray,
    previous_mask: np.ndarray,
    current_image: np.ndarray,
    current_mask: np.ndarray,
    *,
    model: Any | None = None,
    track_mode: str = "greedy_nodiv",
) -> np.ndarray:
    if previous_mask.shape != current_mask.shape:
        raise ValueError(
            f"Mask shapes must match: {previous_mask.shape} vs {current_mask.shape}"
        )

    imgs = np.stack(
        [
            rgb_to_trackastra_image(previous_image),
            rgb_to_trackastra_image(current_image),
        ],
        axis=0,
    )
    masks = np.stack(
        [
            previous_mask.astype(np.uint16),
            current_mask.astype(np.uint16),
        ],
        axis=0,
    )

    tracker = model or get_trackastra_model()
    graph, _ = tracker.track(
        imgs, masks, mode=track_mode, progbar_class=_QuietProgbar
    )

    label_map = _label_map_from_graph(graph, frame_index=1)
    linked = np.zeros(current_mask.shape, dtype=np.uint16)
    next_id = int(previous_mask.max(initial=0)) + 1

    for new_label in np.unique(current_mask):
        if new_label == 0:
            continue
        new_label = int(new_label)
        stable_id = label_map.get(new_label, next_id)
        if new_label not in label_map:
            next_id += 1
        linked[current_mask == new_label] = stable_id

    return linked.astype(np.int32, copy=False)


def track_sequence_with_trackastra(
    frames: list[np.ndarray],
    segment_fn,
    *,
    seed_search_frames: int = 5,
    track_mode: str = "greedy_nodiv",
    model: Any | None = None,
    verbose: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Segment + link across a frame sequence (GUI-style CellSeek tracking)."""
    if not frames:
        return [], {"propagation": "trackastra", "seed_frame": None}

    k = min(int(seed_search_frames), len(frames))
    seed_t = None
    seed_mask = None
    for t in range(k):
        lab = segment_fn(frames[t])
        if int(np.max(lab)) > 0:
            seed_t, seed_mask = t, lab.astype(np.int32, copy=False)
            break

    stats = {
        "propagation": "trackastra",
        "seed_frame": seed_t,
        "seed_search_frames": k,
        "used_per_frame_cellsam_fallback": seed_t is None,
    }

    if seed_t is None:
        return [segment_fn(f).astype(np.int32, copy=False) for f in frames], stats

    pred_masks = [np.zeros(frames[i].shape[:2], dtype=np.int32) for i in range(len(frames))]
    for t in range(seed_t):
        pred_masks[t] = segment_fn(frames[t]).astype(np.int32, copy=False)

    pred_masks[seed_t] = seed_mask.copy()
    prev_image = frames[seed_t]
    prev_mask = seed_mask

    frame_iter = range(seed_t + 1, len(frames))
    if verbose:
        frame_iter = tqdm(frame_iter, desc="trackastra", leave=False)

    tracker = model
    for t in frame_iter:
        current_image = frames[t]
        current_mask = segment_fn(current_image).astype(np.int32, copy=False)
        if int(np.max(prev_mask)) <= 0 or int(np.max(current_mask)) <= 0:
            pred_masks[t] = current_mask.copy()
        else:
            pred_masks[t] = link_masks_with_trackastra(
                prev_image,
                prev_mask,
                current_image,
                current_mask,
                model=tracker,
                track_mode=track_mode,
            )
        prev_image = current_image
        prev_mask = pred_masks[t]

    return pred_masks, stats
