import contextlib
import gc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from tempfile import TemporaryDirectory

from .base import ModelAdapter


class SAM3Adapter(ModelAdapter):
    """
    SAM3 image adapter following SAM3's basic usage:
    build_sam3_image_model -> Sam3Processor -> set_image -> set_text_prompt.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prompt = str(cfg.get("prompt", "cell"))
        self.debug_dtype = bool(cfg.get("debug_dtype", False))
        self._printed_debug = False
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path:
            model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        else:
            model = build_sam3_image_model()
        self.processor = Sam3Processor(model)
        self._sam3_checkpoint_path = checkpoint_path
        self._use_cuda_autocast = torch.cuda.is_available()
        self._amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self._video_predictor = None

        # Video tracking VRAM: segmentation leaves the image model on GPU; the video
        # predictor needs additional memory. Resize + offload image weights before tracking.
        tle = cfg.get("tracking_max_long_edge", 1024)
        self._tracking_max_long_edge = int(tle) if tle is not None and int(tle) > 0 else 0
        tmf = cfg.get("tracking_max_frames")
        self._tracking_max_frames = int(tmf) if tmf is not None and int(tmf) > 0 else None
        self._tracking_offload_image_model = bool(cfg.get("tracking_offload_image_model", True))
        raw_retries = cfg.get("tracking_oom_retry_long_edges")
        if raw_retries is None:
            self._oom_retry_long_edges: list[int] = [768, 512, 384]
        else:
            self._oom_retry_long_edges = [int(x) for x in raw_retries if int(x) > 0]

    def _autocast_ctx(self):
        if not self._use_cuda_autocast:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    @staticmethod
    def _to_pil_rgb(image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            rgb = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] >= 3:
            rgb = image[..., :3]
        else:
            raise ValueError(f"Unsupported image shape for SAM3: {image.shape}")

        if rgb.dtype != np.uint8:
            rgb_min = float(np.min(rgb))
            rgb_max = float(np.max(rgb))
            if rgb_max > rgb_min:
                rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255.0).astype(np.uint8)
            else:
                rgb = np.zeros_like(rgb, dtype=np.uint8)
        return Image.fromarray(rgb)

    @staticmethod
    def _resize_bool_mask_to(mask_bool: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        h0, w0 = out_hw
        if mask_bool.ndim == 3:
            if mask_bool.shape[0] == 1:
                mask_bool = mask_bool[0]
            elif mask_bool.shape[-1] == 1:
                mask_bool = mask_bool[..., 0]
            else:
                mask_bool = mask_bool.max(axis=0)
        mh, mw = int(mask_bool.shape[0]), int(mask_bool.shape[1])
        if mh == h0 and mw == w0:
            return mask_bool.astype(bool, copy=False)
        pil_m = Image.fromarray(mask_bool.astype(np.uint8) * 255)
        resized = pil_m.resize((w0, h0), Image.Resampling.NEAREST)
        return np.asarray(resized) > 127

    @staticmethod
    def _pil_resize_max_long_edge(pil: Image.Image, max_long_edge: int) -> Image.Image:
        if max_long_edge <= 0:
            return pil
        w, h = pil.size
        m = max(w, h)
        if m <= max_long_edge:
            return pil
        scale = max_long_edge / m
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return pil.resize((nw, nh), Image.Resampling.LANCZOS)

    def _clear_cuda_cache(self) -> None:
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()

    def _offload_image_model_for_video(self) -> None:
        if not self._tracking_offload_image_model or not torch.cuda.is_available():
            return
        try:
            self.processor.model.to("cpu")
        except Exception:
            return
        self._clear_cuda_cache()

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        pil_image = self._to_pil_rgb(image)
        if self.debug_dtype and not self._printed_debug:
            print(
                "sam3 adapter debug:"
                f" cuda={torch.cuda.is_available()}"
                f", bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}"
                f", amp_dtype={self._amp_dtype}",
                flush=True,
            )
            self._printed_debug = True

        with self._autocast_ctx():
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=self.prompt,
            )
        masks = output["masks"]

        if len(masks) == 0:
            return np.zeros(image.shape[:2], dtype=np.int32)

        labeled = np.zeros(image.shape[:2], dtype=np.int32)
        for instance_id, mask in enumerate(masks, start=1):
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().to(device="cpu").numpy()
            else:
                mask_np = np.asarray(mask)
            # SAM3 masks can be [H, W], [1, H, W], or [H, W, 1].
            if mask_np.ndim == 3:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                elif mask_np.shape[-1] == 1:
                    mask_np = mask_np[..., 0]
                else:
                    # Keep a deterministic path if an unexpected extra axis appears.
                    mask_np = mask_np.max(axis=0)
            if mask_np.ndim != 2:
                raise RuntimeError(
                    f"SAM3 returned unexpected mask shape {mask_np.shape} for instance {instance_id}"
                )
            labeled[mask_np.astype(bool)] = instance_id
        return labeled

    def _is_cuda_oom(self, err: BaseException) -> bool:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(err, oom_type):
            return True
        msg = str(err).lower()
        return "out of memory" in msg and "cuda" in msg

    def _predict_tracks_with_max_edge(
        self,
        frames: list[np.ndarray],
        max_long_edge: int,
        *,
        return_masks: bool = False,
    ) -> tuple[pd.DataFrame, list[np.ndarray] | None]:
        predictor = self._get_video_predictor()
        out_masks: list[np.ndarray] | None = None
        if return_masks:
            out_masks = [
                np.zeros(frames[i].shape[:2], dtype=np.int32) for i in range(len(frames))
            ]
        with TemporaryDirectory(prefix="sam3_track_") as tmp:
            frame_dir = Path(tmp)
            frame_scales: list[tuple[float, float]] = []
            for i, frame in enumerate(frames):
                pil = self._to_pil_rgb(frame)
                ow, oh = pil.size
                if max_long_edge > 0:
                    pil = self._pil_resize_max_long_edge(pil, max_long_edge)
                nw, nh = pil.size
                sx = float(ow) / float(nw) if nw > 0 else 1.0
                sy = float(oh) / float(nh) if nh > 0 else 1.0
                frame_scales.append((sx, sy))
                pil.save(frame_dir / f"{i:06d}.png")

            session = predictor.handle_request(
                {"type": "start_session", "resource_path": str(frame_dir)}
            )
            session_id = session["session_id"]
            try:
                predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": self.prompt,
                    }
                )

                rows: list[dict] = []
                stream = predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": len(frames),
                    }
                )
                for response in stream:
                    frame_idx = int(response.get("frame_index", -1))
                    outputs = response.get("outputs", {})
                    obj_ids = outputs.get("out_obj_ids", [])
                    bin_masks = outputs.get("out_binary_masks", None)
                    if bin_masks is None:
                        continue
                    if frame_idx < 0 or frame_idx >= len(frame_scales):
                        continue
                    sx, sy = frame_scales[frame_idx]
                    out_hw = frames[frame_idx].shape[:2]

                    if isinstance(obj_ids, torch.Tensor):
                        obj_ids = obj_ids.detach().cpu().numpy()
                    if isinstance(bin_masks, torch.Tensor):
                        bin_masks = bin_masks.detach().cpu().numpy()

                    lab: np.ndarray | None = None
                    if return_masks and out_masks is not None:
                        lab = out_masks[frame_idx]

                    for mi, obj_id in enumerate(obj_ids):
                        mask = np.asarray(bin_masks[mi])
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask[0]
                        elif mask.ndim == 3 and mask.shape[-1] == 1:
                            mask = mask[..., 0]
                        mask = mask.astype(bool)
                        mask_full = self._resize_bool_mask_to(mask, out_hw)
                        if lab is not None and int(mi + 1) > 0:
                            lab[mask_full] = int(mi + 1)
                        # Centroids on the model grid, then scaled to full-res frame coords.
                        ys, xs = np.nonzero(mask)
                        if len(xs) == 0:
                            continue
                        rows.append(
                            {
                                "frame": frame_idx,
                                "track_id": int(obj_id),
                                "x": float(xs.mean()) * sx,
                                "y": float(ys.mean()) * sy,
                            }
                        )
                return (
                    pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"]),
                    out_masks,
                )
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
                self._clear_cuda_cache()

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        if len(frames) == 0:
            return pd.DataFrame(columns=["frame", "track_id", "x", "y"])
        if self._tracking_max_frames is not None:
            frames = frames[: self._tracking_max_frames]

        self._offload_image_model_for_video()
        self._clear_cuda_cache()

        primary_edge = self._tracking_max_long_edge
        edge_candidates: list[int] = []
        if primary_edge > 0:
            edge_candidates.append(primary_edge)
        for e in self._oom_retry_long_edges:
            if e > 0 and e not in edge_candidates:
                edge_candidates.append(e)
        if primary_edge <= 0:
            edge_candidates.insert(0, 0)

        last_err: BaseException | None = None
        for edge in edge_candidates:
            try:
                df, _m = self._predict_tracks_with_max_edge(frames, edge, return_masks=False)
                return df
            except Exception as e:
                if not self._is_cuda_oom(e):
                    raise
                last_err = e
                self._video_predictor = None
                self._clear_cuda_cache()
        assert last_err is not None
        raise last_err

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        if len(frames) == 0:
            return (
                pd.DataFrame(columns=["frame", "track_id", "x", "y"]),
                [],
            )
        if self._tracking_max_frames is not None:
            frames = frames[: self._tracking_max_frames]

        self._offload_image_model_for_video()
        self._clear_cuda_cache()

        primary_edge = self._tracking_max_long_edge
        edge_candidates: list[int] = []
        if primary_edge > 0:
            edge_candidates.append(primary_edge)
        for e in self._oom_retry_long_edges:
            if e > 0 and e not in edge_candidates:
                edge_candidates.append(e)
        if primary_edge <= 0:
            edge_candidates.insert(0, 0)

        last_err: BaseException | None = None
        for edge in edge_candidates:
            try:
                df, masks = self._predict_tracks_with_max_edge(
                    frames, edge, return_masks=True
                )
                if masks is None:
                    masks = [
                        np.zeros(frames[i].shape[:2], dtype=np.int32)
                        for i in range(len(frames))
                    ]
                return df, masks
            except Exception as e:
                if not self._is_cuda_oom(e):
                    raise
                last_err = e
                self._video_predictor = None
                self._clear_cuda_cache()
        assert last_err is not None
        raise last_err

    def _get_video_predictor(self):
        if self._video_predictor is None:
            from sam3.model_builder import build_sam3_video_predictor

            kwargs = {}
            if self._sam3_checkpoint_path:
                kwargs["checkpoint_path"] = self._sam3_checkpoint_path
            self._video_predictor = build_sam3_video_predictor(**kwargs)
        return self._video_predictor
