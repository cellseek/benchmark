import numpy as np
import pandas as pd

from .base import ModelAdapter


def _import_omni_model_class():
    """
    Require Omnipose 2.x (``omnipose.models.OmniModel``), not a namespace-only
    ``omnipose`` folder on PYTHONPATH and not legacy Cellpose-only installs.
    """
    try:
        import omnipose
    except ImportError as e:
        raise ImportError(
            "Omnipose failed to import. Activate conda env `benchmark` and run:\n"
            "  pip install -r requirements-benchmark.txt"
        ) from e

    pkg_file = getattr(omnipose, "__file__", None)
    version = getattr(omnipose, "__version__", None)
    if pkg_file is None and version is None:
        raise ImportError(
            "Imported `omnipose` is not a full Omnipose 2.x install (namespace stub). "
            "Remove stray directories named `omnipose` from PYTHONPATH and install from "
            "the Omnipose repo (`pip install -e .../omnipose` or PYTHONPATH=.../omnipose/src)."
        )

    try:
        from omnipose.models import OmniModel
    except ImportError as e:
        raise ImportError(
            "Could not import `omnipose.models.OmniModel`. "
            "You likely have the wrong `omnipose` package or an incomplete install."
        ) from e

    return omnipose, OmniModel, pkg_file, version


class OmniposeAdapter(ModelAdapter):
    """
    Omnipose 2.x inference via ``omnipose.models.OmniModel``.

    Configure with ``model_type`` (e.g. ``cyto2_omni``, ``bact_phase_omni``) or
    ``pretrained_model``; ``eval_params`` merges into ``eval()`` calls.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.native = True
        omnipose_pkg, OmniModel, pkg_file, version = _import_omni_model_class()

        gpu = bool(cfg.get("use_gpu", True))
        model_type = cfg.get("model_type")
        pretrained = cfg.get("pretrained_model")
        net_avg = bool(cfg.get("net_avg", False))

        init_kw = {"gpu": gpu, "net_avg": net_avg}
        if pretrained and not model_type:
            init_kw["pretrained_model"] = pretrained
        else:
            init_kw["model_type"] = model_type or pretrained or "cyto2_omni"

        for key in ("diam_mean", "nclasses", "dim", "nchan", "device"):
            if key in cfg and cfg[key] is not None:
                init_kw[key] = cfg[key]

        resolved_type = init_kw.get("model_type") or init_kw.get("pretrained_model")
        print(
            f"cellseek-benchmark: Omnipose package={pkg_file or omnipose_pkg.__path__} "
            f"version={version}",
            flush=True,
        )
        print(
            f"cellseek-benchmark: loading OmniModel pretrained={resolved_type!r} "
            f"gpu={gpu} net_avg={net_avg}",
            flush=True,
        )

        self.model = OmniModel(**init_kw)
        device = getattr(self.model, "device", None)
        print(f"cellseek-benchmark: OmniModel ready (device={device}).", flush=True)

        self._eval_params = {
            "rescale_factor": None,
            "diameter": cfg.get("diameter"),
            "mask_threshold": cfg.get("mask_threshold", -2.0),
            "flow_threshold": cfg.get("flow_threshold", 0.0),
            "transparency": False,
            "omni": True,
            "cluster": bool(cfg.get("cluster", True)),
            "resample": bool(cfg.get("resample", True)),
            "verbose": bool(cfg.get("verbose", False)),
            "tile": bool(cfg.get("tile", True)),
            "niter": cfg.get("niter"),
            "augment": bool(cfg.get("augment", False)),
            "affinity_seg": bool(cfg.get("affinity_seg", False)),
            "show_progress": bool(cfg.get("show_progress", False)),
        }
        if isinstance(cfg.get("eval_params"), dict):
            self._eval_params.update(cfg["eval_params"])

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Omnipose adapter is not initialized.")
        img = np.asarray(image, dtype=np.float32)
        out = self.model.eval(img, **self._eval_params)
        if hasattr(out, "masks"):
            masks = out.masks
        else:
            masks = out[0]
        masks = np.asarray(masks)
        if masks.ndim == 3 and masks.shape[0] == 1:
            masks = masks[0]
        return masks.astype(np.int32, copy=False)

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        raise NotImplementedError(
            "Omnipose native tracking is not implemented in this benchmark adapter."
        )
