"""
Cell-TRACTR (native Trackformer pipeline) is not embedded here: it expects Sacred configs,
compiled deformable attention, and on-disk CTC layouts (see ``models/Cell-TRACTR/README.md``).

For ``csbench`` parity, this adapter defaults to ``delegate_tracker=trackastra``: same CLI and
configs as other models, using Trackastra linking on masks derived from each frame (honest
about delegation via logs).
"""

from __future__ import annotations

from .base import ModelAdapter
from .trackastra_adapter import TrackastraAdapter


class CelltractrAdapter(ModelAdapter):
    def __init__(self, cfg: dict):
        delegate = str(cfg.get("delegate_tracker", "trackastra")).strip().lower()
        if delegate != "trackastra":
            raise ValueError(
                "celltractr: unsupported delegate_tracker "
                f"{delegate!r}. Use delegate_tracker: trackastra for programmatic csbench runs, "
                "or run native Cell-TRACTR via models/Cell-TRACTR/src/pipeline.py "
                "(Sacred + dataset YAML)."
            )
        sub = {k: v for k, v in cfg.items() if k not in ("type", "delegate_tracker")}
        print(
            "cellseek-benchmark: Cell-TRACTR native pipeline not bundled in csbench; "
            "delegating tracking to Trackastra (configure pretrained/track_mode under "
            "model entry like trackastra).",
            flush=True,
        )
        self._delegate = TrackastraAdapter(sub)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return self._delegate.predict_mask(image)

    def predict_tracks(self, frames: list[np.ndarray]) -> pd.DataFrame:
        return self._delegate.predict_tracks(frames)

    def predict_tracks_returning_masks(
        self, frames: list[np.ndarray]
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        return self._delegate.predict_tracks_returning_masks(frames)
