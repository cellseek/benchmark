"""Shared tracking result schema helpers."""

from __future__ import annotations

import pandas as pd

TRACK_COLUMNS = ["frame", "track_id", "x", "y"]


def empty_tracks_df() -> pd.DataFrame:
    """Return an empty tracks DataFrame with the benchmark's canonical columns."""

    return pd.DataFrame(columns=TRACK_COLUMNS)
