#!/usr/bin/env python3
"""Aggregate tracking benchmark results (excluding ultrack) into one Excel sheet."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
OUT_XLSX = EXPERIMENTS / "tracking_results_summary.xlsx"

DATASETS = [
    ("hsc", "HSC", "ctc_bf_c2dl_hsc"),
    ("musc", "MuSC", "ctc_bf_c2dl_musc"),
]
MODELS = [
    ("cellseek", "cellseek"),
    ("sam3", "SAM3"),
    ("trackastra", "Trackastra"),
    ("celltractr", "Cell-TRACTR"),
    ("microsam", "micro_sam"),
]

DIR_RE = re.compile(
    r"^(?P<model>cellseek|sam3|trackastra|celltractr|microsam)"
    r"_ctc_bf_c2dl_(?P<ds>hsc|musc)_tracking_(?P<stamp>\d{8}_\d+)$"
)


def _load_tracking_summary(report_path: Path) -> dict | None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    for ds_key, ds_val in data.items():
        if not isinstance(ds_val, dict):
            continue
        for model_key, model_val in ds_val.items():
            if not isinstance(model_val, dict):
                continue
            trk = model_val.get("tracking")
            if not isinstance(trk, dict):
                continue
            if trk.get("error") or trk.get("skipped"):
                return None
            hota = trk.get("HOTA")
            if hota is None or (isinstance(hota, float) and hota != hota):
                if trk.get("n_sequences_scored", 0) == 0:
                    return None
            return {
                "TA": trk.get("TA"),
                "HOTA": trk.get("HOTA"),
                "DetA": trk.get("DetA"),
                "AssA": trk.get("AssA"),
                "n_sequences_scored": trk.get("n_sequences_scored"),
                "experiment_dir": report_path.parent.name,
            }
    return None


def _collect_latest() -> dict[tuple[str, str], dict]:
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for report in EXPERIMENTS.glob("*_tracking_*/report.json"):
        if "ultrack" in report.parent.name:
            continue
        m = DIR_RE.match(report.parent.name)
        if not m:
            continue
        model = m.group("model")
        ds = m.group("ds")
        stamp = m.group("stamp")
        summary = _load_tracking_summary(report)
        if summary is None:
            continue
        key = (ds, model)
        if key not in best or stamp > best[key][0]:
            best[key] = (stamp, summary)
    return {k: v[1] for k, v in best.items()}


def _round_metric(v: object) -> object:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return round(float(v), 4)


def main() -> None:
    results = _collect_latest()
    rows: list[dict[str, object]] = []
    for ds_slug, ds_label, _ in DATASETS:
        for model_slug, model_label in MODELS:
            s = results.get((ds_slug, model_slug))
            row: dict[str, object] = {
                "dataset": ds_label,
                "model": model_label,
                "TRA": "",
                "Cell-HOTA": "",
                "DetA": "",
                "AssA": "",
            }
            if s is not None:
                row["TRA"] = _round_metric(s.get("TA"))
                row["Cell-HOTA"] = _round_metric(s.get("HOTA"))
                row["DetA"] = _round_metric(s.get("DetA"))
                row["AssA"] = _round_metric(s.get("AssA"))
            rows.append(row)

    out = pd.DataFrame(rows, columns=["dataset", "model", "TRA", "Cell-HOTA", "DetA", "AssA"])

    meta_rows = []
    for ds_slug, _, _ in DATASETS:
        for model_slug, model_label in MODELS:
            s = results.get((ds_slug, model_slug))
            if s:
                meta_rows.append(
                    {
                        "dataset": ds_slug,
                        "model": model_label,
                        "experiment_dir": s["experiment_dir"],
                        "n_sequences_scored": s.get("n_sequences_scored"),
                    }
                )
            else:
                meta_rows.append(
                    {"dataset": ds_slug, "model": model_label, "experiment_dir": "", "n_sequences_scored": ""}
                )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="sources", index=False)

    print(f"Wrote {OUT_XLSX}")
    for ds_slug, ds_label, _ in DATASETS:
        for model_slug, model_label in MODELS:
            s = results.get((ds_slug, model_slug))
            status = "OK" if s else "MISSING"
            print(f"  {ds_label} / {model_label}: {status}" + (f"  ({s['experiment_dir']})" if s else ""))


if __name__ == "__main__":
    main()
