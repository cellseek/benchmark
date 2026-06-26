"""Shared benchmark summary aggregation and CSV writing."""

from __future__ import annotations

import csv
from pathlib import Path

SUMMARY_FIELDNAMES = [
    "dataset",
    "model",
    "init_error",
    "seg_error",
    "trk_error",
    "n_seg_samples",
    "precision",
    "recall",
    "f1",
    "Cell-HOTA",
    "DetA",
    "AssA",
    "mask_f1_macro_mean",
    "n_trk_sequences",
    "n_trk_scored",
    "n_trk_error",
]


def segmentation_summary_from_counts(
    agg: dict,
    n_samples: int,
    *,
    extra: dict | None = None,
) -> dict:
    """Build the existing precision/recall/F1 segmentation summary from counts."""

    p = agg["tp"] / (agg["tp"] + agg["fp"] + 1e-8)
    r = agg["tp"] / (agg["tp"] + agg["fn"] + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    summary = {
        "n_samples": n_samples,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(agg["tp"]),
        "fp": int(agg["fp"]),
        "fn": int(agg["fn"]),
    }
    if extra:
        summary.update(extra)
    return summary


def _blank_row(dataset_name: str, model_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "model": model_name,
        "init_error": "",
        "seg_error": "",
        "trk_error": "",
        "n_seg_samples": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "Cell-HOTA": None,
        "DetA": None,
        "AssA": None,
        "mask_f1_macro_mean": None,
        "n_trk_sequences": None,
        "n_trk_scored": None,
        "n_trk_error": None,
    }


def summary_rows_from_report(report: dict) -> list[dict]:
    """Flatten the nested report object into rows for `summary.csv`."""

    rows: list[dict] = []
    for dataset_name, model_data in report.items():
        if not isinstance(model_data, dict):
            continue
        if "__dataset_init_error__" in model_data:
            payload = model_data["__dataset_init_error__"]
            err = str(payload.get("error", payload)) if isinstance(payload, dict) else str(payload)
            row = _blank_row(dataset_name, "__dataset__")
            row["init_error"] = err
            rows.append(row)

        for model_name, results in model_data.items():
            if model_name == "__dataset_init_error__" or not isinstance(results, dict):
                continue
            if "model_init_error" in results:
                row = _blank_row(dataset_name, model_name)
                row["init_error"] = str(results["model_init_error"])
                rows.append(row)
                continue
            if "skipped" in results:
                row = _blank_row(dataset_name, model_name)
                row["seg_error"] = f"SKIPPED: {results['skipped']}"
                row["trk_error"] = f"SKIPPED: {results['skipped']}"
                rows.append(row)
                continue

            seg = results.get("segmentation") if isinstance(results.get("segmentation"), dict) else {}
            trk = results.get("tracking") if isinstance(results.get("tracking"), dict) else {}
            seg_err = seg.get("error")
            trk_err = trk.get("error")
            seg_skip = seg.get("skipped")
            trk_skip = trk.get("skipped")
            row = _blank_row(dataset_name, model_name)
            row.update(
                {
                    "seg_error": ("" if seg_err is None else str(seg_err))
                    or (f"SKIPPED: {seg_skip}" if seg_skip is not None else ""),
                    "trk_error": ("" if trk_err is None else str(trk_err))
                    or (f"SKIPPED: {trk_skip}" if trk_skip is not None else ""),
                    "n_seg_samples": seg.get("n_samples"),
                    "precision": seg.get("precision"),
                    "recall": seg.get("recall"),
                    "f1": seg.get("f1"),
                    "Cell-HOTA": trk.get("Cell-HOTA"),
                    "DetA": trk.get("DetA"),
                    "AssA": trk.get("AssA"),
                    "mask_f1_macro_mean": trk.get("mask_f1_macro_mean"),
                    "n_trk_sequences": trk.get("n_sequences_total"),
                    "n_trk_scored": trk.get("n_sequences_scored"),
                    "n_trk_error": trk.get("n_sequences_error"),
                }
            )
            rows.append(row)
    return rows


def write_summary_csv(report: dict, out_csv: str | Path) -> None:
    rows = summary_rows_from_report(report)
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def simple_tracking_summary(per_seq: list[dict]) -> dict:
    """Build compare-mode's historical macro tracking summary."""

    valid = [r for r in per_seq if "Cell-HOTA" in r]
    if valid:
        return {
            "Cell-HOTA": float(sum(r["Cell-HOTA"] for r in valid) / len(valid)),
            "DetA": float(sum(r["DetA"] for r in valid) / len(valid)),
            "AssA": float(sum(r["AssA"] for r in valid) / len(valid)),
            "n_sequences_total": len(per_seq),
            "n_sequences_scored": len(valid),
            "n_sequences_error": len(per_seq) - len(valid),
        }
    return {
        "Cell-HOTA": None,
        "DetA": None,
        "AssA": None,
        "n_sequences_total": len(per_seq),
        "n_sequences_scored": 0,
        "n_sequences_error": len(per_seq),
    }
