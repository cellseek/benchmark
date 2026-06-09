"""
Single dataset × single model per invocation.

Loads one **experiment YAML** (e.g. ``configs/experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml``)
plus dataset/model catalogs from ``--config-dir``, then runs segmentation and/or tracking.

Each run writes under ``<output_parent>/<run_name>_<YYYYMMDD_HHMMSS>/`` (default
``output_parent`` is ``experiments``). Override with ``output_parent`` / ``run_name``
in the YAML, or ``--output-parent`` / ``--run-name`` on the CLI.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .io_utils import dump_json, ensure_dir, load_yaml
from .metrics.segmentation import prf1_iou
from .metrics.tracking import eval_tracking_hota
from .registry import DATASET_REGISTRY, MODEL_REGISTRY
from .schemas import TrackSequence
from .tracking_eval import (
    build_tracking_summary,
    infer_tracks_for_sequence,
    score_sequence,
    tracking_options,
)


def _normalize_tasks_value(raw) -> list[str]:
    if raw is None:
        raise ValueError("tasks must be set in the experiment YAML or via --tasks.")
    if isinstance(raw, str):
        t = raw.strip().lower()
        if t == "both":
            return ["segmentation", "tracking"]
        if t in ("segmentation", "tracking"):
            return [t]
        raise ValueError(f"Invalid tasks {raw!r}; use segmentation, tracking, or both.")
    if isinstance(raw, list):
        if not raw:
            raise ValueError("tasks list must not be empty.")
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            s = str(item).strip().lower()
            if s == "both":
                for part in ("segmentation", "tracking"):
                    if part not in seen:
                        seen.add(part)
                        out.append(part)
            elif s in ("segmentation", "tracking"):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            else:
                raise ValueError(f"Invalid tasks entry {item!r}")
        if not out:
            raise ValueError("tasks list produced no runnable tasks.")
        return out
    raise ValueError(f"tasks must be string or list, got {type(raw).__name__}.")


def _safe_run_name(name: str) -> str:
    """Filesystem-safe folder prefix."""
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name).strip())
    s = s.strip("._-")
    return s if s else "run"


def _resolve_run_output_root(
    bench_cfg: dict,
    bench_path: Path,
    *,
    output_parent: str | None = None,
    run_name: str | None = None,
) -> Path:
    """
    Default: ``experiments/<run_name>_<YYYYMMDD_HHMMSS>/`` relative to the process cwd.

    ``run_name`` defaults to YAML ``run_name`` (if set), else ``experiment_id``, else the
    benchmark file stem. ``output_parent`` defaults to ``experiments`` or the YAML
    ``output_parent`` key.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        run_name
        if run_name is not None and str(run_name).strip()
        else bench_cfg.get("run_name")
        or bench_cfg.get("experiment_id")
        or bench_path.stem
    )
    base_name = _safe_run_name(str(base_name))
    parent_str = (
        output_parent
        if output_parent is not None and str(output_parent).strip()
        else bench_cfg.get("output_parent")
    )
    if parent_str is None or not str(parent_str).strip():
        parent_str = "experiments"
    return Path(parent_str).expanduser() / f"{base_name}_{stamp}"


def _resolve_single_run(
    bench_cfg: dict,
    dataset_override: str | None,
    model_override: str | None,
    tasks_override: str | None,
) -> tuple[str, str, list[str]]:
    ds = dataset_override or bench_cfg.get("dataset")
    md = model_override or bench_cfg.get("model")
    if not ds or not isinstance(ds, str):
        raise ValueError("experiment YAML must set `dataset`, or pass --dataset.")
    if not md or not isinstance(md, str):
        raise ValueError("experiment YAML must set `model`, or pass --model.")
    if tasks_override is not None:
        tasks = _normalize_tasks_value(tasks_override)
    else:
        tasks = _normalize_tasks_value(bench_cfg.get("tasks"))
    return ds.strip(), md.strip(), tasks


def _select_tasks_for_combo(
    dataset_name: str,
    model_name: str,
    selected_tasks: list[str],
    dataset_type: str | None,
) -> tuple[list[str], str | None]:
    ds = dataset_name.lower()
    model = model_name.lower()
    requested = [t for t in selected_tasks if t in {"segmentation", "tracking"}]

    if ds in {"cellpose", "bbbc038"}:
        tasks = [t for t in requested if t == "segmentation"]
        if not tasks:
            return [], "Only segmentation is allowed on cellpose/BBBC038."
        return tasks, None

    if dataset_type == "ctc_cell_tracking":
        seg_only = {"cellpose", "omnipose", "cellsam_seg"}
        if model.lower() in seg_only:
            tasks = [t for t in requested if t == "segmentation"]
            if not tasks:
                return (
                    [],
                    f"Model {model_name!r} on CTC supports only segmentation here; use tasks: segmentation.",
                )
            if "tracking" in requested:
                print(
                    "cellseek-benchmark: CTC + segmentation-only model — ignoring tracking.",
                    flush=True,
                )
            return tasks, None
        # Other models: CTC supports tracking and/or segmentation. ``iter_segmentation``
        # ``iter_segmentation`` yields sparse SEG/man_seg frames only.
        tasks = [t for t in requested if t in ("segmentation", "tracking")]
        if not tasks:
            return (
                [],
                "CTC: use tasks segmentation, tracking, or both.",
            )
        return tasks, None

    if model in {"cellseek", "sam3", "microsam", "trackastra", "ultrack", "celltractr"}:
        tasks = [t for t in requested if t in {"segmentation", "tracking"}]
        return (tasks, None) if tasks else ([], "No runnable tasks requested.")

    if model == "omnipose":
        tasks = [t for t in requested if t == "segmentation"]
        return (tasks, None) if tasks else ([], "Omnipose is segmentation-only here.")

    if model == "cellsam_seg":
        tasks = [t for t in requested if t == "segmentation"]
        return (tasks, None) if tasks else ([], "cellsam_seg is segmentation-only here.")

    if model == "cellpose":
        tasks = [t for t in requested if t == "segmentation"]
        return (tasks, None) if tasks else ([], "Cellpose is segmentation-only here.")

    return requested, None


def _iter_segmentation_samples(dataset_adapter):
    for sample in dataset_adapter.iter_segmentation():
        if sample.gt_mask is not None:
            yield sample


def run_segmentation_pass(dataset_adapter, model_adapter, out_dir: Path, iou_threshold: float):
    ensure_dir(out_dir)
    per_sample = []
    agg = defaultdict(float)
    n = 0

    for sample in tqdm(
        _iter_segmentation_samples(dataset_adapter),
        desc="segmentation",
        unit="img",
        dynamic_ncols=True,
    ):
        try:
            pred = model_adapter.predict_mask(sample.image)
        except NotImplementedError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"segmentation inference failed for sample {sample.sample_id!r}: {e}"
            ) from e
        m = prf1_iou(sample.gt_mask, pred, iou_threshold=iou_threshold)
        per_sample.append({"sample_id": sample.sample_id, **m})
        agg["tp"] += m["tp"]
        agg["fp"] += m["fp"]
        agg["fn"] += m["fn"]
        n += 1

    p = agg["tp"] / (agg["tp"] + agg["fp"] + 1e-8)
    r = agg["tp"] / (agg["tp"] + agg["fn"] + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    summary = {
        "n_samples": n,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(agg["tp"]),
        "fp": int(agg["fp"]),
        "fn": int(agg["fn"]),
    }
    dump_json(summary, out_dir / "summary.json")
    dump_json(per_sample, out_dir / "per_sample.json")
    return summary


def _iter_tracking_sequences(dataset_adapter):
    for seq in dataset_adapter.iter_tracking():
        if seq.gt_tracks is not None and getattr(seq, "frames", None):
            yield seq


def run_tracking_pass(
    dataset_adapter,
    model_adapter,
    out_dir: Path,
    *,
    metrics_cfg: dict,
    iou_threshold: float,
):
    ensure_dir(out_dir)
    opts = tracking_options(metrics_cfg)
    per_seq = []

    for seq in tqdm(
        _iter_tracking_sequences(dataset_adapter),
        desc="tracking",
        unit="seq",
        dynamic_ncols=True,
    ):
        try:
            pred_tracks, pred_masks, seq_infer = infer_tracks_for_sequence(
                model_adapter,
                seq,
                oom_fallback_max_frames=opts["oom_fallback_max_frames"],
            )
            if not isinstance(pred_tracks, pd.DataFrame):
                pred_tracks = pd.DataFrame(pred_tracks)
            per_seq.append(
                score_sequence(
                    seq,
                    seq_infer,
                    pred_tracks,
                    pred_masks,
                    match_radius_px=opts["match_radius_px"],
                    pred_min_track_frames=opts["pred_min_track_frames"],
                    mask_metrics_on_tracking=opts["mask_metrics_on_tracking"],
                    iou_threshold=iou_threshold,
                    model_adapter=model_adapter,
                )
            )
        except NotImplementedError:
            raise
        except Exception as e:
            meta = seq.meta or {}
            per_seq.append(
                {
                    "seq_id": seq.seq_id,
                    "n_frames": len(seq.frames),
                    "constant_cell_count": meta.get("constant_cell_count"),
                    "error": str(e),
                }
            )
            print(
                f"cellseek-benchmark: sequence {seq.seq_id!r} failed: {e}",
                flush=True,
            )
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    summary = build_tracking_summary(per_seq, buckets=opts["frame_length_buckets"])
    dump_json({"summary": summary, "per_sequence": per_seq}, out_dir / "tracking.json")
    return summary


def _sequence_has_seg_gt(seq: TrackSequence) -> bool:
    if not seq.gt_masks:
        return False
    return any(g is not None for g in seq.gt_masks)


def run_joint_pass(
    dataset_adapter,
    model_adapter,
    seg_out_dir: Path,
    trk_out_dir: Path,
    iou_threshold: float,
    metrics_cfg: dict,
):
    opts = tracking_options(metrics_cfg)
    match_radius_px = opts["match_radius_px"]
    ensure_dir(seg_out_dir)
    ensure_dir(trk_out_dir)
    per_sample = []
    agg = defaultdict(float)
    n_seg = 0
    per_seq_trk = []

    for seq in tqdm(
        _iter_tracking_sequences(dataset_adapter),
        desc="seg+track (joint)",
        unit="seq",
        dynamic_ncols=True,
    ):
        try:
            pred_tracks, pred_masks, seq_infer = infer_tracks_for_sequence(
                model_adapter,
                seq,
                oom_fallback_max_frames=opts["oom_fallback_max_frames"],
            )
        except NotImplementedError:
            raise
        except Exception as e:
            raise RuntimeError(f"joint inference failed for sequence {seq.seq_id!r}: {e}") from e

        if not isinstance(pred_tracks, pd.DataFrame):
            pred_tracks = pd.DataFrame(pred_tracks)
        per_seq_trk.append(
            score_sequence(
                seq,
                seq_infer,
                pred_tracks,
                pred_masks,
                match_radius_px=match_radius_px,
                pred_min_track_frames=opts["pred_min_track_frames"],
                mask_metrics_on_tracking=opts["mask_metrics_on_tracking"],
                iou_threshold=iou_threshold,
                model_adapter=model_adapter,
            )
        )

        if (
            pred_masks
            and _sequence_has_seg_gt(seq_infer)
            and len(pred_masks) == len(seq_infer.frames)
        ):
            sample_ids = seq_infer.meta.get("frame_seg_sample_ids") if seq_infer.meta else None
            for i in range(len(seq_infer.frames)):
                gm = seq_infer.gt_masks[i] if seq_infer.gt_masks else None
                pm = pred_masks[i]
                if gm is None or pm is None:
                    continue
                sid = sample_ids[i] if sample_ids is not None and i < len(sample_ids) else f"{seq_infer.seq_id}/frame{i}"
                m = prf1_iou(gm, pm, iou_threshold=iou_threshold)
                per_sample.append({"sample_id": sid, **m})
                agg["tp"] += m["tp"]
                agg["fp"] += m["fp"]
                agg["fn"] += m["fn"]
                n_seg += 1

    p = agg["tp"] / (agg["tp"] + agg["fp"] + 1e-8)
    r = agg["tp"] / (agg["tp"] + agg["fn"] + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    seg_summary = {
        "n_samples": n_seg,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(agg["tp"]),
        "fp": int(agg["fp"]),
        "fn": int(agg["fn"]),
        "joint_with_tracking_inference": True,
    }
    dump_json(seg_summary, seg_out_dir / "summary.json")
    dump_json(per_sample, seg_out_dir / "per_sample.json")

    trk_summary = build_tracking_summary(
        per_seq_trk, buckets=opts["frame_length_buckets"]
    )
    trk_summary["joint_with_segmentation_inference"] = True
    dump_json({"summary": trk_summary, "per_sequence": per_seq_trk}, trk_out_dir / "tracking.json")
    return seg_summary, trk_summary


def _finalize_report(output_root: Path, report: dict) -> None:
    report_path = output_root / "report.json"
    summary_path = output_root / "summary.csv"
    dump_json(report, report_path)
    _write_summary_csv(report, summary_path)
    print(
        f"cellseek-benchmark: wrote {report_path.resolve()} and {summary_path.resolve()}",
        flush=True,
    )
    _print_report_errors(report)


def _print_report_errors(report: dict) -> None:
    for ds, payload in report.items():
        if "__dataset_init_error__" in payload:
            err = payload["__dataset_init_error__"].get("error", payload["__dataset_init_error__"])
            print(f"cellseek-benchmark: dataset '{ds}' failed to load: {err}", flush=True)
        for model_name, results in payload.items():
            if model_name == "__dataset_init_error__":
                continue
            if isinstance(results, dict) and "model_init_error" in results:
                print(
                    f"cellseek-benchmark: model '{model_name}' on '{ds}' failed to load: {results['model_init_error']}",
                    flush=True,
                )
            if isinstance(results, dict) and "skipped" in results:
                print(
                    f"cellseek-benchmark: model '{model_name}' on '{ds}' skipped: {results['skipped']}",
                    flush=True,
                )


def _write_summary_csv(report: dict, out_csv: Path):
    fieldnames = [
        "dataset",
        "model",
        "init_error",
        "seg_error",
        "trk_error",
        "n_seg_samples",
        "precision",
        "recall",
        "f1",
        "TA",
        "HOTA",
        "DetA",
        "AssA",
        "mask_f1_macro_mean",
        "n_trk_sequences",
        "n_trk_scored",
        "n_trk_error",
    ]
    rows: list[dict] = []

    for dataset_name, model_data in report.items():
        if not isinstance(model_data, dict):
            continue
        if "__dataset_init_error__" in model_data:
            payload = model_data["__dataset_init_error__"]
            err = str(payload.get("error", payload)) if isinstance(payload, dict) else str(payload)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": "__dataset__",
                    "init_error": err,
                    "seg_error": "",
                    "trk_error": "",
                    "n_seg_samples": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "TA": None,
                    "HOTA": None,
                    "DetA": None,
                    "AssA": None,
                    "n_trk_sequences": None,
                    "n_trk_scored": None,
                    "n_trk_error": None,
                }
            )
        for model_name, results in model_data.items():
            if model_name == "__dataset_init_error__":
                continue
            if not isinstance(results, dict):
                continue
            if "model_init_error" in results:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "init_error": str(results["model_init_error"]),
                        "seg_error": "",
                        "trk_error": "",
                        "n_seg_samples": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "TA": None,
                        "HOTA": None,
                        "DetA": None,
                        "AssA": None,
                        "n_trk_sequences": None,
                        "n_trk_scored": None,
                        "n_trk_error": None,
                    }
                )
                continue
            if "skipped" in results:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "init_error": "",
                        "seg_error": f"SKIPPED: {results['skipped']}",
                        "trk_error": f"SKIPPED: {results['skipped']}",
                        "n_seg_samples": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "TA": None,
                        "HOTA": None,
                        "DetA": None,
                        "AssA": None,
                        "n_trk_sequences": None,
                        "n_trk_scored": None,
                        "n_trk_error": None,
                    }
                )
                continue
            seg = results.get("segmentation") if isinstance(results.get("segmentation"), dict) else {}
            trk = results.get("tracking") if isinstance(results.get("tracking"), dict) else {}
            seg_err = seg.get("error")
            trk_err = trk.get("error")
            seg_skip = seg.get("skipped")
            trk_skip = trk.get("skipped")
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "init_error": "",
                    "seg_error": ("" if seg_err is None else str(seg_err))
                    or (f"SKIPPED: {seg_skip}" if seg_skip is not None else ""),
                    "trk_error": ("" if trk_err is None else str(trk_err))
                    or (f"SKIPPED: {trk_skip}" if trk_skip is not None else ""),
                    "n_seg_samples": seg.get("n_samples"),
                    "precision": seg.get("precision"),
                    "recall": seg.get("recall"),
                    "f1": seg.get("f1"),
                    "TA": trk.get("TA"),
                    "HOTA": trk.get("HOTA"),
                    "DetA": trk.get("DetA"),
                    "AssA": trk.get("AssA"),
                    "mask_f1_macro_mean": trk.get("mask_f1_macro_mean"),
                    "n_trk_sequences": trk.get("n_sequences_total"),
                    "n_trk_scored": trk.get("n_sequences_scored"),
                    "n_trk_error": trk.get("n_sequences_error"),
                }
            )

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def run_benchmark(
    config_dir: Path,
    dataset: str | None = None,
    model: str | None = None,
    tasks: str | None = None,
    benchmark_config: str = "experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml",
    datasets_config: str | None = None,
    models_config: str | None = None,
    metrics_config: str | None = None,
    output_parent: str | None = None,
    run_name: str | None = None,
) -> dict:
    """Load YAML from ``config_dir`` and run exactly one dataset × one model."""
    config_dir = Path(config_dir)
    bc = Path(benchmark_config)
    bench_path = bc.resolve() if bc.is_absolute() else (config_dir / bc).resolve()
    if not bench_path.is_file():
        raise FileNotFoundError(
            f"Experiment config not found: {bench_path}\n"
            f"  (--config-dir {config_dir}, --benchmark-config {benchmark_config})"
        )
    bench_cfg = load_yaml(bench_path)
    cfg_root = config_dir.resolve()
    try:
        shown = str(bench_path.resolve().relative_to(cfg_root))
    except ValueError:
        shown = str(bench_path.resolve())
    print(f"cellseek-benchmark: experiment {shown}", flush=True)

    ds_file = Path(datasets_config or bench_cfg.get("datasets_config", "datasets.yaml")).name
    md_file = Path(models_config or bench_cfg.get("models_config", "models.yaml")).name
    mt_file = Path(metrics_config or bench_cfg.get("metrics_config", "metrics.yaml")).name

    datasets_cfg = load_yaml(config_dir / ds_file)["datasets"]
    models_cfg = load_yaml(config_dir / md_file)["models"]
    metrics_cfg = load_yaml(config_dir / mt_file)
    print(f"cellseek-benchmark: datasets={ds_file}, models={md_file}, metrics={mt_file}", flush=True)

    dataset_name, model_name, requested_tasks = _resolve_single_run(
        bench_cfg, dataset, model, tasks
    )

    output_root = _resolve_run_output_root(
        bench_cfg,
        bench_path,
        output_parent=output_parent,
        run_name=run_name,
    )
    ensure_dir(output_root)
    print(f"cellseek-benchmark: writing outputs under {output_root}", flush=True)

    report: dict = {}

    if dataset_name not in datasets_cfg:
        raise ValueError(f"Unknown dataset {dataset_name!r}; keys: {sorted(datasets_cfg)}")
    if model_name not in models_cfg:
        raise ValueError(f"Unknown model {model_name!r}; keys: {sorted(models_cfg)}")

    report[dataset_name] = {}
    d_type = datasets_cfg[dataset_name]["type"]

    if not models_cfg[model_name].get("enabled", True):
        report[dataset_name][model_name] = {"skipped": f"Model {model_name!r} disabled in {md_file}."}
        print(f"cellseek-benchmark: model {model_name!r} skipped (enabled: false).", flush=True)
        _finalize_report(output_root, report)
        return report

    combo_tasks, skip_reason = _select_tasks_for_combo(
        dataset_name, model_name, requested_tasks, d_type
    )
    if not combo_tasks:
        report[dataset_name][model_name] = {"skipped": skip_reason or "No runnable tasks."}
        print(
            f"cellseek-benchmark: dataset={dataset_name!r} model={model_name!r} skipped: {skip_reason}",
            flush=True,
        )
        _finalize_report(output_root, report)
        return report

    try:
        dataset_adapter = DATASET_REGISTRY[d_type](datasets_cfg[dataset_name])
    except Exception as e:
        report[dataset_name]["__dataset_init_error__"] = {"error": str(e)}
        print(f"cellseek-benchmark: dataset init failed for {dataset_name!r}: {e}", flush=True)
        _finalize_report(output_root, report)
        return report

    m_type = models_cfg[model_name]["type"]
    out_dir = output_root / dataset_name / model_name
    ensure_dir(out_dir)

    print(
        f"cellseek-benchmark: dataset={dataset_name!r} model={model_name!r} — loading model …",
        flush=True,
    )
    try:
        model_adapter = MODEL_REGISTRY[m_type](models_cfg[model_name])
    except Exception as e:
        report[dataset_name][model_name] = {"model_init_error": str(e)}
        print(
            f"cellseek-benchmark: model init failed for dataset={dataset_name!r}, model={model_name!r}: {e}",
            flush=True,
        )
        _finalize_report(output_root, report)
        return report

    print(
        f"cellseek-benchmark: dataset={dataset_name!r} model={model_name!r} — tasks={combo_tasks}",
        flush=True,
    )
    result: dict = {}
    iou_thr = metrics_cfg["segmentation"]["iou_thresholds"][0]
    ran_joint = False
    if (
        "segmentation" in combo_tasks
        and "tracking" in combo_tasks
        and getattr(dataset_adapter, "joint_segmentation_with_tracking", False)
        and callable(getattr(model_adapter, "predict_tracks_returning_masks", None))
    ):
        print("cellseek-benchmark: joint segmentation+tracking …", flush=True)
        try:
            seg_summary, trk_summary = run_joint_pass(
                dataset_adapter,
                model_adapter,
                out_dir / "segmentation",
                out_dir / "tracking",
                iou_thr,
                metrics_cfg,
            )
            result["segmentation"] = seg_summary
            result["tracking"] = trk_summary
            ran_joint = True
        except NotImplementedError as e:
            print(f"cellseek-benchmark: joint unavailable ({e}); falling back.", flush=True)
        except Exception as e:
            print(f"cellseek-benchmark: joint failed ({e}); falling back.", flush=True)

    if not ran_joint:
        if "segmentation" in combo_tasks:
            try:
                result["segmentation"] = run_segmentation_pass(
                    dataset_adapter, model_adapter, out_dir / "segmentation", iou_thr
                )
            except NotImplementedError as e:
                result["segmentation"] = {"skipped": str(e)}
                print(f"cellseek-benchmark: segmentation skipped: {e}", flush=True)
            except Exception as e:
                result["segmentation"] = {"error": str(e)}
                print(f"cellseek-benchmark: segmentation failed: {e}", flush=True)

        if "tracking" in combo_tasks:
            try:
                result["tracking"] = run_tracking_pass(
                    dataset_adapter,
                    model_adapter,
                    out_dir / "tracking",
                    metrics_cfg=metrics_cfg,
                    iou_threshold=iou_thr,
                )
            except NotImplementedError as e:
                result["tracking"] = {"skipped": str(e)}
                trk_dir = out_dir / "tracking"
                ensure_dir(trk_dir)
                dump_json({"skipped": str(e), "per_sequence": []}, trk_dir / "skipped.json")
                print(f"cellseek-benchmark: tracking skipped: {e}", flush=True)
            except Exception as e:
                result["tracking"] = {"error": str(e)}
                print(f"cellseek-benchmark: tracking failed: {e}", flush=True)

    report[dataset_name][model_name] = result
    _finalize_report(output_root, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="CellSeek benchmark (one dataset × one model)")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="experiments/cellseek_ctc_bf_c2dl_hsc_tracking.yaml",
        help="Experiment YAML under --config-dir (path relative to config dir or absolute).",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        choices=("segmentation", "tracking", "both"),
    )
    parser.add_argument("--datasets-config", type=str, default=None)
    parser.add_argument("--models-config", type=str, default=None)
    parser.add_argument("--metrics-config", type=str, default=None)
    parser.add_argument(
        "--output-parent",
        type=str,
        default=None,
        help="Directory that will contain this run's folder (default: experiments or YAML output_parent).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Prefix for the run folder (default: experiment_id or benchmark YAML stem).",
    )
    args = parser.parse_args()
    run_benchmark(
        Path(args.config_dir),
        dataset=args.dataset,
        model=args.model,
        tasks=args.tasks,
        benchmark_config=args.benchmark_config,
        datasets_config=args.datasets_config,
        models_config=args.models_config,
        metrics_config=args.metrics_config,
        output_parent=args.output_parent,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
