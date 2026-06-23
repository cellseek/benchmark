"""Shared experiment/catalog config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import load_yaml


@dataclass(frozen=True)
class LoadedBenchmarkConfig:
    bench_cfg: dict
    datasets_cfg: dict
    models_cfg: dict
    metrics_cfg: dict
    bench_path: Path
    datasets_file: str
    models_file: str
    metrics_file: str
    shown_benchmark_path: str


def load_benchmark_configs(
    config_dir: str | Path,
    benchmark_config: str | Path,
    *,
    datasets_config: str | None = None,
    models_config: str | None = None,
    metrics_config: str | None = None,
) -> LoadedBenchmarkConfig:
    """Load one experiment YAML and its dataset/model/metric catalogs."""

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

    ds_file = Path(datasets_config or bench_cfg.get("datasets_config", "datasets.yaml")).name
    md_file = Path(models_config or bench_cfg.get("models_config", "models.yaml")).name
    mt_file = Path(metrics_config or bench_cfg.get("metrics_config", "metrics.yaml")).name

    return LoadedBenchmarkConfig(
        bench_cfg=bench_cfg,
        datasets_cfg=load_yaml(config_dir / ds_file)["datasets"],
        models_cfg=load_yaml(config_dir / md_file)["models"],
        metrics_cfg=load_yaml(config_dir / mt_file),
        bench_path=bench_path,
        datasets_file=ds_file,
        models_file=md_file,
        metrics_file=mt_file,
        shown_benchmark_path=shown,
    )
