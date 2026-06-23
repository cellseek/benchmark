from pathlib import Path
import json
import os
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
CELLSEEK_ROOT = BENCHMARK_ROOT.parent


def resolve_checkpoint_path(path: str | Path | None) -> Path | None:
    """Resolve a checkpoint path from config (absolute, or relative to benchmark/ / cellseek/)."""
    if path is None or not str(path).strip():
        return None
    raw = Path(path).expanduser()
    if raw.is_file():
        return raw.resolve()
    for base in (BENCHMARK_ROOT, CELLSEEK_ROOT, Path.cwd()):
        candidate = (base / raw).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
