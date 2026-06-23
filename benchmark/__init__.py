def __getattr__(name: str):
    if name == "main":
        from .test import main as _main

        return _main
    if name == "run_benchmark":
        from .test import run_benchmark as _run_benchmark

        return _run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main", "run_benchmark"]
