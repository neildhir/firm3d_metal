"""Typed loader for the layout experiment's config.yaml."""

from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML


@dataclass(frozen=True)
class LayoutConfig:
    """Parameters of the layout experiment (see config.yaml for units)."""

    benchmark_config: Path
    kernel_source: Path
    nparticles_max: int
    nparticles: tuple[int, ...]
    tmax_s: float
    tol: float
    steps_per_dispatch: int
    variants: tuple[str, ...]
    threads_per_tg_1t: tuple[int, ...]
    repeats: int
    smoke_nparticles: int
    smoke_tmax_s: float
    data_dir: Path
    field: str


def load_config(path: Path) -> LayoutConfig:
    """Load config.yaml into a typed config; a misspelt key raises TypeError.

    Paths in the file are relative to the file's own directory.
    """
    raw = YAML(typ="safe").load(path.read_text())
    for key in ("benchmark_config", "kernel_source", "data_dir"):
        raw[key] = (path.parent / raw[key]).resolve()  # relative to the config file
    for key in ("nparticles", "threads_per_tg_1t"):
        raw[key] = tuple(int(v) for v in raw[key])
    raw["variants"] = tuple(str(v) for v in raw["variants"])
    raw["tmax_s"] = float(raw["tmax_s"])
    raw["tol"] = float(raw["tol"])
    raw["smoke_tmax_s"] = float(raw["smoke_tmax_s"])
    if raw["field"] not in ("vacuum", "finite_beta"):
        raise ValueError(f"field must be vacuum or finite_beta, got {raw['field']!r}")
    return LayoutConfig(**raw)
