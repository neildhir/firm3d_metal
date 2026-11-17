"""Typed loader for the head-to-head config.yaml."""

from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML


@dataclass(frozen=True)
class H2HConfig:
    """Parameters of the head-to-head (see config.yaml for units)."""

    benchmark_config: Path
    nparticles: int
    tmax_s: float
    tols: tuple[float, ...]
    repeats: int
    warmup_nparticles: int
    warmup_tmax_s: float
    data_dir: Path


def load_config(path: Path) -> H2HConfig:
    """Load config.yaml into a typed config; a misspelt key raises TypeError."""
    raw = YAML(typ="safe").load(path.read_text())
    raw["benchmark_config"] = (
        path.parent / raw["benchmark_config"]
    ).resolve()  # relative to the config file
    raw["data_dir"] = (path.parent / raw["data_dir"]).resolve()
    raw["tols"] = tuple(float(t) for t in raw["tols"])
    return H2HConfig(**raw)
