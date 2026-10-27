"""Configuration loading.

All hyperparameters live in ``configs/default.yaml``. An experiment is the
default file plus a list of ``key.subkey=value`` overrides, so every run in the
paper is fully described by (config file, overrides, fold).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access (cfg.train.lr) that round-trips to YAML."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(raw: str) -> Any:
    # yaml gives us ints, floats, bools, null and lists for free
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config section '{p}' in override '{item}'")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key '{key}' (typo?)")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Config:
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides or [])
    return Config.wrap(cfg)


def save_config(cfg: dict, path: str | Path) -> None:
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def resolve_path(p: str | Path) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def print_config(cfg: Config) -> None:
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


if __name__ == "__main__":
    print_config(load_config())
