"""Shared utilities: seeding, config loading, experiment logging, timing, device."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["Favour", "Against", "None"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the sparse baselines
        pass


# --------------------------------------------------------------------------- config
def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def load_config(config_path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load configs/default.yaml, overlay `config_path`, then apply `key.sub=value` overrides."""
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            cfg = _deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)
    cfg["_config_path"] = str(config_path) if config_path else None
    return cfg


def resolve_device(pref: str = "auto") -> str:
    if pref in ("cpu", "cuda"):
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------- git / hashing
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def stable_hash(*parts: Any, n: int = 12) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()[:n]


# --------------------------------------------------------------------------- timing
@contextmanager
def timer(name: str = "") -> Iterator[dict]:
    box = {"name": name, "seconds": 0.0}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - t0
        if name:
            print(f"[timer] {name}: {box['seconds']:.1f}s", file=sys.stderr)


# --------------------------------------------------------------------------- experiment logger
EXPERIMENT_COLUMNS = [
    "timestamp", "phase", "experiment", "model", "config", "seed", "git_commit",
    "train_rows", "cv_macro_f1_mean", "cv_macro_f1_std", "dev_macro_f1", "dev_accuracy",
    "gold_macro_f1", "gold_accuracy", "challenge_fail_rate", "f1_hi", "f1_bn",
    "train_seconds", "cpu_ms_per_example", "params_m", "notes",
]


class ExperimentLogger:
    """Append-only CSV logger. One row per experiment; missing fields stay empty."""

    def __init__(self, path: str | Path | None = None):
        self.path = abs_path(path or "results/experiments.csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS).writeheader()

    def log(self, **fields: Any) -> dict:
        row = {c: "" for c in EXPERIMENT_COLUMNS}
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["git_commit"] = git_commit()
        extra = {}
        for k, v in fields.items():
            if k in row:
                row[k] = _fmt(v)
            else:
                extra[k] = v
        if extra:
            row["notes"] = (row["notes"] + " " if row["notes"] else "") + json.dumps(extra, default=str, ensure_ascii=False)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS).writerow(row)
        return row


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str, ensure_ascii=False)
    return v


def read_data(path: str | Path, **kw) -> "pd.DataFrame":
    """Read a dataset CSV safely: the label 'None' must NOT become NaN."""
    import pandas as pd

    df = pd.read_csv(abs_path(path), keep_default_na=False, na_values=[""], encoding="utf-8", **kw)
    for c in ("text", "label", "language", "script", "source_type", "id"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def save_json(obj: Any, path: str | Path) -> None:
    path = abs_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    with open(abs_path(path), encoding="utf-8") as f:
        return json.load(f)


def count_params_m(model) -> float:
    try:
        return sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        return float("nan")
