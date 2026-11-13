#!/usr/bin/env python
"""Phase 1: build train/dev/gold datasets.

    python scripts/build_dataset.py --config configs/dataset.yaml
    python scripts/build_dataset.py --config configs/dataset.yaml --set dataset.translation.max_source_rows=50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stance.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="key.sub=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    from stance.data.build import build

    t0 = time.time()
    stats = build(cfg)
    print(f"[build] done in {(time.time() - t0) / 60:.1f} min")
    print({k: stats[k] for k in ("rows", "train_label_counts", "train_source_type_counts", "synthetic_share")})


if __name__ == "__main__":
    main()
