#!/usr/bin/env python
"""Phase 4 driver: run the transformer matrix in configs/transformers_all.yaml, each run in a
fresh subprocess (clean GPU memory), then aggregate results/predictions/metrics_*.json into
results/transformers.csv.

    python scripts/run_transformers.py --config configs/transformers_all.yaml [--only muril_pair_claim-auto]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stance.utils import abs_path, load_config  # noqa: E402

PY = sys.executable


def run_tag(r: dict) -> str:
    tag = f"{r['name']}_{r.get('formulation', 'pair')}"
    if r.get("formulation", "pair") == "pair":
        tag += f"_claim-{r.get('claim_lang', 'auto')}"
    tag += f"_{r.get('loss', 'weighted_ce')}"
    return tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformers_all.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of run tags to run")
    ap.add_argument("--set", nargs="*", default=[], help="extra --set overrides passed to every run")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = cfg["base_config"]
    t0 = time.time()
    for r in cfg["runs"]:
        tag = run_tag(r)
        if args.only and not any(s in tag for s in args.only):
            continue
        if (time.time() - t0) / 60 > cfg.get("max_total_minutes", 600):
            print(f"[skip] time budget exhausted before {tag}")
            continue
        metrics = abs_path(f"results/predictions/metrics_{tag}_s{cfg['seed']}.json")
        if args.skip_existing and metrics.exists():
            print(f"[skip] {tag} exists")
            continue
        sets = [f"model.name={r['name']}", f"model.formulation={r.get('formulation', 'pair')}",
                f"model.claim_lang={r.get('claim_lang', 'auto')}", f"model.loss={r.get('loss', 'weighted_ce')}",
                f"out.model_dir=models/{tag}", f"seeds=[{cfg['seed']}]"]
        for k in ("lr", "epochs", "batch_size", "max_length", "grad_accum"):
            if k in r:
                sets.append(f"model.{k}={r[k]}")
        if r.get("languages"):
            sets.append("data.languages=[" + ",".join(r["languages"]) + "]")
        sets += args.set
        cmd = [PY, str(ROOT / "scripts" / "train.py"), "--config", base, "--set", *sets]
        print(f"\n### {tag}\n{' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"[FAILED] {tag} rc={rc}")
    rows = []
    for p in sorted(abs_path("results/predictions").glob("metrics_*_s*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "history" not in d:
            continue
        rows.append({"run": d["run"], "dev_macro_f1": d["dev"]["macro_f1"], "gold_macro_f1": d["gold"]["macro_f1"],
                     "gold_accuracy": d["gold"]["accuracy"],
                     "gold_f1_hi": d["gold"]["per_language"].get("hi", {}).get("macro_f1"),
                     "gold_f1_bn": d["gold"]["per_language"].get("bn", {}).get("macro_f1"),
                     "challenge_fail_rate": d["challenge"].get("overall_fail_rate"),
                     "epochs": len(d["history"]), "train_minutes": d["train_seconds"] / 60, "params_m": d.get("params_m")})
    tab = pd.DataFrame(rows).sort_values("gold_macro_f1", ascending=False)
    tab.to_csv(abs_path(cfg["out"]["table"]), index=False)
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
