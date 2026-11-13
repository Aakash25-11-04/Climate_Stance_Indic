"""Shared 'auto' model selection used by explain.py, error_analysis.py and significance.py,
so every analysis refers to the same baseline-to-beat and final model.

Selection rules (documented in the README):
* baseline to beat  = best trained baseline (not majority / random / lexicon) by gold macro F1
                      in results/baselines.csv
* final candidate   = results/final.json["final"]["candidate"] (chosen by build_final.py)
* best transformer  = the transformer member of the final model if there is one, otherwise the
                      saved transformer run with the highest DEV macro F1 (dev, not gold, to avoid
                      selecting on the test set)
"""
from __future__ import annotations

import json

import pandas as pd

from stance.utils import abs_path

NOT_TRAINED = ("majority", "stratified_random", "lexicon")


def baseline_to_beat() -> str:
    b = pd.read_csv(abs_path("results/baselines.csv"))
    b = b[~b["model"].isin(NOT_TRAINED)]
    return str(b.sort_values(["gold_macro_f1", "cv_macro_f1_mean"]).iloc[-1]["model"])


def final_candidate() -> str | None:
    p = abs_path("results/final.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["final"].get("candidate")


def transformer_runs() -> pd.DataFrame:
    """One row per saved transformer run: run name, model dir, dev / gold macro F1."""
    rows = []
    for p in sorted(abs_path("results/predictions").glob("metrics_*_s*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "history" not in d:
            continue
        langs = (d.get("gold") or {}).get("per_language") or {}
        if langs and len(langs) < 2:
            continue  # language-restricted run (e.g. BanglaBERT, bn only) cannot be the whole system
        run = d["run"]
        base = run.rsplit("_s", 1)[0]
        cand = [abs_path(f"models/{base}"), abs_path(f"models/{run}")]
        if base == "muril_pair_claim-auto_weighted_ce":
            cand.append(abs_path("models/muril_pair"))  # MVS run saved under its config's out.model_dir
        mdir = next((c for c in cand if (c / "stance_meta.json").exists()), None)
        rows.append({"run": run, "model_dir": str(mdir) if mdir else None,
                     "dev_macro_f1": d["dev"]["macro_f1"], "gold_macro_f1": d["gold"]["macro_f1"]})
    return pd.DataFrame(rows)


def best_transformer() -> tuple[str, str]:
    """(run name, model dir) of the transformer to explain / analyse."""
    runs = transformer_runs().dropna(subset=["model_dir"])
    p = abs_path("results/final.json")
    if p.exists():
        spec = json.loads(p.read_text(encoding="utf-8"))["final"]
        paths = [spec.get("path", "")] + [m.get("path", "") for m in spec.get("members", [])]
        ens = abs_path("models/final/ensemble.json")
        if ens.exists():
            paths += [m.get("path", "") for m in json.loads(ens.read_text(encoding="utf-8"))["members"]]
        for path in paths:
            hit = runs[runs["model_dir"].map(lambda d: str(abs_path(path)) == d if path else False)]
            if len(hit):
                r = hit.sort_values("dev_macro_f1").iloc[-1]
                return r["run"], r["model_dir"]
    r = runs.sort_values("dev_macro_f1").iloc[-1]
    return r["run"], r["model_dir"]
