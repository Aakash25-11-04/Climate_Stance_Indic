"""CheckList-style robustness challenge suite evaluation (Phase 7).

Files in data/challenge/ (all UTF-8 CSV):
  mft.csv          id, text, language, category, label
                   -> minimum functionality tests: fail if prediction != label
  invariance.csv   id, text_a, text_b, language, category
                   -> fail if prediction(text_a) != prediction(text_b)
  directional.csv  id, text_a, text_b, language, category, label_a, label_b
                   -> fail if prediction(text_b) != label_b  (and we also report whether
                      the prediction changed from a to b)

Failure rate per category and overall; a model's `predict(texts, languages)` is enough.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stance.utils import abs_path, read_data


def load_challenge(challenge_dir: str | Path = "data/challenge") -> dict[str, pd.DataFrame]:
    d = abs_path(challenge_dir)
    out = {}
    for name in ("mft", "invariance", "directional"):
        p = d / f"{name}.csv"
        if p.exists():
            out[name] = read_data(p)
    return out


def _predict(model, texts, languages):
    try:
        return list(model.predict(texts, languages))
    except TypeError:
        return list(model.predict(texts))


def evaluate_challenge(model, challenge: dict[str, pd.DataFrame] | None = None,
                       challenge_dir: str | Path = "data/challenge") -> dict:
    """Return {"overall_fail_rate", "n_tests", "per_category": {cat: {fail_rate, n}}, "per_type": ...,
    "rows": DataFrame with per-test outcomes}."""
    challenge = challenge if challenge is not None else load_challenge(challenge_dir)
    if not challenge:
        return {"overall_fail_rate": float("nan"), "n_tests": 0, "per_category": {}, "per_type": {}, "rows": pd.DataFrame()}
    records = []
    if "mft" in challenge:
        df = challenge["mft"]
        preds = _predict(model, df["text"].tolist(), df["language"].tolist())
        for (_, r), p in zip(df.iterrows(), preds):
            records.append({"type": "mft", "id": r["id"], "category": r["category"], "language": r["language"],
                            "text": r["text"], "expected": r["label"], "pred": p, "fail": int(p != r["label"])})
    if "invariance" in challenge:
        df = challenge["invariance"]
        pa = _predict(model, df["text_a"].tolist(), df["language"].tolist())
        pb = _predict(model, df["text_b"].tolist(), df["language"].tolist())
        for (_, r), a, b in zip(df.iterrows(), pa, pb):
            records.append({"type": "invariance", "id": r["id"], "category": r["category"], "language": r["language"],
                            "text": r["text_a"] + "  =>  " + r["text_b"], "expected": a, "pred": b, "fail": int(a != b)})
    if "directional" in challenge:
        df = challenge["directional"]
        pa = _predict(model, df["text_a"].tolist(), df["language"].tolist())
        pb = _predict(model, df["text_b"].tolist(), df["language"].tolist())
        for (_, r), a, b in zip(df.iterrows(), pa, pb):
            records.append({"type": "directional", "id": r["id"], "category": r["category"], "language": r["language"],
                            "text": r["text_a"] + "  =>  " + r["text_b"], "expected": r["label_b"], "pred": b,
                            "fail": int(b != r["label_b"]), "pred_a": a, "expected_a": r["label_a"]})
    rows = pd.DataFrame(records)
    per_cat = {c: {"fail_rate": float(g["fail"].mean()), "n": int(len(g))} for c, g in rows.groupby("category")}
    per_type = {c: {"fail_rate": float(g["fail"].mean()), "n": int(len(g))} for c, g in rows.groupby("type")}
    per_lang = {c: {"fail_rate": float(g["fail"].mean()), "n": int(len(g))} for c, g in rows.groupby("language")}
    return {
        "overall_fail_rate": float(rows["fail"].mean()),
        "n_tests": int(len(rows)),
        "per_category": per_cat,
        "per_type": per_type,
        "per_language": per_lang,
        "rows": rows,
    }
