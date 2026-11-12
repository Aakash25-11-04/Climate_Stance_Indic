#!/usr/bin/env python
"""Phase 10: predict stance labels for a CSV with columns `id,text` (optional `language`).

    python scripts/predict.py --input my.csv --output preds.csv --model final
    python scripts/predict.py --input my.csv --output preds.csv --model efficient --device cpu

--model final      : the final submission model (models/final/ -> see models/final/MODEL.json)
--model efficient  : the hybrid sparse+dense baseline (models/hybrid_baseline.pkl) or the
                     dynamically quantised transformer if present (models/efficient/)
Output columns: id, label, p_Favour, p_Against, p_None.
Handles empty strings, emoji-only text and very long text (truncated at max_length) without crashing.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.preprocessing.normalize import detect_script  # noqa: E402
from stance.utils import LABELS, abs_path, read_data  # noqa: E402


def guess_language(text: str) -> str:
    s = detect_script(text)
    if s == "bengali":
        return "bn"
    if s == "devanagari":
        return "hi"
    # Latin / mixed: cheap heuristic on common Banglish vs Hinglish function words
    t = f" {text.lower()} "
    bn_hits = sum(t.count(f" {w} ") for w in ("ami", "amra", "ache", "nei", "kintu", "keno", "kore", "bhalo", "khub", "tumi", "hobe", "sob", "eta"))
    hi_hits = sum(t.count(f" {w} ") for w in ("hai", "nahi", "kya", "aur", "bhi", "mein", "hum", "yeh", "bahut", "sab", "log", "kuch", "toh"))
    return "bn" if bn_hits > hi_hits else "hi"


class FinalModel:
    """Loads whatever models/final/MODEL.json points to (transformer, ensemble, or hybrid)."""

    def __init__(self, kind: str = "final", device: str = "auto"):
        self.kind = kind
        spec_path = abs_path(f"models/{kind}/MODEL.json")
        if kind == "efficient" and not spec_path.exists():
            self.spec = {"type": "hybrid", "path": "models/hybrid_baseline.pkl"}
        else:
            with open(spec_path, encoding="utf-8") as f:
                self.spec = json.load(f)
        t = self.spec["type"]
        if t == "transformer":
            from stance.models.transformer import TransformerStanceModel

            self.model = TransformerStanceModel.load(self.spec["path"], device=device)
        elif t == "hybrid":
            with open(abs_path(self.spec["path"]), "rb") as f:
                self.model = pickle.load(f)
            for m in getattr(self.model, "base_models", {}).values():
                if hasattr(m, "embedder"):
                    m.embedder.device = device if device != "auto" else m.embedder.device
        elif t == "ensemble":
            from stance.models.ensemble import EnsembleModel

            self.model = EnsembleModel.load(self.spec["path"], device=device)
        elif t == "onnx":
            from stance.models.efficient import OnnxStanceModel

            self.model = OnnxStanceModel.load(self.spec["path"])
        elif t == "quantized":
            from stance.models.efficient import QuantizedStanceModel

            self.model = QuantizedStanceModel.load(self.spec["path"])
        else:
            raise ValueError(f"unknown model type {t}")
        self.calibration = self.spec.get("calibration")  # {"temperature": T, "thresholds": [..]} optional

    def predict_proba(self, texts, languages):
        texts = [t if isinstance(t, str) else "" for t in texts]
        texts = [t if t.strip() else "." for t in texts]
        try:
            p = self.model.predict_proba(texts, languages)
        except TypeError:
            p = self.model.predict_proba(texts)
        p = np.asarray(p, dtype=np.float64)
        if self.calibration:
            from stance.models.calibration import apply_calibration

            p = apply_calibration(p, self.calibration)
        return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="final", choices=["final", "efficient"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    df = read_data(args.input)
    if "text" not in df.columns:
        raise SystemExit("input CSV needs a 'text' column")
    if "id" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    df["text"] = df["text"].fillna("").astype(str)
    if "language" not in df.columns or (df["language"] == "").all():
        df["language"] = df["text"].map(guess_language)
    fm = FinalModel(args.model, device=args.device)
    probs = []
    for b in range(0, len(df), args.batch_size):
        chunk = df.iloc[b : b + args.batch_size]
        probs.append(fm.predict_proba(chunk["text"].tolist(), chunk["language"].tolist()))
    probs = np.vstack(probs) if probs else np.zeros((0, 3))
    out = pd.DataFrame({"id": df["id"], "label": [LABELS[i] for i in probs.argmax(1)]})
    for i, l in enumerate(LABELS):
        out[f"p_{l}"] = probs[:, i]
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} rows) with model={args.model} type={fm.spec['type']}")
    print(out["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
