#!/usr/bin/env python
"""Phase 10: predict stance labels for a CSV with columns `id,text` (optional `language`).

    python scripts/predict.py --input my.csv --output preds.csv --model final
    python scripts/predict.py --input my.csv --output preds.csv --model efficient --device cpu
    python scripts/predict.py --input my.csv --output preds.csv --hub Aakash25-11-04/Climate_Stance_Indic

--hub REPO         : download the final model from the Hugging Face Hub (use this on a new computer,
                     where models/ is empty). A private repo needs `hf auth login` first.

--model final      : the final submission model (models/final/ -> see models/final/MODEL.json)
--model efficient  : the hybrid sparse+dense baseline (models/hybrid_baseline.pkl) or the
                     dynamically quantised transformer if present (models/efficient/)
Output columns: id, text, language, label, p_Favour, p_Against, p_None.
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

import re  # noqa: E402

from stance.preprocessing.normalize import _BENGALI, _DEVANAGARI, detect_script  # noqa: E402,F401
from stance.utils import LABELS, abs_path, read_data  # noqa: E402


# Frequent function words that separate Romanized Bengali from Romanized Hindi. Words shared by both
# languages ("ki", "na", "bhai") are left out on purpose.
_BN_WORDS = set("""ami amra amar amader tumi tomar tomader apni apnar ora oder ache achhe achi nei noy kintu keno kore
korun korte korchi korbo korlam kori bhalo khub hobe hoy hoye holo hocche hochhe chilo chhilo sob shob eta ota ei oi
ekta ekti theke kotha keu kichu kichhu ekhon ekhoni ekdin jemon jodi tobe tai jonno moto bolche bole bolo dekhi dekhini
jani jane thik gorom thanda chara r ar shudhu uchit onujayi somossa mithye bachate kom ekebare amake tomake kono er der ta kemon bhabe nio""".split())
_HI_WORDS = set("""hai hain ho raha rahe rahi tha thi the ka ke ko se pe par nahi nahin kya aur bhi mein me hum ham yeh ye
woh wo bahut sab log kuch toh to gaya gayi karo karna karne kar wala wali wale yaar abhi kab kaise kyun kyon padti padta
hota hoti hoga diya liya apna apni mujhe tum aap unka uska isme usme jab tak ek dono""".split())
_WORD = re.compile(r"[a-z]+")


def guess_language(text: str) -> str:
    """hi or bn. Native script decides when present (also inside mixed-script text such as
    "গরম এত বেড়েছে ... global warming is no joke"); Romanized text is decided by function words."""
    text = text or ""
    d, b = len(_DEVANAGARI.findall(text)), len(_BENGALI.findall(text))
    if d or b:
        return "bn" if b > d else "hi"
    words = _WORD.findall(text.lower())
    bn_hits = sum(w in _BN_WORDS for w in words)
    hi_hits = sum(w in _HI_WORDS for w in words)
    return "bn" if bn_hits > hi_hits else "hi"


class FinalModel:
    """Loads whatever models/final/MODEL.json points to (transformer, ensemble, or hybrid)."""

    def __init__(self, kind: str = "final", device: str = "auto", hub: str | None = None):
        self.kind = kind
        if hub:
            self._load_from_hub(hub, device)
            return
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

    def _load_from_hub(self, repo: str, device: str) -> None:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo))
        self.spec = json.loads((root / "MODEL.json").read_text(encoding="utf-8"))
        if (root / "ensemble" / "ensemble.json").exists():
            from stance.models.ensemble import EnsembleModel

            self.model = EnsembleModel.from_dir(root, device=device)
        else:
            from stance.models.transformer import TransformerStanceModel

            self.model = TransformerStanceModel.load(root, device=device)
        self.calibration = self.spec.get("calibration")

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
    ap.add_argument("--hub", default=None, help="Hugging Face repo id to download the final model from")
    args = ap.parse_args()

    df = read_data(args.input)
    if "text" not in df.columns:
        alias = next((c for c in ("comment", "comments", "Comment", "Text", "sentence") if c in df.columns), None)
        if alias is None:
            raise SystemExit("input CSV needs a 'text' (or 'comment') column")
        df = df.rename(columns={alias: "text"})
    if "id" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    df["text"] = df["text"].fillna("").astype(str)
    if "language" not in df.columns or (df["language"] == "").all():
        df["language"] = df["text"].map(guess_language)
    if not args.hub and not abs_path(f"models/{args.model}/MODEL.json").exists():
        raise SystemExit("No local model found. On a new computer use --hub Aakash25-11-04/Climate_Stance_Indic")
    fm = FinalModel(args.model, device=args.device, hub=args.hub)
    probs = []
    for b in range(0, len(df), args.batch_size):
        chunk = df.iloc[b : b + args.batch_size]
        probs.append(fm.predict_proba(chunk["text"].tolist(), chunk["language"].tolist()))
    probs = np.vstack(probs) if probs else np.zeros((0, 3))
    out = pd.DataFrame({"id": df["id"], "text": df["text"], "language": df["language"],
                        "label": [LABELS[i] for i in probs.argmax(1)]})
    for i, l in enumerate(LABELS):
        out[f"p_{l}"] = probs[:, i]
    out.to_csv(args.output, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel shows Hindi/Bengali correctly
    print(f"wrote {args.output} ({len(out)} rows) with model={args.hub or args.model} type={fm.spec['type']}")
    print(out["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
