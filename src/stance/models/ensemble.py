"""Ensembles (Phase 8): probability averaging and stacking over heterogeneous saved models.

An EnsembleModel is described by a JSON spec:
{
  "members": [
     {"name": "muril_pair", "type": "transformer", "path": "models/muril_pair", "weight": 1.0},
     {"name": "tfidf_lr",   "type": "pickle",      "path": "models/tfidf_lr.pkl", "weight": 0.5},
     {"name": "emb_labse",  "type": "pickle",      "path": "models/emb_labse_lr.pkl", "weight": 0.5}
  ],
  "method": "average" | "stack",
  "meta": {"coef": [[...]], "intercept": [...]}     # only for stacking (LogisticRegression weights)
}
Members are loaded lazily and one GPU model at a time to fit the 4 GB card.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from stance.utils import LABELS, abs_path


def _load_member(spec: dict, device: str = "auto"):
    t = spec["type"]
    if t == "transformer":
        from stance.models.transformer import TransformerStanceModel

        return TransformerStanceModel.load(spec["path"], device=device)
    if t == "pickle":
        with open(abs_path(spec["path"]), "rb") as f:
            m = pickle.load(f)
        if hasattr(m, "embedder") and device != "auto":
            m.embedder.device = device
        return m
    if t == "onnx":
        from stance.models.efficient import OnnxStanceModel

        return OnnxStanceModel.load(spec["path"])
    raise ValueError(t)


def _proba(m, texts, languages):
    try:
        return np.asarray(m.predict_proba(texts, languages), dtype=np.float64)
    except TypeError:
        return np.asarray(m.predict_proba(texts), dtype=np.float64)


class EnsembleModel:
    def __init__(self, spec: dict, device: str = "auto"):
        self.spec = spec
        self.device = device
        self.name = spec.get("name", "ensemble")
        self._members: dict[str, object] = {}

    @classmethod
    def from_dir(cls, root: str | Path, device: str = "auto") -> "EnsembleModel":
        """Load the ensemble from a downloaded Hugging Face snapshot (or any folder) whose
        `ensemble/ensemble.json` stores member paths relative to that folder."""
        root = Path(root)
        spec = json.loads((root / "ensemble" / "ensemble.json").read_text(encoding="utf-8"))
        for m in spec["members"]:
            m["path"] = str((root / m["path"]).resolve())
        return cls(spec, device=device)

    def member(self, s: dict):
        if s["name"] not in self._members:
            self._members[s["name"]] = _load_member(s, self.device)
        return self._members[s["name"]]

    def member_probas(self, texts, languages) -> dict[str, np.ndarray]:
        out = {}
        for s in self.spec["members"]:
            out[s["name"]] = _proba(self.member(s), texts, languages)
        return out

    def combine(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        if self.spec.get("method", "average") == "stack":
            X = np.hstack([probas[s["name"]] for s in self.spec["members"]])
            coef = np.asarray(self.spec["meta"]["coef"]); b = np.asarray(self.spec["meta"]["intercept"])
            z = X @ coef.T + b
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        w = np.array([s.get("weight", 1.0) for s in self.spec["members"]], dtype=float)
        w = w / w.sum()
        return sum(wi * probas[s["name"]] for wi, s in zip(w, self.spec["members"]))

    def predict_proba(self, texts, languages=None):
        languages = languages or [None] * len(texts)
        return self.combine(self.member_probas(texts, languages))

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]

    def save(self, path: str | Path):
        path = abs_path(path); path.mkdir(parents=True, exist_ok=True)
        with open(path / "ensemble.json", "w", encoding="utf-8") as f:
            json.dump(self.spec, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        path = abs_path(path)
        p = path / "ensemble.json" if path.is_dir() else path
        with open(p, encoding="utf-8") as f:
            return cls(json.load(f), device=device)


def average_probs(prob_list: list[np.ndarray], weights=None) -> np.ndarray:
    w = np.ones(len(prob_list)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, prob_list))
