"""Evaluation metrics: Macro F1 (primary), per-class P/R/F1, accuracy, confusion matrix,
per-language scores. All functions work on label strings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)

from stance.utils import LABELS


def evaluate(y_true, y_pred, languages=None, labels=LABELS) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_labels": list(labels),
    }
    if languages is not None:
        langs = pd.Series(list(languages))
        out["per_language"] = {}
        for lang in sorted(langs.unique()):
            m = (langs == lang).values
            if m.sum() == 0:
                continue
            yt = [y for y, k in zip(y_true, m) if k]
            yp = [y for y, k in zip(y_pred, m) if k]
            out["per_language"][lang] = {
                "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(yt, yp)),
                "n": int(m.sum()),
            }
    return out


def evaluate_by(y_true, y_pred, groups, labels=LABELS) -> dict:
    """Macro F1 / accuracy per group value (e.g. gold category or source_type)."""
    g = pd.Series(list(groups))
    res = {}
    for val in sorted(g.unique(), key=str):
        m = (g == val).values
        yt = [y for y, k in zip(y_true, m) if k]
        yp = [y for y, k in zip(y_pred, m) if k]
        res[str(val)] = {
            "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(yt, yp)),
            "n": int(m.sum()),
        }
    return res


def summarize(metrics: dict) -> str:
    pc = metrics["per_class"]
    lines = [f"macro F1 {metrics['macro_f1']:.4f}  acc {metrics['accuracy']:.4f}  n={metrics['n']}"]
    for lab, d in pc.items():
        lines.append(f"  {lab:8s} P {d['precision']:.3f} R {d['recall']:.3f} F1 {d['f1']:.3f} (n={d['support']})")
    if "per_language" in metrics:
        for lang, d in metrics["per_language"].items():
            lines.append(f"  [{lang}] macro F1 {d['macro_f1']:.4f} acc {d['accuracy']:.4f} n={d['n']}")
    return "\n".join(lines)


def probs_to_labels(probs: np.ndarray, labels=LABELS) -> list[str]:
    return [labels[i] for i in np.asarray(probs).argmax(axis=1)]
