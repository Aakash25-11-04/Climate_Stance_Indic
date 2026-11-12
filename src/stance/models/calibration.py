"""Temperature scaling and per-class threshold tuning (Phase 8).

Both are fitted ONLY on out-of-fold (or dev) probabilities, never on gold / challenge data.

apply_calibration(probs, spec) is used by predict.py:
    spec = {"temperature": T, "class_bias": [b_Favour, b_Against, b_None]}
    p' = softmax(log(p) / T + bias) ; label = argmax p'
The additive log-space bias is equivalent to per-class decision thresholds for argmax
decisions, and keeps the output a proper probability vector.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from stance.utils import LABELS


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Minimise NLL of softmax(log p / T) on held-out probabilities."""
    logp = np.log(np.clip(probs, 1e-9, 1.0))

    def nll(T):
        q = _softmax(logp / T)
        return -np.mean(np.log(np.clip(q[np.arange(len(y_idx)), y_idx], 1e-9, 1.0)))

    res = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded")
    return float(res.x)


def fit_class_bias(probs: np.ndarray, y_idx: np.ndarray, temperature: float = 1.0,
                   grid=np.linspace(-1.5, 1.5, 13)) -> list[float]:
    """Grid-search additive log-space biases (one per class, last fixed at 0) to maximise
    macro F1 on held-out data."""
    logp = np.log(np.clip(probs, 1e-9, 1.0)) / temperature
    best, best_bias = -1.0, [0.0] * len(LABELS)
    for combo in itertools.product(grid, repeat=len(LABELS) - 1):
        bias = np.array(list(combo) + [0.0])
        pred = (logp + bias).argmax(axis=1)
        f = f1_score(y_idx, pred, average="macro")
        if f > best + 1e-9:
            best, best_bias = f, bias.tolist()
    return best_bias


def apply_calibration(probs: np.ndarray, spec: dict) -> np.ndarray:
    T = float(spec.get("temperature", 1.0))
    bias = np.array(spec.get("class_bias", [0.0] * len(LABELS)), dtype=float)
    logp = np.log(np.clip(probs, 1e-9, 1.0)) / T + bias
    return _softmax(logp)


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)
