"""Stratified K-fold cross-validation (by label x language), out-of-fold probabilities,
inner-fold grid search, and the stacking meta-learner (hybrid baseline j).

No leakage: vectorisers / classifiers / hyper-parameters are fitted only on the training
part of each fold; the meta-learner is trained on OOF probabilities only.
"""
from __future__ import annotations

import itertools
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from stance.evaluation.metrics import evaluate
from stance.utils import LABEL2ID, LABELS


def strata(df: pd.DataFrame) -> np.ndarray:
    return np.array((df["label"].astype(str) + "|" + df["language"].astype(str)).tolist(), dtype=object)


def group_aware_folds(df: pd.DataFrame, n_splits: int, seed: int):
    """Stratified folds by label x language, but rows sharing a group_id (a source sentence
    and its augmented copies) always land in the same fold."""
    if "group_id" in df.columns and df["group_id"].notna().any():
        g = df.groupby("group_id")
        gdf = g.agg(label=("label", "first"), language=("language", "first")).reset_index()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        gstrata = np.array((gdf["label"].astype(str) + "|" + gdf["language"].astype(str)).tolist(), dtype=object)
        gid_to_fold = {}
        for f, (_, te) in enumerate(skf.split(gdf, gstrata)):
            for gid in gdf.loc[te, "group_id"]:
                gid_to_fold[gid] = f
        folds = np.array(df["group_id"].map(gid_to_fold).tolist(), dtype=int)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = np.zeros(len(df), dtype=int)
        for f, (_, te) in enumerate(skf.split(df, strata(df))):
            folds[te] = f
    for f in range(n_splits):
        yield np.where(folds != f)[0], np.where(folds == f)[0]


def _fit_predict(model_factory: Callable, df: pd.DataFrame, tr: np.ndarray, te: np.ndarray, **fkw):
    m = model_factory(**fkw)
    m.fit(df["text"].iloc[tr].tolist(), df["label"].iloc[tr].tolist(), df["language"].iloc[tr].tolist())
    return m.predict_proba(df["text"].iloc[te].tolist(), df["language"].iloc[te].tolist()) \
        if "languages" in m.predict_proba.__code__.co_varnames else m.predict_proba(df["text"].iloc[te].tolist())


def cross_validate(model_factory: Callable, df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                   grid: dict | None = None, inner_splits: int = 3, log=print) -> dict:
    """Return OOF probabilities, per-fold metrics, mean/std macro F1 and chosen params.

    `grid`: {param: [values]} searched with an inner CV on each outer training fold
    (so test folds never influence hyper-parameters).
    """
    df = df.reset_index(drop=True)
    oof = np.zeros((len(df), len(LABELS)), dtype=np.float32)
    fold_scores, chosen, t0 = [], [], time.time()
    for k, (tr, te) in enumerate(group_aware_folds(df, n_splits, seed)):
        best_params = {}
        if grid:
            best, best_params = -1.0, None
            keys = list(grid)
            inner_df = df.iloc[tr].reset_index(drop=True)
            for combo in itertools.product(*[grid[k_] for k_ in keys]):
                params = dict(zip(keys, combo))
                scores = []
                for itr, ite in group_aware_folds(inner_df, inner_splits, seed + 1):
                    p = _fit_predict(model_factory, inner_df, itr, ite, **params)
                    scores.append(evaluate(inner_df["label"].iloc[ite], [LABELS[i] for i in p.argmax(1)])["macro_f1"])
                s = float(np.mean(scores))
                if s > best:
                    best, best_params = s, params
            chosen.append(best_params)
        p = _fit_predict(model_factory, df, tr, te, **(best_params or {}))
        oof[te] = p
        m = evaluate(df["label"].iloc[te], [LABELS[i] for i in p.argmax(1)], df["language"].iloc[te])
        fold_scores.append(m["macro_f1"])
        log(f"  fold {k}: macro F1 {m['macro_f1']:.4f} params={best_params}")
    overall = evaluate(df["label"], [LABELS[i] for i in oof.argmax(1)], df["language"])
    return {
        "oof": oof,
        "fold_macro_f1": fold_scores,
        "cv_macro_f1_mean": float(np.mean(fold_scores)),
        "cv_macro_f1_std": float(np.std(fold_scores)),
        "oof_metrics": overall,
        "chosen_params": chosen,
        "seconds": time.time() - t0,
    }


class StackingModel:
    """j) Logistic-regression meta-learner over concatenated base-model probabilities.
    Base models are trained on the full training set; the meta-learner on OOF probabilities."""

    def __init__(self, base_factories: dict, oof_probs: dict, labels: list[str], C: float = 1.0, seed: int = 42):
        self.base_factories = base_factories
        self.names = list(base_factories)
        X = np.hstack([oof_probs[n] for n in self.names])
        y = np.array([LABEL2ID[l] for l in labels])
        self.meta = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed).fit(X, y)
        self.base_models = {}
        self.name = "stack(" + "+".join(self.names) + ")"

    def __getstate__(self):  # factories are closures (not picklable); fitted base models are enough
        d = dict(self.__dict__); d["base_factories"] = None
        return d

    def fit(self, texts, labels, languages=None):
        for n, f in self.base_factories.items():
            self.base_models[n] = f().fit(texts, labels, languages)
        return self

    def predict_proba(self, texts, languages=None):
        feats = []
        for n in self.names:
            m = self.base_models[n]
            try:
                feats.append(m.predict_proba(texts, languages))
            except TypeError:
                feats.append(m.predict_proba(texts))
        p = self.meta.predict_proba(np.hstack(feats))
        out = np.zeros((len(texts), len(LABELS)))
        out[:, self.meta.classes_] = p
        return out

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]
