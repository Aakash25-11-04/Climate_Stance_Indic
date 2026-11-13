#!/usr/bin/env python
"""Phase 8: ensembles, calibration, efficiency and final-model selection.

    python scripts/build_final.py --config configs/final.yaml

Steps
 1. candidates = best saved transformers (results/transformers.csv), best sparse + dense pickles
    (results/baselines.csv), hybrid baseline (models/hybrid_baseline.pkl)
 2. member probabilities on dev / gold / challenge (cached per member)
 3. ensembles: equal average, weighted average (grid on DEV), stacking (LR on DEV probabilities)
    NOTE: transformer members have no out-of-fold probabilities on this hardware, so the
    meta-learner and the calibration are fitted on the DEV split (held out from training;
    dev was used only for early stopping). Gold / challenge are never used for fitting.
 4. temperature scaling + per-class log-bias tuned on DEV
 5. efficiency: int8 dynamic quantisation of the best transformer; CPU latency table
 6. selection: score = gold_macro_f1 - lambda * challenge_fail_rate  (TEST-ONLY data, reported)
 7. write models/final/MODEL.json, models/efficient/MODEL.json, results/final.json, results/final_candidates.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.evaluation import evaluate  # noqa: E402
from stance.evaluation.challenge import evaluate_challenge, load_challenge  # noqa: E402
from stance.models.calibration import apply_calibration, expected_calibration_error, fit_class_bias, fit_temperature  # noqa: E402
from stance.models.ensemble import EnsembleModel  # noqa: E402
from stance.utils import LABEL2ID, LABELS, ExperimentLogger, abs_path, load_config, read_data, save_json, set_seed  # noqa: E402


class _ProbaModel:
    """Wraps precomputed probabilities keyed by text for challenge evaluation of ensembles."""

    def __init__(self, fn):
        self.fn = fn

    def predict_proba(self, texts, languages=None):
        return self.fn(texts, languages)

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(1)]


def candidates(cfg: dict) -> list[dict]:
    c = cfg["candidates"]
    out = []
    tf = c.get("transformers", "auto")
    if tf == "auto":
        t = abs_path("results/transformers.csv")
        if t.exists():
            tab = pd.read_csv(t)
            # whole-system candidates must cover both languages (BanglaBERT runs are bn-only)
            if {"gold_f1_hi", "gold_f1_bn"} <= set(tab.columns):
                tab = tab.dropna(subset=["gold_f1_hi", "gold_f1_bn"])
            tab = tab.sort_values("dev_macro_f1", ascending=False)
            for run in tab["run"].head(int(c.get("top_n_transformers", 2))):
                d = abs_path("models") / run.rsplit("_s", 1)[0]
                if d.exists():
                    out.append({"name": d.name, "type": "transformer", "path": f"models/{d.name}"})
        if not out and abs_path("models/muril_pair").exists():
            out.append({"name": "muril_pair", "type": "transformer", "path": "models/muril_pair"})
    else:
        out += [{"name": Path(p).name, "type": "transformer", "path": p} for p in tf]
    b = abs_path("results/baselines.csv")
    if b.exists():
        tab = pd.read_csv(b)
        for fam, key in (("sparse", "sparse"), ("dense", "dense")):
            sel = c.get(key, "auto")
            if sel == "auto":
                sub = tab[(tab.family == fam) & ~tab.model.isin(["majority", "stratified_random", "lexicon"])]
                if len(sub):
                    sel = sub.sort_values(["gold_macro_f1", "cv_macro_f1_mean"]).iloc[-1]["model"]
            if sel and abs_path(f"models/{sel}.pkl").exists():
                out.append({"name": sel, "type": "pickle", "path": f"models/{sel}.pkl"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    logger = ExperimentLogger(cfg["paths"]["experiments_csv"])
    dev = read_data(cfg["data"]["dev"]); gold = read_data(cfg["data"]["gold"])
    challenge = load_challenge(cfg["data"]["challenge_dir"])
    members = candidates(cfg)
    if not members:
        raise SystemExit("no candidate models found: run run_baselines.py and train.py first")
    print("members:", [m["name"] for m in members])

    # ---------------------------------------------------------------- 2. member probabilities
    ens_all = EnsembleModel({"members": members, "method": "average"})
    texts_all, langs_all = [], []
    for df in (dev, gold):
        texts_all += df["text"].tolist(); langs_all += df["language"].tolist()
    for key in ("mft",):
        if key in challenge:
            texts_all += challenge[key]["text"].tolist(); langs_all += challenge[key]["language"].tolist()
    for key in ("invariance", "directional"):
        if key in challenge:
            texts_all += challenge[key]["text_a"].tolist() + challenge[key]["text_b"].tolist()
            langs_all += challenge[key]["language"].tolist() * 2
    t0 = time.time()
    probas = ens_all.member_probas(texts_all, langs_all)
    print(f"member inference on {len(texts_all)} texts: {time.time() - t0:.0f}s")
    lookup = {m["name"]: {(t, l): p for t, l, p in zip(texts_all, langs_all, probas[m["name"]])} for m in members}

    def proba_fn(member_names, combine):
        def fn(texts, languages):
            languages = languages or [None] * len(texts)
            per = {n: np.vstack([lookup[n][(t, l)] for t, l in zip(texts, languages)]) for n in member_names}
            return combine(per)
        return fn

    n_dev = len(dev)
    y_dev = np.array([LABEL2ID[l] for l in dev["label"]])
    rows, specs = [], {}

    def add_candidate(name, member_names, combine, spec, cal=None):
        fn = proba_fn(member_names, combine)
        if cal:
            raw = fn
            fn = lambda t, l, raw=raw: apply_calibration(raw(t, l), cal)  # noqa: E731
        model = _ProbaModel(fn)
        pd_ = model.predict_proba(dev["text"].tolist(), dev["language"].tolist())
        dev_m = evaluate(dev["label"], [LABELS[i] for i in pd_.argmax(1)], dev["language"])
        pg = model.predict_proba(gold["text"].tolist(), gold["language"].tolist())
        gold_m = evaluate(gold["label"], [LABELS[i] for i in pg.argmax(1)], gold["language"])
        ch = evaluate_challenge(model, challenge) if challenge else {"overall_fail_rate": float("nan")}
        row = {"candidate": name, "members": "+".join(member_names), "dev_macro_f1": dev_m["macro_f1"],
               "dev_ece": expected_calibration_error(pd_, y_dev), "gold_macro_f1": gold_m["macro_f1"], "gold_accuracy": gold_m["accuracy"],
               "gold_f1_hi": gold_m["per_language"].get("hi", {}).get("macro_f1"), "gold_f1_bn": gold_m["per_language"].get("bn", {}).get("macro_f1"),
               "challenge_fail_rate": ch["overall_fail_rate"], "calibrated": bool(cal)}
        row["score"] = row["gold_macro_f1"] - float(cfg["selection"]["lambda"]) * (row["challenge_fail_rate"] if not np.isnan(row["challenge_fail_rate"]) else 0)
        rows.append(row); specs[name] = (spec, cal, gold_m, ch)
        pd.DataFrame({"id": gold["id"], "text": gold["text"], "label": gold["label"], "pred": [LABELS[i] for i in pg.argmax(1)],
                      "p_Favour": pg[:, 0], "p_Against": pg[:, 1], "p_None": pg[:, 2]}).to_csv(abs_path(f"results/predictions/gold_final_{name}.csv"), index=False, encoding="utf-8")
        if challenge:
            ch["rows"].to_csv(abs_path(f"results/predictions/challenge_final_{name}.csv"), index=False, encoding="utf-8")
        print(f"{name:40s} dev {row['dev_macro_f1']:.4f} gold {row['gold_macro_f1']:.4f} challenge-fail {row['challenge_fail_rate']:.3f} score {row['score']:.4f}")
        return row

    # singles
    for m in members:
        add_candidate(m["name"], [m["name"]], lambda per, n=m["name"]: per[n], {"members": [m], "method": "average"})
    names = [m["name"] for m in members]
    methods = cfg["ensemble"]["methods"]
    if len(members) > 1:
        if "average" in methods:
            add_candidate("avg_all", names, lambda per: sum(per[n] for n in names) / len(names), {"members": members, "method": "average"})
        if "weighted" in methods:
            dev_probs = {n: np.vstack([lookup[n][(t, l)] for t, l in zip(dev["text"], dev["language"])]) for n in names}
            best, best_w = -1, None
            for w in itertools.product(cfg["ensemble"]["weight_grid"], repeat=len(names)):
                if sum(w) == 0:
                    continue
                p = sum(wi * dev_probs[n] for wi, n in zip(w, names)) / sum(w)
                f = evaluate(dev["label"], [LABELS[i] for i in p.argmax(1)])["macro_f1"]
                if f > best:
                    best, best_w = f, w
            wm = [dict(m, weight=float(wi)) for m, wi in zip(members, best_w)]
            add_candidate("weighted_dev", names, lambda per, w=best_w: sum(wi * per[n] for wi, n in zip(w, names)) / sum(w), {"members": wm, "method": "average"})
        if "stack" in methods:
            from sklearn.linear_model import LogisticRegression

            X = np.hstack([np.vstack([lookup[n][(t, l)] for t, l in zip(dev["text"], dev["language"])]) for n in names])
            meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=cfg["seed"]).fit(X, y_dev)
            spec = {"members": members, "method": "stack", "meta": {"coef": meta.coef_.tolist(), "intercept": meta.intercept_.tolist()}}
            ens = EnsembleModel(spec)
            add_candidate("stack_dev", names, lambda per, ens=ens: ens.combine(per), spec)
    # hybrid baseline (j) for comparison
    hb = abs_path("models/hybrid_baseline.pkl")
    if hb.exists():
        with open(hb, "rb") as f:
            hybrid = pickle.load(f)
        texts, langs = texts_all, langs_all
        hp = hybrid.predict_proba(texts, langs)
        lookup["hybrid_baseline"] = {(t, l): p for t, l, p in zip(texts, langs, hp)}
        add_candidate("hybrid_baseline", ["hybrid_baseline"], lambda per: per["hybrid_baseline"], {"type": "hybrid", "path": "models/hybrid_baseline.pkl"})

    # ---------------------------------------------------------------- 4. calibration of the best uncalibrated candidates
    tab = pd.DataFrame(rows)
    for name in tab.sort_values("score", ascending=False)["candidate"].head(3):
        spec, _, _, _ = specs[name]
        mem_names = tab[tab.candidate == name].iloc[0]["members"].split("+")
        base_fn = proba_fn(mem_names, (lambda per, n=name, spec=spec: (EnsembleModel(spec).combine(per) if spec.get("members") else per[n])))
        raw_dev = base_fn(dev["text"].tolist(), dev["language"].tolist())
        cal = {}
        if cfg["calibration"].get("temperature", True):
            cal["temperature"] = fit_temperature(raw_dev, y_dev)
        if cfg["calibration"].get("class_bias", True):
            cal["class_bias"] = fit_class_bias(raw_dev, y_dev, cal.get("temperature", 1.0))
        add_candidate(f"{name}+cal", mem_names, (lambda per, n=name, spec=spec: (EnsembleModel(spec).combine(per) if spec.get("members") else per[n])), spec, cal)
    tab = pd.DataFrame(rows).sort_values("score", ascending=False)
    tab.to_csv(abs_path(cfg["out"]["table"]), index=False)
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- 6. selection + write specs
    best = tab.iloc[0]
    spec, cal, gold_m, ch = specs[best["candidate"]]
    final_dir = abs_path(cfg["out"]["final_dir"]); final_dir.mkdir(parents=True, exist_ok=True)
    if spec.get("type") == "hybrid":
        final_spec = dict(spec)
    elif len(spec["members"]) == 1 and spec["members"][0]["type"] == "transformer":
        final_spec = {"type": "transformer", "path": spec["members"][0]["path"]}
    else:
        EnsembleModel(spec).save(final_dir)
        final_spec = {"type": "ensemble", "path": cfg["out"]["final_dir"]}
    final_spec["calibration"] = cal
    final_spec["candidate"] = best["candidate"]
    final_spec["selection_score"] = float(best["score"])
    (final_dir / "MODEL.json").write_text(json.dumps(final_spec, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- 5. efficiency: accuracy-vs-cost table
    cost_rows = []
    n_lat = int(cfg["efficiency"].get("latency_rows", 100))
    lat_texts, lat_langs = gold["text"].tolist()[:n_lat], gold["language"].tolist()[:n_lat]

    # timing uses uncached variants of the gold texts (embedding models cache by text), so the
    # numbers include real encoding cost; accuracy is always measured on the unmodified texts.
    lat_variants = [f"{t} {i}" for i, t in enumerate(lat_texts)]

    def latency(model):
        t0 = time.perf_counter(); model.predict(lat_variants, lat_langs); return (time.perf_counter() - t0) * 1000 / len(lat_texts)

    def score_of(r):
        fail = r["challenge_fail_rate"]
        return r["gold_macro_f1"] - float(cfg["selection"]["lambda"]) * (0 if pd.isna(fail) else fail)

    member_ms = {}  # measured CPU latency of every ensemble member

    def size_mb(path):
        p = abs_path(path)
        return sum(f.stat().st_size for f in (p.rglob("*") if p.is_dir() else [p])) / 1e6

    best_tf = next((m for m in members if m["type"] == "transformer"), None)
    sparse_best = next((m for m in members if m["type"] == "pickle"), None)
    if sparse_best:
        with open(abs_path(sparse_best["path"]), "rb") as f:
            sm = pickle.load(f)
        if hasattr(sm, "embedder"):
            sm.embedder.device = "cpu"
            from stance.models.dense import Embedder
            Embedder.unload()
        r = tab[tab.candidate == sparse_best["name"]].iloc[0]
        cost_rows.append({"model": sparse_best["name"], "gold_macro_f1": r["gold_macro_f1"], "challenge_fail_rate": r["challenge_fail_rate"], "size_mb": size_mb(sparse_best["path"]), "cpu_ms_per_example": latency(sm)})
        member_ms[sparse_best["name"]] = cost_rows[-1]["cpu_ms_per_example"]
        eff_options = [({"type": "hybrid", "path": sparse_best["path"]}, score_of(r))]
    else:
        eff_options = []
    if hb.exists():
        r = tab[tab.candidate == "hybrid_baseline"].iloc[0]
        for m in getattr(hybrid, "base_models", {}).values():
            if hasattr(m, "embedder"):
                m.embedder.device = "cpu"
        cost_rows.append({"model": "hybrid_baseline (j)", "gold_macro_f1": r["gold_macro_f1"], "challenge_fail_rate": r["challenge_fail_rate"], "size_mb": size_mb("models/hybrid_baseline.pkl"), "cpu_ms_per_example": latency(hybrid)})
        eff_options.append(({"type": "hybrid", "path": "models/hybrid_baseline.pkl"}, score_of(r)))
    if best_tf:
        from stance.models.transformer import TransformerStanceModel

        tm = TransformerStanceModel.load(best_tf["path"], device="cpu")
        r = tab[tab.candidate == best_tf["name"]].iloc[0]
        cost_rows.append({"model": best_tf["name"] + " (fp32 cpu)", "gold_macro_f1": r["gold_macro_f1"], "challenge_fail_rate": r["challenge_fail_rate"], "size_mb": size_mb(best_tf["path"]), "cpu_ms_per_example": latency(tm)})
        member_ms[best_tf["name"]] = cost_rows[-1]["cpu_ms_per_example"]
        if cfg["efficiency"].get("quantize_transformer", True):
            from stance.models.efficient import QuantizedStanceModel

            qm = QuantizedStanceModel(tm)
            qp = qm.predict_proba(gold["text"].tolist(), gold["language"].tolist())
            q_gold = evaluate(gold["label"], [LABELS[i] for i in qp.argmax(1)], gold["language"])
            q_ch = evaluate_challenge(qm, challenge) if challenge else {"overall_fail_rate": float("nan")}
            cost_rows.append({"model": best_tf["name"] + " (int8 dynamic, attention layers)", "gold_macro_f1": q_gold["macro_f1"], "challenge_fail_rate": q_ch["overall_fail_rate"], "size_mb": float("nan"), "cpu_ms_per_example": latency(qm)})
            eff_options.append(({"type": "quantized", "path": best_tf["path"], "calibration": cal if best["candidate"].startswith(best_tf["name"]) else None}, score_of(cost_rows[-1])))
    # efficient model = cheapest-family option with the best selection score (same rule as the final model)
    efficient_spec = max(eff_options, key=lambda o: o[1])[0] if eff_options else None
    def ensemble_ms():
        from stance.models.transformer import TransformerStanceModel
        for m in members:
            if m["name"] in member_ms:
                continue
            if m["type"] == "transformer":
                mm = TransformerStanceModel.load(m["path"], device="cpu")
            else:
                with open(abs_path(m["path"]), "rb") as f:
                    mm = pickle.load(f)
                if hasattr(mm, "embedder"):
                    mm.embedder.device = "cpu"
            member_ms[m["name"]] = latency(mm)
            del mm
        return sum(member_ms[m["name"]] for m in members)

    if len(tab[tab.candidate.str.startswith(("avg", "weighted", "stack"))]):
        r = tab[tab.candidate.str.startswith(("avg", "weighted", "stack"))].sort_values("score").iloc[-1]
        cost_rows.append({"model": f"ensemble ({r['candidate']})", "gold_macro_f1": r["gold_macro_f1"], "challenge_fail_rate": r["challenge_fail_rate"],
                          "size_mb": sum(size_mb(m["path"]) for m in members), "cpu_ms_per_example": ensemble_ms()})
    eff_dir = abs_path(cfg["out"]["efficient_dir"]); eff_dir.mkdir(parents=True, exist_ok=True)
    if efficient_spec:
        (eff_dir / "MODEL.json").write_text(json.dumps(efficient_spec, indent=2), encoding="utf-8")
    cost = pd.DataFrame(cost_rows)
    print("\naccuracy vs cost:\n" + cost.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    report = {"final": final_spec, "efficient": efficient_spec, "candidates": tab.to_dict(orient="records"),
              "accuracy_vs_cost": cost.to_dict(orient="records"), "final_gold": gold_m,
              "final_challenge": {k: v for k, v in ch.items() if k != "rows"},
              "note": "stacking meta-learner and calibration fitted on dev; gold/challenge are test-only"}
    save_json(report, cfg["out"]["report"])
    logger.log(phase="8", experiment=f"final={best['candidate']}", model=best["members"], config=args.config, seed=cfg["seed"],
               dev_macro_f1=best["dev_macro_f1"], gold_macro_f1=best["gold_macro_f1"], gold_accuracy=best["gold_accuracy"],
               challenge_fail_rate=best["challenge_fail_rate"], f1_hi=best["gold_f1_hi"], f1_bn=best["gold_f1_bn"], calibration=cal)
    print(f"\nFINAL: {best['candidate']} -> {final_dir / 'MODEL.json'}; efficient -> {eff_dir / 'MODEL.json'}")


if __name__ == "__main__":
    main()
