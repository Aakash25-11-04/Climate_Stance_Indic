#!/usr/bin/env python
"""Phase 3: run all baselines with stratified 5-fold CV, evaluate on the gold set and the
challenge suite, write one comparison table and log every run to results/experiments.csv.

    python scripts/run_baselines.py --config configs/baselines.yaml
    python scripts/run_baselines.py --config configs/baselines.yaml --set data.cv_rows=500 models=[majority,tfidf_lr]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stance.evaluation import evaluate, evaluate_by, summarize  # noqa: E402
from stance.evaluation.challenge import evaluate_challenge, load_challenge  # noqa: E402
from stance.models.cv import StackingModel, cross_validate  # noqa: E402
from stance.models.dense import Embedder, EmbeddingLRModel, TargetAwareEmbeddingModel, ZeroShotNLIModel  # noqa: E402
from stance.models.sparse import (LexiconModel, MajorityModel, NBSVMModel, StratifiedRandomModel,  # noqa: E402
                                  TfidfModel, TransliteratedCopyModel)
from stance.utils import ExperimentLogger, abs_path, load_config, read_data, save_json, set_seed  # noqa: E402


def registry(cfg: dict) -> dict:
    pre = cfg.get("preprocessing")
    seed = cfg["seed"]
    claims = cfg["claim"]
    return {
        "majority": (lambda **kw: MajorityModel(), None, "sparse"),
        "stratified_random": (lambda **kw: StratifiedRandomModel(seed), None, "sparse"),
        "lexicon": (lambda **kw: LexiconModel(), None, "sparse"),
        "tfidf_lr": (lambda **kw: TfidfModel("lr", preprocessing=pre, seed=seed, **kw), "tfidf_lr", "sparse"),
        "tfidf_svm": (lambda **kw: TfidfModel("svm", preprocessing=pre, seed=seed, **kw), "tfidf_svm", "sparse"),
        "tfidf_cnb": (lambda **kw: TfidfModel("cnb", preprocessing=pre, seed=seed, **kw), "tfidf_cnb", "sparse"),
        "nbsvm": (lambda **kw: NBSVMModel(preprocessing=pre, seed=seed, **kw), "nbsvm", "sparse"),
        # e) no inner grid search: doubled-length text makes nested CV exhaust 16 GB RAM (ASSUMPTIONS #14);
        #    C is fixed to the value the base model selected in its own nested CV (tfidf_lr C=1.0, nbsvm C=0.5).
        "tfidf_lr_translit": (lambda **kw: TransliteratedCopyModel(TfidfModel("lr", preprocessing=pre, seed=seed, **{"C": 1.0, **kw})), None, "sparse"),
        "nbsvm_translit": (lambda **kw: TransliteratedCopyModel(NBSVMModel(preprocessing=pre, seed=seed, **{"C": 0.5, **kw})), None, "sparse"),
        "emb_labse_lr": (lambda **kw: EmbeddingLRModel("labse", preprocessing=pre, seed=seed, **kw), "emb_lr", "dense"),
        "emb_e5base_lr": (lambda **kw: EmbeddingLRModel("e5-base", preprocessing=pre, seed=seed, **kw), "emb_lr", "dense"),
        "emb_e5small_lr": (lambda **kw: EmbeddingLRModel("e5-small", preprocessing=pre, seed=seed, **kw), "emb_lr", "dense"),
        "emb_bgem3_lr": (lambda **kw: EmbeddingLRModel("bge-m3", preprocessing=pre, seed=seed, **kw), "emb_lr", "dense"),
        "emb_labse_target_lr": (lambda **kw: TargetAwareEmbeddingModel(claims, "labse", preprocessing=pre, seed=seed, **kw), "emb_lr", "dense"),
        "emb_e5base_target_lr": (lambda **kw: TargetAwareEmbeddingModel(claims, "e5-base", preprocessing=pre, seed=seed, **kw), "emb_lr", "dense"),
        "zeroshot_nli": (lambda **kw: ZeroShotNLIModel(), None, "nli"),
    }


def cpu_latency_ms(model, texts: list[str], languages: list[str], n: int = 200) -> float:
    texts, languages = texts[:n], languages[:n]
    t0 = time.perf_counter()
    try:
        model.predict(texts, languages)
    except TypeError:
        model.predict(texts)
    return (time.perf_counter() - t0) * 1000 / max(len(texts), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pre_cfg = load_config("configs/preprocessing.yaml")
    cfg.setdefault("preprocessing", pre_cfg["preprocessing"])
    set_seed(cfg["seed"])
    logger = ExperimentLogger(cfg["paths"]["experiments_csv"])

    train = read_data(cfg["data"]["train"])
    dev = read_data(cfg["data"]["dev"])
    gold = read_data(cfg["data"]["gold"])
    challenge = load_challenge(cfg["data"]["challenge_dir"])
    if cfg["data"].get("cv_rows"):
        train = train.sample(n=min(len(train), int(cfg["data"]["cv_rows"])), random_state=cfg["seed"]).reset_index(drop=True)
    print(f"train {len(train)}  dev {len(dev)}  gold {len(gold)}  challenge files {list(challenge)}")
    # CV runs on train; dev is used as an extra held-out check and (later) for threshold tuning
    reg = registry(cfg)
    oof_dir = abs_path(cfg["out"]["oof_dir"]); oof_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = abs_path(cfg["out"]["predictions_dir"]); pred_dir.mkdir(parents=True, exist_ok=True)
    rows, oofs, family = [], {}, {}
    prev = abs_path(cfg["out"]["table"])
    if prev.exists():  # resume: keep rows (and OOF files) of models not re-run now
        for _, r in pd.read_csv(prev).iterrows():
            if r["model"] in cfg["models"] and r["model"] != "stack_best":
                continue
            if r["model"] == "stack_best":
                continue
            if (oof_dir / f"{r['model']}.npy").exists():
                oofs[r["model"]] = np.load(oof_dir / f"{r['model']}.npy"); family[r["model"]] = r["family"]
                rows.append(r.to_dict())
                print(f"[resume] kept {r['model']} (gold {r['gold_macro_f1']:.4f})")
    models_to_run = [m for m in cfg["models"] if m != "stack_best"]
    for name in models_to_run:
        if name not in reg:
            print(f"[skip] unknown model {name}")
            continue
        factory, grid_key, fam = reg[name]
        grid = cfg.get("grids", {}).get(grid_key) if grid_key else None
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            if name in ("zeroshot_nli",):
                # no training: evaluate directly on train (as pseudo-OOF), dev, gold
                m = factory()
                oof = m.predict_proba(train["text"].tolist(), train["language"].tolist())
                oof_metrics = evaluate(train["label"], m.predict(train["text"].tolist(), train["language"].tolist()), train["language"])
                res = {"oof": oof, "cv_macro_f1_mean": oof_metrics["macro_f1"], "cv_macro_f1_std": 0.0,
                       "oof_metrics": oof_metrics, "chosen_params": [], "seconds": time.time() - t0}
                final = m
            else:
                res = cross_validate(factory, train, n_splits=cfg["cv"]["n_splits"], seed=cfg["seed"], grid=grid,
                                     inner_splits=cfg["cv"]["inner_splits"])
                params = {}
                if res["chosen_params"]:
                    # most frequent parameter combo across folds
                    combos = [json.dumps(p, sort_keys=True) for p in res["chosen_params"]]
                    params = json.loads(max(set(combos), key=combos.count))
                final = factory(**params).fit(train["text"].tolist(), train["label"].tolist(), train["language"].tolist())
                res["final_params"] = params
        except Exception as e:  # noqa: BLE001
            print(f"[FAILED] {name}: {type(e).__name__}: {e}")
            logger.log(phase="3", experiment=name, model=name, config=args.config, seed=cfg["seed"], notes=f"FAILED {e}")
            continue
        train_seconds = time.time() - t0
        oofs[name] = res["oof"]; family[name] = fam
        np.save(oof_dir / f"{name}.npy", res["oof"])

        def _pred(df):
            try:
                return final.predict(df["text"].tolist(), df["language"].tolist())
            except TypeError:
                return final.predict(df["text"].tolist())

        def _proba(df):
            try:
                return final.predict_proba(df["text"].tolist(), df["language"].tolist())
            except TypeError:
                return final.predict_proba(df["text"].tolist())

        dev_m = evaluate(dev["label"], _pred(dev), dev["language"])
        gold_pred = _pred(gold)
        gold_m = evaluate(gold["label"], gold_pred, gold["language"])
        gold_cat = evaluate_by(gold["label"], gold_pred, gold["category"]) if "category" in gold.columns else {}
        pd.DataFrame({"id": gold["id"], "text": gold["text"], "language": gold["language"], "label": gold["label"],
                      "pred": gold_pred, **{f"p_{l}": _proba(gold)[:, i] for i, l in enumerate(["Favour", "Against", "None"])}}
                     ).to_csv(pred_dir / f"gold_{name}.csv", index=False, encoding="utf-8")
        ch = evaluate_challenge(final, challenge) if challenge else {"overall_fail_rate": float("nan"), "per_category": {}}
        if challenge:
            ch["rows"].to_csv(pred_dir / f"challenge_{name}.csv", index=False, encoding="utf-8")
        # CPU latency: dense models are measured with the encoder on CPU
        if fam == "dense":
            Embedder.unload()
            final.embedder.device = "cpu"
            final.embedder.cache = type(final.embedder.cache)("data/cache/_latency_tmp.sqlite")
            lat = cpu_latency_ms(final, gold["text"].tolist(), gold["language"].tolist(), n=64)
            Embedder.unload()
        elif fam == "nli":
            lat = float("nan")
        else:
            lat = cpu_latency_ms(final, gold["text"].tolist(), gold["language"].tolist())
        row = {
            "model": name, "family": fam,
            "cv_macro_f1_mean": res["cv_macro_f1_mean"], "cv_macro_f1_std": res["cv_macro_f1_std"],
            "oof_f1_hi": res["oof_metrics"].get("per_language", {}).get("hi", {}).get("macro_f1", float("nan")),
            "oof_f1_bn": res["oof_metrics"].get("per_language", {}).get("bn", {}).get("macro_f1", float("nan")),
            "dev_macro_f1": dev_m["macro_f1"], "gold_macro_f1": gold_m["macro_f1"], "gold_accuracy": gold_m["accuracy"],
            "gold_f1_hi": gold_m["per_language"].get("hi", {}).get("macro_f1", float("nan")),
            "gold_f1_bn": gold_m["per_language"].get("bn", {}).get("macro_f1", float("nan")),
            "challenge_fail_rate": ch["overall_fail_rate"],
            "train_seconds": train_seconds, "cpu_ms_per_example": lat,
            "params": json.dumps(res.get("final_params", {})),
        }
        rows.append(row)
        if fam in ("sparse", "dense") and name not in ("majority", "stratified_random"):
            import pickle

            abs_path("models").mkdir(exist_ok=True)
            if fam == "dense":
                Embedder.unload()
            with open(abs_path(f"models/{name}.pkl"), "wb") as f:
                pickle.dump(final, f)
        print(f"CV macro F1 {row['cv_macro_f1_mean']:.4f} ± {row['cv_macro_f1_std']:.4f} | dev {row['dev_macro_f1']:.4f} | "
              f"gold {row['gold_macro_f1']:.4f} (hi {row['gold_f1_hi']:.3f} bn {row['gold_f1_bn']:.3f}) | "
              f"challenge fail {row['challenge_fail_rate']:.3f} | {lat:.1f} ms/ex")
        print(summarize(gold_m))
        save_json({"cv": {k: v for k, v in res.items() if k != "oof"}, "dev": dev_m, "gold": gold_m,
                   "gold_by_category": gold_cat, "challenge": {k: v for k, v in ch.items() if k != "rows"}},
                  pred_dir / f"metrics_{name}.json")
        logger.log(phase="3", experiment=name, model=name, config=args.config, seed=cfg["seed"], train_rows=len(train),
                   cv_macro_f1_mean=row["cv_macro_f1_mean"], cv_macro_f1_std=row["cv_macro_f1_std"],
                   dev_macro_f1=row["dev_macro_f1"], dev_accuracy=dev_m["accuracy"], gold_macro_f1=row["gold_macro_f1"],
                   gold_accuracy=row["gold_accuracy"], challenge_fail_rate=row["challenge_fail_rate"],
                   f1_hi=row["gold_f1_hi"], f1_bn=row["gold_f1_bn"], train_seconds=train_seconds,
                   cpu_ms_per_example=lat, params=row["params"])
        pd.DataFrame(rows).to_csv(abs_path(cfg["out"]["table"]), index=False)

    # ------------------------------------------------------------------ j) stacking
    if "stack_best" in cfg["models"] and rows:
        tab = pd.DataFrame(rows)
        best_sparse = cfg["stacking"].get("sparse")
        best_dense = cfg["stacking"].get("dense")
        if best_sparse in (None, "auto"):
            cand = tab[(tab.family == "sparse") & ~tab.model.isin(["majority", "stratified_random", "lexicon"])]
            best_sparse = cand.sort_values("cv_macro_f1_mean").iloc[-1]["model"] if len(cand) else None
        if best_dense in (None, "auto"):
            cand = tab[tab.family == "dense"]
            best_dense = cand.sort_values("cv_macro_f1_mean").iloc[-1]["model"] if len(cand) else None
        if best_sparse and best_dense:
            name = "stack_best"
            print(f"\n=== {name}: {best_sparse} + {best_dense} ===")
            t0 = time.time()
            params = {n: json.loads(tab[tab.model == n].iloc[0]["params"]) for n in (best_sparse, best_dense)}
            factories = {n: (lambda n=n: reg[n][0](**params[n])) for n in (best_sparse, best_dense)}
            stack = StackingModel(factories, {n: oofs[n] for n in factories}, train["label"].tolist(), seed=cfg["seed"])
            # OOF estimate of the stack itself: meta-learner CV on the OOF features
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            X = np.hstack([oofs[n] for n in factories]); y = np.array(train["label"].tolist(), dtype=object)
            meta_oof = cross_val_predict(LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000),
                                         X, y, cv=StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]), method="predict")
            stack_cv = evaluate(y, meta_oof, train["language"])
            stack.fit(train["text"].tolist(), train["label"].tolist(), train["language"].tolist())
            dev_m = evaluate(dev["label"], stack.predict(dev["text"].tolist(), dev["language"].tolist()), dev["language"])
            gold_pred = stack.predict(gold["text"].tolist(), gold["language"].tolist())
            gold_m = evaluate(gold["label"], gold_pred, gold["language"])
            gp = stack.predict_proba(gold["text"].tolist(), gold["language"].tolist())
            pd.DataFrame({"id": gold["id"], "text": gold["text"], "language": gold["language"], "label": gold["label"],
                          "pred": gold_pred, "p_Favour": gp[:, 0], "p_Against": gp[:, 1], "p_None": gp[:, 2]}
                         ).to_csv(pred_dir / f"gold_{name}.csv", index=False, encoding="utf-8")
            ch = evaluate_challenge(stack, challenge) if challenge else {"overall_fail_rate": float("nan"), "per_category": {}}
            if challenge:
                ch["rows"].to_csv(pred_dir / f"challenge_{name}.csv", index=False, encoding="utf-8")
            Embedder.unload()
            for m in stack.base_models.values():
                if hasattr(m, "embedder"):
                    m.embedder.device = "cpu"
            lat = cpu_latency_ms(stack, gold["text"].tolist(), gold["language"].tolist(), n=64)
            row = {"model": name, "family": "hybrid", "cv_macro_f1_mean": stack_cv["macro_f1"], "cv_macro_f1_std": float("nan"),
                   "oof_f1_hi": stack_cv["per_language"].get("hi", {}).get("macro_f1"), "oof_f1_bn": stack_cv["per_language"].get("bn", {}).get("macro_f1"),
                   "dev_macro_f1": dev_m["macro_f1"], "gold_macro_f1": gold_m["macro_f1"], "gold_accuracy": gold_m["accuracy"],
                   "gold_f1_hi": gold_m["per_language"].get("hi", {}).get("macro_f1"), "gold_f1_bn": gold_m["per_language"].get("bn", {}).get("macro_f1"),
                   "challenge_fail_rate": ch["overall_fail_rate"], "train_seconds": time.time() - t0, "cpu_ms_per_example": lat,
                   "params": json.dumps({"sparse": best_sparse, "dense": best_dense})}
            rows.append(row)
            print(f"stack CV(meta OOF) {row['cv_macro_f1_mean']:.4f} | dev {row['dev_macro_f1']:.4f} | gold {row['gold_macro_f1']:.4f} | challenge fail {row['challenge_fail_rate']:.3f}")
            print(summarize(gold_m))
            save_json({"cv": stack_cv, "dev": dev_m, "gold": gold_m,
                       "gold_by_category": evaluate_by(gold["label"], gold_pred, gold["category"]),
                       "challenge": {k: v for k, v in ch.items() if k != "rows"}}, pred_dir / f"metrics_{name}.json")
            logger.log(phase="3", experiment=name, model=name, config=args.config, seed=cfg["seed"], train_rows=len(train),
                       cv_macro_f1_mean=row["cv_macro_f1_mean"], dev_macro_f1=row["dev_macro_f1"], gold_macro_f1=row["gold_macro_f1"],
                       gold_accuracy=row["gold_accuracy"], challenge_fail_rate=row["challenge_fail_rate"], f1_hi=row["gold_f1_hi"],
                       f1_bn=row["gold_f1_bn"], train_seconds=row["train_seconds"], cpu_ms_per_example=lat, params=row["params"])
            import pickle

            with open(abs_path("models/hybrid_baseline.pkl"), "wb") as f:
                pickle.dump(stack, f)
    tab = pd.DataFrame(rows)
    tab.to_csv(abs_path(cfg["out"]["table"]), index=False)
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(tab):
        best = tab.sort_values(["gold_macro_f1", "cv_macro_f1_mean"]).iloc[-1]
        print(f"\nBASELINE TO BEAT (by gold macro F1, tie-break CV): {best['model']} gold {best['gold_macro_f1']:.4f} CV {best['cv_macro_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
