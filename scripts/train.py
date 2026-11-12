#!/usr/bin/env python
"""Phase 4: fine-tune a transformer stance model from a YAML config.

    python scripts/train.py --config configs/transformer_muril_pair.yaml
    python scripts/train.py --config configs/transformer_muril_pair.yaml --set model.name=xlmr seeds=[1,2,3]

Writes: models/<name>/ (if out.save_model), results/predictions/{dev,gold,challenge}_<run>.csv,
results/predictions/metrics_<run>.json and a row in results/experiments.csv per seed.
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
from stance.models.transformer import TransformerStanceModel  # noqa: E402
from stance.utils import ExperimentLogger, abs_path, count_params_m, load_config, read_data, save_json, set_seed  # noqa: E402


def run_name(cfg: dict, seed: int) -> str:
    m = cfg["model"]
    tag = f"{m['name']}_{m['formulation']}"
    if m["formulation"] == "pair":
        tag += f"_claim-{m['claim_lang']}"
    tag += f"_{m['loss']}"
    if cfg["data"].get("exclude_source_types"):
        tag += "_no-" + "-".join(cfg["data"]["exclude_source_types"])
    if cfg.get("run_tag"):
        tag += f"_{cfg['run_tag']}"
    return f"{tag}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer_muril_pair.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pre_cfg = load_config("configs/preprocessing.yaml")
    cfg.setdefault("preprocessing", pre_cfg["preprocessing"])
    logger = ExperimentLogger(cfg["paths"]["experiments_csv"])

    train = read_data(cfg["data"]["train"])
    dev = read_data(cfg["data"]["dev"])
    gold = read_data(cfg["data"]["gold"])
    challenge = load_challenge(cfg["data"].get("challenge_dir", "data/challenge"))
    excl = cfg["data"].get("exclude_source_types") or []
    if excl:
        train = train[~train["source_type"].isin(excl)].reset_index(drop=True)
    langs = cfg["data"].get("languages")
    if langs:  # e.g. BanglaBERT trained and evaluated on Bengali only
        train = train[train["language"].isin(langs)].reset_index(drop=True)
        dev = dev[dev["language"].isin(langs)].reset_index(drop=True)
        gold = gold[gold["language"].isin(langs)].reset_index(drop=True)
        challenge = {k: v[v["language"].isin(langs)].reset_index(drop=True) for k, v in challenge.items()}
    if cfg["data"].get("max_train_rows"):
        train = train.sample(n=min(len(train), int(cfg["data"]["max_train_rows"])), random_state=cfg["seed"]).reset_index(drop=True)
    extra_train = cfg["data"].get("extra_train")
    if extra_train:  # e.g. English originals for the cross-lingual ablation
        train = pd.concat([train, read_data(extra_train)], ignore_index=True)
    print(f"train {len(train)} dev {len(dev)} gold {len(gold)} challenge {list(challenge)}")
    pred_dir = abs_path(cfg["out"]["predictions_dir"]); pred_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in cfg.get("seeds", [cfg["seed"]]):
        set_seed(seed)
        name = run_name(cfg, seed)
        print(f"\n=== {name} ===")
        m = cfg["model"]
        model = TransformerStanceModel(
            model_name=m["name"], formulation=m["formulation"], claim_lang=m.get("claim_lang", "auto"),
            claims=cfg["claim"], max_length=m.get("max_length", 128), batch_size=m.get("batch_size", 16),
            grad_accum=m.get("grad_accum", 1), lr=float(m.get("lr", 2e-5)), epochs=m.get("epochs", 3),
            warmup_ratio=m.get("warmup_ratio", 0.1), weight_decay=m.get("weight_decay", 0.01),
            loss=m.get("loss", "weighted_ce"), focal_gamma=m.get("focal_gamma", 2.0),
            class_weight_power=m.get("class_weight_power", 0.5), patience=m.get("patience", 2),
            fp16=m.get("fp16", True), seed=seed, device=cfg.get("device", "auto"),
            preprocessing=cfg["preprocessing"], max_train_minutes=m.get("max_train_minutes", 170),
            freeze_embeddings=m.get("freeze_embeddings", True))
        t0 = time.time()
        model.fit(train["text"].tolist(), train["label"].tolist(), train["language"].tolist(),
                  dev=(dev["text"].tolist(), dev["label"].tolist(), dev["language"].tolist()))
        train_seconds = time.time() - t0

        def _eval(df, split):
            probs = model.predict_proba(df["text"].tolist(), df["language"].tolist())
            preds = [["Favour", "Against", "None"][i] for i in probs.argmax(1)]
            out = pd.DataFrame({"id": df["id"], "text": df["text"], "language": df["language"], "label": df["label"],
                                "pred": preds, "p_Favour": probs[:, 0], "p_Against": probs[:, 1], "p_None": probs[:, 2]})
            out.to_csv(pred_dir / f"{split}_{name}.csv", index=False, encoding="utf-8")
            return evaluate(df["label"], preds, df["language"]), preds

        dev_m, _ = _eval(dev, "dev")
        gold_m, gold_pred = _eval(gold, "gold")
        gold_cat = evaluate_by(gold["label"], gold_pred, gold["category"]) if "category" in gold.columns else {}
        ch = evaluate_challenge(model, challenge) if challenge else {"overall_fail_rate": float("nan"), "per_category": {}}
        if challenge:
            ch["rows"].to_csv(pred_dir / f"challenge_{name}.csv", index=False, encoding="utf-8")
        print(f"dev macro F1 {dev_m['macro_f1']:.4f} | gold macro F1 {gold_m['macro_f1']:.4f} | challenge fail {ch['overall_fail_rate']:.3f} | {train_seconds / 60:.1f} min")
        print(summarize(gold_m))
        params_m = count_params_m(model.model)
        save_json({"run": name, "config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "history": model.history,
                   "dev": dev_m, "gold": gold_m, "gold_by_category": gold_cat,
                   "challenge": {k: v for k, v in ch.items() if k != "rows"}, "train_seconds": train_seconds,
                   "params_m": params_m}, pred_dir / f"metrics_{name}.json")
        logger.log(phase="4", experiment=name, model=m["name"], config=args.config, seed=seed, train_rows=len(train),
                   dev_macro_f1=dev_m["macro_f1"], dev_accuracy=dev_m["accuracy"], gold_macro_f1=gold_m["macro_f1"],
                   gold_accuracy=gold_m["accuracy"], challenge_fail_rate=ch["overall_fail_rate"],
                   f1_hi=gold_m["per_language"].get("hi", {}).get("macro_f1"), f1_bn=gold_m["per_language"].get("bn", {}).get("macro_f1"),
                   train_seconds=train_seconds, params_m=params_m,
                   formulation=m["formulation"], claim_lang=m.get("claim_lang"), loss=m["loss"], epochs_run=len(model.history),
                   exclude_source_types=excl)
        results.append({"seed": seed, "dev": dev_m["macro_f1"], "gold": gold_m["macro_f1"], "challenge": ch["overall_fail_rate"]})
        if cfg["out"].get("save_model", True):
            model_dir = abs_path(cfg["out"]["model_dir"])
            if len(cfg.get("seeds", [seed])) > 1:
                model_dir = model_dir.parent / f"{model_dir.name}_s{seed}"
            model.save(model_dir)
            print(f"saved -> {model_dir}")
        del model
        import torch

        torch.cuda.empty_cache()
    if len(results) > 1:
        r = pd.DataFrame(results)
        print("\nseeds summary: dev {:.4f}±{:.4f} gold {:.4f}±{:.4f} challenge-fail {:.3f}±{:.3f}".format(
            r.dev.mean(), r.dev.std(), r.gold.mean(), r.gold.std(), r.challenge.mean(), r.challenge.std()))


if __name__ == "__main__":
    main()
