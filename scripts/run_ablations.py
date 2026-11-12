#!/usr/bin/env python
"""Phase 6: augmentation ablations on the sparse baseline-to-beat (5-fold CV + gold + challenge)
and on the transformer (dev + gold + challenge, one seed).

    python scripts/run_ablations.py --config configs/ablations.yaml [--only no_synthetic plus_char_noise] [--no_transformer]

Each ablation produces a modified training frame written to data/cache/ablation_<name>.csv and a
row in results/ablations.csv with deltas against the `none` reference.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stance.augmentation import augment_frame  # noqa: E402
from stance.evaluation import evaluate  # noqa: E402
from stance.evaluation.challenge import evaluate_challenge, load_challenge  # noqa: E402
from stance.models.cv import cross_validate  # noqa: E402
from stance.utils import ExperimentLogger, abs_path, load_config, read_data, save_json, set_seed  # noqa: E402


def make_frame(train: pd.DataFrame, ab: dict, rng: random.Random, translator_factory) -> pd.DataFrame:
    df = train.copy()
    if ab.get("remove"):
        df = df[~df["source_type"].isin(ab["remove"])]
    if ab.get("add"):
        # negation flips come from every row (only 14 hand-approved flips exist, most from templates);
        # other augmentations perturb real (original / translated) rows only
        base = df if ab["add"] == "negation_flip" else df[df["source_type"].isin(["original", "translated"])]
        tr = translator_factory() if ab["add"] == "backtranslate" else None
        aug = augment_frame(base, ab["add"], rng, share=float(ab.get("share", 0.3)), translator=tr)
        if tr is not None:  # release NLLB's VRAM before the transformer subprocess starts (4 GB card)
            tr.unload()
            del tr
            import gc

            gc.collect()
        df = pd.concat([df, aug], ignore_index=True)
    if ab.get("add_synthetic"):
        from stance.data.dedup import deduplicate
        from stance.data.synth_templates import generate_dataset
        from stance.preprocessing.normalize import detect_script

        syn = generate_dataset(int(ab["add_synthetic"]), seed=1234)  # different seed from the build
        syn["group_id"] = [f"synthetic_extra_{i}" for i in range(len(syn))]
        syn["script"] = syn["text"].map(detect_script)
        both, _ = deduplicate(pd.concat([df, syn], ignore_index=True))
        df = both
    if ab.get("add_file"):
        df = pd.concat([df, read_data(ab["add_file"])], ignore_index=True)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablations.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no_transformer", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pre = load_config("configs/preprocessing.yaml")["preprocessing"]
    set_seed(cfg["seed"])
    logger = ExperimentLogger(cfg["paths"]["experiments_csv"])
    train = read_data(cfg["data"]["train"]); dev = read_data(cfg["data"]["dev"]); gold = read_data(cfg["data"]["gold"])
    challenge = load_challenge(cfg["data"]["challenge_dir"])
    # the sparse baseline to beat: best sparse model by gold macro F1 in results/baselines.csv
    sparse_name = cfg.get("sparse_model", "tfidf_lr")
    bl = abs_path("results/baselines.csv")
    params = {}
    if bl.exists():
        b = pd.read_csv(bl)
        b = b[(b.family == "sparse") & ~b.model.isin(["majority", "stratified_random", "lexicon"])]
        if len(b):
            best = b.sort_values(["gold_macro_f1", "cv_macro_f1_mean"]).iloc[-1]
            sparse_name = best["model"]; params = json.loads(best["params"]) if isinstance(best["params"], str) else {}
    from run_baselines import registry

    reg = registry({**cfg, "preprocessing": pre})
    factory = reg[sparse_name][0]
    print(f"sparse baseline to beat: {sparse_name} {params}")

    def translator_factory():
        from stance.data.indictrans import IndicTranslator

        return IndicTranslator(batch_size=16)

    rows = []
    for ab in cfg["ablations"]:
        if args.only and ab["name"] not in args.only:
            continue
        name = ab["name"]
        rng = random.Random(cfg["seed"])
        df = make_frame(train, ab, rng, translator_factory)
        path = abs_path(f"data/cache/ablation_{name}.csv"); df.to_csv(path, index=False, encoding="utf-8")
        print(f"\n=== ablation {name}: {len(df)} rows (train {len(train)}) ===")
        t0 = time.time()
        res = cross_validate(lambda **kw: factory(**params), df, n_splits=cfg["cv"]["n_splits"], seed=cfg["seed"])
        m = factory(**params).fit(df["text"].tolist(), df["label"].tolist(), df["language"].tolist())
        gold_m = evaluate(gold["label"], m.predict(gold["text"].tolist(), gold["language"].tolist()), gold["language"])
        ch = evaluate_challenge(m, challenge) if challenge else {"overall_fail_rate": float("nan"), "per_category": {}}
        row = {"ablation": name, "rows": len(df), "sparse_model": sparse_name, "sparse_cv_macro_f1": res["cv_macro_f1_mean"],
               "sparse_cv_std": res["cv_macro_f1_std"], "sparse_gold_macro_f1": gold_m["macro_f1"],
               "sparse_challenge_fail": ch["overall_fail_rate"], "sparse_seconds": time.time() - t0}
        logger.log(phase="6", experiment=f"ablation_{name}_{sparse_name}", model=sparse_name, config=args.config, seed=cfg["seed"], train_rows=len(df),
                   cv_macro_f1_mean=res["cv_macro_f1_mean"], cv_macro_f1_std=res["cv_macro_f1_std"], gold_macro_f1=gold_m["macro_f1"],
                   gold_accuracy=gold_m["accuracy"], challenge_fail_rate=ch["overall_fail_rate"], train_seconds=row["sparse_seconds"],
                   f1_hi=gold_m["per_language"].get("hi", {}).get("macro_f1"), f1_bn=gold_m["per_language"].get("bn", {}).get("macro_f1"))
        save_json({"ablation": ab, "cv": {k: v for k, v in res.items() if k != "oof"}, "gold": gold_m,
                   "challenge": {k: v for k, v in ch.items() if k != "rows"}}, f"results/predictions/metrics_ablation_{name}_{sparse_name}.json")
        if cfg["transformer"].get("run", True) and not args.no_transformer:
            tag = f"ablation_{name}"
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", cfg["transformer"]["base_config"], "--set",
                   f"data.train={path.as_posix()}", f"model.epochs={cfg['transformer'].get('epochs', 2)}", f"run_tag={tag}",
                   "out.save_model=false", f"seeds=[{cfg['seed']}]", *cfg["transformer"].get("extra_set", [])]
            print(" ".join(cmd))
            rc = subprocess.call(cmd, cwd=ROOT)
            mp = sorted(abs_path("results/predictions").glob(f"metrics_*_{tag}_s{cfg['seed']}.json"))
            if rc == 0 and mp:
                d = json.loads(mp[-1].read_text(encoding="utf-8"))
                row.update({"tf_dev_macro_f1": d["dev"]["macro_f1"], "tf_gold_macro_f1": d["gold"]["macro_f1"],
                            "tf_challenge_fail": d["challenge"].get("overall_fail_rate"), "tf_minutes": d["train_seconds"] / 60})
        rows.append(row)
        pd.DataFrame(rows).to_csv(abs_path(cfg["out"]["table"]), index=False)
    tab = pd.DataFrame(rows)
    if "none" in set(tab["ablation"]):
        ref = tab[tab.ablation == "none"].iloc[0]
        for c in [c for c in tab.columns if c.endswith("macro_f1") or c.endswith("fail")]:
            tab[f"d_{c}"] = tab[c] - ref[c]
        tab["keep_sparse"] = (tab["d_sparse_cv_macro_f1"] > 0) & (tab["d_sparse_challenge_fail"] < 0)
        if "tf_gold_macro_f1" in tab:
            tab["keep_tf"] = (tab["d_tf_dev_macro_f1"] > 0) & (tab["d_tf_challenge_fail"] < 0)
    tab.to_csv(abs_path(cfg["out"]["table"]), index=False)
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
