#!/usr/bin/env python
"""Phase 5: local LLM zero-shot / few-shot (kNN) stance classification + rationales.

    python scripts/run_llm.py --config configs/llm.yaml
    python scripts/run_llm.py --config configs/llm.yaml --set llm.max_eval_rows=100   # quick check

Logs Macro F1 on dev (subsample) and the full gold set + challenge suite, latency and GPU memory to
results/experiments.csv and results/llm.csv; writes rationales for 40 gold rows to results/llm_rationales.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.evaluation import evaluate, summarize  # noqa: E402
from stance.evaluation.challenge import evaluate_challenge, load_challenge  # noqa: E402
from stance.models.llm import LocalLLMClassifier  # noqa: E402
from stance.utils import ExperimentLogger, abs_path, load_config, read_data, save_json, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    logger = ExperimentLogger(cfg["paths"]["experiments_csv"])
    L = cfg["llm"]
    train = read_data(cfg["data"]["train"]); dev = read_data(cfg["data"]["dev"]); gold = read_data(cfg["data"]["gold"])
    challenge = load_challenge(cfg["data"]["challenge_dir"])
    n = L.get("max_eval_rows")
    dev_s = dev.sample(n=min(len(dev), int(L.get("dev_rows", 300))), random_state=cfg["seed"])
    if n:
        gold = gold.sample(n=min(len(gold), int(n)), random_state=cfg["seed"])
        dev_s = dev_s.head(int(n))
        challenge = {k: v.head(int(n) // 4) for k, v in challenge.items()}
    pool = train[train["source_type"].isin(["original", "translated"])]
    rows = []
    for k in L["k_shots"]:
        m = LocalLLMClassifier(L["model_name"], k_shots=k, load_in_4bit=L.get("load_in_4bit", False))
        m.fit(pool["text"].tolist(), pool["label"].tolist())
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        dev_m = evaluate(dev_s["label"], m.predict(dev_s["text"].tolist()), dev_s["language"])
        gold_pred = m.predict(gold["text"].tolist())
        gold_m = evaluate(gold["label"], gold_pred, gold["language"])
        lat = (time.time() - t0) * 1000 / (len(dev_s) + len(gold))
        ch = evaluate_challenge(m, challenge) if challenge else {"overall_fail_rate": float("nan"), "per_category": {}}
        if challenge:
            ch["rows"].to_csv(abs_path(f"results/predictions/challenge_{m.name}.csv"), index=False, encoding="utf-8")
        pd.DataFrame({"id": gold["id"], "text": gold["text"], "label": gold["label"], "pred": gold_pred}).to_csv(
            abs_path(f"results/predictions/gold_{m.name}.csv"), index=False, encoding="utf-8")
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
        row = {"model": m.name, "k_shots": k, "dev_macro_f1": dev_m["macro_f1"], "gold_macro_f1": gold_m["macro_f1"],
               "gold_f1_hi": gold_m["per_language"].get("hi", {}).get("macro_f1"), "gold_f1_bn": gold_m["per_language"].get("bn", {}).get("macro_f1"),
               "challenge_fail_rate": ch["overall_fail_rate"], "ms_per_example_gpu": lat, "peak_gpu_gb": mem}
        rows.append(row)
        print(f"\n=== {m.name} ===\n{summarize(gold_m)}\nchallenge fail {ch['overall_fail_rate']:.3f} | {lat:.0f} ms/ex | {mem:.2f} GB")
        save_json({"dev": dev_m, "gold": gold_m, "challenge": {kk: v for kk, v in ch.items() if kk != "rows"}, "latency_ms": lat, "peak_gpu_gb": mem},
                  f"results/predictions/metrics_{m.name}.json")
        logger.log(phase="5", experiment=m.name, model=L["model_name"], config=args.config, seed=cfg["seed"], train_rows=len(pool) if k else 0,
                   dev_macro_f1=dev_m["macro_f1"], gold_macro_f1=gold_m["macro_f1"], gold_accuracy=gold_m["accuracy"],
                   challenge_fail_rate=ch["overall_fail_rate"], f1_hi=row["gold_f1_hi"], f1_bn=row["gold_f1_bn"],
                   cpu_ms_per_example=float("nan"), gpu_ms_per_example=lat, peak_gpu_gb=mem, k_shots=k)
        if k == L["k_shots"][-1] and L.get("rationales", 0):
            rs = []
            for _, r in gold.sample(n=min(int(L["rationales"]), len(gold)), random_state=cfg["seed"]).iterrows():
                pred = m.predict([r["text"]])[0]
                rs.append({"id": r["id"], "text": r["text"], "label": r["label"], "llm_pred": pred, "rationale": m.rationale(r["text"], pred)})
            pd.DataFrame(rs).to_csv(abs_path("results/llm_rationales.csv"), index=False, encoding="utf-8")
            print("rationales -> results/llm_rationales.csv")
        del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(abs_path("results/llm.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
