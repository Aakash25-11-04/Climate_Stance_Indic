#!/usr/bin/env python
"""If official challenge data appears in data/official/, make it the main train/test data.

    python scripts/prepare_official.py [--train data/official/train.csv --test data/official/test.csv]

Behaviour
* data/official/train.csv (columns id,text,label[,language]) -> becomes the primary training set:
  data/processed/train.csv = official train (source_type = "official") + our built data as extra rows,
  dev = a stratified 10 % split of the OFFICIAL train rows (so model selection follows the official distribution).
  The previous built files are kept as data/processed/train_built.csv / dev_built.csv.
* data/official/test.csv (id,text[,label]) -> data/official/test_prepared.csv with a language guess; if labels exist
  it is also copied to data/gold/test_official.csv for evaluation.
* The hand-written gold set and the challenge suite stay as robustness test sets.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stance.data.build import group_split  # noqa: E402
from stance.data.dedup import cross_split_leakage, deduplicate  # noqa: E402
from stance.preprocessing.normalize import detect_script  # noqa: E402
from stance.utils import LABELS, abs_path, read_data  # noqa: E402

LABEL_ALIASES = {"favor": "Favour", "favour": "Favour", "for": "Favour", "support": "Favour", "pro": "Favour",
                 "against": "Against", "deny": "Against", "oppose": "Against", "con": "Against",
                 "none": "None", "neutral": "None", "neither": "None", "unrelated": "None", "other": "None"}


def norm_label(x: str) -> str:
    s = str(x).strip()
    return LABEL_ALIASES.get(s.lower(), s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/official/train.csv")
    ap.add_argument("--test", default="data/official/test.csv")
    ap.add_argument("--dev_share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from predict import guess_language

    tr_path = abs_path(args.train)
    if tr_path.exists():
        off = read_data(tr_path)
        off["label"] = off["label"].map(norm_label)
        off = off[off["label"].isin(LABELS)].copy()
        if "language" not in off.columns or (off["language"] == "").all():
            off["language"] = off["text"].map(guess_language)
        if "id" not in off.columns:
            off["id"] = [f"official_{i:06d}" for i in range(len(off))]
        off["source"] = "official"; off["source_type"] = "official"; off["style"] = ""
        off["group_id"] = off["id"]; off["script"] = off["text"].map(detect_script)
        off, rep = deduplicate(off)
        print("official train after dedup:", len(off), rep)
        off_train, off_dev = group_split(off, args.dev_share, args.seed)
        built_train = abs_path("data/processed/train.csv"); built_dev = abs_path("data/processed/dev.csv")
        if built_train.exists() and not abs_path("data/processed/train_built.csv").exists():
            shutil.copy(built_train, abs_path("data/processed/train_built.csv"))
            shutil.copy(built_dev, abs_path("data/processed/dev_built.csv"))
        built = read_data("data/processed/train_built.csv") if abs_path("data/processed/train_built.csv").exists() else pd.DataFrame()
        built_dev_df = read_data("data/processed/dev_built.csv") if abs_path("data/processed/dev_built.csv").exists() else pd.DataFrame()
        extra = pd.concat([built, built_dev_df], ignore_index=True)
        if len(extra):
            leak = cross_split_leakage(extra, off_dev)
            print("built-data rows near-duplicating official dev:", leak)
        train = pd.concat([off_train, extra], ignore_index=True)
        train.to_csv(built_train, index=False, encoding="utf-8")
        off_dev.to_csv(built_dev, index=False, encoding="utf-8")
        print(f"train.csv = {len(off_train)} official + {len(extra)} built = {len(train)}; dev.csv = {len(off_dev)} official rows")
    else:
        print(f"no official train at {tr_path}; nothing changed")
    te_path = abs_path(args.test)
    if te_path.exists():
        te = read_data(te_path)
        if "language" not in te.columns or (te["language"] == "").all():
            te["language"] = te["text"].map(guess_language)
        te.to_csv(abs_path("data/official/test_prepared.csv"), index=False, encoding="utf-8")
        if "label" in te.columns and te["label"].map(norm_label).isin(LABELS).all():
            te["label"] = te["label"].map(norm_label)
            te.to_csv(abs_path("data/gold/test_official.csv"), index=False, encoding="utf-8")
            print("official test has labels -> data/gold/test_official.csv")
        print(f"official test prepared: {len(te)} rows -> data/official/test_prepared.csv (run scripts/predict.py on it)")


if __name__ == "__main__":
    main()
