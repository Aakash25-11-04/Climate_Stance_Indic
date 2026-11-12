#!/usr/bin/env python
"""Phase 7: build the CheckList-style challenge suite into data/challenge/.

    python scripts/build_challenge.py [--seed 7] [--min_per_category 30]

Also writes data/challenge/README.md with counts and a leakage check against train.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stance.data.challenge_suite import build_suite  # noqa: E402
from stance.data.dedup import cross_split_leakage  # noqa: E402
from stance.utils import abs_path, read_data, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min_per_category", type=int, default=30)
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--out_dir", default="data/challenge")
    ap.add_argument("--clean_train", action="store_true", default=True)
    args = ap.parse_args()
    mft, inv, dire = build_suite(seed=args.seed, min_per_category=args.min_per_category)
    out = abs_path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    mft.to_csv(out / "mft.csv", index=False, encoding="utf-8")
    inv.to_csv(out / "invariance.csv", index=False, encoding="utf-8")
    dire.to_csv(out / "directional.csv", index=False, encoding="utf-8")
    stats = {
        "mft_total": int(len(mft)),
        "mft_per_category_language": {f"{c}/{l}": int(n) for (c, l), n in mft.groupby(["category", "language"]).size().items()},
        "mft_labels": {k: int(v) for k, v in mft["label"].value_counts().items()},
        "mft_origin": {k: int(v) for k, v in mft["origin"].value_counts().items()},
        "invariance_total": int(len(inv)),
        "invariance_per_kind": {k: int(v) for k, v in inv["category"].value_counts().items()},
        "directional_total": int(len(dire)),
    }
    train_path = abs_path(args.train)
    if train_path.exists():
        train = read_data(train_path)
        if args.clean_train:
            # the challenge suite is test-only: drop any TRAIN row that exactly / nearly duplicates an MFT test
            from datasketch import MinHashLSH

            from stance.data.dedup import _minhash, _shingles
            from stance.preprocessing.normalize import strip_for_dedup

            lsh = MinHashLSH(threshold=0.85, num_perm=128)
            canon_m = mft["text"].map(strip_for_dedup)
            for i, c in enumerate(canon_m):
                if len(c) >= 3:
                    lsh.insert(str(i), _minhash(_shingles(c), 128))
            cset = set(canon_m)
            drop = [idx for idx, c in zip(train.index, train["text"].map(strip_for_dedup))
                    if c in cset or (len(c) >= 3 and lsh.query(_minhash(_shingles(c), 128)))]
            if drop:
                train = train.drop(index=drop)
                train.to_csv(train_path, index=False, encoding="utf-8")
            stats["train_rows_removed_for_leakage"] = len(drop)
            print(f"removed {len(drop)} train rows that near-duplicate challenge tests")
        stats["leakage_vs_train_mft"] = cross_split_leakage(train, mft)
    save_json(stats, out / "stats.json")
    lines = ["# Robustness challenge suite (TEST-ONLY)", "",
             "CheckList-style tests built by `scripts/build_challenge.py` from hand-written seeds in",
             "`src/stance/data/challenge_suite.py`. Never used for training, tuning or model selection", "",
             f"* MFT tests: {stats['mft_total']} ({stats['mft_origin']})",
             f"* Invariance tests: {stats['invariance_total']} ({stats['invariance_per_kind']})",
             f"* Directional (negation flip) tests: {stats['directional_total']}", "",
             "| category | hi | bn |", "|---|---|---|"]
    for c in sorted(mft["category"].unique()):
        lines.append(f"| {c} | {stats['mft_per_category_language'].get(c + '/hi', 0)} | {stats['mft_per_category_language'].get(c + '/bn', 0)} |")
    if "leakage_vs_train_mft" in stats:
        lines += ["", f"Leakage check vs train.csv (exact / near duplicates): {stats['leakage_vs_train_mft']}"]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
