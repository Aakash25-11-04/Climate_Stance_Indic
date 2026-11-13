"""Exact and near-duplicate removal across ALL rows before any split is made.

* exact: canonical form from `strip_for_dedup` (lowercase, no URLs/mentions/emoji/punct).
* near:  MinHash LSH over character 4-gram shingles of the canonical form; rows whose
         estimated Jaccard similarity >= `threshold` to an earlier row are dropped.
         Rows are processed in a priority order (handcrafted/real first, then original,
         translated, ...) so that the most valuable copy survives.

The function also returns a `dup_report` dict used in the EDA leakage check.
"""
from __future__ import annotations

import pandas as pd
from datasketch import MinHash, MinHashLSH

from stance.preprocessing.normalize import strip_for_dedup

PRIORITY = {"handcrafted": 0, "real": 1, "original": 2, "translated": 3, "transliterated": 4,
            "codemixed": 5, "synthetic": 6}


def _shingles(s: str, n: int = 4) -> set[str]:
    s = s.replace(" ", "_")
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _minhash(sh: set[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for x in sh:
        m.update(x.encode("utf-8"))
    return m


def deduplicate(df: pd.DataFrame, threshold: float = 0.85, num_perm: int = 128,
                min_chars: int = 3) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["_canon"] = df["text"].map(strip_for_dedup)
    df["_prio"] = df["source_type"].map(PRIORITY).fillna(9)
    df = df.sort_values(["_prio"], kind="stable")
    n0 = len(df)
    # handcrafted rows are kept even when very short (emoji-only tests are deliberate)
    df = df[(df["_canon"].str.len() >= min_chars) | (df["source_type"] == "handcrafted")]
    n_short = n0 - len(df)
    # exact (within a language: the hi and bn versions of one source are different rows)
    before = len(df)
    df = df.drop_duplicates(subset=["language", "_canon"], keep="first")
    n_exact = before - len(df)
    # near (within the same language only: translations of one source are legitimately
    # similar across languages but are kept in the same split via group_id)
    keep_mask = []
    near_pairs = []
    for lang, sub in df.groupby("language", sort=False):
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for idx, canon in zip(sub.index, sub["_canon"]):
            if len(canon) < min_chars:  # emoji-only: nothing to hash
                keep_mask.append((idx, True))
                continue
            m = _minhash(_shingles(canon), num_perm)
            hits = lsh.query(m)
            if hits:
                keep_mask.append((idx, False))
                near_pairs.append((idx, hits[0]))
            else:
                lsh.insert(idx, m)
                keep_mask.append((idx, True))
    keep = pd.Series(dict(keep_mask))
    df = df[keep.reindex(df.index).fillna(True).astype(bool)]
    n_near = len(near_pairs)
    report = {
        "input_rows": int(n0),
        "dropped_too_short": int(n_short),
        "dropped_exact": int(n_exact),
        "dropped_near": int(n_near),
        "output_rows": int(len(df)),
        "near_threshold": threshold,
        "example_near_pairs": [(int(a), int(b)) for a, b in near_pairs[:10]],
    }
    return df.drop(columns=["_canon", "_prio"]).sort_index(), report


def cross_split_leakage(train: pd.DataFrame, other: pd.DataFrame, threshold: float = 0.85,
                        num_perm: int = 128) -> dict:
    """Count rows of `other` that have an exact or near duplicate in `train`."""
    tr_canon = train["text"].map(strip_for_dedup)
    ot_canon = other["text"].map(strip_for_dedup)
    exact = int(ot_canon.isin(set(tr_canon)).sum())
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, c in enumerate(tr_canon):
        lsh.insert(str(i), _minhash(_shingles(c), num_perm))
    near = 0
    for c in ot_canon:
        if lsh.query(_minhash(_shingles(c), num_perm)):
            near += 1
    return {"exact": exact, "near_or_exact": near, "n_other": int(len(other))}
