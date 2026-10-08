"""Analysis helpers for the corpus built by build_corpus.py."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

LABELS = ["favour", "against", "none"]

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
TOKEN = re.compile(r"[a-z][a-z']+")
BENGALI = re.compile(r"[\u0980-\u09FF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
NEGATION = re.compile(r"\b(?:not|no|never|nothing|nobody|isn't|aren't|wasn't|don't|doesn't|didn't|can't|won't|n't)\b")
WE = re.compile(r"\b(?:we|us|our|ours)\b")

# Very common function words carry no class signal and dominate counts.
STOP = set("""
a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those i you he she they them his her their my your me him
do does did have has had will would can could should may might so than then there here
just also very more most about into over out up rt user amp semst
""".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def normalise_for_dedup(text: str) -> str:
    text = URL.sub("", text.lower())
    text = MENTION.sub("", text)
    return re.sub(r"\W+", " ", text).strip()


def surface_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"]
    return pd.DataFrame({
        "words": text.str.split().str.len(),
        "has_url": text.str.contains(URL),
        "has_mention": text.str.contains(MENTION),
        "has_hashtag": text.str.contains(HASHTAG),
        "has_emoji": text.str.contains(EMOJI),
        "is_question": text.str.contains(r"\?"),
        "has_exclaim": text.str.contains("!"),
        "has_negation": text.str.lower().str.contains(NEGATION),
        "has_we": text.str.lower().str.contains(WE),
    }, index=df.index)


def overview(df: pd.DataFrame) -> pd.DataFrame:
    feats = surface_features(df)
    g = df.assign(**feats).groupby("source")
    table = pd.DataFrame({
        "rows": g.size(),
        "lang": g["lang"].first(),
        "domain": g["domain"].first(),
        "quality": g["quality"].first(),
        "median_words": g["words"].median(),
        "p95_words": g["words"].quantile(0.95),
        "pct_url": 100 * g["has_url"].mean(),
        "pct_hashtag": 100 * g["has_hashtag"].mean(),
        "pct_mention": 100 * g["has_mention"].mean(),
        "pct_emoji": 100 * g["has_emoji"].mean(),
    })
    dist = pd.crosstab(df["source"], df["label"], normalize="index")[LABELS] * 100
    return table.join(dist.add_prefix("pct_")).round(1)


def duplicates(df: pd.DataFrame) -> pd.DataFrame:
    key = df["text"].map(normalise_for_dedup)
    d = df.assign(key=key)
    within = d.groupby("source")["key"].apply(lambda s: s.duplicated().sum())
    conflicting = (
        d.groupby(["source", "key"])["label"].nunique().gt(1).groupby("source").sum()
    )
    first_source = d.drop_duplicates("key").set_index("key")["source"]
    cross = d[d["source"] != d["key"].map(first_source)].groupby("source").size()
    return pd.DataFrame({
        "exact_dups_within": within,
        "dup_groups_with_conflicting_labels": conflicting,
        "rows_also_in_earlier_source": cross,
    }).fillna(0).astype(int)


def reddit_kappa(a1: pd.Series, a2: pd.Series) -> float:
    return cohen_kappa_score(a1, a2)


def log_odds_dirichlet(df: pd.DataFrame, label: str, top: int = 15,
                       min_count: int = 5) -> pd.DataFrame:
    """Monroe et al. (2008) weighted log-odds with an informative Dirichlet prior.

    Compares words in `label` against all other classes; returns the most
    distinctive words with their z-scores.
    """
    in_class = Counter(t for txt in df.loc[df["label"] == label, "text"] for t in tokens(txt))
    rest = Counter(t for txt in df.loc[df["label"] != label, "text"] for t in tokens(txt))
    prior = in_class + rest
    vocab = [w for w, c in prior.items() if c >= min_count]
    a0 = sum(prior[w] for w in vocab)
    n1 = sum(in_class[w] for w in vocab)
    n2 = sum(rest[w] for w in vocab)
    rows = []
    for w in vocab:
        aw = prior[w]
        y1, y2 = in_class[w], rest[w]
        delta = (np.log((y1 + aw) / (n1 + a0 - y1 - aw))
                 - np.log((y2 + aw) / (n2 + a0 - y2 - aw)))
        var = 1 / (y1 + aw) + 1 / (y2 + aw)
        rows.append((w, delta / np.sqrt(var), y1))
    out = pd.DataFrame(rows, columns=["word", "z", "count_in_class"])
    return out.sort_values("z", ascending=False).head(top).reset_index(drop=True)


def cue_rates_by_label(df: pd.DataFrame) -> pd.DataFrame:
    feats = surface_features(df)[["has_negation", "is_question", "has_exclaim", "has_we", "has_url"]]
    return (100 * feats.groupby(df["label"]).mean()).loc[LABELS].round(1)


def vocab_overlap(df: pd.DataFrame, sources: list[str], k: int = 3000) -> pd.DataFrame:
    tops = {
        s: {w for w, _ in Counter(t for txt in df.loc[df["source"] == s, "text"]
                                   for t in tokens(txt)).most_common(k)}
        for s in sources
    }
    return pd.DataFrame(
        [[len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for b in sources] for a in sources],
        index=sources, columns=sources,
    ).round(2)


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           preprocessor=lambda t: HASHTAG.sub(lambda m: m.group()[1:],
                                                               MENTION.sub(" ", URL.sub(" ", t.lower()))))


def cross_source_transfer(df: pd.DataFrame, sources: list[str], seed: int = 13) -> pd.DataFrame:
    """Train TF-IDF + LR on 80% of one source, report macro F1 on 20% of each source.

    The diagonal is in-domain performance; off-diagonal cells show how much a
    model trained on one source degrades on another.
    """
    splits = {}
    for s in sources:
        part = df[df["source"] == s]
        tr, te = train_test_split(part, test_size=0.2, stratify=part["label"], random_state=seed)
        splits[s] = (tr, te)
    result = pd.DataFrame(index=sources, columns=sources, dtype=float)
    for s_train in sources:
        tr, _ = splits[s_train]
        vec = _vectorizer()
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(vec.fit_transform(tr["text"]), tr["label"])
        for s_test in sources:
            _, te = splits[s_test]
            pred = clf.predict(vec.transform(te["text"]))
            result.loc[s_train, s_test] = f1_score(te["label"], pred, labels=LABELS,
                                                   average="macro", zero_division=0)
    result.index.name = "train \\ test"
    return result.round(3)


def source_identifiability(df: pd.DataFrame, sources: list[str], seed: int = 13) -> float:
    """Accuracy of predicting which dataset a text came from. High = strong domain signal."""
    part = df[df["source"].isin(sources)]
    part = pd.concat(g.sample(min(len(g), 300), random_state=seed)
                     for _, g in part.groupby("source"))
    tr, te = train_test_split(part, test_size=0.3, stratify=part["source"], random_state=seed)
    vec = _vectorizer()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(vec.fit_transform(tr["text"]), tr["source"])
    return float((clf.predict(vec.transform(te["text"])) == te["source"]).mean())


def hashtag_only_f1(df: pd.DataFrame, source: str, seed: int = 13) -> tuple[float, float]:
    """Macro F1 using only hashtags vs only the remaining text. Detects shortcut features."""
    part = df[df["source"] == source]
    tr, te = train_test_split(part, test_size=0.2, stratify=part["label"], random_state=seed)
    scores = []
    for view in (lambda t: " ".join(HASHTAG.findall(t.lower())) or "__none__",
                 lambda t: HASHTAG.sub(" ", t.lower())):
        vec = TfidfVectorizer(token_pattern=r"[#\w]+", min_df=2)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(vec.fit_transform(tr["text"].map(view)), tr["label"])
        pred = clf.predict(vec.transform(te["text"].map(view)))
        scores.append(f1_score(te["label"], pred, labels=LABELS, average="macro", zero_division=0))
    return scores[0], scores[1]


def script_profile(text: pd.Series) -> pd.DataFrame:
    def shares(t: str) -> tuple[float, float, float]:
        b, d, l = len(BENGALI.findall(t)), len(DEVANAGARI.findall(t)), len(LATIN.findall(t))
        total = max(b + d + l, 1)
        return b / total, d / total, l / total
    return pd.DataFrame(text.map(shares).tolist(), columns=["bengali", "devanagari", "latin"],
                        index=text.index)
