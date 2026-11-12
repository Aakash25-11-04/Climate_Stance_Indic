"""End-to-end dataset builder (Phase 1).

Pipeline
--------
1. load sources (English stance datasets + Bengali Dhoroni)            -> source_type original
2. clean English text (drop URLs/mentions), translate en->hi, en->bn,
   back-translate, filter by LaBSE similarity                          -> source_type translated
   (optionally Dhoroni bn->hi as well)
3. deduplicate exact + near duplicates across everything, INCLUDING the gold set
   (gold rows have priority, so any train row that duplicates a gold row is dropped)
4. group-wise stratified split into train / dev  (a group = one source sentence and all
   of its translations, so no sentence leaks across the split)
5. realism layers on TRAIN groups only: Romanized copies, code-mixed copies, template
   synthetic data capped at `synthetic.max_share`
6. write train.csv, dev.csv, train_en.csv, test_handcrafted.csv, build_stats.json
"""
from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np
import pandas as pd

from stance.data.codemix import codemix, romanize_full
from stance.data.dedup import cross_split_leakage, deduplicate
from stance.data.gold_bn import GOLD_BN
from stance.data.gold_hi import GOLD_HI
from stance.data.loaders import load_all_sources
from stance.data.synth_templates import generate_dataset
from stance.preprocessing.normalize import detect_script
from stance.utils import abs_path, save_json

COLUMNS = ["id", "text", "language", "script", "source_type", "label", "source", "style",
           "group_id", "bt_similarity", "text_en", "orig_label"]

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION = re.compile(r"(?<![\w@])@\w{1,30}")
_AMP = re.compile(r"&amp;|&lt;|&gt;")


def clean_english(text: str, cfg: dict) -> str:
    t = str(text).replace("\r", " ").replace("\n", " ")
    t = _AMP.sub(lambda m: {"&amp;": "&", "&lt;": "<", "&gt;": ">"}[m.group(0)], t)
    if cfg.get("drop_urls", True):
        t = _URL.sub(" ", t)
    if cfg.get("drop_mentions", True):
        t = _MENTION.sub(" ", t)
    if cfg.get("strip_hashtag_symbol", True):
        t = re.sub(r"#(\w+)", lambda m: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(1)), t)
    t = re.sub(r"\s+", " ", t).strip(" \t-:|")
    return t.strip()


def gold_frame() -> pd.DataFrame:
    rows = []
    for lang, items in (("hi", GOLD_HI), ("bn", GOLD_BN)):
        for i, (text, label, cat, just) in enumerate(items):
            rows.append({"id": f"gold_{lang}_{i:04d}", "text": text, "language": lang,
                         "script": detect_script(text), "source_type": "handcrafted",
                         "label": label, "source": "handcrafted", "style": cat,
                         "group_id": f"gold_{lang}_{i:04d}", "category": cat, "justification": just})
    return pd.DataFrame(rows)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c == "bt_similarity" else ""
    df["script"] = df["text"].map(detect_script)
    return df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]


def group_split(df: pd.DataFrame, dev_share: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign whole groups to dev, stratified by (label, source) at the group level."""
    rng = random.Random(seed)
    groups = df.groupby("group_id").agg(label=("label", "first"), source=("source", "first"))
    dev_groups = set()
    for (_, _), g in groups.groupby(["label", "source"]):
        ids = list(g.index)
        rng.shuffle(ids)
        n_dev = int(round(len(ids) * dev_share))
        dev_groups.update(ids[:n_dev])
    is_dev = df["group_id"].isin(dev_groups)
    return df[~is_dev].reset_index(drop=True), df[is_dev].reset_index(drop=True)


def build(cfg: dict, log=print) -> dict:
    seed = int(cfg.get("seed", 42))
    dcfg = cfg["dataset"]
    rng = random.Random(seed)
    stats: dict = {"seed": seed}

    # ------------------------------------------------------------------ 1. sources
    src = load_all_sources(dcfg["sources"], log=log)
    src["group_id"] = [f"{s}_{i}" for i, s in enumerate(src["source"])]
    pre = dcfg.get("pre_translation", {})
    en = src[src["language"] == "en"].copy()
    en["text"] = en["text"].map(lambda t: clean_english(t, pre))
    en = en[(en["text"].str.len() >= pre.get("min_chars", 12)) & (en["text"].str.len() <= pre.get("max_chars", 600))]
    if dcfg["translation"].get("max_source_rows"):
        en = en.sample(n=min(len(en), int(dcfg["translation"]["max_source_rows"])), random_state=seed)
    indic_orig = src[src["language"] != "en"].copy()
    stats["sources"] = {s: int(n) for s, n in src["source"].value_counts().items()}
    stats["english_after_cleaning"] = int(len(en))
    stats["source_label_counts"] = {f"{s}/{l}": int(n) for (s, l), n in src.groupby(["source", "label"]).size().items()}

    # ------------------------------------------------------------------ 2. translation
    from stance.data.indictrans import IndicTranslator, translate_and_filter

    tcfg = dcfg["translation"]
    translator = IndicTranslator(batch_size=int(tcfg.get("batch_size", 16)),
                                 max_length=int(tcfg.get("max_length", 200)),
                                 num_beams=int(tcfg.get("num_beams", 4)),
                                 backend=tcfg.get("backend", "auto"))
    stats["translation_backend"] = translator.backend
    stats["translation_models"] = sorted({translator.model_name("en"), translator.model_name("hi")})
    translated, tstats, all_translations = translate_and_filter(
        en, translator, tgt_langs=("hi", "bn"), threshold=float(tcfg.get("bt_threshold", 0.75)), log=log)
    stats["translation"] = tstats
    all_translations.to_csv(abs_path(dcfg["out"]["translations_all"]), index=False, encoding="utf-8")

    frames = [translated, indic_orig]
    if tcfg.get("translate_dhoroni_to_hi", True) and len(indic_orig):
        bn_rows = indic_orig[indic_orig["language"] == "bn"].copy()
        log(f"[translate] bn->hi for {len(bn_rows)} Dhoroni rows")
        translator2 = IndicTranslator(batch_size=int(tcfg.get("batch_size", 16)),
                                      max_length=int(tcfg.get("max_length", 200)),
                                      num_beams=int(tcfg.get("num_beams", 4)),
                                      backend=tcfg.get("backend", "auto"))
        hi_txt = translator2.translate(bn_rows["text"].tolist(), "bn", "hi")
        back = translator2.translate(hi_txt, "hi", "bn")
        translator2.unload()
        from stance.data.indictrans import BackTranslationFilter

        scorer = BackTranslationFilter()
        sims = scorer.similarity(bn_rows["text"].tolist(), back)
        del scorer
        d = bn_rows.copy()
        d["text_en"] = d["text"]
        d["text"] = hi_txt
        d["language"] = "hi"
        d["source_type"] = "translated"
        d["bt_similarity"] = sims
        keep = d["bt_similarity"] >= float(tcfg.get("bt_threshold", 0.75))
        stats["dhoroni_bn_to_hi"] = {"n": int(len(d)), "kept": int(keep.sum())}
        frames.append(d[keep])
    pool = pd.concat(frames, ignore_index=True)
    pool["style"] = pool.get("style", "")
    pool = _finalize(pool)

    # ------------------------------------------------------------------ 3. dedup (with gold)
    gold = gold_frame()
    everything = pd.concat([gold, pool], ignore_index=True)
    deduped, dup_report = deduplicate(everything, threshold=float(dcfg["dedup"]["near_threshold"]))
    stats["dedup"] = dup_report
    pool = deduped[deduped["source_type"] != "handcrafted"].reset_index(drop=True)
    gold = deduped[deduped["source_type"] == "handcrafted"].reset_index(drop=True)
    stats["gold_rows_after_dedup"] = int(len(gold))

    # ------------------------------------------------------------------ 4. split
    train, dev = group_split(pool, float(dcfg.get("dev_share", 0.1)), seed)
    stats["split_before_augmentation"] = {"train": int(len(train)), "dev": int(len(dev))}

    # ------------------------------------------------------------------ 5. realism layers (train only)
    native = train[train["script"].isin(["devanagari", "bengali", "mixed"])]
    n_tl = int(len(native) * float(dcfg.get("transliterate_share", 0.25)))
    tl_rows = native.sample(n=n_tl, random_state=seed).copy()
    tl_rows["text"] = [romanize_full(t, l, rng) for t, l in zip(tl_rows["text"], tl_rows["language"])]
    tl_rows["source_type"] = "transliterated"
    tl_rows["id"] = ""
    n_cm = int(len(native) * float(dcfg.get("codemix_share", 0.20)))
    cm_rows = native.sample(n=n_cm, random_state=seed + 1).copy()
    cm_rows["text"] = [codemix(t, l, rng) for t, l in zip(cm_rows["text"], cm_rows["language"])]
    cm_rows["source_type"] = "codemixed"
    cm_rows["id"] = ""
    train = pd.concat([train, tl_rows, cm_rows], ignore_index=True)

    scfg = dcfg.get("synthetic", {})
    if scfg.get("enabled", True):
        synth = generate_dataset(int(scfg.get("n_per_label_lang", 400)), seed=seed)
        synth["group_id"] = [f"synthetic_{i}" for i in range(len(synth))]
        synth = _finalize(synth)
        max_share = float(scfg.get("max_share", 0.4))
        max_rows = int(max_share / (1 - max_share) * len(train))
        if len(synth) > max_rows:
            synth = synth.sample(n=max_rows, random_state=seed)
        synth, _ = deduplicate(pd.concat([gold, train, synth], ignore_index=True),
                               threshold=float(dcfg["dedup"]["near_threshold"]))
        synth = synth[synth["source_type"] == "synthetic"]
        stats["synthetic_rows"] = int(len(synth))
        train = pd.concat([train, synth], ignore_index=True)

    # final exact dedup inside train (augmented copies can coincide with originals)
    train, rep2 = deduplicate(train, threshold=float(dcfg["dedup"]["near_threshold"]))
    stats["dedup_after_augmentation"] = rep2
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train["id"] = [f"train_{i:06d}" for i in range(len(train))]
    dev = dev.reset_index(drop=True)
    dev["id"] = [f"dev_{i:05d}" for i in range(len(dev))]
    train, dev, gold = _finalize(train), _finalize(dev), _finalize(gold)

    # ------------------------------------------------------------------ 6. leakage + write
    stats["leakage_train_vs_dev"] = cross_split_leakage(train, dev)
    stats["leakage_train_vs_gold"] = cross_split_leakage(train, gold)
    stats["train_label_counts"] = {k: int(v) for k, v in train["label"].value_counts().items()}
    stats["train_language_counts"] = {k: int(v) for k, v in train["language"].value_counts().items()}
    stats["train_source_type_counts"] = {k: int(v) for k, v in train["source_type"].value_counts().items()}
    stats["train_script_counts"] = {k: int(v) for k, v in train["script"].value_counts().items()}
    stats["synthetic_share"] = float((train["source_type"] == "synthetic").mean())
    stats["dev_label_counts"] = {k: int(v) for k, v in dev["label"].value_counts().items()}
    stats["dev_language_counts"] = {k: int(v) for k, v in dev["language"].value_counts().items()}
    stats["gold_label_counts"] = {k: int(v) for k, v in gold["label"].value_counts().items()}
    stats["gold_category_counts"] = {k: int(v) for k, v in gold["category"].value_counts().items()}
    stats["rows"] = {"train": int(len(train)), "dev": int(len(dev)), "gold": int(len(gold))}

    out = dcfg["out"]
    for key, df in (("train", train), ("dev", dev), ("gold", gold)):
        p = abs_path(out[key])
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding="utf-8")
        log(f"[build] wrote {p} ({len(df)} rows)")
    en_out = _finalize(en.assign(id=[f"en_{i:05d}" for i in range(len(en))]))
    en_out.to_csv(abs_path(out["train_en"]), index=False, encoding="utf-8")
    save_json(stats, out["stats"])
    log(f"[build] stats -> {out['stats']}")
    return stats
