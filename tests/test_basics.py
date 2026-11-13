# -*- coding: utf-8 -*-
"""Fast unit tests (no model downloads): `.venv/Scripts/python -m pytest -q tests`."""
import random

import numpy as np
import pandas as pd

from stance.data.codemix import codemix
from stance.data.dedup import deduplicate
from stance.data.gold_bn import GOLD_BN
from stance.data.gold_hi import GOLD_HI
from stance.data.synth_templates import generate_dataset
from stance.data.transliterate import native_to_roman, romanized_to_native
from stance.evaluation.metrics import evaluate
from stance.models.calibration import apply_calibration, fit_class_bias, fit_temperature
from stance.models.sparse import LexiconModel, TfidfModel
from stance.preprocessing.normalize import Preprocessor, code_mix_ratio, detect_script
from stance.utils import LABELS, load_config


def test_detect_script():
    assert detect_script("पर्यावरण और जलवायु") == "devanagari"
    assert detect_script("আজ আবহাওয়া কেমন") == "bengali"
    assert detect_script("global warming is a major crisis") == "latin"
    assert detect_script("global warming एक गंभीर समस्या है") == "mixed"
    assert detect_script("👍👍 🔥🔥") == "other"

def test_preprocessor_keeps_negation_and_handles_noise():
    p = Preprocessor()
    out = p("Bahuuuut garmi hai yaaar!!!! @modi https://t.co/x #ClimateChange 🔥🔥🔥🔥 नहीं")
    assert "नहीं" in out and "@user" in out and "url" in out.lower()
    assert "bahuut" in out and "🔥🔥" in out and "🔥🔥🔥" not in out
    assert p("") == "" and p(None) == ""


def test_transliteration_roundtrip_shape():
    r = native_to_roman("जलवायु परिवर्तन एक गंभीर समस्या है", "hi")
    assert r.isascii() and "jalvaayu" in r and r.endswith("hai")
    b = native_to_roman("জলবায়ু পরিবর্তন একটি গুরুতর সমস্যা", "bn")
    assert b.isascii() and "poriborton" in b
    back = romanized_to_native("bahut garmi hai", "hi")
    assert detect_script(back) == "devanagari"


def test_codemix_changes_text_and_keeps_label_cues():
    rng = random.Random(0)
    t = "जलवायु परिवर्तन एक गंभीर समस्या है"
    outs = {codemix(t, "hi", rng) for _ in range(10)}
    assert any(o != t for o in outs)
    assert all(0 <= code_mix_ratio(o) <= 1 for o in outs)


def test_templates_balanced_and_unique():
    df = generate_dataset(30, seed=1)
    assert set(df.label) == set(LABELS) and set(df.language) == {"hi", "bn"}
    assert df.groupby(["language", "label"]).size().nunique() == 1
    assert df.text.str.len().min() > 3


def test_gold_sets_valid():
    for items in (GOLD_HI, GOLD_BN):
        assert len(items) >= 150
        assert all(lab in LABELS for _, lab, _, _ in items)
        texts = [t for t, _, _, _ in items]
        assert len(texts) == len(set(texts))


def test_dedup_removes_near_duplicates_but_keeps_handcrafted():
    df = pd.DataFrame({"text": ["जलवायु परिवर्तन एक गंभीर समस्या है।", "जलवायु परिवर्तन एक गंभीर समस्या है!!", "👍👍", "👍👍"],
                       "language": ["hi", "hi", "hi", "bn"], "source_type": ["original", "translated", "handcrafted", "handcrafted"]})
    out, rep = deduplicate(df)
    assert len(out) == 3 and rep["dropped_exact"] + rep["dropped_near"] == 1


def test_lexicon_and_tfidf_on_tiny_data():
    texts = ["जलवायु परिवर्तन असली है और गंभीर खतरा है", "ग्लोबल वार्मिंग झूठ है, साजिश है", "कल बारिश होगी",
             "climate change sach hai, khatra hai", "global warming hoax hai scam hai", "video acha tha"] * 5
    labels = ["Favour", "Against", "None"] * 10
    lx = LexiconModel()
    assert lx.predict(texts[:3]) == ["Favour", "Against", "None"]
    m = TfidfModel("lr", min_df=1).fit(texts, labels)
    p = m.predict_proba(texts[:3])
    assert p.shape == (3, 3) and np.allclose(p.sum(1), 1)
    assert m.predict(texts[:3]) == ["Favour", "Against", "None"]


def test_metrics_and_calibration():
    y = ["Favour", "Against", "None", "Favour"]
    m = evaluate(y, ["Favour", "Against", "None", "None"], ["hi", "hi", "bn", "bn"])
    assert 0 < m["macro_f1"] < 1 and m["per_language"]["hi"]["accuracy"] == 1.0
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])
    yi = np.array([0, 1, 2, 0])
    T = fit_temperature(probs, yi); assert 0.25 <= T <= 5
    bias = fit_class_bias(probs, yi, T); assert len(bias) == 3
    out = apply_calibration(probs, {"temperature": T, "class_bias": bias})
    assert np.allclose(out.sum(1), 1)


def test_config_overrides():
    cfg = load_config("configs/dataset.yaml", ["dataset.dev_share=0.2", "seed=7"])
    assert cfg["dataset"]["dev_share"] == 0.2 and cfg["seed"] == 7 and cfg["claim"]["en"]
