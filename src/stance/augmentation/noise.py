"""Augmentation operators for the Phase 6 ablations.

  char_noise      : keyboard-neighbour typos / swaps on 1-3 characters
  drop_matras     : delete 1-2 vowel signs (matras) - a very common Indic typing error
  repeat_chars    : "bahut" -> "bahuuut", "!!!" - social-media emphasis
  romanize        : native -> Romanized copy (Hinglish / Banglish)        [uses data.transliterate]
  codemix         : synthetic code-mixing                                  [uses data.codemix]
  backtranslate   : hi/bn -> en -> hi/bn paraphrase via the cached translator
  negation_flip   : template-based contrast pairs (label flips), written by hand and verified

augment_frame(df, op, rng, share) applies one operator to a random `share` of rows and
returns the new rows with source_type = f"aug_{op}".
"""
from __future__ import annotations

import random
import re

import pandas as pd

MATRAS = "ािीुूेैोौািীুূেৈোৌ"
_KEYMAP = {"a": "s", "e": "r", "i": "o", "o": "p", "n": "m", "h": "j", "t": "y", "r": "t", "s": "d", "l": "k", "u": "i", "k": "l"}


def char_noise(text: str, rng: random.Random, n: int = 2) -> str:
    chars = list(text)
    idx = [i for i, c in enumerate(chars) if c.isalpha()]
    if not idx:
        return text
    for i in rng.sample(idx, k=min(n, len(idx))):
        op = rng.random()
        c = chars[i]
        if op < 0.4 and c.lower() in _KEYMAP:
            chars[i] = _KEYMAP[c.lower()]
        elif op < 0.7 and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op < 0.85:
            chars[i] = ""
        else:
            chars[i] = c + c
    return "".join(chars)


def drop_matras(text: str, rng: random.Random, n: int = 2) -> str:
    idx = [i for i, c in enumerate(text) if c in MATRAS]
    if not idx:
        return char_noise(text, rng, 1)
    drop = set(rng.sample(idx, k=min(n, len(idx))))
    return "".join(c for i, c in enumerate(text) if i not in drop)


def repeat_chars(text: str, rng: random.Random) -> str:
    m = list(re.finditer(r"[aeiouाीूাীু]", text))
    if m:
        k = rng.choice(m)
        text = text[: k.end()] + k.group(0) * rng.randint(2, 4) + text[k.end():]
    return text.rstrip("।.!") + rng.choice(["!!!", "...", "!!", "?!"])


# ----------------------------------------------------------------------------- negation flips
# (pattern, replacement, from_label, to_label). Applied only when the pattern matches a
# training text of from_label; the result gets to_label. Hand-written and verified: each
# edit reverses the polarity of the matched claim.
# Whole-word boundaries that work for Indic scripts (\b fails on combining vowel signs).
_L = r"(?<![ऀ-ॿঀ-৿\w])"
_R = r"(?![ऀ-ॿঀ-৿\w])"
NEG_FLIPS = {
    "hi": [
        (_L + r"गंभीर समस्या है" + _R, "कोई गंभीर समस्या नहीं है", "Favour", "Against"),
        (_L + r"असली है" + _R, "असली नहीं है", "Favour", "Against"),
        (_L + r"(बड़ा|गंभीर) खतरा है" + _R, r"कोई \1 खतरा नहीं है", "Favour", "Against"),
        (_L + r"झूठ है" + _R, "झूठ नहीं है", "Against", "Favour"),
        (_L + r"साजिश है" + _R, "साजिश नहीं है", "Against", "Favour"),
        (_L + r"कोई खतरा नहीं है" + _R, "बहुत बड़ा खतरा है", "Against", "Favour"),
        (_L + r"कुछ नहीं होने वाला" + _R, "बहुत कुछ होने वाला है", "Against", "Favour"),
    ],
    "bn": [
        (_L + r"গুরুতর সমস্যা" + _R + r"(?! না| নয়| নেই)", "কোনো গুরুতর সমস্যা না", "Favour", "Against"),
        (_L + r"আসল" + _R + r"(?! না| নয়)", "আসল না", "Favour", "Against"),
        (_L + r"(বড়|গুরুতর) বিপদ" + _R + r"(?! না| নয়| নেই)", r"কোনো \1 বিপদ না", "Favour", "Against"),
        (_L + r"মিথ্যা" + _R + r"(?! না| নয়)", "মিথ্যা না", "Against", "Favour"),
        (_L + r"ষড়যন্ত্র" + _R + r"(?! না| নয়)", "ষড়যন্ত্র না", "Against", "Favour"),
        (_L + r"কোনো বিপদ (না|নেই)" + _R, "অনেক বড় বিপদ", "Against", "Favour"),
        (_L + r"কিছুই হবে না" + _R, "অনেক কিছু হবে", "Against", "Favour"),
    ],
}
CLIMATE_WORDS = ("जलवायु", "ग्लोबल वार्मिंग", "क्लाइमेट", "तापमान", "धरती", "জলবায়ু", "গ্লোবাল ওয়ার্মিং", "ক্লাইমেট", "উষ্ণায়ন", "তাপমাত্রা", "পৃথিবী")
NEG_FLIP_MAX_WORDS = 15


def negation_flip(text: str, lang: str, label: str) -> tuple[str, str] | None:
    """Flip only short texts that mention climate, so the flipped clause IS the stance-bearing claim.
    Long translated texts are excluded: their flipped clause is often not the author's stance."""
    if len(text.split()) > NEG_FLIP_MAX_WORDS or not any(w in text for w in CLIMATE_WORDS):
        return None
    for pat, rep, frm, to in NEG_FLIPS.get(lang, []):
        if label == frm and re.search(pat, text):
            return re.sub(pat, rep, text, count=1), to
    return None


def load_approved_flips(path: str = "data/augmentation/negation_flips_reviewed.csv") -> pd.DataFrame | None:
    """Hand-reviewed flips (column keep=1). If the file exists, only approved flips are used."""
    from stance.utils import abs_path, read_data

    p = abs_path(path)
    if not p.exists():
        return None
    d = read_data(p)
    return d[d["keep"].astype(str) == "1"]


# ----------------------------------------------------------------------------- frame-level API
def augment_frame(df: pd.DataFrame, op: str, rng: random.Random, share: float = 0.3,
                  translator=None) -> pd.DataFrame:
    base = df[df["script"].isin(["devanagari", "bengali", "mixed"])] if op in ("drop_matras", "romanize", "codemix", "backtranslate") else df
    if op == "negation_flip":
        rows = []
        approved = load_approved_flips()
        ok = set(approved["flipped"]) if approved is not None else None
        for _, r in df.iterrows():
            res = negation_flip(r["text"], r["language"], r["label"])
            if res and (ok is None or res[0] in ok):
                rows.append({**r.to_dict(), "text": res[0], "label": res[1], "source_type": "aug_negation_flip"})
        return pd.DataFrame(rows)
    sub = base.sample(n=int(len(base) * share), random_state=rng.randint(0, 10**6)).copy()
    if op == "char_noise":
        sub["text"] = [char_noise(t, rng) for t in sub["text"]]
    elif op == "drop_matras":
        sub["text"] = [drop_matras(t, rng) for t in sub["text"]]
    elif op == "repeat_chars":
        sub["text"] = [repeat_chars(t, rng) for t in sub["text"]]
    elif op == "romanize":
        from stance.data.transliterate import native_to_roman

        sub["text"] = [native_to_roman(t, l, rng, "casual") for t, l in zip(sub["text"], sub["language"])]
    elif op == "codemix":
        from stance.data.codemix import codemix

        sub["text"] = [codemix(t, l, rng) for t, l in zip(sub["text"], sub["language"])]
    elif op == "backtranslate":
        assert translator is not None
        out = []
        for lang in ("hi", "bn"):
            m = sub["language"] == lang
            if m.any():
                en = translator.translate(sub.loc[m, "text"].tolist(), lang, "en")
                back = translator.translate(en, "en", lang)
                out.append(pd.Series(back, index=sub.index[m]))
        sub["text"] = pd.concat(out).reindex(sub.index)
    else:
        raise ValueError(op)
    sub["source_type"] = f"aug_{op}"
    sub = sub[sub["text"].str.strip().str.len() > 0]
    return sub
