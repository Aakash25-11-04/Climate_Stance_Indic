"""Transliteration between native Indic scripts and social-media Romanized spelling.

native_to_roman(text, lang, rng, style)
    Devanagari/Bengali -> casual Latin spelling ("Hinglish"/"Banglish"). Uses the ITRANS
    scheme of `indic_transliteration` and then applies the spelling habits that real
    users have: schwa deletion (desha -> desh), long-vowel simplification (A -> aa/a),
    Bengali inherent vowel o (kare -> kore), and optional random spelling variation
    (v/w, ph/f, ee/i, oo/u, z/j, dropped final h) when style == "casual".

romanized_to_native(text, lang)
    Rough inverse used only for the ablation "add a native-script copy of Romanized
    text" (Phase 3e / Phase 6). It is deliberately simple; it does not have to be
    perfect, only consistent, so that native and Romanized spellings share features.
"""
from __future__ import annotations

import random
import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _xlit

from stance.preprocessing.normalize import detect_script

_SCHEME = {"hi": sanscript.DEVANAGARI, "bn": sanscript.BENGALI}
_VOWELS = "aeiouAEIOU"
_WORD = re.compile(r"[ऀ-ॣ०-ॿঀ-৿]+")
_BN_FIX = {"য়": "য", "ড়": "র", "ঢ়": "র", "়": ""}


def _itrans_to_casual(word: str, lang: str, rng: random.Random | None, style: str) -> str:
    w = word
    # ITRANS multi-char symbols
    w = re.sub(r"M(?=[bpm])", "m", w)
    w = w.replace("~n", "n").replace(".n", "n").replace("M", "n").replace("H", "")
    w = w.replace("chh", "chh").replace("Ch", "chh").replace("ch", "ch")
    w = w.replace("Sh", "sh").replace("sh", "sh").replace("N", "n").replace("T", "t")
    w = w.replace("D", "d").replace("R", "r").replace("L", "l").replace("j~n", "gy")
    w = w.replace("RRi", "ri").replace("R^i", "ri").replace("^", "").replace(".", "")
    w = w.replace("Y", "y").replace("G", "g").replace("J", "j").replace("K", "k")
    w = w.replace("z", "z").replace("f", "f")
    if lang == "bn":
        # Bengali: inherent vowel sounds like "o"; long A is written "a"; v -> b
        w = re.sub(r"(?<=[^aeiouAEIOU])a(?=[^aeiouAEIOU]|$)", "o", w)
        w = w.replace("A", "a").replace("v", "b").replace("w", "b")
        w = w.replace("I", "i").replace("U", "u").replace("ii", "i").replace("uu", "u")
        w = w.replace("ai", "oi").replace("au", "ou")
        w = re.sub(r"(?<=[aeiou][^aeiou])o(?=[^aeiou][aeiou])", "", w)
        # final inherent vowel is usually silent
        w = re.sub(r"o$", "", w) if len(w) > 3 and not w.endswith("oo") else w
    else:
        # Hindi: schwa deletion (V C a C V -> V C C V), final schwa, long vowels
        w = re.sub(r"a$", "", w) if len(w) > 2 and w[-2] not in _VOWELS else w
        w = re.sub(r"(?<=[aeiouAEIOU][^aeiouAEIOU])a(?=[^aeiouAEIOU][aeiouAEIOU])", "", w)
        w = re.sub(r"eM$|en$", "ein", w)  # meM -> mein, nahIM -> nahin handled above
        w = w.replace("A", "aa").replace("I", "i").replace("U", "u")
        w = w.replace("aaa", "aa").replace("ii", "i")
    w = w.lower()
    if style == "casual" and rng is not None:
        subs = [("aa", "a", 0.5), ("ee", "i", 0.5), ("oo", "u", 0.5), ("v", "w", 0.3),
                ("ph", "f", 0.6), ("z", "j", 0.4), ("ksh", "x", 0.3), ("sh", "s", 0.15),
                ("ai", "e", 0.2), ("au", "o", 0.2)]
        for a, b, p in subs:
            if a in w and rng.random() < p:
                w = w.replace(a, b)
        if w.endswith("h") and len(w) > 3 and rng.random() < 0.15:
            w = w[:-1]
    return w


def native_to_roman(text: str, lang: str, rng: random.Random | None = None,
                    style: str = "standard") -> str:
    """Romanize the native-script words of `text`, leaving everything else untouched."""
    if lang not in _SCHEME:
        return text
    scheme = _SCHEME[lang]

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        if lang == "hi":
            raw = (raw.replace("ऑ", "ओ").replace("ॉ", "ो")   # candra O -> O
                      .replace("ऍ", "ए").replace("ॅ", "े"))  # candra E -> E
        if lang == "bn":
            for a, b in _BN_FIX.items():
                raw = raw.replace(a, b)
        it = _xlit(raw, scheme, sanscript.ITRANS)
        return _itrans_to_casual(it, lang, rng, style)

    out = _WORD.sub(repl, text)
    # Devanagari/Bengali punctuation
    return out.replace("।", ".").replace("॥", ".")


# --------------------------------------------------------------------------- reverse
_ROMAN_WORD = re.compile(r"[A-Za-z]+")
_REV_SUBS = [("aa", "A"), ("ee", "I"), ("oo", "U"), ("ii", "I"), ("uu", "U"), ("ph", "f"),
             ("chh", "Ch"), ("sh", "sh"), ("w", "v"), ("x", "ksh"), ("ck", "k"), ("q", "k")]


def romanized_to_native(text: str, lang: str = "hi") -> str:
    """Rough Latin -> Devanagari/Bengali conversion of every Latin word (ablation aid)."""
    if lang not in _SCHEME:
        lang = "hi"
    scheme = _SCHEME[lang]

    def repl(m: re.Match) -> str:
        w = m.group(0).lower()
        for a, b in _REV_SUBS:
            w = w.replace(a, b)
        if lang == "bn":
            # Banglish writes inherent vowel as "o": kore -> kare
            w = re.sub(r"(?<=[^aeiouAEIOU])o(?=[^aeiouAEIOU])", "a", w)
        w = re.sub(r"c(?!h)", "k", w)
        try:
            return _xlit(w, sanscript.ITRANS, scheme)
        except Exception:
            return m.group(0)

    return _ROMAN_WORD.sub(repl, text)


def is_romanized_indic(text: str) -> bool:
    """Heuristic: Latin-script text that is NOT plain English (used for EDA statistics)."""
    if detect_script(text) != "latin":
        return False
    hints = ("hai", "nahi", "nahin", "kya", "aur", "bhi", "mein", "main", "hum", "toh", "yeh",
             "ami", "amra", "ache", "nei", "kintu", "keno", "kore", "bhalo", "khub", "tumi",
             "bahut", "sab", "log", "kuch", "hota", "hoga", "chahiye", "lagta")
    toks = set(re.findall(r"[a-z]+", text.lower()))
    return len(toks & set(hints)) >= 2
