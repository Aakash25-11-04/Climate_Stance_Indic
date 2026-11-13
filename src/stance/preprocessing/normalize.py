"""Configurable text normalisation for Hindi / Bengali / Romanized social-media text.

Design notes
------------
* Negation words and stopwords are NEVER removed: stance flips on "nahi"/"না", and
  neural models need the full context. Removing them would destroy the signal
  that distinguishes Favour from Against ("climate change is real" vs "is not real").
* Every step is a flag so that ablations (emoji kept vs demojised, transliteration on/off)
  can be driven from YAML.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import emoji as emoji_lib
import regex

# Unicode blocks
_DEVANAGARI = regex.compile(r"\p{Devanagari}")
_BENGALI = regex.compile(r"\p{Bengali}")
_LATIN = regex.compile(r"\p{Latin}")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION = re.compile(r"(?<![\w@])@\w{1,30}")
_HASHTAG = re.compile(r"#(\w+)")
_NUMBER = re.compile(r"(?<![\wऀ-ॿঀ-৿])[+-]?\d+(?:[.,]\d+)*(?![\w])")
_MULTISPACE = re.compile(r"\s+")
_ZW = re.compile("[​‌‍⁠﻿]")
# 3+ repeats of the same grapheme -> 2 (works for any script, incl. matras and emoji)
_ELONG = regex.compile(r"(\X)\1{2,}")
# repeated punctuation !!!!! -> !!
_PUNCT_REP = re.compile("([!?.।,])\\1{2,}")

SCRIPTS = ("devanagari", "bengali", "latin", "mixed", "other")


def detect_script(text: str) -> str:
    """Return the dominant script of text: devanagari / bengali / latin / mixed / other.

    mixed = at least 20% of alphabetic characters in a second script (true code-mixing
    at the script level, e.g. Hindi + English words).
    """
    if not text:
        return "other"
    d = len(_DEVANAGARI.findall(text))
    b = len(_BENGALI.findall(text))
    l = len(_LATIN.findall(text))
    total = d + b + l
    if total == 0:
        return "other"
    counts = {"devanagari": d, "bengali": b, "latin": l}
    top, _ = max(counts.items(), key=lambda kv: kv[1])
    second = sorted(counts.values())[-2]
    if second / total >= 0.20:
        return "mixed"
    return top


def script_ratio(text: str) -> dict:
    d = len(_DEVANAGARI.findall(text))
    b = len(_BENGALI.findall(text))
    l = len(_LATIN.findall(text))
    tot = max(d + b + l, 1)
    return {"devanagari": d / tot, "bengali": b / tot, "latin": l / tot}


def strip_for_dedup(text: str) -> str:
    """Aggressive canonical form used ONLY for duplicate detection."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = emoji_lib.replace_emoji(t, replace=" ")
    t = regex.sub(r"[^\p{L}\p{N}\s]", " ", t)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


@dataclass
class Preprocessor:
    """Normalisation pipeline. Each flag is an ablation switch (see configs/preprocessing.yaml)."""

    unicode_nfc: bool = True
    indic_normalize: bool = True       # indic-nlp-library normaliser (nukta, chandrabindu, etc.)
    remove_zero_width: bool = True
    lowercase_latin: bool = True
    url_token: str | None = "URL"      # None -> keep URL as is
    mention_token: str | None = "@user"
    keep_hashtag_text: bool = True     # "#ClimateHoax" -> "ClimateHoax"
    number_token: str | None = None    # e.g. "NUM"; None keeps numbers (years matter: "2050")
    emoji_mode: str = "keep"           # keep | demojize | remove
    reduce_elongation: bool = True     # "bahuuuut" -> "bahuut" (2 repeats keep emphasis signal)
    reduce_punct_repeat: bool = True
    transliterate_romanized: bool = False  # Romanized Hindi/Bengali -> native script (ablation)
    transliterate_lang: str = "auto"       # hi | bn | auto (uses the language column if given)
    strip: bool = True
    _normalizers: dict = field(default_factory=dict, repr=False)

    # ---- helpers -------------------------------------------------------------
    def _indic_norm(self, text: str, lang: str) -> str:
        if lang not in ("hi", "bn"):
            return text
        if lang not in self._normalizers:
            try:
                from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

                self._normalizers[lang] = IndicNormalizerFactory().get_normalizer(lang)
            except Exception:  # library missing -> no-op, but never crash
                self._normalizers[lang] = None
        norm = self._normalizers[lang]
        return norm.normalize(text) if norm is not None else text

    def _transliterate(self, text: str, lang: str) -> str:
        from stance.data.transliterate import romanized_to_native

        return romanized_to_native(text, lang)

    # ---- main entry ----------------------------------------------------------
    def __call__(self, text: str, language: str | None = None) -> str:
        if text is None:
            return ""
        t = str(text)
        if self.unicode_nfc:
            t = unicodedata.normalize("NFC", t)
        if self.remove_zero_width:
            t = _ZW.sub("", t)
        if self.url_token is not None:
            t = _URL.sub(f" {self.url_token} ", t)
        if self.mention_token is not None:
            t = _MENTION.sub(f" {self.mention_token} ", t)
        if self.keep_hashtag_text:
            t = _HASHTAG.sub(r"\1", t)
        if self.number_token is not None:
            t = _NUMBER.sub(f" {self.number_token} ", t)
        if self.emoji_mode == "demojize":
            t = emoji_lib.demojize(t, delimiters=(" :", ": "))
        elif self.emoji_mode == "remove":
            t = emoji_lib.replace_emoji(t, replace=" ")
        if self.reduce_elongation:
            t = _ELONG.sub(r"\1\1", t)
        if self.reduce_punct_repeat:
            t = _PUNCT_REP.sub(r"\1\1", t)
        scr = detect_script(t)
        lang = language or ("hi" if scr == "devanagari" else "bn" if scr == "bengali" else "")
        if self.indic_normalize and lang in ("hi", "bn"):
            t = self._indic_norm(t, lang)
        if self.lowercase_latin:
            # lowercase only ASCII letters; Indic scripts have no case
            t = "".join(c.lower() if c.isascii() else c for c in t)
        if self.transliterate_romanized:
            if self.transliterate_lang == "auto":
                tl = language if language in ("hi", "bn") else "hi"
            else:
                tl = self.transliterate_lang
            if detect_script(t) in ("latin", "mixed"):
                t = self._transliterate(t, tl)
        t = _MULTISPACE.sub(" ", t)
        return t.strip() if self.strip else t

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Preprocessor":
        cfg = dict(cfg or {})
        allowed = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        return cls(**{k: v for k, v in cfg.items() if k in allowed})


def code_mix_ratio(text: str) -> float:
    """Fraction of alphabetic whitespace tokens that are Latin-script.
    0 = pure native script, 1 = pure Latin (Romanized or English). Used in EDA/challenge."""
    toks = [w for w in text.split() if regex.search(r"\p{L}", w)]
    if not toks:
        return 0.0
    latin = sum(
        1 for w in toks if _LATIN.search(w) and not (_DEVANAGARI.search(w) or _BENGALI.search(w))
    )
    return latin / len(toks)
