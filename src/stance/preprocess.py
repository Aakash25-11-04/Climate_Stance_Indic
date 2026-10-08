"""Text normalisation and script detection shared by data preparation and modelling."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")

SCRIPTS = ("bengali", "devanagari", "latin")
# Danda and double danda sit in the Devanagari block but also end Bengali
# sentences, so they carry no information about which script a text uses.
SCRIPT_NEUTRAL = {"।", "॥"}


def _is_invisible(ch: str) -> bool:
    # Zero-width joiners (Cf) shape conjuncts inside a word and variation selectors
    # only pick emoji presentation, so neither may become a word boundary.
    return unicodedata.category(ch) == "Cf" or "︀" <= ch <= "️"


def _is_word_char(ch: str) -> bool:
    # Python's \w excludes combining marks (Mn, Mc), which would split every
    # Devanagari and Bengali word at its vowel signs and viramas.
    category = unicodedata.category(ch)
    return ch == "_" or category[0] in "LN" or category in ("Mn", "Mc")


def dedup_key(text: str) -> str:
    """Normalise text for duplicate detection: case, URLs, mentions, punctuation, spacing."""
    text = unicodedata.normalize("NFC", text.lower())
    text = MENTION.sub(" ", URL.sub(" ", text))
    kept = (ch if _is_word_char(ch) else " " for ch in text if not _is_invisible(ch))
    return " ".join("".join(kept).split())


def script_of(ch: str) -> str | None:
    if ch in SCRIPT_NEUTRAL:
        return None
    if "ऀ" <= ch <= "ॿ":
        return "devanagari"
    if "ঀ" <= ch <= "৿":
        return "bengali"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return None


def script_shares(text: str) -> dict[str, float]:
    """Share of Bengali, Devanagari and Latin characters among all characters of those scripts."""
    counts = Counter(script for script in map(script_of, text) if script is not None)
    total = max(sum(counts.values()), 1)
    return {script: counts[script] / total for script in SCRIPTS}


def dominant_script(text: str, threshold: float = 0.8) -> str:
    """The script holding at least `threshold` of the letters, "mixed" if none does,
    "none" for texts without letters (emoji, numbers, punctuation)."""
    shares = script_shares(text)
    if not any(shares.values()):
        return "none"
    script, share = max(shares.items(), key=lambda item: item[1])
    return script if share >= threshold else "mixed"
