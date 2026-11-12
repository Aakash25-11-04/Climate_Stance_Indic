"""Synthetic code-mixing (Hindi-English / Bengali-English) of native-script text.

Three operations, each driven by a seeded random generator so the output is reproducible:
1. lexical substitution: replace native words that have a common English equivalent used
   in everyday speech (जलवायु परिवर्तन -> climate change, सरकार -> government, ...).
2. discourse markers: insert English fillers that bilingual speakers use
   ("actually", "honestly", "bro", "I think", "seriously").
3. partial romanization: romanize a random contiguous span of the remaining native words,
   producing mixed-script text (Devanagari + Latin) that is typical of comment sections.

The lexicons below were written by hand for this project; they only contain high-frequency
climate/opinion vocabulary, which is what appears in real code-mixed comments.
"""
from __future__ import annotations

import random
import re

from stance.data.transliterate import native_to_roman

HI_LEXICON = {
    "जलवायु परिवर्तन": "climate change", "ग्लोबल वार्मिंग": "global warming",
    "वैश्विक तापमान": "global warming", "वैज्ञानिक": "scientists", "वैज्ञानिकों": "scientists",
    "सरकार": "government", "समस्या": "problem", "गंभीर": "serious", "सच": "true",
    "झूठ": "fake", "झूठा": "fake", "प्रदूषण": "pollution", "पर्यावरण": "environment",
    "तापमान": "temperature", "मौसम": "weather", "बारिश": "rain", "गर्मी": "heat",
    "भविष्य": "future", "बच्चों": "kids", "बच्चे": "kids", "दुनिया": "world",
    "धरती": "earth", "पृथ्वी": "earth", "पैसा": "money", "पैसे": "money",
    "एजेंडा": "agenda", "साजिश": "conspiracy", "रिपोर्ट": "report", "खबर": "news",
    "कार्बन": "carbon", "ऊर्जा": "energy", "पेड़": "trees", "जंगल": "forest",
    "बाढ़": "flood", "सूखा": "drought", "खतरा": "danger", "चिंता": "concern",
    "असली": "real", "नकली": "fake", "मुद्दा": "issue", "बहस": "debate", "डर": "fear",
    "प्लास्टिक": "plastic", "कंपनियां": "companies", "कंपनी": "company", "नेता": "leaders",
    "सिस्टम": "system", "लोग": "people", "सबूत": "proof", "डेटा": "data",
}
BN_LEXICON = {
    "জলবায়ু পরিবর্তন": "climate change", "বৈশ্বিক উষ্ণায়ন": "global warming",
    "গ্লোবাল ওয়ার্মিং": "global warming", "বিজ্ঞানীরা": "scientists", "বিজ্ঞানী": "scientist",
    "সরকার": "government", "সমস্যা": "problem", "গুরুতর": "serious", "সত্য": "true",
    "সত্যি": "true", "মিথ্যা": "fake", "ভুয়া": "fake", "দূষণ": "pollution",
    "পরিবেশ": "environment", "তাপমাত্রা": "temperature", "আবহাওয়া": "weather",
    "বৃষ্টি": "rain", "গরম": "heat", "ভবিষ্যৎ": "future", "বাচ্চাদের": "kids",
    "পৃথিবী": "earth", "দুনিয়া": "world", "টাকা": "money", "এজেন্ডা": "agenda",
    "ষড়যন্ত্র": "conspiracy", "রিপোর্ট": "report", "খবর": "news", "কার্বন": "carbon",
    "শক্তি": "energy", "গাছ": "trees", "বন": "forest", "বন্যা": "flood", "খরা": "drought",
    "বিপদ": "danger", "উদ্বেগ": "concern", "আসল": "real", "নকল": "fake", "ইস্যু": "issue",
    "বিতর্ক": "debate", "ভয়": "fear", "প্লাস্টিক": "plastic", "কোম্পানি": "company",
    "নেতারা": "leaders", "সিস্টেম": "system", "মানুষ": "people", "প্রমাণ": "proof",
    "তথ্য": "data",
}
MARKERS_START = ["actually", "honestly", "bro", "seriously", "I think", "frankly", "dekho",
                 "listen", "tbh", "lol", "bhai", "guys", "matlab", "basically"]
MARKERS_END = ["seriously", "period.", "no doubt", "obviously", "for real", "100%", "lol",
               "smh", "idk", "right?", "yaar", "bhai", "full stop"]
_WORD_RE = re.compile(r"[ऀ-ॿঀ-৿]+")


def _substitute(text: str, lexicon: dict, rng: random.Random, p: float) -> str:
    # longest keys first so multiword entries win
    for k in sorted(lexicon, key=len, reverse=True):
        if k in text and rng.random() < p:
            # whole-word match: no Indic letter/sign directly before or after the key
            text = re.sub(r"(?<![ऀ-ॣ०-ॿঀ-৿])" + re.escape(k) + r"(?![ऀ-ॣ०-ॿঀ-৿])",
                          lexicon[k], text)
    return text


def _partial_romanize(text: str, lang: str, rng: random.Random, max_span: int = 4) -> str:
    words = list(_WORD_RE.finditer(text))
    if len(words) < 2:
        return text
    n = rng.randint(1, min(max_span, len(words)))
    start = rng.randint(0, len(words) - n)
    span = words[start : start + n]
    s, e = span[0].start(), span[-1].end()
    return text[:s] + native_to_roman(text[s:e], lang, rng, style="casual") + text[e:]


def codemix(text: str, lang: str, rng: random.Random, p_sub: float = 0.6, p_marker: float = 0.5,
            p_roman_span: float = 0.5) -> str:
    """Return a code-mixed variant of a native-script Hindi/Bengali text."""
    lex = HI_LEXICON if lang == "hi" else BN_LEXICON
    out = _substitute(text, lex, rng, p_sub)
    if rng.random() < p_roman_span:
        out = _partial_romanize(out, lang, rng)
    if rng.random() < p_marker:
        if rng.random() < 0.5:
            out = f"{rng.choice(MARKERS_START)} {out}"
        else:
            out = f"{out.rstrip('.!। ')} {rng.choice(MARKERS_END)}"
    return out


def romanize_full(text: str, lang: str, rng: random.Random) -> str:
    """Fully Romanized (Hinglish/Banglish) version with casual spelling variation."""
    return native_to_roman(text, lang, rng, style="casual")
