"""Tier 1 + Tier 2 baselines.

a) MajorityModel, StratifiedRandomModel
b) LexiconModel  - hand-built Hindi / Bengali / Romanized cue lists with negation handling
c) TfidfModel    - word 1-2 + char_wb 2-5 TF-IDF, LogisticRegression / LinearSVC / ComplementNB
d) NBSVMModel    - NB log-count-ratio scaled features + linear classifier (Wang & Manning 2012)
e) TransliteratedCopyModel - same as (c)/(d) but each text is concatenated with its
   Romanized / native copy so that both spellings share features
"""
from __future__ import annotations

import random
import re

import numpy as np
from scipy import sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC

from stance.preprocessing.normalize import Preprocessor, detect_script
from stance.utils import LABEL2ID, LABELS


class BaseModel:
    name = "base"

    def fit(self, texts, labels, languages=None):
        raise NotImplementedError

    def predict_proba(self, texts, languages=None) -> np.ndarray:
        raise NotImplementedError

    def predict(self, texts, languages=None) -> list[str]:
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]


# --------------------------------------------------------------------------- Tier 1
class MajorityModel(BaseModel):
    name = "majority"

    def fit(self, texts, labels, languages=None):
        counts = {l: 0 for l in LABELS}
        for l in labels:
            counts[l] += 1
        self.major = max(counts, key=counts.get)
        return self

    def predict_proba(self, texts, languages=None):
        p = np.zeros((len(texts), len(LABELS)))
        p[:, LABEL2ID[self.major]] = 1.0
        return p


class StratifiedRandomModel(BaseModel):
    name = "stratified_random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, texts, labels, languages=None):
        counts = np.array([sum(1 for l in labels if l == lab) for lab in LABELS], dtype=float)
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, texts, languages=None):
        rng = np.random.default_rng(self.seed)
        draws = rng.choice(len(LABELS), size=len(texts), p=self.prior)
        p = np.full((len(texts), len(LABELS)), 0.0)
        p[np.arange(len(texts)), draws] = 1.0
        return p


# --------------------------------------------------------------------------- Lexicon
FAVOUR_CUES = [
    # Hindi
    "गंभीर", "खतरा", "खतरनाक", "असली", "सच", "सच्चाई", "बचाओ", "बचाना", "जागो", "कदम उठा", "पेड़ लगा",
    "ग्लेशियर", "पिघल", "समुद्र का स्तर", "रिकॉर्ड", "चिंता", "भविष्य", "बच्चों", "जिम्मेदार", "ज़िम्मेदार",
    "कार्बन कम", "उत्सर्जन", "सौर", "सोलर", "चेतावनी", "वैज्ञानिक सही", "डर लगता", "बर्बाद", "सूख",
    # Bengali
    "গুরুতর", "বিপদ", "বিপজ্জনক", "আসল", "সত্যি", "সত্য", "বাঁচাও", "বাঁচাতে", "জাগো", "পদক্ষেপ", "গাছ লাগা",
    "হিমবাহ", "গলছে", "সমুদ্রের", "রেকর্ড", "উদ্বেগ", "ভবিষ্যৎ", "সন্তান", "দায়ী", "নিঃসরণ", "সৌর", "সোলার",
    "সতর্ক", "ভয় লাগে", "নষ্ট", "শুকিয়ে", "ডুবছে",
    # Romanized / English
    "serious", "real", "danger", "khatra", "khatarnak", "sach", "sotti", "asli", "asol", "bachao", "jago",
    "act now", "save", "warning", "glacier", "pighal", "record", "future", "bacho", "responsib", "emission",
    "solar", "ped lagao", "gach lagao", "dar lagta", "bhoy lage", "barbad", "nosto",
]
AGAINST_CUES = [
    # Hindi
    "झूठ", "झूठा", "साजिश", "स्कैम", "प्रोपेगैंडा", "एजेंडा", "ड्रामा", "नाटक", "अफवाह", "धोखा", "धंधा",
    "पैसा कमा", "फंडिंग", "बेवजह", "डराने", "हमेशा बदलता", "हर साल गर्मी", "कोई खतरा नहीं", "कोई सबूत नहीं",
    "सूरज", "प्राकृतिक", "मज़ाक", "मजाक", "टैक्स", "हल्ला",
    # Bengali
    "মিথ্যা", "ভুয়া", "ষড়যন্ত্র", "স্ক্যাম", "প্রোপাগান্ডা", "এজেন্ডা", "নাটক", "গুজব", "ধোঁকা", "ব্যবসা",
    "টাকা কামা", "ফান্ডিং", "অযথা", "ভয় দেখা", "সবসময় বদলা", "প্রতি বছর গরম", "কোনো বিপদ নেই",
    "কোনো প্রমাণ নেই", "সূর্যের", "প্রাকৃতিক", "কৌতুক", "মজা", "ট্যাক্স", "হইচই",
    # Romanized / English
    "jhooth", "jhuth", "jhut", "mittha", "mitha", "hoax", "scam", "fake", "sajish", "conspiracy", "agenda",
    "propaganda", "drama", "natak", "afwah", "gujob", "dhandha", "dhanda", "business", "funding", "paisa kama",
    "taka kama", "darane", "bhoy dekha", "koi khatra nahi", "kono bipod nei", "natural cycle", "sooraj",
    "tax", "mazak", "majak", "jhut",
]
NEGATORS = ["नहीं", "नही", "ना ", "मत ", "गलत", "না ", "নেই", "নয়", "ভুল", "nahi", "nahin", "nhi", "nai",
            "na ", "mat ", "galat", "nei", "noy", "bhul", "not ", "no ", "never"]
NEUTRAL_CUES = ["बैठक", "सेमिनार", "रिपोर्ट", "वीडियो", "मतलब", "निबंध", "पूर्वानुमान", "मौसम विभाग", "कल ",
                "सम्मेलन", "बৈঠক", "সেমিনার", "রিপোর্ট", "ভিডিও", "মানে", "রচনা", "পূর্বাভাস", "আবহাওয়া অফিস",
                "কাল ", "webinar", "seminar", "report", "video", "link", "essay", "forecast", "meeting", "summit",
                "match", "song", "subscribe", "gaana", "gaan", "trailer", "?"]


class LexiconModel(BaseModel):
    """Count cue hits per class; a negator within 3 tokens before a cue flips Favour<->Against."""

    name = "lexicon"

    def __init__(self):
        self.pre = Preprocessor(emoji_mode="keep", indic_normalize=False)

    def fit(self, texts, labels, languages=None):
        return self

    def _score(self, text: str) -> np.ndarray:
        t = " " + self.pre(text) + " "
        fav = sum(t.count(c) for c in FAVOUR_CUES)
        agn = sum(t.count(c) for c in AGAINST_CUES)
        neu = sum(t.count(c) for c in NEUTRAL_CUES)
        # negation flips: "जलवायु परिवर्तन झूठ नहीं है" -> Favour
        neg = any(n in t for n in NEGATORS)
        if neg and agn > fav:
            fav, agn = agn, fav
        elif neg and fav > agn and ("खतरा" in t or "bipod" in t or "khatra" in t or "বিপদ" in t):
            fav, agn = agn, fav
        s = np.array([fav, agn, neu * 0.6], dtype=float) + 1e-3
        s[2] += 0.25 if (fav + agn) == 0 else 0.0  # nothing matched -> None
        return s / s.sum()

    def predict_proba(self, texts, languages=None):
        return np.vstack([self._score(t) for t in texts])


# --------------------------------------------------------------------------- Tier 2
def make_vectorizer(word_ngram=(1, 2), char_ngram=(2, 5), min_df=2, max_features_word=200_000,
                    max_features_char=300_000):
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=word_ngram, min_df=min_df, sublinear_tf=True,
                        lowercase=False, token_pattern=r"(?u)\b\w+\b", max_features=max_features_word),
        TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, min_df=min_df, sublinear_tf=True,
                        lowercase=False, max_features=max_features_char),
    )


def _make_clf(kind: str, C: float, seed: int):
    if kind == "lr":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
    if kind == "svm":
        # CalibratedClassifierCV gives probabilities for stacking / ensembling
        return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", random_state=seed), cv=3)
    if kind == "cnb":
        return ComplementNB(alpha=C)
    raise ValueError(kind)


class TfidfModel(BaseModel):
    """c) TF-IDF word+char n-grams with LR / linear SVM / Complement NB."""

    def __init__(self, clf: str = "lr", C: float = 1.0, word_ngram=(1, 2), char_ngram=(2, 5),
                 min_df: int = 2, preprocessing: dict | None = None, seed: int = 42):
        self.clf_kind, self.C, self.seed = clf, C, seed
        self.word_ngram, self.char_ngram, self.min_df = tuple(word_ngram), tuple(char_ngram), min_df
        self.pre = Preprocessor.from_config(preprocessing)
        self.name = f"tfidf_{clf}"

    def _prep(self, texts, languages=None):
        langs = languages or [None] * len(texts)
        return [self.pre(t, l) for t, l in zip(texts, langs)]

    def fit(self, texts, labels, languages=None):
        self.vec = make_vectorizer(self.word_ngram, self.char_ngram, self.min_df)
        X = self.vec.fit_transform(self._prep(texts, languages))
        y = np.array([LABEL2ID[l] for l in labels])
        self.clf = _make_clf(self.clf_kind, self.C, self.seed).fit(X, y)
        return self

    def predict_proba(self, texts, languages=None):
        X = self.vec.transform(self._prep(texts, languages))
        p = self.clf.predict_proba(X)
        out = np.zeros((len(texts), len(LABELS)))
        out[:, self.clf.classes_] = p
        return out

    def top_features(self, k: int = 25) -> dict:
        """Top positive coefficients per class (for explainability)."""
        if self.clf_kind != "lr":
            return {}
        names = np.concatenate([v.get_feature_names_out() for _, v in self.vec.transformer_list])
        out = {}
        for i, lab in enumerate(LABELS):
            coef = self.clf.coef_[i]
            idx = np.argsort(coef)[::-1][:k]
            out[lab] = [(str(names[j]), float(coef[j])) for j in idx]
        return out


class NBSVMModel(BaseModel):
    """d) NB-SVM: one-vs-rest; features scaled by the NB log-count ratio r, then LR/SVM."""

    def __init__(self, C: float = 1.0, alpha: float = 1.0, clf: str = "lr", word_ngram=(1, 2),
                 char_ngram=(2, 5), min_df: int = 2, preprocessing: dict | None = None, seed: int = 42):
        self.C, self.alpha, self.clf_kind, self.seed = C, alpha, clf, seed
        self.word_ngram, self.char_ngram, self.min_df = tuple(word_ngram), tuple(char_ngram), min_df
        self.pre = Preprocessor.from_config(preprocessing)
        self.name = "nbsvm"

    def _prep(self, texts, languages=None):
        langs = languages or [None] * len(texts)
        return [self.pre(t, l) for t, l in zip(texts, langs)]

    def fit(self, texts, labels, languages=None):
        self.vec = make_vectorizer(self.word_ngram, self.char_ngram, self.min_df)
        X = self.vec.fit_transform(self._prep(texts, languages))
        X = (X > 0).astype(np.float32)  # binarised presence, as in Wang & Manning
        y = np.array([LABEL2ID[l] for l in labels])
        self.r, self.models = [], []
        for c in range(len(LABELS)):
            pos = np.asarray(X[y == c].sum(axis=0)).ravel() + self.alpha
            neg = np.asarray(X[y != c].sum(axis=0)).ravel() + self.alpha
            r = np.log((pos / pos.sum()) / (neg / neg.sum()))
            Xr = X.multiply(r).tocsr()
            yb = (y == c).astype(int)
            if self.clf_kind == "svm":
                m = CalibratedClassifierCV(LinearSVC(C=self.C, class_weight="balanced", random_state=self.seed), cv=3)
            else:
                m = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed)
            m.fit(Xr, yb)
            self.r.append(r)
            self.models.append(m)
        return self

    def predict_proba(self, texts, languages=None):
        X = (self.vec.transform(self._prep(texts, languages)) > 0).astype(np.float32)
        scores = np.column_stack([m.predict_proba(X.multiply(r).tocsr())[:, 1] for m, r in zip(self.models, self.r)])
        return scores / scores.sum(axis=1, keepdims=True)


class TransliteratedCopyModel(BaseModel):
    """e) Wrap (c)/(d): every text becomes `text ||| transliterated_copy` so native-script and
    Romanized spellings of the same word share features. Native -> Romanized copy for native
    text; Romanized -> native copy for Latin text (language from the `languages` column)."""

    def __init__(self, base: BaseModel):
        self.base = base
        self.name = f"{base.name}+translit"

    @staticmethod
    def _augment(texts, languages):
        from stance.data.transliterate import native_to_roman, romanized_to_native

        out = []
        langs = languages or ["hi"] * len(texts)
        for t, l in zip(texts, langs):
            l = l if l in ("hi", "bn") else "hi"
            scr = detect_script(t)
            if scr in ("devanagari", "bengali", "mixed"):
                copy = native_to_roman(t, l)
            else:
                copy = romanized_to_native(t, l)
            out.append(f"{t} ||| {copy}")
        return out

    def fit(self, texts, labels, languages=None):
        self.base.fit(self._augment(texts, languages), labels, languages)
        return self

    def predict_proba(self, texts, languages=None):
        return self.base.predict_proba(self._augment(texts, languages), languages)
