"""Tier 3 baselines: frozen multilingual sentence embeddings + Logistic Regression.

f) EmbeddingLRModel         : embed(text) -> LR
g) TargetAwareEmbeddingModel: [u, v, |u-v|, u*v] with v = embedding of the claim in the
                              comment's language -> LR  (InferSent/SBERT-style pair features)
h) ZeroShotNLIModel         : multilingual XNLI model, no training

Embeddings are cached on disk per (model, text) so that CV folds and later ensembles never
re-encode a sentence.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from stance.preprocessing.normalize import Preprocessor
from stance.utils import LABEL2ID, LABELS, abs_path, resolve_device

EMBEDDING_MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-small": "intfloat/multilingual-e5-small",
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


class EmbeddingCache:
    """SQLite cache of float16 vectors keyed by sha1(model + text)."""

    def __init__(self, path: str | Path = "data/cache/embeddings.sqlite"):
        self.path = abs_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = None

    @property
    def con(self):
        if self._con is None:
            self._con = sqlite3.connect(self.path)
            self._con.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, dim INTEGER, v BLOB)")
            self._con.commit()
        return self._con

    def __getstate__(self):  # picklable: drop the live connection
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]; self._con = None

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys):
        out = {}
        for i in range(0, len(keys), 900):
            chunk = keys[i : i + 900]
            q = f"SELECT k, dim, v FROM e WHERE k IN ({','.join('?' * len(chunk))})"
            for k, dim, blob in self.con.execute(q, chunk):
                out[k] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)[:dim]
        return out

    def put_many(self, items: dict[str, np.ndarray]):
        rows = [(k, int(v.shape[0]), v.astype(np.float16).tobytes()) for k, v in items.items()]
        self.con.executemany("INSERT OR REPLACE INTO e (k, dim, v) VALUES (?, ?, ?)", rows)
        self.con.commit()


class Embedder:
    _models: dict = {}

    def __init__(self, name: str = "labse", device: str = "auto", batch_size: int = 64,
                 cache_path: str = "data/cache/embeddings.sqlite"):
        self.alias = name
        self.model_name = EMBEDDING_MODELS.get(name, name)
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_path)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()}

    def _model(self):
        if self.model_name not in Embedder._models:
            from sentence_transformers import SentenceTransformer

            Embedder._models.clear()  # keep only one encoder in memory (4 GB GPU / low RAM)
            Embedder._models[self.model_name] = SentenceTransformer(self.model_name, device=self.device)
        return Embedder._models[self.model_name]

    def _prefix(self, texts):
        # E5 expects "query: " prefix for symmetric tasks
        if "e5" in self.model_name:
            return [f"query: {t}" for t in texts]
        return list(texts)

    def encode(self, texts, show_progress: bool = False) -> np.ndarray:
        texts = [t if isinstance(t, str) and t.strip() else "." for t in texts]
        keys = [self.cache.key(self.model_name, t) for t in texts]
        cached = self.cache.get_many(list(set(keys)))
        todo_idx = [i for i, k in enumerate(keys) if k not in cached]
        if todo_idx:
            model = self._model()
            todo_texts = self._prefix([texts[i] for i in todo_idx])
            vecs = model.encode(todo_texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=show_progress)
            new = {keys[i]: vecs[j] for j, i in enumerate(todo_idx)}
            self.cache.put_many(new)
            cached.update(new)
        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    @staticmethod
    def unload():
        Embedder._models.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


class EmbeddingLRModel:
    """f) frozen sentence embedding -> LogisticRegression."""

    def __init__(self, embedder: str = "labse", C: float = 1.0, preprocessing: dict | None = None,
                 seed: int = 42, device: str = "auto"):
        self.embedder = Embedder(embedder, device=device)
        self.C, self.seed = C, seed
        self.pre = Preprocessor.from_config(preprocessing)
        self.name = f"emb_{embedder}_lr"

    def _feats(self, texts, languages=None):
        langs = languages or [None] * len(texts)
        return self.embedder.encode([self.pre(t, l) for t, l in zip(texts, langs)])

    def fit(self, texts, labels, languages=None):
        X = self._feats(texts, languages)
        y = np.array([LABEL2ID[l] for l in labels])
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000, random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, texts, languages=None):
        p = self.clf.predict_proba(self._feats(texts, languages))
        out = np.zeros((len(texts), len(LABELS)))
        out[:, self.clf.classes_] = p
        return out

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]


class TargetAwareEmbeddingModel(EmbeddingLRModel):
    """g) [u, v, |u-v|, u*v] where v = claim embedding in the comment's language."""

    def __init__(self, claims: dict, embedder: str = "labse", C: float = 1.0, preprocessing=None,
                 seed: int = 42, device: str = "auto"):
        super().__init__(embedder, C, preprocessing, seed, device)
        self.claims = claims  # {"en": ..., "hi": ..., "bn": ...}
        self.name = f"emb_{embedder}_target_lr"

    def _feats(self, texts, languages=None):
        langs = [l if l in self.claims else "en" for l in (languages or ["en"] * len(texts))]
        u = self.embedder.encode([self.pre(t, l) for t, l in zip(texts, langs)])
        claim_vecs = {l: self.embedder.encode([self.claims[l]])[0] for l in set(langs)}
        v = np.vstack([claim_vecs[l] for l in langs])
        return np.hstack([u, v, np.abs(u - v), u * v])


class ZeroShotNLIModel:
    """h) Zero-shot NLI: P(entail | comment, hypothesis_label) for three hypotheses.

    Hypotheses are written in the comment's language (hi / bn) with an English fallback.
    The 'None' hypothesis is explicit so the three scores can be normalised.
    """

    HYPOTHESES = {
        "en": {"Favour": "This comment agrees that climate change is a serious concern.",
               "Against": "This comment disagrees that climate change is a serious concern.",
               "None": "This comment does not express an opinion about climate change."},
        "hi": {"Favour": "यह टिप्पणी मानती है कि जलवायु परिवर्तन एक गंभीर चिंता है।",
               "Against": "यह टिप्पणी नहीं मानती कि जलवायु परिवर्तन एक गंभीर चिंता है।",
               "None": "यह टिप्पणी जलवायु परिवर्तन पर कोई राय नहीं देती।"},
        "bn": {"Favour": "এই মন্তব্য মানে যে জলবায়ু পরিবর্তন একটি গুরুতর উদ্বেগ।",
               "Against": "এই মন্তব্য মানে না যে জলবায়ু পরিবর্তন একটি গুরুতর উদ্বেগ।",
               "None": "এই মন্তব্য জলবায়ু পরিবর্তন নিয়ে কোনো মতামত দেয় না।"},
    }

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                 device: str = "auto", batch_size: int = 16, hypothesis_lang: str = "auto"):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.hypothesis_lang = hypothesis_lang
        self.name = "zeroshot_nli"
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = [i for i, l in id2label.items() if l.startswith("entail")][0]
        self._loaded = True

    def fit(self, texts, labels, languages=None):
        return self

    def predict_proba(self, texts, languages=None):
        import torch

        self._load()
        langs = languages or ["en"] * len(texts)
        scores = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        pairs = []
        for i, (t, l) in enumerate(zip(texts, langs)):
            hl = l if (self.hypothesis_lang == "auto" and l in self.HYPOTHESES) else (
                self.hypothesis_lang if self.hypothesis_lang in self.HYPOTHESES else "en")
            for j, lab in enumerate(LABELS):
                pairs.append((i, j, t if t.strip() else ".", self.HYPOTHESES[hl][lab]))
        for b in range(0, len(pairs), self.batch_size):
            chunk = pairs[b : b + self.batch_size]
            enc = self.tok([c[2] for c in chunk], [c[3] for c in chunk], truncation=True, max_length=256,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits
                ent = torch.softmax(logits, dim=-1)[:, self.entail_idx].cpu().numpy()
            for (i, j, _, _), e in zip(chunk, ent):
                scores[i, j] = e
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]
