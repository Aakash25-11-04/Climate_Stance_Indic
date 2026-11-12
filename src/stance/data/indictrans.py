"""IndicTrans2 translation with a pure-Python pre/post-processor and a disk cache.

Why a re-implementation of IndicProcessor?
    The official `IndicTransToolkit` is a Cython extension that needs MSVC build tools,
    which are not available on this machine. The logic below mirrors
    IndicTransToolkit/processor.pyx (punctuation normalisation, Moses tokenisation for
    English, indic-nlp normalisation + tokenisation + transliteration to the unified
    Devanagari script for Indic inputs, and the `src_lang tgt_lang` tag prefix).

Models (Tier C+, 4 GB VRAM): ai4bharat/indictrans2-en-indic-dist-200M and
ai4bharat/indictrans2-indic-en-dist-200M.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from stance.utils import abs_path, stable_hash

FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng"}
ISO = {v: k for k, v in FLORES.items()}

_PUNC_REPLACEMENTS = [
    (re.compile(r"\r"), ""),
    (re.compile(r"\(\s*"), "("),
    (re.compile(r"\s*\)"), ")"),
    (re.compile(r"\s:\s?"), ":"),
    (re.compile(r"\s;\s?"), ";"),
    (re.compile("[`\u00b4\u2018\u201a\u2019]"), "'"),
    (re.compile("[\u201e\u201c\u201d\u00ab\u00bb]"), '"'),
    (re.compile("[\u2013\u2014]"), "-"),
    (re.compile(r" %"), "%"),
    (re.compile(r" [?!;]"), lambda m: m.group(0).strip()),
]
_MULTISPACE = re.compile(r"[ ]{2,}")
_DIGITS = {}
for _block in (0x0966, 0x09E6):  # Devanagari, Bengali digit blocks -> ASCII
    for _i in range(10):
        _DIGITS[_block + _i] = str(_i)


def punc_norm(text: str) -> str:
    for pat, rep in _PUNC_REPLACEMENTS:
        text = pat.sub(rep, text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


class IndicProcessorPy:
    """Minimal pure-Python replacement for IndicTransToolkit.IndicProcessor."""

    def __init__(self):
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

        self._factory = IndicNormalizerFactory()
        self._normalizers = {}
        self._xlit = UnicodeIndicTransliterator()
        self._en_tok = MosesTokenizer(lang="en")
        self._en_norm = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")

    def _normalizer(self, iso: str):
        if iso not in self._normalizers:
            self._normalizers[iso] = self._factory.get_normalizer(iso)
        return self._normalizers[iso]

    def preprocess_batch(self, batch: Iterable[str], src_lang: str, tgt_lang: str) -> list[str]:
        from indicnlp.tokenize import indic_tokenize

        iso = ISO.get(src_lang, "hi")
        out = []
        for sent in batch:
            sent = punc_norm(str(sent)).translate(_DIGITS)
            if iso == "en":
                normed = self._en_norm.normalize(sent.strip())
                proc = " ".join(self._en_tok.tokenize(normed, escape=False))
            else:
                normed = self._normalizer(iso).normalize(sent.strip())
                toks = " ".join(indic_tokenize.trivial_tokenize(normed, iso))
                proc = self._xlit.transliterate(toks, iso, "hi").replace(" \u094d ", "\u094d")
            out.append(f"{src_lang} {tgt_lang} {proc.strip()}")
        return out

    def postprocess_batch(self, sents: Iterable[str], lang: str) -> list[str]:
        from indicnlp.tokenize import indic_detokenize

        iso = ISO.get(lang, "hi")
        out = []
        for s in sents:
            if lang == "eng_Latn":
                out.append(self._en_detok.detokenize(s.split(" ")))
            else:
                x = self._xlit.transliterate(s, "hi", iso)
                out.append(indic_detokenize.trivial_detokenize(x, iso))
        return out


# --------------------------------------------------------------------------- cache
class TranslationCache:
    """SQLite key-value cache: (model, src, tgt, text) -> translation."""

    def __init__(self, path: str | Path = "data/cache/translations.sqlite"):
        self.path = abs_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT)")
        self.con.commit()

    @staticmethod
    def key(model: str, src: str, tgt: str, text: str) -> str:
        return stable_hash(model, src, tgt, text, n=32)

    def get_many(self, keys: list[str]) -> dict[str, str]:
        out = {}
        for i in range(0, len(keys), 900):
            chunk = keys[i : i + 900]
            q = f"SELECT k, v FROM t WHERE k IN ({','.join('?' * len(chunk))})"
            out.update(dict(self.con.execute(q, chunk).fetchall()))
        return out

    def put_many(self, items: dict[str, str]) -> None:
        self.con.executemany("INSERT OR REPLACE INTO t (k, v) VALUES (?, ?)", list(items.items()))
        self.con.commit()


# --------------------------------------------------------------------------- translator
def hf_token_available() -> bool:
    """True if a Hugging Face token is configured (needed for the gated IndicTrans2 repos)."""
    import os

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


class IndicTranslator:
    """Batched, cached translation (en->hi/bn and hi/bn->en).

    backend = "indictrans2" : ai4bharat/indictrans2-*-dist-200M (gated repos: the user must
                              accept the terms on the Hub and run `huggingface-cli login`).
    backend = "nllb"        : facebook/nllb-200-distilled-600M (open, CC-BY-NC-4.0). Used
                              automatically when no HF token is configured.
    backend = "auto"        : indictrans2 if a token is available, else nllb.
    """

    EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
    INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
    NLLB = "facebook/nllb-200-distilled-600M"

    def __init__(self, device: str = "auto", batch_size: int = 32, max_length: int = 256,
                 cache_path: str = "data/cache/translations.sqlite", num_beams: int = 4,
                 backend: str = "auto"):
        from stance.utils import resolve_device

        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.cache = TranslationCache(cache_path)
        if backend == "auto":
            backend = "indictrans2" if hf_token_available() else "nllb"
        self.backend = backend
        self.proc = IndicProcessorPy() if backend == "indictrans2" else None
        self._models: dict[str, tuple] = {}

    def model_name(self, src: str) -> str:
        if self.backend == "nllb":
            return self.NLLB
        return self.EN_INDIC if src == "en" else self.INDIC_EN

    def _load(self, name: str):
        if name in self._models:
            return self._models[name]
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        remote = self.backend == "indictrans2"
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=remote)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=remote, dtype=dtype)
        model.to(self.device).eval()
        self._models[name] = (tok, model)
        return tok, model

    def unload(self) -> None:
        self._models.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def translate(self, texts: list[str], src: str, tgt: str, show_progress: bool = True) -> list[str]:
        """Translate a list of texts; src/tgt are ISO codes (en, hi, bn)."""
        import torch
        from tqdm import tqdm

        if src == tgt:
            return list(texts)
        name = self.model_name(src)
        src_fl, tgt_fl = FLORES[src], FLORES[tgt]
        keys = [self.cache.key(name, src, tgt, t) for t in texts]
        cached = self.cache.get_many(list(set(keys)))
        todo = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in cached]
        results = [cached.get(k, "") for k in keys]
        if todo:
            tok, model = self._load(name)
            # sort by length for efficient batching
            todo.sort(key=lambda it: len(it[1]))
            it = range(0, len(todo), self.batch_size)
            if show_progress:
                it = tqdm(it, desc=f"translate {src}->{tgt} [{self.backend}]", unit="batch")
            new: dict[str, str] = {}
            for b in it:
                chunk = todo[b : b + self.batch_size]
                raw = [t if t.strip() else "." for _, t in chunk]
                if self.backend == "indictrans2":
                    pre = self.proc.preprocess_batch(raw, src_fl, tgt_fl)
                    enc = tok(pre, truncation=True, padding="longest", max_length=self.max_length,
                              return_tensors="pt").to(self.device)
                    gen_kwargs = {}
                else:
                    tok.src_lang = src_fl
                    pre = [punc_norm(t) for t in raw]
                    enc = tok(pre, truncation=True, padding="longest", max_length=self.max_length,
                              return_tensors="pt").to(self.device)
                    gen_kwargs = {"forced_bos_token_id": tok.convert_tokens_to_ids(tgt_fl)}
                with torch.inference_mode():
                    gen = model.generate(**enc, max_length=self.max_length, num_beams=self.num_beams,
                                         num_return_sequences=1, use_cache=True, **gen_kwargs)
                dec = tok.batch_decode(gen.detach().cpu(), skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
                if self.backend == "indictrans2":
                    post = self.proc.postprocess_batch(dec, tgt_fl)
                else:
                    post = [d.strip() for d in dec]
                for (i, t), out in zip(chunk, post):
                    results[i] = out
                    new[keys[i]] = out
                if len(new) >= 256:
                    self.cache.put_many(new)
                    new = {}
            if new:
                self.cache.put_many(new)
        return results


# --------------------------------------------------------------------------- quality filter
class BackTranslationFilter:
    """LaBSE cosine similarity between the English source and the back-translation of the
    Indic output. Rows below `threshold` are dropped (reported in the datasheet)."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str = "auto",
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        from stance.utils import resolve_device

        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

    def similarity(self, a: list[str], b: list[str]) -> list[float]:
        import numpy as np

        ea = self.model.encode(a, batch_size=self.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        eb = self.model.encode(b, batch_size=self.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        return [float(x) for x in np.sum(ea * eb, axis=1)]


def translate_and_filter(df, translator: IndicTranslator, tgt_langs=("hi", "bn"),
                         threshold: float = 0.75, filter_device: str = "auto", log=print):
    """Translate every English row of `df` into tgt_langs; back-translate; filter by LaBSE.

    Returns (translated_df, stats) where translated_df has source_type == 'translated' and an
    extra column `bt_similarity`.
    """
    import pandas as pd

    en = df[df["language"] == "en"].reset_index(drop=True)
    stats = {"threshold": threshold}
    outs = []
    for tgt in tgt_langs:
        log(f"[translate] en->{tgt}: {len(en)} rows")
        fwd = translator.translate(en["text"].tolist(), "en", tgt)
        log(f"[translate] {tgt}->en back-translation")
        back = translator.translate(fwd, tgt, "en")
        d = en.copy()
        d["text_en"] = d["text"]
        d["text"] = fwd
        d["back_translation"] = back
        d["language"] = tgt
        d["source_type"] = "translated"
        outs.append(d)
    translator.unload()
    allt = pd.concat(outs, ignore_index=True)
    log(f"[translate] scoring {len(allt)} rows with LaBSE")
    scorer = BackTranslationFilter(device=filter_device)
    sims = []
    for i in range(0, len(allt), 512):
        sims.extend(scorer.similarity(allt["text_en"].iloc[i:i + 512].tolist(),
                                      allt["back_translation"].iloc[i:i + 512].tolist()))
    allt["bt_similarity"] = sims
    del scorer
    keep = allt["bt_similarity"] >= threshold
    stats["n_translated"] = int(len(allt))
    stats["n_kept"] = int(keep.sum())
    stats["kept_share"] = float(keep.mean())
    stats["per_lang_kept"] = {l: float(keep[allt["language"] == l].mean()) for l in tgt_langs}
    stats["sim_mean"] = float(allt["bt_similarity"].mean())
    log(f"[translate] kept {stats['n_kept']}/{stats['n_translated']} (threshold {threshold})")
    return allt[keep].reset_index(drop=True), stats, allt
