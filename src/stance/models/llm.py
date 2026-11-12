"""Phase 5: local instruction-tuned LLM (no API) for zero-/few-shot stance classification
and short rationales. Tier C+: Qwen/Qwen2.5-1.5B-Instruct in fp16 (~3.1 GB) on the 4 GB GPU.

a) zero-shot / few-shot prompting with a strict stance definition; few-shot examples are
   retrieved by kNN over cached sentence embeddings (LaBSE) of the labelled training data.
   Output is constrained by scoring the three label strings with the LM (log-likelihood
   of "Favour" / "Against" / "None" as the next tokens), so parsing never fails.
c) rationale(text) -> one short sentence explaining the predicted label (explainability aid,
   NOT ground truth).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from stance.utils import LABELS, resolve_device

SYSTEM = (
    "You are a careful annotator. Classify the STANCE of a social-media comment (Hindi, Bengali, "
    "Romanized or code-mixed) toward the claim: \"Climate change and global warming is a serious concern.\"\n"
    "Labels:\n"
    "Favour = agrees the claim, supports action, treats the threat as real, mocks deniers.\n"
    "Against = denies or minimises the concern: hoax, natural cycle, scam, agenda, no danger, mocks the worried.\n"
    "None = no stance: weather/news/questions/off-topic/praise of the video/unclear.\n"
    "Resolve negation and sarcasm to the intended meaning. Answer with exactly one word: Favour, Against or None."
)


class LocalLLMClassifier:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto", k_shots: int = 0,
                 embedder: str = "labse", max_new_tokens: int = 48, load_in_4bit: bool = False):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.k_shots = k_shots
        self.embedder_name = embedder
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.name = f"llm_{model_name.split('/')[-1]}_{k_shots}shot"
        self._loaded = False
        self.pool = None  # (texts, labels, embeddings) for kNN

    def _load(self):
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        kw = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)
        if not self.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        # token ids of each label (first token after a space-less start)
        self.label_token_ids = [self.tok(l, add_special_tokens=False)["input_ids"] for l in LABELS]
        self._loaded = True

    # ------------------------------------------------------------------ kNN few-shot pool
    def fit(self, texts, labels, languages=None):
        if self.k_shots > 0:
            from stance.models.dense import Embedder

            emb = Embedder(self.embedder_name)  # GPU is free here: the LLM is loaded lazily afterwards
            self.pool = (list(texts), list(labels), emb.encode(list(texts)))
            Embedder.unload()
        return self

    def _shots(self, text: str):
        if not self.pool or self.k_shots == 0:
            return []
        from stance.models.dense import Embedder

        # CPU: LaBSE next to the LLM would overflow the 4 GB GPU (query embeddings are cached on disk anyway)
        q = Embedder(self.embedder_name, device="cpu").encode([text])[0]
        sims = self.pool[2] @ q
        idx = np.argsort(-sims)[: self.k_shots]
        return [(self.pool[0][i], self.pool[1][i]) for i in idx]

    def _prompt(self, text: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}]
        for ex_t, ex_l in self._shots(text):
            msgs.append({"role": "user", "content": f"Comment: {ex_t}\nLabel:"})
            msgs.append({"role": "assistant", "content": ex_l})
        msgs.append({"role": "user", "content": f"Comment: {text}\nLabel:"})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ constrained scoring
    @torch.inference_mode()
    def predict_proba(self, texts, languages=None) -> np.ndarray:
        self._load()
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        for i, t in enumerate(texts):
            prompt = self._prompt(t if t.strip() else ".")
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)
            scores = []
            for lab_ids in self.label_token_ids:
                full = torch.cat([ids, torch.tensor([lab_ids], device=ids.device)], dim=1)
                logits = self.model(full).logits[0, ids.shape[1] - 1 : -1].float()
                lp = torch.log_softmax(logits, dim=-1)
                scores.append(sum(lp[j, tid].item() for j, tid in enumerate(lab_ids)) / len(lab_ids))
            s = np.array(scores)
            s = np.exp(s - s.max())
            out[i] = s / s.sum()
        return out

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]

    @torch.inference_mode()
    def rationale(self, text: str, label: str | None = None) -> str:
        self._load()
        label = label or self.predict([text])[0]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Comment: {text}\nThe label is {label}. In one short English sentence, say which words or cues support this label."}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)
        gen = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def benchmark_latency(self, texts, n: int = 20) -> float:
        t0 = time.perf_counter()
        self.predict(list(texts)[:n])
        return (time.perf_counter() - t0) * 1000 / max(1, min(n, len(texts)))
