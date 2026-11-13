"""Efficient CPU inference (Phase 8): dynamic int8 quantisation of a fine-tuned transformer.

QuantizedStanceModel.load(path) loads the saved fp32 transformer, quantises the self-attention nn.Linear layers to
int8 with torch.ao.quantization.quantize_dynamic (CPU only), and exposes the usual
predict_proba(texts, languages). Nothing extra is stored on disk: MODEL.json just records
{"type": "quantized", "path": <transformer dir>}. ONNX export via optimum is attempted by
export_onnx() but is optional; if it fails the quantised torch path is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stance.models.transformer import TransformerStanceModel
from stance.utils import LABELS, abs_path


class QuantizedStanceModel:
    def __init__(self, base: TransformerStanceModel):
        self.base = base
        self.base.device = "cpu"
        self.base.fp16 = False
        self.base.model.to("cpu").eval()
        # Only the self-attention projections are quantised. Quantising every nn.Linear collapsed
        # MuRIL (gold macro F1 0.74 -> 0.33): the feed-forward output projections receive large
        # activation outliers that per-tensor int8 activations cannot represent (Bondarenko et al., 2021).
        # Attention-only int8 kept probabilities within 0.02 of fp32 on a gold sample.
        names = {n for n, m in self.base.model.named_modules() if isinstance(m, torch.nn.Linear) and ".attention." in n}
        self.base.model = torch.ao.quantization.quantize_dynamic(self.base.model, names, dtype=torch.qint8)
        self.name = f"{base.name}_int8"

    @classmethod
    def load(cls, path: str | Path):
        return cls(TransformerStanceModel.load(path, device="cpu"))

    def predict_proba(self, texts, languages=None, batch_size: int = 16) -> np.ndarray:
        return self.base.predict_proba(texts, languages, batch_size=batch_size)

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]


def export_onnx(model_dir: str | Path, out_dir: str | Path) -> Path | None:
    """Optional ONNX export (optimum). Returns the output dir or None if export is unavailable."""
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        out = abs_path(out_dir)
        ort = ORTModelForSequenceClassification.from_pretrained(str(abs_path(model_dir)), export=True)
        ort.save_pretrained(out)
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(str(abs_path(model_dir))).save_pretrained(out)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[efficient] ONNX export skipped: {type(e).__name__}: {e}")
        return None


class OnnxStanceModel:
    """ONNX Runtime inference for an exported model dir (uses the same encoding as the torch model)."""

    def __init__(self, ort_model, base_meta_dir: str | Path):
        self.ort = ort_model
        self.base = TransformerStanceModel.load(base_meta_dir, device="cpu")
        self.name = f"{self.base.name}_onnx"

    @classmethod
    def load(cls, path: str | Path):
        from optimum.onnxruntime import ORTModelForSequenceClassification

        p = abs_path(path)
        return cls(ORTModelForSequenceClassification.from_pretrained(str(p)), p.parent / p.name.replace("_onnx", ""))

    def predict_proba(self, texts, languages=None, batch_size: int = 16):
        out = []
        for b in range(0, len(texts), batch_size):
            enc = self.base._encode(texts[b : b + batch_size], (languages or [None] * len(texts))[b : b + batch_size])
            logits = self.ort(**enc).logits
            out.append(torch.softmax(torch.as_tensor(logits), dim=-1).numpy())
        return np.vstack(out) if out else np.zeros((0, 3))

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]
