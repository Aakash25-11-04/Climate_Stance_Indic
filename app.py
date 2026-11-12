
#!/usr/bin/env python
"""Gradio demo: stance of a Hindi / Bengali comment toward
"Climate change and global warming is a serious concern."

    python app.py                      # local, uses models/final (falls back to models/efficient)
    STANCE_MODEL_REPO=<user>/indic-climate-stance python app.py   # load weights from the Hub (Space)

Shows the predicted stance, class probabilities and highlighted influential tokens
(Integrated Gradients for the transformer, LIME as fallback / for non-transformer models).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from stance.utils import LABELS  # noqa: E402

CLAIM = "Climate change and global warming is a serious concern."
EXAMPLES = [
    ["जलवायु परिवर्तन एक गंभीर समस्या है, हमें अभी कदम उठाने चाहिए।"],
    ["ग्लोबल वार्मिंग सब झूठ है भाई, पैसा कमाने का धंधा है"],
    ["কাল ঢাকায় হালকা বৃষ্টি হবে বলে আবহাওয়া অফিস জানিয়েছে।"],
    ["হ্যাঁ হ্যাঁ, পৃথিবী গরম হচ্ছে, তাই ডিসেম্বরে এত ঠান্ডা 🙄"],
    ["climate change real hai yaar, delhi me april me hi 45 degree ho gaya"],
    ["ये कहना गलत है कि जलवायु परिवर्तन एक झूठ है।"],
]


def _load_model():
    repo = os.environ.get("STANCE_MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        from stance.models.transformer import TransformerStanceModel

        path = snapshot_download(repo)
        return TransformerStanceModel.load(path, device="auto"), "transformer"
    from predict import FinalModel

    for kind in ("final", "efficient"):
        try:
            fm = FinalModel(kind, device="auto")
            return fm, fm.spec["type"]
        except Exception as e:  # noqa: BLE001
            print(f"[app] could not load models/{kind}: {e}")
    raise SystemExit("No model found. Run run_all.ps1 (or scripts/build_final.py) first.")


MODEL, MODEL_TYPE = _load_model()


def _inner_transformer():
    m = getattr(MODEL, "model", MODEL)
    return m if hasattr(m, "tok") and hasattr(m, "model") else None


def classify(text: str, language: str, explain: str):
    from predict import guess_language

    text = (text or "").strip()
    if not text:
        return {l: 1 / 3 for l in LABELS}, "<i>Enter a comment.</i>", ""
    lang = language if language in ("hi", "bn") else guess_language(text)
    probs = np.asarray(MODEL.predict_proba([text], [lang]))[0]
    pred = LABELS[int(probs.argmax())]
    probs_d = {l: float(p) for l, p in zip(LABELS, probs)}
    html_out = ""
    try:
        from stance.explainability import integrated_gradients, lime_explanation, render_html

        tm = _inner_transformer()
        if explain == "Integrated Gradients" and tm is not None:
            expl = integrated_gradients(tm, text, lang)
        else:
            expl = lime_explanation(MODEL, text, lang, num_samples=200)
        html_out = render_html(expl, title="Token influence:")
    except Exception as e:  # noqa: BLE001
        html_out = f"<i>explanation unavailable: {e}</i>"
    info = f"language: **{lang}** &nbsp;|&nbsp; model type: **{MODEL_TYPE}** &nbsp;|&nbsp; prediction: **{pred}**"
    return probs_d, html_out, info


with gr.Blocks(title="Indic Climate Stance") as demo:
    gr.Markdown(f"## Indic Climate Stance Detection\nStance of a Hindi / Bengali (native, Romanized or code-mixed) comment toward the claim: **\"{CLAIM}\"**")
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="Comment", lines=4, placeholder="जलवायु परिवर्तन ... / জলবায়ু পরিবর্তন ... / climate change ...")
            with gr.Row():
                lang = gr.Radio(["auto", "hi", "bn"], value="auto", label="Language")
                expl = gr.Radio(["Integrated Gradients", "LIME"], value="Integrated Gradients", label="Explanation")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=2):
            out_label = gr.Label(label="Stance probabilities", num_top_classes=3)
            info = gr.Markdown()
    out_html = gr.HTML(label="Influential tokens")
    gr.Examples(EXAMPLES, inputs=[txt])
    btn.click(classify, inputs=[txt, lang, expl], outputs=[out_label, out_html, info])
    txt.submit(classify, inputs=[txt, lang, expl], outputs=[out_label, out_html, info])
    gr.Markdown("Labels: **Favour** = agrees the claim / supports action; **Against** = denies or minimises; "
                "**None** = neutral, unrelated or no clear stance. See the model card for limitations.")

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), share=False)
