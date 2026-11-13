#!/usr/bin/env python
"""Assemble a ready-to-push Hugging Face Space folder in space/ (does NOT upload anything).

    python scripts/prepare_space.py --model_repo <your-user>/indic-climate-stance

space/ contains app.py, the stance package (src/stance), scripts/predict.py (language guessing),
requirements.txt (from requirements-space.txt) and a README.md with the Space metadata header.
The Space downloads the transformer weights from the model repo at start-up
(environment variable STANCE_MODEL_REPO, written into the README header and app defaults).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_repo", required=True)
    ap.add_argument("--out", default="space")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "stance", out / "src" / "stance", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "scripts" / "predict.py", out / "scripts" / "predict.py")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    app = app.replace('repo = os.environ.get("STANCE_MODEL_REPO")', f'repo = os.environ.get("STANCE_MODEL_REPO", "{args.model_repo}")')
    app = app.replace('demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), share=False)',
                      'demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), share=False)')
    (out / "app.py").write_text(app, encoding="utf-8")
    shutil.copy(ROOT / "requirements-space.txt", out / "requirements.txt")
    (out / "README.md").write_text(f"""---
title: Indic Climate Stance
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

Stance of Hindi / Bengali comments toward "Climate change and global warming is a serious concern".
Model: https://huggingface.co/{args.model_repo}
""", encoding="utf-8")
    print(f"-> {out}  (push it as a Gradio Space, see MANUAL_TASKS.md step 7)")


if __name__ == "__main__":
    main()
