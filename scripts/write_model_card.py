#!/usr/bin/env python
"""Generate models/MODEL_CARD.md (Hugging Face model card) from results/final.json,
results/baselines.csv, results/transformers.csv, results/robustness.csv and build_stats.json.

    python scripts/write_model_card.py [--repo <user>/indic-climate-stance]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stance.utils import abs_path  # noqa: E402


def _load(p):
    p = abs_path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="<your-user>/indic-climate-stance")
    args = ap.parse_args()
    final = _load("results/final.json")
    stats = _load("data/processed/build_stats.json")
    spec = final.get("final", {})
    gold = final.get("final_gold", {})
    ch = final.get("final_challenge", {})
    cands = pd.DataFrame(final.get("candidates", []))
    cost = pd.DataFrame(final.get("accuracy_vs_cost", []))
    rob = abs_path("results/robustness.csv")
    rob_tab = pd.read_csv(rob, index_col=0) if rob.exists() else pd.DataFrame()
    pc = gold.get("per_class", {})
    lines = [
        "---", "language: [hi, bn, en]", "license: mit", "tags: [stance-detection, climate-change, hindi, bengali, code-mixed, text-classification]",
        "base_model: google/muril-base-cased" if "muril" in json.dumps(spec) else "", "pipeline_tag: text-classification",
        "metrics: [f1]", "---", "",
        "# Indic Climate Stance (Hindi / Bengali)", "",
        "Stance of a Hindi or Bengali social-media comment (native script, Romanized or code-mixed) toward the claim",
        "**\"Climate change and global warming is a serious concern.\"** -> `Favour` / `Against` / `None`.", "",
        f"Final configuration: **{spec.get('candidate', 'n/a')}** (`{spec.get('type', '')}`), calibration: `{json.dumps(spec.get('calibration'))}`.",
        "", "## Intended use",
        "Research and educational analysis of public climate discourse in Hindi and Bengali (comment sections, surveys).",
        "Not intended for moderation decisions about individuals, for inferring personal attributes, or for languages other than Hindi/Bengali/English mixes.",
        "", "## Training data",
        "No official data existed; the training set was built from public English stance datasets (SemEval-2016 Task 6 climate target,",
        "GWSD, ClimateStance) translated to Hindi and Bengali with NLLB-200 (back-translation filtered by LaBSE cosine >= 0.75),",
        "Bengali headlines from Dhoroni, Romanized and code-mixed copies, and template-synthetic comments capped at 40 %.",
        f"Train rows: {stats.get('rows', {}).get('train', 'n/a')} (synthetic share {stats.get('synthetic_share', float('nan')):.2f}); dev rows: {stats.get('rows', {}).get('dev', 'n/a')}.",
        "See `docs/datasheet.md` in the repository for sources, licences and known biases.",
        "", "## Evaluation (hand-written gold set, test-only, 300+ comments, 14 categories)", "",
        f"| metric | value |", "|---|---|",
        f"| Macro F1 | {gold.get('macro_f1', float('nan')):.4f} |", f"| Accuracy | {gold.get('accuracy', float('nan')):.4f} |",
    ]
    for lab, d in pc.items():
        lines.append(f"| {lab} P / R / F1 | {d['precision']:.3f} / {d['recall']:.3f} / {d['f1']:.3f} (n={d['support']}) |")
    for lang, d in gold.get("per_language", {}).items():
        lines.append(f"| Macro F1 ({lang}) | {d['macro_f1']:.4f} (n={d['n']}) |")
    lines += ["", f"Challenge-suite failure rate (CheckList-style, {ch.get('n_tests', 'n/a')} tests): **{ch.get('overall_fail_rate', float('nan')):.3f}**", ""]
    if ch.get("per_category"):
        lines += ["| category | fail rate | n |", "|---|---|---|"]
        for c, d in sorted(ch["per_category"].items(), key=lambda kv: -kv[1]["fail_rate"]):
            lines.append(f"| {c} | {d['fail_rate']:.2f} | {d['n']} |")
    if len(cands):
        lines += ["", "## Candidates compared (final selection)", "", cands[["candidate", "dev_macro_f1", "gold_macro_f1", "challenge_fail_rate", "score"]].to_markdown(index=False, floatfmt=".4f")]
    if len(cost):
        lines += ["", "## Accuracy vs cost (CPU)", "", cost.to_markdown(index=False, floatfmt=".3f")]
    lines += ["", "## Limitations and biases",
              "* Training text is mostly machine-translated from English tweets/news; style differs from real Hindi/Bengali comments (see the synthetic-to-handcrafted gap analysis in the report).",
              "* The gold set and lexicon were written by the project author; scores on it are indicative, not a substitute for real annotated data.",
              "* Sarcasm, implicit stance and stance toward a related-but-different target remain the hardest categories (see the robustness table).",
              "* Only 1 seed per configuration could be afforded on a 4 GB GPU; variance across seeds is not reported for most runs.",
              "* The model may reflect political framings present in the English source data (US/UK climate debate).",
              "", "## How to use", "", "```python",
              "from stance.models.transformer import TransformerStanceModel",
              "m = TransformerStanceModel.load('models/final_model_dir')  # or snapshot_download('" + args.repo + "')",
              "m.predict(['जलवायु परिवर्तन एक गंभीर समस्या है'], ['hi'])", "```",
              "Or: `python scripts/predict.py --input comments.csv --output preds.csv --model final`.", "",
              "## Citation", "", "If you use this model or dataset pipeline, cite the repository and the source datasets listed in the datasheet.", ""]
    out = abs_path("models/MODEL_CARD.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
