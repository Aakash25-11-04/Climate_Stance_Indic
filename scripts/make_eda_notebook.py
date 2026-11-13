#!/usr/bin/env python
"""Create notebooks/eda.ipynb (Phase 2) from scripts/eda.py outputs and execute it with nbconvert.

    python scripts/make_eda_notebook.py [--execute]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

MD_INTRO = """# Phase 2: Exploratory data analysis

Dataset built by `scripts/build_dataset.py` (see `docs/datasheet.md`). This notebook re-runs the
analysis functions in `scripts/eda.py`, shows the saved figures and records the manual label audit.
Every number printed below is computed live from `data/processed/*.csv`."""

MD_AUDIT = """## Manual label-noise audit (50 random train rows, seed 42)

Inspected by the author (file `results/eda_label_audit_sample.csv`). Findings on the first build:

* **~8 / 50 rows (about 15 %) have questionable labels**, almost all inherited from the English source labels
  rather than created by translation: spam/advert tweets labelled *Favour* in ClimateStance, a sarcastic
  tweet ("I died in the great ozone cut of 1974") labelled *Favour*, a "brainwashing / mass hysteria" tweet
  labelled *None* (it is *Against*), a Dhoroni tourism headline labelled *Favour*.
* Translation quality itself was mostly adequate for stance (meaning preserved), but long tweets with
  hashtags/usernames produce clumsy sentences ("re के डी शांति सेम एसटी" = leftover handle fragments).
* Two generator bugs surfaced and were fixed before the final build: candra-vowel characters leaking into
  Romanized text ("rikaॉrd"), and code-mix substitution inside words ("peopleों").
* Synthetic template rows are grammatical but repetitive; a few slot fills are awkward
  ("ग्लेशियर पिघलना आती थी").

Implication: label noise of roughly 10-15 % bounds the achievable CV macro F1 on `translated` rows and
makes the **hand-written gold set and the challenge suite the decisive evaluations**, not CV."""

MD_INSIGHTS = """## Insights that change our modelling decisions

1. **Distribution shift is the main risk, not data size.** Train text (median ~107 chars, 58 % machine-translated
   from English tweets/news) is longer, more formal and more "Western" than the gold set (median ~62 chars, written
   as Indian/Bangladeshi comments). -> Select and tune models on gold + challenge results, use dev only for early stopping.
2. **Against is rare in train (11.5 %) but ~1/3 of the hidden test is plausibly Against.** -> class-weighted loss /
   focal loss, per-class threshold tuning on held-out data, and the Against-heavy conspiracy/sarcasm challenge categories.
3. **Scripts**: 18 % of train is Latin-script and 8 % mixed-script (all synthetic Romanization / code-mixing).
   Real Romanized text has more spelling variation than our rules produce -> character n-grams for sparse models,
   MuRIL (pre-trained on transliterated text) first, typo/matra-drop augmentations in Phase 6.
4. **Lexicon agrees with only 40 % of train labels** but reaches a much higher score on gold: translated news-style
   text carries stance implicitly; comments carry it in explicit cue words. A model must do both -> keep the
   target-aware pair input (explicit claim context) and ensemble a cue-sensitive sparse model with the transformer.
5. **No leakage**: 0 exact / near duplicates between train and dev, gold or challenge after group-wise splitting.
6. **Label noise ~15 % in translated rows** (audit above) -> report CV with error bars, do not chase CV decimals,
   and prefer augmentations that help the challenge suite."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    nb = nbf.v4.new_notebook()
    c = [nbf.v4.new_markdown_cell(MD_INTRO),
         nbf.v4.new_code_cell("""import sys, json, os
sys.path.insert(0, os.path.abspath('../src')); sys.path.insert(0, os.path.abspath('../scripts'))
os.chdir('..')
import pandas as pd
from IPython.display import Image, display
from stance.utils import read_data
import eda
train, dev, gold = read_data('data/processed/train.csv'), read_data('data/processed/dev.csv'), read_data('data/gold/test_handcrafted.csv')
stats = json.load(open('results/eda_stats.json', encoding='utf-8'))
print({k: stats[k] for k in ['rows', 'train_len_median', 'gold_len_median', 'train_codemix_mean', 'gold_codemix_mean']})"""),
         nbf.v4.new_markdown_cell("## Label distribution: overall, per language, per source_type"),
         nbf.v4.new_code_cell("""display(pd.crosstab(train.language, train.label, margins=True))
display(pd.crosstab(train.source_type, train.label, margins=True))
display(pd.crosstab(train.source, train.label))
for f in ['label_distribution.png', 'label_by_language.png', 'label_by_source_type.png']: display(Image(f'results/figures/{f}'))"""),
         nbf.v4.new_markdown_cell("## Script distribution (Devanagari / Bengali / Latin / mixed)"),
         nbf.v4.new_code_cell("""display(pd.crosstab(train.script, train.source_type)); print('gold:', gold.script.value_counts().to_dict())
display(Image('results/figures/script_distribution.png'))"""),
         nbf.v4.new_markdown_cell("## Text length and code-mixing ratio"),
         nbf.v4.new_code_cell("""from stance.preprocessing.normalize import code_mix_ratio
print(train.text.str.len().describe().round(1)); print(gold.text.str.len().describe().round(1))
print('mean code-mix ratio per source_type:'); print(train.assign(cm=train.text.map(code_mix_ratio)).groupby('source_type').cm.mean().round(3))
display(Image('results/figures/length_and_codemix.png')); display(Image('results/figures/source_type_vs_gold.png'))"""),
         nbf.v4.new_markdown_cell("## Top n-grams per class and language"),
         nbf.v4.new_code_cell("""for k, v in stats['top_ngrams'].items(): print(k, [w for w, _ in v[:10]])
display(Image('results/figures/top_ngrams.png'))"""),
         nbf.v4.new_markdown_cell("## Emoji, URL, hashtag, mention frequency; Romanized share"),
         nbf.v4.new_code_cell("""display(pd.DataFrame({'train': stats['surface_train'], 'gold': stats['surface_gold']}).round(3)); display(Image('results/figures/surface_features.png'))"""),
         nbf.v4.new_markdown_cell("## Duplicates and leakage across splits"),
         nbf.v4.new_code_cell("""print(json.dumps(stats['leakage'], indent=1)); print(json.load(open('data/processed/build_stats.json', encoding='utf-8'))['dedup'])"""),
         nbf.v4.new_markdown_cell("## Hard / ambiguous examples (lexicon disagrees with the label on translated rows)"),
         nbf.v4.new_code_cell("""pd.set_option('display.max_colwidth', 160); display(read_data('results/eda_hard_examples.csv').head(20))"""),
         nbf.v4.new_markdown_cell(MD_AUDIT),
         nbf.v4.new_code_cell("""display(read_data('results/eda_label_audit_sample.csv'))"""),
         nbf.v4.new_markdown_cell(MD_INSIGHTS)]
    nb["cells"] = c
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    out = ROOT / "notebooks" / "eda.ipynb"
    nbf.write(nb, out)
    print(f"-> {out}")
    if args.execute:
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                               "--ExecutePreprocessor.timeout=600", "--ExecutePreprocessor.kernel_name=stance-venv", str(out)], cwd=ROOT / "notebooks")


if __name__ == "__main__":
    main()
