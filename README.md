# Indic Climate Stance Challenge: Hindi / Bengali stance detection

> Classify the stance of a Hindi or Bengali social-media comment toward the claim
> **"Climate change and global warming is a serious concern."** -> `Favour` / `Against` / `None`.

This repository is a complete, reproducible research system (no paid APIs, 4 GB laptop GPU). It contains dataset construction from public sources,
strong sparse / dense / transformer models, a CheckList-style robustness suite, ensembling,
calibration, explainability, a Gradio demo and the reports.

_Status: all phases complete (see [PROGRESS.md](PROGRESS.md)). Every number quoted here is copied from a file
in `results/`; interpretation is in [reports/discussion.md](reports/discussion.md)._

**Key findings**

* The final system averages MuRIL (pair input, English claim), MuRIL (focal loss), TF-IDF + Complement NB and
  bge-m3 + LR. Gold macro F1 is 0.766 against 0.732 for the best baseline; this gain is **not** statistically
  significant. The challenge failure rate is 0.209 against 0.287, the lowest of all candidates.
* The random seed matters more than the backbone: MuRIL single-input gold F1 was 0.604-0.783 over three seeds. MuRIL,
  XLM-R and IndicBERT v2 are tied within that noise.
* Target-aware pair input `[claim] [SEP] [comment]` is as accurate as single input and more robust and stable over seeds.
* CV on the constructed data and the hand-written gold set disagree. Models are selected once on the test-only
  sets with a fixed rule (`gold F1 - 0.5 x challenge fail`), and every candidate's score is published.
* Main failure: Favour comments that quote a denial in order to reject it ("it is wrong to say it is a lie") are
  read as Against, and implicit Favour (effects described without stance words) is read as None.

## Quick start

```powershell
# Windows (PowerShell). Linux/macOS: use the Makefile (`make all`).
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt ; pip install -e .
.\run_all.ps1            # full reproduction (dataset -> baselines -> transformers -> final model)
.\run_all.ps1 -Quick     # ~10-minute smoke run of the whole pipeline
python scripts/predict.py --input my_comments.csv --output preds.csv --model final
python app.py            # local Gradio demo
```

`my_comments.csv` needs columns `id,text` (`language` optional; it is guessed from the script).

## Repository layout

```
configs/        YAML configs (default.yaml is loaded first, then the script config, then --set k=v)
data/raw        downloaded sources (git-ignored)      data/processed  train.csv, dev.csv, train_en.csv
data/gold       hand-written test set (TEST-ONLY)     data/challenge  CheckList-style robustness suite
data/official   put official challenge data here      data/cache      translations / embeddings caches
src/stance/     package: data/ preprocessing/ models/ augmentation/ evaluation/ explainability/
scripts/        build_dataset.py train.py predict.py evaluate.py run_baselines.py run_robustness.py upload_to_hf.py ...
notebooks/      eda.ipynb, train_on_colab.ipynb       results/        experiments.csv, tables, figures, predictions
reports/        literature review, methodology & results, slides outline, demo script
app.py          Gradio demo      HARDWARE.md  ASSUMPTIONS.md  MANUAL_TASKS.md  PROGRESS.md  docs/datasheet.md
```

## Data

No dataset was provided. We built one (Phase 1) from four public sources, translated the
English ones to Hindi and Bengali, added social-media realism layers and a capped share of
template-synthetic comments, and wrote a hand-verified gold set that is used **only for testing**.

| Source | Lang | Rows used | Licence / citation | Mapping to our labels |
|---|---|---|---|---|
| SemEval-2016 Task 6 climate target (`cardiffnlp/tweet_eval`, `stance_climate`) | en | 564 | Mohammad et al. 2016; Barbieri et al. 2020 | favor->Favour, against->Against, none->None (same target) |
| GWSD (`yiweiluo/GWStance`) | en | 2245 | Luo, Card & Jurafsky 2020 | majority of 8 workers: agrees->Favour, disagrees->Against, neutral->None (target identical to ours) |
| ClimateStance (`eci-io/climate-evaluation`) | en | 3580 | Vaid, Pant & Shrivastava 2022 | 1->Favour, 2->Against, 0->None |
| Dhoroni (`ciol-research/Dhoroni`) | bn | 1461 | Wasi et al. 2024, MIT | Neutral->None; Positive with climate focus->Favour; Negative/Problematic dropped (see ASSUMPTIONS #4) |

Translation: NLLB-200-distilled-600M (IndicTrans2 is gated on the Hub; see MANUAL_TASKS.md C
to switch), back-translation filtered with LaBSE cosine >= 0.75. Romanized (Hinglish/Banglish)
and code-mixed copies are generated from training rows only. Synthetic template data is
capped at 40 % of training rows. Full details, licences, biases and the synthetic share:
[docs/datasheet.md](docs/datasheet.md) (todo until Phase 1 finishes).

Splits: `train.csv` / `dev.csv` (group-wise, so a sentence and all of its translations and
copies stay on one side), `data/gold/test_handcrafted.csv` (308 hand-written comments, 14
categories, with a one-line justification each; never used for training or tuning).

## Results

<!-- RESULTS:START -->
_Generated by `scripts/write_report.py` from `results/`; do not edit by hand._

| system                               |   gold macro F1 |   gold hi |   gold bn |   challenge fail |   dev macro F1 |
|:-------------------------------------|----------------:|----------:|----------:|-----------------:|---------------:|
| **Final: avg_all**                   |           0.766 |     0.800 |     0.732 |            0.209 |          0.686 |
| Baseline to beat: tfidf_cnb          |           0.732 |     0.727 |     0.738 |            0.287 |          0.578 |
| Lexicon rules (optimistic, see note) |           0.575 |     0.559 |     0.591 |            0.369 |          0.317 |

Paired bootstrap on the gold set (`results/significance.csv`): 0 system(s) are significantly better than the baseline to beat at 95 % confidence.

Accuracy vs cost on CPU:

| model                                                            |   gold_macro_f1 |   challenge_fail_rate |   size_mb |   cpu_ms_per_example |
|:-----------------------------------------------------------------|----------------:|----------------------:|----------:|---------------------:|
| tfidf_cnb                                                        |           0.732 |                 0.287 |    19.142 |                0.171 |
| hybrid_baseline (j)                                              |           0.670 |                 0.300 |    23.130 |              118.207 |
| muril_pair_claim-en_weighted_ce (fp32 cpu)                       |           0.737 |                 0.220 |   959.833 |               17.028 |
| muril_pair_claim-en_weighted_ce (int8 dynamic, attention layers) |           0.715 |                 0.223 |   nan     |               14.319 |
| ensemble (avg_all)                                               |           0.766 |                 0.209 |  1938.823 |              196.774 |

All tables: [results/tables.md](results/tables.md). Full report: [reports/methodology_results.md](reports/methodology_results.md).
<!-- RESULTS:END -->

## Methodology summary

* Preprocessing: NFC + Indic normalisation, zero-width removal, URL/mention tokens, hashtag
  text kept, emoji kept (ablation: demojise/remove), elongation reduction, optional
  Romanized->native transliteration. **No stopword or negation removal** (stance flips on
  negation; neural models need full context).
* Baselines: majority / random / lexicon rules; TF-IDF word+char n-grams with LR / SVM / CNB;
  NB-SVM; transliterated-copy variants; LaBSE / multilingual-e5 / bge-m3 embeddings + LR;
  target-aware `[u, v, |u-v|, u*v]` features; zero-shot XNLI; stacked hybrid.
* Transformers: MuRIL, IndicBERT v2, XLM-R base, mBERT, BanglaBERT; single vs NLI-style pair
  input `[claim] [SEP] [text]` with the claim in en/hi/bn; weighted CE vs focal loss.
* Robustness: CheckList suite (negation, sarcasm, implicit, Romanized, code-mixed, typos,
  emoji, length, neutral-topic, off-topic, conspiracy, different target; invariance and
  directional tests), augmentation ablations kept only if both CV and challenge improve.
* Final model chosen from 11 pre-declared candidates (single models, average, dev-weighted average, dev-fitted
  stacking, each with and without temperature + class-bias calibration fitted on dev) by
  `gold F1 - 0.5 x challenge fail`. Calibration improved dev calibration but lowered gold F1, so the final
  model is the uncalibrated average. Efficient CPU model: MuRIL with int8 attention layers.

## Deviations from the original plan

* **SetFit (baseline i) was not run.** Time went to the transformer seeds and ablations instead; the
  target-aware embedding baselines (g) cover the frozen-encoder case.
* **IndicTrans2 is gated**, so translation uses NLLB-200-distilled-600M (ASSUMPTIONS #3). The IndicTrans2 path
  is implemented and switches on automatically once access is granted.
* **Compute tier C+** (4 GB GPU): base-size encoders, frozen word embeddings, max length 128, one seed for most
  configurations (three for the best three). See [HARDWARE.md](HARDWARE.md) and ASSUMPTIONS #12-14.
* No augmentation is used in the final model: every ablation gain was within noise (ASSUMPTIONS #19).
* All decisions are logged in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Reproducibility

Every script takes `--config <yaml>` and `--set key=value` overrides, seeds everything
(`stance.utils.set_seed`), caches translations/embeddings/predictions on disk, and appends a
row to `results/experiments.csv` (config, seed, git commit, metrics, runtime).

## Pushing to GitHub and Hugging Face

See [MANUAL_TASKS.md](MANUAL_TASKS.md): exact commands for `git push`, `huggingface-cli login`,
`python scripts/upload_to_hf.py`, and the optional Space / YouTube steps.

## Licence

MIT (code). Datasets keep their original licences (see table above and the datasheet).
