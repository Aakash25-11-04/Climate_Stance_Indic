# Methodology

This section describes what was built and why. It contains design parameters only; every
measured number is in the results section, which is generated from `results/` by
`scripts/make_tables.py` and `scripts/write_report.py`.

## 1. Problem framing

Three-way stance classification of a Hindi or Bengali comment toward one fixed claim,
"Climate change and global warming is a serious concern" (Favour / Against / None), scored by
macro F1. The hidden test set is noisy: Romanized and code-mixed text, spelling variation,
sarcasm, negation, implicit stance and off-topic comments. Two consequences shaped every decision:

1. **There is no in-domain training data.** All training text is either translated from English
   stance datasets, Bengali news headlines, or generated. The training distribution therefore
   differs from the test distribution, and validation scores on translated text are a weak proxy
   for the hidden test.
2. **Robustness matters more than in-distribution accuracy.** We built two test-only resources that
   look like the hidden test: a hand-written gold set (306 comments, 14 phenomena) and a
   CheckList-style challenge suite (1,084 tests). Model selection uses them as the final arbiter,
   while all fitting (vectorisers, hyper-parameters, early stopping, calibration, stacking) uses
   training folds or the dev split only.

## 2. Data construction (Phase 1)

See `docs/datasheet.md` for full detail.

* **Sources.** SemEval-2016 Task 6 climate target, GWSD and ClimateStance (English); Dhoroni
  (Bengali news). Label mappings are justified in `src/stance/data/loaders.py`; Dhoroni's
  "Negative" (negative environmental news) is *not* mapped to Against.
* **Translation.** English -> Hindi and Bengali with NLLB-200-distilled-600M (IndicTrans2 is gated;
  the pipeline switches to it automatically when a Hub token exists). Each translation is
  back-translated and kept only if the LaBSE cosine between source and back-translation is at
  least 0.75. Bengali Dhoroni headlines are additionally translated to Hindi.
* **Social-media realism (train only).** A rule-based Romanizer (ITRANS plus Hindi schwa
  deletion, Bengali inherent-o, and random casual spelling variants) and a code-mixer (whole-word
  English substitution from a hand-written lexicon, English discourse markers, partial
  Romanization) create Hinglish/Banglish copies of training rows.
* **Template synthesis.** Hand-written Hindi and Bengali templates for each label and ten styles
  (formal, slang, sarcastic, implicit, negated, Romanized, code-mixed, emoji, short, long),
  capped at 40 % of training data. The compute tier allowed a 1.5B local LLM, but template
  generation was kept as the default so that every synthetic label is correct by construction;
  the LLM is used for classification and rationales instead (Phase 5).
* **Leakage control.** Exact and near-duplicate removal (MinHash over character 4-grams, Jaccard
  >= 0.85, within language) runs over all rows *including the gold set* before splitting. Splits are
  group-wise: a source sentence and all its translations and copies stay in one split. Train rows
  that near-duplicate any challenge test are removed.

## 3. Preprocessing (Phase 3)

Unicode NFC, Indic normalisation (indic-nlp-library), zero-width character removal, URL and
mention placeholders, hashtag text kept, elongation and punctuation-repeat reduction, Latin
lowercasing. Emoji are kept as tokens (an ablation flag allows demojising or removal), and a
flag transliterates Romanized text back to native script. **Negation words and stopwords are never
removed**: stance often flips on a single negator ("नहीं", "না", "nahi"), and transformers need
the full context.

## 4. Models

**Tier 1 reference points.** Majority class, stratified random, and a hand-built lexicon with
negation handling (Hindi, Bengali and Romanized cue lists). The lexicon was written by the same
author as the gold set, so its gold score is optimistic.

**Tier 2 sparse models.** TF-IDF word 1-2-grams plus char_wb 2-5-grams (sublinear tf), with Logistic
Regression, calibrated Linear SVM and Complement Naive Bayes; NB-SVM (Wang and Manning 2012);
and a transliterated-copy variant where every text is concatenated with its Romanized or native
copy, so both spellings share features. Class weights are balanced and hyper-parameters are chosen by
an inner 3-fold CV inside each outer fold.

**Tier 3 dense models.** Frozen LaBSE, multilingual-E5-base and BGE-M3 embeddings with Logistic
Regression; a target-aware variant using `[u, v, |u-v|, u*v]` features, where `v` is the claim
embedding in the comment's language; and zero-shot NLI with a multilingual mDeBERTa checkpoint,
with hypotheses written in Hindi and Bengali.

**Tier 4 hybrid.** Stacking of the best sparse and the best dense model on out-of-fold
probabilities with a Logistic Regression meta-learner.

**Transformers (Phase 4).** MuRIL (pre-trained on transliterated Indic text), IndicBERT v2, XLM-R
base, mBERT, BanglaBERT (Bengali only), and the fine-tuned multilingual NLI checkpoint. Two
input formulations:

* single: `[CLS] comment [SEP]`;
* **pair (target-aware, NLI-style): `[CLS] claim [SEP] comment [SEP]`**, with the claim in English,
  Hindi, Bengali, or the comment's own language. This tests whether stating the target explicitly
  helps, and in which language.

Training: weighted cross-entropy (inverse class frequency to the power 0.5) or focal loss,
AdamW, linear warm-up, fp16, max length 128, batch 16, early stopping on dev macro F1. On the
4 GB GPU the word-embedding matrix is frozen (MuRIL has a 197k vocabulary), which keeps the run in
VRAM (ASSUMPTIONS #12). Three seeds are run for the strongest configurations; one seed elsewhere.

**Local LLM (Phase 5).** Qwen2.5-1.5B-Instruct in fp16. Classification uses constrained scoring:
the log-likelihood of each label word after a stance-definition prompt, so output parsing cannot
fail. Few-shot examples are retrieved by LaBSE kNN from labelled training data. The LLM also
writes one-sentence rationales, an explainability aid that is not ground truth. QLoRA was not run
because it does not fit in 4 GB.

## 5. Augmentation ablations (Phase 6)

One change at a time against the current training set, on both the sparse baseline to beat (5-fold CV,
gold, challenge) and MuRIL (dev, gold, challenge): remove synthetic, Romanized or code-mixed
rows; add character noise, dropped vowel signs (matras), repeated characters, back-translation
paraphrases, hand-verified negation-flip contrast pairs, English source data (cross-lingual), or
three times more template data. An augmentation is **kept only if it improves both CV macro F1 and
the challenge-suite failure rate**. Negation flips: 64 automatic candidates were reviewed by hand
and only 14 were kept (ASSUMPTIONS #15), because in most of the others the rest of the comment still
argued the original side.

## 6. Robustness challenge suite (Phase 7)

Following CheckList (Ribeiro et al. 2020): minimum-functionality tests in 13 categories x 2
languages with at least 30 each (negation, sarcasm, implicit stance, Romanized, code-mixed, typos,
emoji cues, very short, very long, topic-related neutral, off-topic, conspiracy framing,
different target); invariance tests (typos, emoji, URLs, mentions, elongation, whitespace,
partial Romanization, punctuation must not change the label); and directional tests where negation
must flip the label. The suite is generated from hand-written seeds with templates that differ
from the training templates, and it is never used for fitting.

## 7. Ensembles, calibration and efficiency (Phase 8)

Candidates: the top transformers by dev F1, the best sparse and dense baselines, and the stacked
hybrid. Combination by equal averaging, dev-tuned weighted averaging, and stacking. Transformers have
no out-of-fold predictions on this hardware, so the meta-learner, temperature scaling and
per-class log-space biases (equivalent to per-class thresholds) are fitted on the **dev** split,
which was used only for early stopping. The final model is chosen by
`gold macro F1 - 0.5 x challenge failure rate`; every candidate's score is reported. An efficient
option is provided (dynamically quantised int8 transformer, or the hybrid sparse+dense model),
with a CPU accuracy-vs-cost table.

## 8. Explainability and error analysis (Phase 9)

Integrated Gradients (Captum, `[PAD]` baseline, claim segment masked from the display) and LIME for
the transformer; the largest TF-IDF coefficients per class for the sparse model; 10 correct and 10
wrong gold examples per language are rendered as HTML. Errors on gold and dev (more than 100) are tagged
with rule-based categories (sarcasm, implicit, negation, Romanized, code-mixed, short, long,
translation artefact, synthetic artefact, plus the gold category). The analysis compares, per category,
which errors the final model fixes or introduces relative to the baseline to beat.

## 9. Evaluation protocol and statistics

Primary metric: macro F1. We also report per-class precision, recall and F1, accuracy, confusion
matrices and per-language scores. CV is stratified 5-fold by label x language with group-aware
folds. Differences on the 306-item gold set are tested with a paired bootstrap (2,000 resamples; 95 %
CI of the difference against the baseline to beat), because single-seed differences of a few points
are within noise at this sample size.

## 10. Reproducibility

Every script takes a YAML config plus `--set key=value` overrides, seeds Python, NumPy and torch, caches
translations, embeddings and predictions on disk, and appends a row to `results/experiments.csv` (config,
seed, git commit, metrics, runtime). `run_all.ps1` (Windows) and the `Makefile` reproduce the
pipeline; `scripts/job_queue.py` ran the long jobs sequentially on the 4 GB GPU.
