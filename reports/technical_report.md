---
title: "Target-Aware, Robustness-Selected Stance Detection for Hindi and Bengali Climate Discourse Without In-Domain Training Data"
subtitle: "Technical report: Indic Climate Stance Challenge"
author: "Aakash Rauniar"
date: "November 2026"
---

# Abstract

We describe a system for three-way stance detection (Favour / Against / None) of Hindi and Bengali
social-media comments toward the claim *"Climate change and global warming is a serious concern."*
No in-domain training data exists, so we build a 19,278-row training set from four public
English and Bengali sources. English text is machine-translated with NLLB-200 and kept only when a LaBSE
back-translation check passes. We add rule-based Romanized and code-mixed copies and template
comments, and remove duplicates and leakage across splits. Because validation on constructed data is a poor proxy
for real comments, we evaluate on two test-only resources built for this task: a hand-written gold set
of 306 comments in 14 linguistic categories, and a CheckList-style suite of 1,084 behavioural tests.
We compare 16 classical and embedding baselines, 18 fine-tuned transformer runs (six encoders,
single vs target-aware pair input, two losses, up to three seeds), a local 1.5B-parameter LLM and 11
augmentation ablations. The final system is an equal-weight average of two MuRIL models, a TF-IDF
Complement Naive Bayes model and a BGE-M3 embedding classifier. It reaches macro F1 0.766 on the gold
set (Hindi 0.800, Bengali 0.732) against 0.732 for the strongest baseline. That gain is not statistically
significant. The challenge-suite failure rate falls from 0.287 to 0.209. Our main empirical findings are that seed variance
(gold F1 0.60–0.78 for one configuration) exceeds the differences between backbones, that target-aware
pair input is at least as accurate as single input and more robust and stable, and that cross-validation on constructed data
ranks models differently from hand-written comments. The dominant remaining errors are negated denials,
sarcasm and implicit stance.

# 1. Introduction

Stance detection asks whether a text supports, opposes or is neutral toward a given target
(Mohammad et al., 2016). Public opinion on climate change in India and Bangladesh is expressed on
social media in Hindi and Bengali. This text is often Romanized ("Hinglish", "Banglish"), code-mixed with
English, informally spelled, and frequently sarcastic. The Indic Climate Stance Challenge asks for a
classifier for this setting, scored by macro F1 over three labels, but provides no training data.

The task has three difficulties:

1. **No in-domain labels.** Labelled climate-stance data exists in English (SemEval-2016 Task 6, GWSD,
   ClimateStance). For Bengali there is only news data with a different stance target (Dhoroni).
2. **Distribution shift.** The hidden test contains Romanized and code-mixed comments, typos,
   emoji, negation and sarcasm. These are rare or absent in translated news and tweets.
3. **Hardware.** All work runs on one laptop with a 4 GB GPU and no paid APIs.

Our contributions are:

* A documented, leakage-controlled pipeline that turns English stance data into Hindi and Bengali training
  data, with filtered translation, Romanization, code-mixing and capped template synthesis (Section 3).
* Two test-only evaluation resources that resemble the hidden test: a 306-comment gold set in 14
  categories and a 1,084-test CheckList-style suite (Section 3.4).
* A systematic comparison of sparse, dense, transformer and LLM approaches, including a target-aware
  pair-input formulation with the claim in four language settings and multi-seed runs (Sections 4–6).
* A transparent model-selection rule that weighs robustness (`gold F1 − 0.5 × challenge failure rate`),
  with every candidate's score published, and statistical testing of the final gain (Section 6).

# 2. Related work

**Stance detection.** SemEval-2016 Task 6 (Mohammad et al., 2016) introduced tweet stance toward
targets including "Climate Change is a Real Concern". TweetEval (Barbieri et al., 2020) redistributes
it. Mohammad, Sobhani and Kiritchenko (2017) show that stance and sentiment differ. GWSD (Luo, Card and Jurafsky,
2020) labels news sentences toward exactly our claim, and ClimateStance (Vaid, Pant and Shrivastava, 2022)
provides COP25-era tweets. Dhoroni (Wasi et al., 2024) is a Bengali environmental news dataset whose
stance label refers to the environment in general. Küçük and Can (2020) and ALDayel and Magdy (2021) survey
the field.

**Target-aware and cross-lingual stance.** Conditional encoding of the target (Augenstein et al., 2016)
and target-specific attention (Du et al., 2017) predate transformers. Encoding target and text as a
sentence pair is now standard for zero-shot and cross-target stance (Allaway and McKeown, 2020).
Hardalov et al. (2022) study few-shot cross-lingual stance. Zero-shot classification by entailment
(Yin, Hay and Roth, 2019), with multilingual NLI checkpoints (Laurer et al., 2024), is our zero-shot baseline.

**Indic and multilingual encoders.** MuRIL (Khanuja et al., 2021) is pre-trained on 17 Indian languages
including transliterated text. IndicBERT v2 (Doddapaneni et al., 2023) and BanglaBERT (Bhattacharjee et al.,
2022) are Indic- and Bengali-specific. XLM-R (Conneau et al., 2020) and mBERT (Devlin et al., 2019) are general
multilingual models. LaBSE (Feng et al., 2022), multilingual E5 (Wang et al., 2024) and BGE-M3
(Chen et al., 2024) provide sentence embeddings. Translation uses NLLB-200 (NLLB Team, 2022), because the
IndicTrans2 (Gala et al., 2023) checkpoints are gated.

**Robustness and augmentation.** CheckList (Ribeiro et al., 2020) defines minimum-functionality,
invariance and directional tests. Contrast sets (Gardner et al., 2020) and back-translation (Sennrich, Haddow
and Birch, 2016) motivate our augmentation ablations. GLUECoS (Khanuja et al., 2020) shows that
transliteration-aware pre-training matters for code-mixed text.

# 3. Data

## 3.1 Sources and label mapping

| Source | Language | Rows loaded | Mapping to Favour / Against / None |
|---|---|---|---|
| SemEval-2016 T6 climate target (via TweetEval) | en | 564 | favor / against / none |
| GWSD (Luo et al., 2020) | en | 2,245 | majority of 8 annotators (share ≥ 0.5) |
| ClimateStance (Vaid et al., 2022) | en | 3,580 | favor / against / ambiguous → None |
| Dhoroni (Wasi et al., 2024) | bn | 1,461 | Neutral → None; Positive with a climate focus → Favour; Negative/Problematic dropped |

*Table 1. Labelled sources.* Dhoroni's "Negative" label marks bad environmental news, not denial, so it
is not mapped to Against. Against is rare in every source (2,225 of 19,278 training rows, 11.5 %).

## 3.2 Construction pipeline

1. **Translation.** English sentences are translated into Hindi and Bengali with
   NLLB-200-distilled-600M. Each output is back-translated into English, and the pair is kept only if the LaBSE cosine
   similarity between source and back-translation is at least 0.75. 11,888 of 12,778 translations passed (93.0 %; Hindi 95.0 %, Bengali
   91.1 %). Dhoroni Bengali headlines are also translated into Hindi (1,347 of 1,461 kept).
2. **Romanization.** A rule-based ITRANS Romanizer with Hindi schwa deletion, Bengali inherent-*o*
   and random casual spelling variants creates Hinglish and Banglish copies.
3. **Code-mixing.** Whole-word English substitution from a hand-written lexicon, English discourse
   markers and partial Romanization create code-mixed copies.
4. **Templates.** Hand-written Hindi and Bengali templates for each of 3 labels × 10 styles (formal,
   slang, sarcastic, implicit, negated, Romanized, code-mixed, emoji, short, long), capped at 40 % of training
   data (actual share 8.9 %).
5. **Leakage control.** MinHash LSH removes exact and near duplicates (character 4-grams, Jaccard ≥ 0.85)
   within each language, across all rows *including the gold set*, before splitting. Splits are group-wise: a source
   sentence and all its translations and copies stay in one split. Training rows that near-duplicate
   any challenge test are removed (3 rows).

## 3.3 Resulting splits

| Split | Rows | Hindi | Bengali | Favour | Against | None |
|---|---|---|---|---|---|---|
| train | 19,278 | 9,788 | 9,490 | 10,982 | 2,225 | 6,071 |
| dev | 1,402 | 712 | 690 | 825 | 130 | 447 |
| gold (test-only) | 306 | 153 | 153 | 110 | 104 | 92 |

*Table 2. Splits.* By origin, training rows are translated (11,372), transliterated (3,165), synthetic
(1,723), code-mixed (1,720) and original Bengali (1,298). By script they are Devanagari 7,279, Bengali 7,088,
Latin 3,449 and mixed 1,462.

**Data quality.** A manual audit of 50 translated rows found about 8 questionable labels (≈15 %),
inherited from the sources or introduced by translation. Translated training text is about twice as long as
the gold comments (median 107 vs 62 characters).

## 3.4 Test-only evaluation resources

**Gold set.** 306 comments (153 per language) were written by hand before any model was trained. They cover 14
categories: explicit, implicit, negation, sarcasm, Romanized, code-mixed, typos, emoji, short,
long, topic-related neutral, off-topic, conspiracy framing and different target. The labels are near
balanced. The set was deduplicated against training data (0 duplicates). Because the gold set and the
templates share an author, the gold set approximates the hidden test but is not a sample of it.

**Challenge suite.** Following CheckList, it has 780 minimum-functionality tests (13 categories × 2 languages), 240
invariance tests (typos, emoji, URLs, mentions, elongation, whitespace, partial Romanization and
punctuation must not change the label) and 64 directional tests (negating a statement must flip the label). It is generated from
hand-written seeds with templates distinct from the training templates.

**Protocol.** All fitting (vectorisers, hyper-parameters, early stopping, ensemble weights,
calibration, stacking) uses training folds or the dev split. The gold set and challenge suite are used only to compare
a small, pre-declared set of candidates, and every candidate's score is published.

# 4. Methods

## 4.1 Preprocessing

The pipeline applies Unicode NFC, Indic normalisation (Indic NLP Library), zero-width removal, URL and mention placeholders,
elongation reduction and Latin lowercasing. Emoji and hashtag text are kept. **Negators and stopwords are never
removed**, because a single "नहीं" / "না" / "nahi" flips stance.

## 4.2 Baselines

* **Reference points:** majority class, stratified random, and a hand-written cue lexicon with negation
  handling.
* **Sparse:** TF-IDF on word 1–2-grams plus char_wb 2–5-grams with Logistic Regression, calibrated linear SVM or
  Complement Naive Bayes (CNB); NB-SVM (Wang and Manning, 2012); and "transliterated-copy" variants that
  concatenate each text with its Romanized or native-script copy. Class weights are balanced, and C is chosen by
  inner CV.
* **Dense:** frozen LaBSE, multilingual-E5-base and BGE-M3 embeddings with Logistic Regression. Target-aware
  variants use `[u, v, |u−v|, u·v]` features (Reimers and Gurevych, 2019), where *v* embeds the claim.
* **Zero-shot NLI:** a multilingual mDeBERTa-v3 XNLI checkpoint (Laurer et al., 2024) with Hindi and Bengali
  hypotheses.
* **Stacking:** the best sparse and best dense model by CV, combined by a logistic-regression meta-learner on out-of-fold probabilities.

All baselines use 5-fold CV, stratified by label × language and group-aware.

## 4.3 Fine-tuned transformers

We fine-tune MuRIL, IndicBERT v2, XLM-R base, mBERT, BanglaBERT (Bengali only) and the XNLI-tuned
mDeBERTa. Two input formulations:

* **single:** `[CLS] comment [SEP]`
* **pair (target-aware):** `[CLS] claim [SEP] comment [SEP]`, with the claim in English, Hindi, Bengali or the
  comment's own language ("auto").

Training uses AdamW (lr 2e-5, weight decay 0.01, 10 % linear warm-up) for 3 epochs, batch 16, max length 128, fp16,
early stopping on dev macro F1 (patience 2), and class-weighted cross-entropy (inverse frequency^0.5) or focal loss
(γ = 2). To fit into 4 GB of VRAM, the word-embedding matrix is frozen. Without this, MuRIL's 197k-token embeddings and
their optimiser states spill into system memory and training slows from 0.25 to 8 s per step. Three seeds (42, 43, 44) are run for
the three strongest MuRIL configurations; other configurations use seed 42.

## 4.4 Local LLM

Qwen2.5-1.5B-Instruct (Qwen Team, 2024) runs locally in fp16. It classifies by *constrained scoring*: the
log-likelihood of each label word after a stance-definition prompt, so output parsing cannot fail. Few-shot
examples (k = 4) are retrieved by LaBSE nearest-neighbour search over training data.

## 4.5 Ensembling, calibration and selection

Candidates are the top two transformers by dev F1 (whole-system models only; Bengali-only BanglaBERT
is excluded), the best sparse and best dense baselines by gold, and the stacked hybrid. They are combined by
equal averaging, dev-tuned weighted averaging or dev-fitted stacking, each with or without temperature
scaling plus per-class log-bias calibration (Guo et al., 2017) fitted on dev. This gives 11
candidates, ranked by the selection score

  `score = gold macro F1 − 0.5 × challenge failure rate`.

For CPU deployment we also consider dynamic int8 quantisation of the best transformer.

## 4.6 Statistics

Differences on the gold set are tested with a paired bootstrap (2,000 resamples; Efron and Tibshirani,
1993), with 95 % confidence intervals for each model's F1 and for its difference from the baseline. With 306 items the 95 % CI of a single
macro F1 is about ±0.05.

# 5. Experimental setup

**Hardware:** Intel i5-12500H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB). **Software:** Python 3.11,
PyTorch 2.x, Hugging Face Transformers, scikit-learn. Every script takes a YAML config, seeds Python, NumPy
and PyTorch, caches translations, embeddings and predictions, and logs one row per experiment (config, seed,
git commit, metrics, runtime) to `results/experiments.csv`. `run_all.ps1` reproduces the pipeline, and its
`-Quick` mode was verified end to end in a clean checkout. Fine-tuning one MuRIL configuration takes about 15
minutes.

# 6. Results

## 6.1 Baselines

| Model | CV F1 | Gold F1 | Gold hi | Gold bn | Challenge fail |
|---|---|---|---|---|---|
| Majority | 0.242 | 0.176 | 0.176 | 0.176 | 0.542 |
| Lexicon (optimistic†) | 0.337 | 0.575 | 0.559 | 0.591 | 0.369 |
| Zero-shot NLI (mDeBERTa-XNLI) | 0.405 | 0.413 | 0.362 | 0.432 | 0.453 |
| TF-IDF + LR | 0.666 | 0.635 | 0.664 | 0.606 | 0.320 |
| TF-IDF + LR, transliterated copy | **0.675** | 0.550 | 0.679 | 0.396 | 0.391 |
| NB-SVM | 0.653 | 0.612 | 0.656 | 0.568 | 0.356 |
| **TF-IDF + CNB** | 0.626 | **0.732** | 0.727 | **0.738** | 0.287 |
| LaBSE + LR | 0.610 | 0.612 | 0.590 | 0.635 | 0.304 |
| LaBSE target-aware + LR | 0.611 | 0.643 | 0.637 | 0.649 | 0.296 |
| e5-base target-aware + LR | 0.626 | 0.671 | 0.682 | 0.661 | 0.316 |
| BGE-M3 + LR | 0.624 | 0.719 | 0.722 | 0.716 | **0.266** |
| Stacked hybrid | 0.669 | 0.670 | 0.715 | 0.625 | 0.300 |

*Table 3. Selected baselines (full table in `results/tables.md`).* † Written by the gold-set author.
TF-IDF + CNB is the baseline to beat. Target-aware features add about +0.03 gold F1 to both LaBSE and e5.

## 6.2 Transformers and input formulation

| Configuration (pair, claim language) | Seeds | Gold F1 | Dev F1 | Challenge fail |
|---|---|---|---|---|
| MuRIL single | 42 / 43 / 44 | 0.783 / 0.604 / 0.768 (mean 0.718 ± 0.099) | 0.665 | 0.272 |
| MuRIL pair, en | 42 / 43 / 44 | 0.737 / 0.667 / 0.774 (mean 0.726 ± 0.054) | 0.678 | 0.247 |
| MuRIL pair, bn | 42 / 43 | 0.777 / 0.771 (mean 0.774 ± 0.005)‡ | 0.670 | 0.238 |
| MuRIL pair, hi | 42 | 0.728 | 0.660 | 0.246 |
| MuRIL pair, auto | 42 | 0.674 | 0.654 | 0.291 |
| MuRIL pair, auto, focal loss | 42 | 0.732 | 0.672 | 0.271 |
| XLM-R pair, auto | 42 | 0.776 | 0.671 | 0.256 |
| IndicBERT v2 pair, auto | 42 | 0.774 | 0.667 | 0.210 |
| mDeBERTa-XNLI pair, auto | 42 | 0.700 | 0.649 | 0.252 |
| mBERT pair, auto | 42 | 0.642 | 0.609 | 0.352 |
| BanglaBERT pair, auto (bn only) | 42 | 0.797 (Bengali half) | 0.682 | 0.201 |

*Table 4. Fine-tuned transformers.* Dev F1 and challenge fail are means over the listed seeds. ‡ Seed 44 failed to train: its loss stayed flat at about 1.05 and it predicted only the majority class
(dev F1 0.455, gold 0.267). Such runs are detected with a dev-only rule (dev F1 < 0.5) and reported separately,
a known BERT fine-tuning instability (Mosbach et al., 2021).

**Findings.** (i) **Seed variance dominates.** One configuration spans 0.604–0.783 gold F1 while
its dev F1 is nearly constant, so MuRIL, XLM-R and IndicBERT v2 (0.774–0.783, one seed) are statistically
tied. (ii) **Pair input with a fixed claim language** is at least as accurate as single input over seeds,
with lower challenge failure and lower variance. A claim in the comment's own language ("auto") was the
weakest variant. (iii) mBERT is clearly weaker than Indic-aware or larger encoders.

## 6.3 Local LLM

| Setting | Gold F1 | Challenge fail | Latency (GPU) |
|---|---|---|---|
| Qwen2.5-1.5B, zero-shot | 0.481 | 0.493 | 0.36 s / comment |
| Qwen2.5-1.5B, 4-shot kNN | 0.414 | 0.551 | 9.7 s / comment |

*Table 5. Local LLM.* The small LLM is far below the fine-tuned encoders, with Against recall near 0.2.
Few-shot prompts lower accuracy and overflow 4 GB of VRAM (5.6 GB peak).

## 6.4 Augmentation ablations

Each variant retrains TF-IDF + CNB (5-fold CV) and MuRIL single (2 epochs, 1 seed).

| Variant | Sparse CV | Sparse gold | Sparse fail | MuRIL dev | MuRIL gold | MuRIL fail |
|---|---|---|---|---|---|---|
| reference | 0.626 | 0.732 | 0.287 | 0.674 | 0.717 | 0.272 |
| − synthetic templates | 0.537 | **0.419** | 0.475 | 0.650 | **0.504** | 0.385 |
| − transliterated | 0.634 | 0.704 | 0.303 | 0.658 | 0.741 | 0.214 |
| − code-mixed | 0.630 | 0.722 | 0.280 | 0.675 | 0.748 | 0.259 |
| + character noise | 0.611 | 0.728 | 0.292 | 0.681 | 0.739 | 0.229 |
| + dropped matras | 0.615 | 0.710 | 0.303 | 0.675 | 0.746 | 0.259 |
| + repeated characters | 0.613 | 0.714 | 0.289 | 0.668 | 0.699 | 0.257 |
| + negation flips (14 hand-approved) | 0.627 | 0.729 | 0.284 | 0.656 | 0.695 | 0.287 |
| + back-translation | 0.619 | 0.726 | 0.305 | 0.649 | 0.665 | 0.272 |
| + English source data | 0.614 | 0.725 | 0.293 | 0.678 | 0.695 | 0.273 |
| + 3× templates | 0.700 | 0.756 | 0.270 | 0.668 | 0.756 | 0.227 |

*Table 6. Ablations.* The pre-declared rule (keep a variant only if CV/dev **and** challenge failure both improve) flags five variants. Every
flagged gain is within noise: transformer deltas (−0.05 to +0.04) are within seed spread; the negation-flip CV gain
is +0.0002 against a CV std of 0.006; and the +0.074 CV gain from more templates is mechanical, since template rows
are almost trivially learnable (out-of-fold F1 0.94–0.99). **No augmentation is used in the final model.** The
one clear effect is that removing templates collapses gold F1 for both models.

## 6.5 Final system

| Candidate | Dev F1 | Dev ECE | Gold F1 | Gold hi | Gold bn | Challenge fail | Score |
|---|---|---|---|---|---|---|---|
| **Average of 4 (selected)** | 0.686 | 0.065 | **0.766** | **0.800** | 0.732 | **0.209** | **0.662** |
| MuRIL pair-en | 0.683 | 0.090 | 0.737 | 0.769 | 0.704 | 0.220 | 0.627 |
| MuRIL pair-en + calibration | 0.685 | 0.046 | 0.730 | 0.762 | 0.698 | 0.218 | 0.622 |
| Dev-weighted average + calibration | 0.699 | 0.031 | 0.731 | 0.769 | 0.693 | 0.220 | 0.621 |
| Average + calibration | 0.700 | 0.028 | 0.740 | 0.769 | 0.711 | 0.241 | 0.619 |
| Dev-weighted average | 0.699 | 0.064 | 0.725 | 0.760 | 0.688 | 0.211 | 0.619 |
| MuRIL pair-auto, focal | 0.672 | 0.121 | 0.732 | 0.736 | 0.729 | 0.271 | 0.597 |
| TF-IDF + CNB | 0.578 | 0.144 | 0.732 | 0.727 | 0.738 | 0.287 | 0.588 |
| BGE-M3 + LR | 0.634 | 0.039 | 0.719 | 0.722 | 0.716 | 0.266 | 0.586 |
| Stacking (dev) | 0.672 | 0.041 | 0.684 | 0.718 | 0.651 | 0.232 | 0.567 |
| Hybrid baseline | 0.638 | 0.028 | 0.670 | 0.715 | 0.625 | 0.300 | 0.521 |

*Table 7. Final-model candidates.* The selected system averages MuRIL pair-en, MuRIL pair-auto (focal),
TF-IDF + CNB and BGE-M3 + LR. Calibration improves dev calibration error (0.065 → 0.028) but lowers
gold F1 by about 0.03, so the final model is uncalibrated.

**Significance.** The gold-F1 gain over TF-IDF + CNB is +0.034, with a paired-bootstrap 95 % CI of [−0.019, +0.094]:
**not significant**. The final model's own 95 % CI is [0.715, 0.810]. No model is significantly better than the baseline; only
models scoring below about 0.645 are significantly worse.

**Robustness.** The challenge failure rate falls from 0.287 to 0.209, the lowest among all candidates. The largest
reductions are on directional negation (0.59 → 0.31), different target (0.68 → 0.32), topic-related
neutral text (0.45 → 0.13) and conspiracy framing (0.50 → 0.33).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Favour | 0.840 | 0.573 | 0.681 | 110 |
| Against | 0.797 | 0.904 | 0.847 | 104 |
| None | 0.699 | 0.859 | 0.771 | 92 |

*Table 8. Final model on gold, per class.* Confusion (rows true, columns predicted F/A/N): Favour 63/21/26,
Against 2/94/8, None 10/3/79.

## 6.6 Efficiency

| Model | Gold F1 | Challenge fail | Size | CPU latency |
|---|---|---|---|---|
| TF-IDF + CNB | 0.732 | 0.287 | 19 MB | 0.2 ms |
| MuRIL pair-en, fp32 | 0.737 | 0.220 | 960 MB | 17 ms |
| MuRIL pair-en, int8 (attention layers) | 0.715 | 0.223 | – | 14 ms |
| Final ensemble | 0.766 | 0.209 | 1.9 GB | 197 ms |

*Table 9. Accuracy vs cost on CPU (per comment, timed on uncached text).* Dynamic int8 quantisation of
*all* linear layers collapsed MuRIL (gold 0.74 → 0.33), also with per-channel weights. The cause is activation
outliers in the feed-forward output projections (Bondarenko et al., 2021). Quantising only the attention
projections keeps probabilities within 0.02 of fp32. The int8 model is the recommended CPU option.

# 7. Analysis

## 7.1 Validation does not predict test behaviour

Cross-validation on constructed data and the gold set rank models differently. Transliterated-copy TF-IDF has the
best CV (0.675) but among the worst gold scores (0.550; Bengali 0.396). TF-IDF + CNB has one of the lowest CVs
(0.626) but the best classical gold score. Out-of-fold F1 by text origin shows why: 0.94–0.99 on template rows,
0.51–0.60 on translated rows, and 0.41–0.46 on original Bengali headlines. CV mostly measures fit to translationese
and templates, not to short, explicit, noisy comments. This motivates selection on test-only resources with
a fixed, published rule.

## 7.2 Error analysis

| Gold category | Accuracy | n | | Gold category | Accuracy | n |
|---|---|---|---|---|---|---|
| implicit | 0.43 | 30 | | neutral topic | 0.77 | 30 |
| negation | 0.68 | 22 | | explicit | 0.87 | 46 |
| different target | 0.69 | 16 | | code-mixed | 0.91 | 22 |
| sarcasm | 0.70 | 20 | | conspiracy | 0.93 | 14 |
| Romanized | 0.72 | 32 | | off-topic | 1.00 | 16 |

*Table 10. Final-model accuracy on selected gold categories.*

The dominant error is **Favour predicted as Against** (21 of 110). About half of these are negated denials or sarcasm,
for example "ये कहना गलत है कि जलवायु परिवर्तन एक झूठ है" ("it is wrong to say climate change is a lie"). LIME
shows that "झूठ" (lie) is the strongest Against cue, while "गलत" (wrong) does not push toward Favour.
The model has not learned that reported or negated denial reverses stance. **Implicit Favour**, where effects are
described without stance words (heat records, receding glaciers), is often labelled None, because the
English sources state stance explicitly. Relative to the baseline, the ensemble fixes 6 of 7 baseline errors on
conspiracy framing, 6 of 13 on topic-related neutral text, 5 of 6 on code-mixed and all 3 on off-topic comments,
but introduces new errors on implicit comments (8 broken vs 1 fixed).

# 8. Limitations and ethics

* **Proxy evaluation.** The gold set is author-written and shares its author with the templates. It approximates
  the hidden test but is not a sample of it. Ablation evidence about templates is confounded by this.
* **Small test sets.** With 306 gold items, differences under about 0.05 macro F1 are not evidence.
* **Limited seeds.** Most configurations have one seed, and seed variance is large.
* **Translated training data** carries US/UK framings of the climate debate and roughly 15 % inherited label noise.
* **Resource constraints.** Frozen embeddings, base-size encoders and max length 128 were forced by the 4 GB GPU.
  NLLB replaced the gated IndicTrans2. SetFit and LLM fine-tuning (QLoRA) were not run.
* **Ethics.** No personal data is stored; the optional comment collector hashes usernames. The model is
  meant for aggregate analysis of public discourse, not for judging individuals. Errors on sarcasm and
  implicit stance mean individual predictions should not be trusted without review.

# 9. Conclusion

Without in-domain training data, a carefully constructed and leakage-controlled training set plus robustness-weighted
selection yields a stance classifier for Hindi and Bengali climate comments. It scores 0.766 macro F1 on
hand-written comments and fails 21 % of behavioural tests (baseline 29 %). The accuracy gain over a strong TF-IDF
baseline is within noise, while the robustness gain is substantial. Seed variance is larger than the differences between
backbones, and validation on constructed data is a poor guide to real comments. Both points argue for evaluation on
data that resembles deployment. The most valuable next steps are a few hundred real annotated comments,
hand-written negation and reported-speech examples, and more seeds per configuration.

# Reproducibility statement

Code, configs, processed data, test resources, per-run predictions and metrics are in the repository.
`run_all.ps1` (Windows) or `make all` reproduces the pipeline; `results/experiments.csv` links every number to its
config, seed and git commit. All decisions and deviations are logged in `ASSUMPTIONS.md` (19 entries).

# Acknowledgements

All models used are open and ran locally; no paid APIs were used. The author thanks the creators of the public
datasets and models listed in the references.

# References

* ALDayel, A. and Magdy, W. (2021). Stance detection on social media: State of the art and trends. *Information Processing & Management*.
* Allaway, E. and McKeown, K. (2020). Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations. *EMNLP*.
* Augenstein, I., Rocktäschel, T., Vlachos, A. and Bontcheva, K. (2016). Stance Detection with Bidirectional Conditional Encoding. *EMNLP*.
* Barbieri, F., Camacho-Collados, J., Espinosa Anke, L. and Neves, L. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *Findings of EMNLP*.
* Bhattacharjee, A. et al. (2022). BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla. *Findings of NAACL*.
* Bondarenko, Y., Nagel, M. and Blankevoort, T. (2021). Understanding and Overcoming the Challenges of Efficient Transformer Quantization. *EMNLP*.
* Chen, J. et al. (2024). BGE M3-Embedding. arXiv:2402.03216.
* Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL*.
* Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
* Doddapaneni, S. et al. (2023). Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages. *ACL*.
* Du, J., Xu, R., He, Y. and Gui, L. (2017). Stance Classification with Target-Specific Neural Attention Networks. *IJCAI*.
* Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
* Feng, F., Yang, Y., Cer, D., Arivazhagan, N. and Wang, W. (2022). Language-agnostic BERT Sentence Embedding. *ACL*.
* Gala, J. et al. (2023). IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages. *TMLR*.
* Gardner, M. et al. (2020). Evaluating Models' Local Decision Boundaries via Contrast Sets. *Findings of EMNLP*.
* Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
* Hardalov, M., Arora, A., Nakov, P. and Augenstein, I. (2022). Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training. *AAAI*.
* Khanuja, S. et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730.
* Khanuja, S., Dandapat, S., Srinivasan, A., Sitaram, S. and Choudhury, M. (2020). GLUECoS: An Evaluation Benchmark for Code-Switched NLP. *ACL*.
* Küçük, D. and Can, F. (2020). Stance Detection: A Survey. *ACM Computing Surveys*.
* Laurer, M., van Atteveldt, W., Casas, A. and Welbers, K. (2024). Less Annotating, More Classifying. *Political Analysis*.
* Luo, Y., Card, D. and Jurafsky, D. (2020). Detecting Stance in Media on Global Warming. *Findings of EMNLP*.
* Mohammad, S. M., Kiritchenko, S., Sobhani, P., Zhu, X. and Cherry, C. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets. *SemEval*.
* Mohammad, S. M., Sobhani, P. and Kiritchenko, S. (2017). Stance and Sentiment in Tweets. *ACM Transactions on Internet Technology*.
* Mosbach, M., Andriushchenko, M. and Klakow, D. (2021). On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines. *ICLR*.
* NLLB Team, Costa-jussà, M. R. et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672.
* Qwen Team (2024). Qwen2.5 Technical Report. arXiv.
* Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
* Ribeiro, M. T., Wu, T., Guestrin, C. and Singh, S. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. *ACL*.
* Sennrich, R., Haddow, B. and Birch, A. (2016). Improving Neural Machine Translation Models with Monolingual Data. *ACL*.
* Vaid, R., Pant, K. and Shrivastava, M. (2022). Towards Fine-grained Classification of Climate Change related Social Media Text. *ACL Student Research Workshop*.
* Wang, L. et al. (2024). Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
* Wang, S. and Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. *ACL*.
* Wasi, A. T. et al. (2024). Dhoroni: a Bengali climate and environmental news dataset with multi-perspective annotation. arXiv:2410.17225. (Title paraphrased; see the arXiv entry.)
* Yin, W., Hay, J. and Roth, D. (2019). Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. *EMNLP*.

# Appendix A. Artefacts

| Artefact | Location |
|---|---|
| All results tables | `results/tables.md` |
| Per-run predictions and metrics | `results/predictions/` |
| Significance tests | `results/significance.csv` |
| Robustness per category | `results/robustness.csv`, `results/figures/robustness_heatmap.png` |
| Explanations (IG, LIME) | `results/explanations/index.html` |
| Datasheet / model card | `docs/datasheet.md`, `models/MODEL_CARD.md` |
| Decisions log | `ASSUMPTIONS.md` |
