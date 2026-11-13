# Literature review: stance detection for Hindi/Bengali climate comments

_All citations below are real publications. Entries marked **[verify]** still need their exact
venue/year/pages checked before quoting them in a submission._

## 1. Stance detection: task and benchmarks

**SemEval-2016 Task 6** (Mohammad, Kiritchenko, Sobhani, Zhu & Cherry, 2016, *SemEval*)
introduced tweet-level stance detection toward five targets, including "Climate Change is a
Real Concern", with labels FAVOR / AGAINST / NONE and macro-F1 over FAVOR and AGAINST as the
metric. The climate target is famously skewed: very few AGAINST tweets. The same data is
redistributed in **TweetEval** (Barbieri, Camacho-Collados, Espinosa Anke & Neves, 2020,
*Findings of EMNLP*), which we load from the Hugging Face Hub. Mohammad, Sobhani and
Kiritchenko (2017, *ACM TOIT*, "Stance and Sentiment in Tweets") show that stance is not
sentiment: a negative-sentiment tweet can be in favour of the target (e.g. anger at deniers).
This distinction motivates our None/Favour/Against decision rules for sarcasm and
"different target" cases.

**GWSD / DeSMOG** (Luo, Card & Jurafsky, 2020, *Findings of EMNLP*, "Detecting Stance in Media
on Global Warming") labels 2,050+ news sentences with agree / neutral / disagree toward the
target "climate change/global warming is a serious concern", i.e. exactly our claim. They
collect 8 crowd ratings per sentence and fit a Bayesian model of annotator reliability; they
also fine-tune BERT and study opinion-framing devices. We use the majority vote.

**ClimateStance / ClimateEng** (Vaid, Pant & Shrivastava, 2022, *ACL Student Research
Workshop*, "Towards Fine-grained Classification of Climate Change related Social Media Text")
provides 3,777 COP25-era tweets labelled for stance (favor / against / ambiguous) and topic,
plus pseudo-labelled Reddit data. It is the largest English climate-stance tweet set.

**Climate-FEVER** (Diggelmann et al., 2020, NeurIPS Climate Change AI workshop) and
**ClimaText** (Varini et al., 2020) are related climate NLP resources (claim verification and
topic detection) that we do not use for labels but cite for context. **ClimateBERT**
(Webersinke, Kraus, Bingler & Leippold, 2021, arXiv) is a domain-adapted model for English
climate text; it is not multilingual, so we did not use it.

**Dhoroni** (Wasi, Faisal, Ahmad, Rahman & Islam, 2024, arXiv:2410.17225) is a Bengali climate
and environmental news dataset of 2,300 articles annotated from ten perspectives including a
stance label (Positive / Neutral / Negative / Problematic) and fine-tuned BanglaBERT
baselines. Its stance is toward the environment in general, not toward our claim, which is why
we map only part of it (ASSUMPTIONS.md #4).

Surveys: Küçük and Can (2020, *ACM Computing Surveys*, "Stance Detection: A Survey") and
ALDayel and Magdy (2021, *Information Processing & Management*, "Stance detection on social
media: State of the art and trends") summarise target-specific, cross-target and zero-shot
stance detection, the role of target-aware encoders, and the sentiment/stance distinction.

## 2. Target-aware and cross-lingual stance detection

Augenstein, Rocktäschel, Vlachos and Bontcheva (2016, *EMNLP*, "Stance Detection with Bidirectional
Conditional Encoding") condition the tweet encoding on the target, the ancestor of our pair
input. Du, Xu, He and Gui (2017, *IJCAI*, "Stance Classification with Target-Specific Neural
Attention Networks") add target-specific attention. With transformers, feeding the target and
the text as a sentence pair (NLI-style) became standard; Allaway and McKeown (2020, *EMNLP*,
"Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations",
VAST) and Allaway, Srikanth and McKeown (2021, *NAACL*, "Adversarial Learning for Zero-Shot
Stance Detection on Social Media") show that target conditioning generalises to unseen
targets. Hardalov, Arora, Nakov and Augenstein (2021, *EMNLP*, "Cross-Domain Label-Adaptive
Stance Detection"; 2022, *AAAI*, "Few-Shot Cross-Lingual Stance Detection with Sentiment-Based
Pre-Training") study cross-lingual stance with a 15-dataset, 12-language benchmark and show
that label-name prompting and pattern-exploiting training help low-resource languages.
Our experiment (ii) - `[claim] [SEP] [text]` with the claim in English, Hindi or Bengali -
tests whether target conditioning in the comment's own language helps Indic encoders.

Zero-shot NLI classification (Yin, Hay & Roth, 2019, *EMNLP*, "Benchmarking Zero-shot Text
Classification: Datasets, Evaluation and Entailment Approach") is the basis of baseline (h):
a multilingual NLI model (mDeBERTa-v3-base fine-tuned on XNLI + multilingual NLI data by
Laurer et al., 2022/2024, *Political Analysis*, "Less Annotating, More Classifying") scores each
comment against label hypotheses. XNLI (Conneau et al., 2018, *EMNLP*) includes Hindi; Bengali
is covered by the multilingual NLI mixtures used for the checkpoint.

## 3. Multilingual sentence embeddings

**LaBSE** (Feng, Yang, Cer, Arivazhagan & Wang, 2022, *ACL*, "Language-agnostic BERT Sentence
Embedding") is trained with translation ranking on 109 languages (Hindi and Bengali included)
and is the standard choice for cross-lingual similarity; we use it both as a feature extractor
(baseline f/g) and as the back-translation quality filter. **Multilingual E5** (Wang, Yang,
Huang, Yang, Majumder & Wei, 2024, arXiv:2402.05672, "Multilingual E5 Text Embeddings") offers
small/base/large contrastively trained encoders with a `query:` prefix convention. **BGE-M3**
(Chen et al., 2024, arXiv:2402.03216, "BGE M3-Embedding") supports dense, sparse and
multi-vector retrieval across 100+ languages. Reimers and Gurevych (2019, *EMNLP*,
Sentence-BERT) introduced the `[u, v, |u-v|, u*v]` pair-feature scheme we reuse for
target-aware embedding features (baseline g); SetFit (Tunstall et al., 2022, arXiv:2209.11055)
adds contrastive few-shot fine-tuning of such encoders.

## 4. Indic language models and tools

**MuRIL** (Khanuja et al., 2021, arXiv:2103.10730, "MuRIL: Multilingual Representations for
Indian Languages") is a BERT-base model trained on 17 Indian languages including
**transliterated (Romanized) text** and translation pairs, which is why it is our first choice
for Hinglish/Banglish. **IndicBERT v2** (Doddapaneni et al., 2023, *ACL*, "Towards Leaving No
Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic
Languages") is trained on IndicCorp v2 (24 languages) and evaluated on IndicXTREME; the
MLM-only checkpoint is used here. **BanglaBERT** (Bhattacharjee et al., 2022, *Findings of
NAACL*, "BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language
Understanding Evaluation in Bangla") is an ELECTRA-style Bengali model that leads the BLUB
benchmark. **XLM-R** (Conneau et al., 2020, *ACL*, "Unsupervised Cross-lingual Representation
Learning at Scale") and **mBERT** (Devlin et al., 2019, *NAACL*) are the general multilingual
reference points. **IndicTrans2** (Gala et al., 2023, *TMLR*, "IndicTrans2: Towards High-Quality
and Accessible Machine Translation Models for all 22 Scheduled Indian Languages") is the
state-of-the-art open Indic MT system; its Hub checkpoints are gated, so this project falls
back to **NLLB-200** (NLLB Team, Costa-jussà et al., 2022, arXiv:2207.04672, "No Language Left
Behind") distilled-600M, which supports hin_Deva and ben_Beng. Indic normalisation and
tokenisation use the **Indic NLP Library** (Kunchukuttan, 2020); Romanization uses
`indic-transliteration` (ITRANS scheme) with our own casual-spelling post-processing.

## 5. Code-mixed and Romanized NLP

Code-mixing of Hindi/Bengali with English in social media is well documented: Bali, Sharma,
Choudhury and Vyas (2014, *CodeSwitch workshop at EMNLP*, "I am borrowing ya mixing?") analyse
Hindi-English Facebook posts; Patra, Das and Das (2018, SAIL/ICON, "Sentiment Analysis of
Code-Mixed Indian Languages") and the SemEval-2020 Task 9 **SentiMix** (Patwa et al., 2020)
provide Hinglish sentiment data; the **GLUECoS** benchmark (Khanuja, Dandapat, Srinivasan,
Sitaram & Choudhury, 2020, *ACL*) evaluates models on Hindi-English and Spanish-English
code-mixed tasks and shows that transliteration-aware pre-training (like MuRIL's) matters.
Roy et al. (2013, FIRE) and Gupta, Choudhury and Bali (2012) **[verify exact titles]** study
transliterated search and spelling variation in Romanized Hindi, which motivates our
character n-gram features, the transliterated-copy baseline (e) and the typo/matra-drop
augmentations. Krishnan, Anastasopoulos, Purohit and Rangwala (2021, **[verify venue]**)
propose multilingual code-switching data augmentation for low-resource transfer, close to our
synthetic code-mixing layer.

## 6. Robustness evaluation and augmentation

**CheckList** (Ribeiro, Wu, Guestrin & Singh, 2020, *ACL*, "Beyond Accuracy: Behavioral Testing
of NLP Models with CheckList") defines minimum functionality tests, invariance tests and
directional expectation tests; our challenge suite follows this design with 13 categories,
invariance perturbations (typos, emoji, URLs, mentions, elongation, partial Romanization) and
negation-flip directional tests. Data-augmentation surveys (Feng et al., 2021, *Findings of
ACL*, "A Survey of Data Augmentation Approaches for NLP") cover back-translation (Sennrich,
Haddow & Birch, 2016, *ACL*), EDA-style noise (Wei & Zou, 2019, *EMNLP*) and contrast sets
(Gardner et al., 2020, *Findings of EMNLP*, "Evaluating Models' Local Decision Boundaries via
Contrast Sets"), which inspired our negation-flip pairs. Class imbalance is handled with
weighted cross-entropy and **focal loss** (Lin, Goyal, Girshick, He & Dollár, 2017, *ICCV*).
Calibration follows Guo, Pleiss, Sun and Weinberger (2017, *ICML*, "On Calibration of Modern
Neural Networks"): temperature scaling on held-out data.

## 7. Explainability

**Integrated Gradients** (Sundararajan, Taly & Yan, 2017, *ICML*) attributes a prediction to
input embeddings along a path from a baseline; we use the Captum implementation (Kokhlikyan et
al., 2020, arXiv:2009.07896) with a `[PAD]` baseline. **LIME** (Ribeiro, Singh & Guestrin, 2016,
*KDD*, "Why Should I Trust You?") fits a local linear surrogate on perturbed inputs and is
model-agnostic, so it also explains the sparse and ensemble models. For linear TF-IDF models we
report the largest coefficients per class directly.

## 8. Datasets documentation

**Datasheets for Datasets** (Gebru et al., 2021, *Communications of the ACM*; arXiv 2018)
is the template for `docs/datasheet.md`. **Model Cards** (Mitchell et al., 2019, *FAT\**) is
the template for `models/MODEL_CARD.md`.

## 9. Gap this project addresses

To my knowledge there is no public stance-labelled dataset of Hindi or Bengali social-media
comments toward the climate-concern claim: SemEval-2016, GWSD and ClimateStance are English,
Dhoroni is Bengali news with a different stance target. The hidden test set of the challenge
is noisy, Romanized and code-mixed. The contributions here are therefore (1) a documented,
reproducible pipeline that turns English stance data into Indic training data with
quality-filtered translation, Romanization and code-mixing, (2) a target-aware pair-input
formulation with the claim in the comment's language, evaluated against strong sparse and
embedding baselines, and (3) a CheckList-style robustness suite and hand-written gold set
for the categories the hidden test is known to contain.
