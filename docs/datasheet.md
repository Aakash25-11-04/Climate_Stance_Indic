# Datasheet: Indic Climate Stance training data (built 2026-11-11)

Format follows Gebru et al., "Datasheets for Datasets" (CACM 2021). Numbers come from
`data/processed/build_stats.json` and `results/eda_stats.json`.

## Motivation

**Why was the dataset created?** The Indic Climate Stance Challenge asks for a Hindi/Bengali
stance classifier toward the claim "Climate change and global warming is a serious concern",
but provides no training data. This dataset was built so that a model could be trained,
evaluated and stress-tested reproducibly without any paid API.

**Who created it and who funded it?** Built by the repository owner, Aakash Rauniar, on a personal laptop. No funding.

## Composition

**What do the instances represent?** Short texts (comments, tweets, news sentences/headlines)
in Hindi or Bengali, in native script, Romanized (Hinglish/Banglish) or code-mixed with
English, each labelled Favour / Against / None toward the claim.

**How many instances?**

| split | rows | hi | bn | Favour | Against | None |
|---|---|---|---|---|---|---|
| train | 19,278 | 9,788 | 9,490 | 10,982 | 2,225 | 6,071 |
| dev | 1,402 | 712 | 690 | 825 | 130 | 447 |
| gold (hand-written, test-only) | 306 | 153 | 153 | 110 | 104 | 92 |
| challenge suite (test-only) | 780 MFT + 240 invariance + 64 directional | | | | | |

(v2 build, after the Romanizer and code-mixer fixes; three train rows were removed after the build because they near-duplicated challenge tests.)

**Train rows by `source_type`** (how the text came to exist):

| source_type | rows | meaning |
|---|---|---|
| translated | 11,372 | English source sentence machine-translated to hi/bn (NLLB-200-distilled-600M), plus Dhoroni bn headlines translated to hi; kept only if LaBSE cosine(source, back-translation) >= 0.75 |
| transliterated | 3,165 | Romanized copy of a native-script train row (rule-based, casual spelling variation) |
| codemixed | 1,720 | synthetic Hindi-English / Bengali-English mixing of a train row (lexical substitution, discourse markers, partial Romanization) |
| synthetic | 1,723 | template-generated comments (10 styles x 3 labels x 2 languages), **8.9 % of train** (cap was 40 %; not binding) |
| original | 1,298 | Bengali headlines / first sentences from Dhoroni |

**Scripts in train**: Devanagari 7,279; Bengali 7,088; Latin 3,449; mixed 1,462.

**Sources of the underlying labelled text**

| source | rows loaded | licence / citation | label mapping |
|---|---|---|---|
| SemEval-2016 Task 6 climate target via `cardiffnlp/tweet_eval` (`stance_climate`) | 564 | research use; Mohammad et al. 2016, Barbieri et al. 2020 | favor->Favour, against->Against, none->None |
| GWSD, `yiweiluo/GWStance` | 2,245 | no licence file; Luo, Card & Jurafsky 2020 (cite) | majority of 8 workers, share >= 0.5 |
| ClimateStance, `eci-io/climate-evaluation` | 3,580 | research use; Vaid et al. 2022 | 1->Favour, 2->Against, 0->None |
| Dhoroni, `ciol-research/Dhoroni` | 1,461 | MIT; Wasi et al. 2024 | Neutral->None; Positive & climate focus->Favour; Negative/Problematic dropped |

**Is there a label-noise estimate?** Source labels are crowd/expert annotations of the
original English text; translation can change meaning. 93.0 % of translations passed the
back-translation filter (hi 95.0 %, bn 91.1 %; mean similarity 0.907). A 50-row random audit
sample is in `results/eda_label_audit_sample.csv` and is discussed in `notebooks/eda.ipynb`.
The rule-based lexicon agrees with only 40 % of train labels, which mostly reflects the
lexicon's weakness on translated news-style text rather than label noise.

**Relationships between instances**: every row carries a `group_id`; all translations,
Romanized and code-mixed copies of one source sentence share it and are always in the same
split. Exact and near duplicates (MinHash Jaccard >= 0.85, char 4-grams, within language)
were removed before splitting (596 exact + 26 near dropped). Leakage check train vs dev,
train vs gold, train vs challenge: 0 exact, 0 near.

**Recommended splits**: train / dev as given; gold and challenge are **test-only** and must
never be used for training, tuning or model selection.

## Collection process

* Public datasets downloaded from the Hugging Face Hub and GitHub (`src/stance/data/loaders.py`).
* URLs, @mentions and `#` symbols were stripped from English tweets before translation.
* Translation: NLLB-200 distilled 600M, beam 4, max length 200, fp16 on an RTX 3050, cached
  in `data/cache/translations.sqlite`. IndicTrans2 was the intended system but its Hub
  repositories are gated (see ASSUMPTIONS.md #3).
* No scraping, no personal data; the optional YouTube collector (`scripts/collect_youtube.py`)
  hashes usernames and stores only comment text, but it was **not** run for this release.

## Preprocessing / cleaning

See `src/stance/preprocessing/normalize.py`: NFC + Indic normalisation, zero-width removal,
URL/mention tokens, elongation reduction. Raw text is kept in the CSVs; preprocessing is
applied at model time so ablations can switch it.

## Uses

Intended: training and evaluating stance classifiers for Hindi/Bengali climate discourse;
studying robustness to Romanization and code-mixing. Not intended: inferring attributes of
individuals, content moderation decisions, or claims about the prevalence of denial in any
population (the label distribution is an artefact of the sources, not of Indian/Bangladeshi
social media).

## Known biases and limitations

1. **Translationese**: 58 % of train rows are machine translations of English tweets and
   news; they are longer (median 107 vs 62 characters in the gold set), more formal, and
   carry US/UK framings (Trump, Greta, COP). Real comments are shorter and more colloquial.
2. **Class imbalance**: Against is 11.5 % of train (sources have few deniers), while the
   hidden test is expected to be more balanced; models use class weighting and threshold tuning.
3. **Synthetic artefacts**: Romanization is rule-based (one spelling system plus random
   variation); code-mixing uses a fixed lexicon; template comments repeat structures. Their
   effect is measured in the Phase 6 ablations.
4. **Dhoroni** is news, not social media, and its stance target differs from ours.
5. The gold set and lexicon were written by the same author, so lexicon results on gold are
   optimistic.
6. No real, human-annotated Hindi/Bengali comments are included; MANUAL_TASKS.md D describes
   how to add them.

## Distribution and maintenance

The built CSVs are committed to the repository (MIT for our code and annotations; the source
texts remain under their original terms). Rebuild with
`python scripts/build_dataset.py --config configs/dataset.yaml` (about 5 hours on this laptop,
10 minutes from cache). Maintained in the GitHub repository (see MANUAL_TASKS.md).
