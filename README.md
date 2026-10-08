# Climate Stance Detection for Hindi and Bengali YouTube Comments

This project classifies the stance of Hindi and Bengali YouTube comments toward one claim:

> **"Climate change and global warming are serious concerns."**

Every comment gets one of three labels:

| Label | Meaning |
|---|---|
| `favour` | agrees that climate change is real and serious |
| `against` | denies it, calls it exaggerated, a hoax or a conspiracy |
| `none` | unrelated, neutral, or no stance can be inferred |

Models are scored by **macro F1**, so the rare `against` class counts as much as the other two.

**Status (October 2026):** Phase 0 is complete. The Phase 1 collector is built and tested,
and the first live collection run is next. See [Phases and outcomes](#phases-and-outcomes).

---

## The core problem

There is **no public labelled climate-stance data in Hindi, and none from YouTube**. Every
labelled dataset we could find differs from the test set in language, in platform, or in both:

| What we have | What we are tested on |
|---|---|
| English tweets, news sentences, Reddit comments | Hindi and Bengali YouTube comments |
| One Bengali dataset, formal news articles (median 333 words) | Short, informal comments |
| Standard scripts | Devanagari, Bengali, Romanized text, and code-mixing with English |

This mismatch shapes every decision in the project. A model that scores well on English
validation data tells us little about how it will do on Hindi or Bengali comments.

## Methodology

1. **Measure before modelling.** Before training anything, we audit every public source for
   label quality, duplicates, spam and shortcuts. The audit is in
   [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb).
2. **Prefer data whose target matches our claim.** Sources are split into a **core** tier
   (target matches the claim) and an **auxiliary** tier (related target, other language or
   domain). Experiments start on the core tier, and auxiliary data is added one source at a
   time as an ablation, never by default.
3. **Build a native evaluation set.** Real Hindi and Bengali YouTube comments are collected
   through the official API and hand-labelled. This native dev set is the only signal used to
   choose between models.
4. **Label English numbers honestly.** Any score on English data is reported as
   *"English held-out, not a proxy for test"*.
5. **Guard against shortcuts.** Splits are grouped so that near-duplicates and related texts
   never sit on both sides of a split, and the cleaning rules remove the artefacts the audit
   found.
6. **Keep everything reproducible.** Scripts rebuild every dataset from public sources and
   produce byte-identical outputs, versions are pinned, and every rule cites the analysis that
   motivated it.

## Phases and outcomes

| Phase | Goal | Status |
|---|---|---|
| 0 | Build, audit and clean a public training corpus | **Done** |
| 1 | Collect Hindi and Bengali YouTube comments | **Tooling done**, live run pending |
| 2 | Annotate a native dev and test set | Planned |
| 3 | Evaluation harness and baselines | Planned |

### Phase 0: public corpus (done)

**What we did.** [data_prep/build_corpus.py](data_prep/build_corpus.py) downloads five public
climate-stance datasets and maps their labels onto our claim (20,467 raw rows).
[data_prep/clean_corpus.py](data_prep/clean_corpus.py) applies five cleaning rules.

| Source | Text | Tier | Rows after cleaning |
|---|---|---|---|
| SemEval-2016 Task 6 | English tweets | core | 562 |
| GWSD | English news sentences | core | 2,044 |
| ClimateReddit | Reddit comments (both annotators agree) | core | 303 |
| CASE 2024 | English tweets about climate activism | auxiliary | 7,459 |
| Dhoroni | Bengali news articles | auxiliary | 2,193 |
| **Total** | | | **12,561** |

The core tier has **2,909 rows, of which 483 are `against`**. Sources, licences, citations
and the full label mapping are in [data/README.md](data/README.md).

**What we found:**

- **`against` is rare everywhere:** 4.6% of SemEval, 9.6% of CASE, 6.2% of Dhoroni, and at
  most 19.8% in Reddit. Under macro F1 this class will dominate the score.
- **Artefacts inflate scores.** 69% of CASE `against` rows were one repeated spam tweet.
  Removing it dropped CASE's in-domain F1 from 0.68 to 0.46: the higher score came from
  memorisation, not stance. GWSD repeated five crowd-sourcing screening sentences 50 times each.
- **Each source is its own domain.** A bag-of-words model can tell which source a text came
  from 85% of the time. TF-IDF baselines score 0.40–0.61 macro F1 on their own source and
  0.31–0.50 on other English sources, before any change of language.
- **Framing cue.** Sceptics tend to say "global warming" and supporters "climate change".
  Hindi and Bengali comments borrow "global warming" as an English loanword, so this cue
  could mislead a model on the test set.

**Cleaning rules, in order:**

1. Drop model-generated (silver) labels.
2. Drop the CASE spam template.
3. Drop English rows with fewer than 3 words once hashtags, URLs and mentions are removed.
4. Drop every copy of a text that appears with different labels.
5. Keep one copy of each remaining duplicate.

### Phase 1: YouTube collection (tooling done)

**What we built.** [data_prep/collect_youtube.py](data_prep/collect_youtube.py) collects
comments on climate-related videos found by 16 search queries (8 Hindi, 8 Bengali) listed in
[configs/youtube.yaml](configs/youtube.yaml).

- **Official API only.** It uses the YouTube Data API v3 within YouTube's Terms of Service.
  There is no scraping.
- **Quota-safe.** Every API response is cached. When the daily quota runs out, the script
  saves partial output and resumes from the cache on the next run. The full run is estimated at
  4,016 units, within one day's quota.
- **Privacy.** The outputs keep no usernames or commenter channel ids, only a salted hash used
  to cap comments per author. Raw API responses are never committed.
- **Tags.** Each comment is tagged with its query language, query type, writing system
  (Bengali, Devanagari, Latin, mixed), length, and whether it repeats another comment.
- **Query types and annotation pools.** Queries are either `general` or `sceptic`.
  - `sceptic` queries look for denial content, to find enough rare `against` examples.
  - The RANDOM annotation pool draws only from comments that general queries found.
  - Comments found only by sceptic queries go to a TARGETED pool, which is never used to
    estimate class proportions.

**Expected outcome:** an unlabelled pool of comments, plus count tables
(by language, query type, writing system, video, length and repeats) written by
[data_prep/youtube_counts.py](data_prep/youtube_counts.py).

### Phase 2: annotation (planned)

- A **sampler** that draws comments from the RANDOM and TARGETED pools and writes a blind file
  for annotators and a separate key file.
- **Annotation guidelines** with worked examples for each label, including sarcasm,
  code-mixing and comments about something other than the claim.
- A browser **annotation tool** (`tools/annotate.html`).
- **`agreement.py`** to measure agreement between annotators.

**Expected outcome:** a hand-labelled native Hindi and Bengali dev and test set, the only
trustworthy way to choose between models.

### Phase 3: evaluation harness and baselines (planned)

- An evaluation harness that reports macro F1 and per-class scores.
- CPU baselines trained on the **core tier only** and evaluated on a **grouped English
  held-out split**, every number marked *"English held-out, not a proxy for test"*.

**Expected outcome:** a fixed evaluation pipeline and reference scores that later models must
beat on the native dev set.

## How we are making this project

- **Phases and gates.** Each phase ends with a check before the next starts. For example,
  Phase 1 stops after collection to review the counts before the sampler is built.
- **Small, reviewed commits.** Each commit covers one concern and is reviewed before it is
  pushed.
- **Tested shared code.** Text handling that every stage relies on (duplicate detection,
  script detection, the API client) lives in the `stance` package with unit tests. The tests
  cover Devanagari and Bengali edge cases and the API quota and cache logic, and need no
  network.
- **Code quality.** `ruff` and `black` (line length 100) keep the style consistent, and
  `pytest` must pass before each commit.
- **No hidden data.** No raw or processed data is committed. Every dataset is rebuilt by a
  script, and only count tables and figures are tracked.
- **Secrets stay local.** API keys and the hashing salt live in `.env`, which is gitignored.

## Repository layout

```
configs/youtube.yaml        search queries, quota budget and limits for collection
data/README.md              sources, licences, label mapping, known issues
data_prep/
  build_corpus.py           download public datasets -> data/processed/corpus_v1.csv
  clean_corpus.py           cleaning rules -> data/processed/corpus_v1_clean.csv
  explore.py                analysis functions used by the notebook
  collect_youtube.py        YouTube Data API collection with cache and quota ledger
  youtube_counts.py         count tables for collected comments
notebooks/
  01_data_exploration.ipynb audit of the public corpora
results/eda/                figures from the exploration notebook
src/stance/
  preprocess.py             duplicate keys and script detection (Devanagari, Bengali, Latin)
  youtube.py                API client: header-only key, cache, quota ledger, comment parsing
tests/                      unit tests (no network needed)
```

## Setup

Requires Python 3.11 and, for later GPU phases, an NVIDIA GPU with CUDA 12.6.

```
conda create -n stance python=3.11
conda activate stance
pip install -r requirements.txt
pip install -e .
pytest
```

## Reproducing the data

```
# Phase 0: public corpus
python data_prep/build_corpus.py
python data_prep/clean_corpus.py

# Phase 1: YouTube comments (needs an API key)
copy .env.example .env          # set YT_API_KEY and YT_AUTHOR_SALT
python data_prep/collect_youtube.py --dry-run
python data_prep/collect_youtube.py
python data_prep/youtube_counts.py
```

On macOS or Linux, use `cp` instead of `copy`. If the collector exits with code 75, the daily
quota ran out. Run it again after midnight Pacific time and it resumes from the cache.

## Data and ethics

- Public datasets are used for research under their licences and with citation; see
  [data/README.md](data/README.md).
- YouTube data is collected only through the official API, within YouTube's Terms of Service.
- Commenter identities are pseudonymised, and no raw comment data is published in this
  repository.
