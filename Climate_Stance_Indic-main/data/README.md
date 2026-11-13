# Data

No raw data is committed. `data_prep/build_corpus.py` downloads every source into `data/raw/`
and writes `data/processed/corpus_v1.csv`. `data_prep/clean_corpus.py` writes
`data/processed/corpus_v1_clean.csv`. The analysis behind each cleaning rule is in
`notebooks/01_data_exploration.ipynb`.

```
python data_prep/build_corpus.py
python data_prep/clean_corpus.py
```

## Sources

| source | rows (raw → clean) | language / domain | origin | licence | citation |
|---|---|---|---|---|---|
| semeval16 | 564 → 562 | en, tweets | TweetEval repo, `datasets/stance/climate` (github.com/cardiffnlp/tweeteval) | no licence file; research use with citation | Mohammad et al., SemEval-2016 Task 6; Barbieri et al., TweetEval, Findings of EMNLP 2020 |
| gwsd | 2,300 → 2,044 | en, news sentences | `GWSD.tsv` (github.com/yiweiluo/GWStance) | no licence file; research use with citation | Luo, Card & Jurafsky, Findings of EMNLP 2020 |
| climate_reddit | 303 → 303 | en, Reddit | `ClimateReddit/Reddit-Annotator{1,2}.tsv` (github.com/roopalv54/finegrained-climate-change-social-media) | no licence file; research use with citation | Vaid, Pant & Shrivastava, ACL SRW 2022 |
| climate_reddit_pseudo | 6,262 → 0 | en, Reddit | same repo, `Pseudo-labeled-Reddit.tsv` | as above | as above |
| case2024 | 8,845 → 7,459 | en, tweets | CASE 2024 subtask C train + val. Official files are on Google Drive (github.com/therealthapa/case2024-climate); we download a participant's mirror (github.com/Salman1804102/CASE-EACL-2024) because Drive is not scriptable | check organisers' terms | Shiwakoti et al., ClimaConvo, 2024; Thapa et al., CASE 2024 |
| dhoroni | 2,193 → 2,193 | bn, news articles | `dataset.csv` + `dataset_files.zip` (github.com/ciol-researchlab/Dhoroni) | MIT | Wasi et al., 2024, arXiv:2410.17225 |

Not included: ClimateStance tweets (Vaid et al., 2022) are released as tweet IDs only.
ClimateMiSt is released as tweet IDs and has two labels only.

## Label mapping

All labels are relative to the claim "Climate change and global warming are serious concerns."

| source | original target | original → favour / against / none |
|---|---|---|
| semeval16 | Climate Change is a Real Concern | favor / against / none |
| gwsd | global warming is a serious concern | agrees / disagrees / neutral |
| climate_reddit | climate change prevention | 1 / −1 / 0, rows where both annotators agree (303 of 329, κ = 0.86) |
| case2024 | climate activism | support (1) / oppose (2) / neutral (3) |
| dhoroni | climate change is real and pressing | Positive / Negative / Neutral; "Problematic" (107) dropped |

GWSD labels use the authors' released posterior probabilities where present, otherwise the
majority of eight crowd votes. The confidence is kept in the `agreement` column.

## Columns

`uid, text, label, source, lang, domain, target, orig_label, orig_split, quality, agreement`;
the clean file adds `tier` (`core` = semeval16, gwsd, climate_reddit; `auxiliary` = case2024, dhoroni).

## Known issues

- CASE *against* means opposition to activism, which does not always mean opposition to the
  claim. Spot-check before using it as *against* training data.
- GWSD repeats five crowd-sourcing screening sentences 50 times each; the clean file keeps one copy.
- No source contains Hindi, YouTube comments, Romanized text, or code-mixing.

## YouTube comments (unlabelled, for annotation)

`data_prep/collect_youtube.py` collects Hindi and Bengali comments on climate-related videos
through the official YouTube Data API v3 only, within YouTube's Terms of Service; nothing is
scraped. Search queries and limits are in `configs/youtube.yaml`.

```
cp .env.example .env        # then set YT_API_KEY and YT_AUTHOR_SALT
python data_prep/collect_youtube.py --dry-run
python data_prep/collect_youtube.py
python data_prep/youtube_counts.py
```

- Every API response is cached in `data/raw/youtube/cache/`; quota spent per Pacific-time day
  is recorded in `data/raw/youtube/quota.json`. When the daily budget is reached the script
  writes partial outputs, exits with status 75, and resumes from the cache on the next run.
- Outputs: `data/processed/youtube_videos_v1.csv` and `data/processed/youtube_comments_v1.csv`.
  `data_prep/youtube_counts.py` writes count tables (no text, no authors) to `results/youtube/`.
- Privacy: outputs keep no usernames or commenter channel ids, only `author_hash`, a salted
  SHA-256 used to cap comments per author. The salt is `YT_AUTHOR_SALT` in `.env`. The raw
  cache does contain author names; like all of `data/raw/`, it is gitignored and never committed.
- `script` is the writing system holding at least 80% of a comment's letters (`bengali`,
  `devanagari`, `latin`, else `mixed`; `none` without letters). `latin` covers both English and
  Romanized Hindi/Bengali. `duplicate` marks repeated copies of a text after normalisation.

### Query types and annotation pools

Every search query has a `query_type`, and every video and comment row carries the type of the
queries that found its video:

| `query_type` | meaning |
|---|---|
| `general` | found only by general climate queries |
| `sceptic` | found only by queries aimed at sceptical content (e.g. "global warming is a lie") |
| `both` | found by at least one general and at least one sceptic query |

**Pool rule.** The RANDOM annotation pool draws only from `general` and `both` comments; it is
meant to estimate what a general climate search surfaces, and a video a general query returns
belongs to that population whether or not a sceptic query also returned it. `sceptic` comments
may only enter the TARGETED pool, which over-samples likely *against* comments and is never used
to estimate class priors.
