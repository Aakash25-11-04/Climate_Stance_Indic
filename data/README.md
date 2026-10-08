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
