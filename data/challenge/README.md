# Robustness challenge suite (TEST-ONLY)

CheckList-style tests built by `scripts/build_challenge.py` from hand-written seeds in
`src/stance/data/challenge_suite.py`. Never used for training, tuning or model selection

* MFT tests: 780 ({'template': 500, 'seed_filled': 200, 'seed': 80})
* Invariance tests: 240 ({'inv_typo': 30, 'inv_emoji_append': 30, 'inv_url_append': 30, 'inv_mention_prepend': 30, 'inv_elongation': 30, 'inv_whitespace': 30, 'inv_roman_span': 30, 'inv_punct_strip': 30})
* Directional (negation flip) tests: 64

| category | hi | bn |
|---|---|---|
| codemixed | 30 | 30 |
| conspiracy | 30 | 30 |
| different_target | 30 | 30 |
| emoji | 30 | 30 |
| implicit | 30 | 30 |
| long | 30 | 30 |
| negation | 30 | 30 |
| neutral_topic | 30 | 30 |
| offtopic | 30 | 30 |
| romanized | 30 | 30 |
| sarcasm | 30 | 30 |
| short | 30 | 30 |
| typos | 30 | 30 |

Leakage check vs train.csv (exact / near duplicates): {'exact': 0, 'near_or_exact': 0, 'n_other': 780}
