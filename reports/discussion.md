# Discussion: what the results mean

All numbers come from files in `results/` (named in brackets). "Gold" is the 306-comment hand-written
test set (153 Hindi, 153 Bengali) and "fail" is the challenge-suite failure rate (1,084 tests). Neither was
used for training or tuning; both were used only to compare a small, pre-declared set of candidates.

## 1. Headline

The final system is an equal-weight average of four models: MuRIL with an English claim
(`[claim] [SEP] [comment]`), MuRIL trained with focal loss, TF-IDF with Complement Naive Bayes, and
bge-m3 embeddings with logistic regression. It reaches **gold macro F1 0.766** (Hindi 0.800, Bengali 0.732)
and the **lowest challenge failure rate of any candidate, 0.209** [final_candidates.csv]. The baseline to
beat, TF-IDF + Complement NB, scores 0.732 with failure rate 0.287.

The gold improvement (+0.034) is **not statistically significant**: the paired-bootstrap 95 % interval of the
difference is [-0.019, +0.094] [significance.csv]. The robustness improvement is large and consistent:
the ensemble fails 21 % of challenge tests against 29 % for the baseline. The failure rate falls most on
directional negation tests (0.31 vs 0.59), on comments about a different target (0.32 vs 0.68), and on
topic-related neutral text (0.13 vs 0.45) [robustness.csv]. Because the hidden test is described as noisy,
code-mixed social media, we weight this robustness gain more than the gold-F1 gain.

## 2. Seeds matter more than architecture

Three seeds of MuRIL with single input gave gold F1 0.783, 0.604 and 0.769. Dev F1 barely moved (0.66-0.67).
The same configuration therefore spans 0.18 F1 on gold depending only on the random seed. The
single-seed differences between MuRIL, XLM-R and IndicBERT v2 (0.774-0.783) are well inside this range.
We do not claim any backbone is better than another on this evidence.

One seed (MuRIL, claim in Bengali, seed 44) never left the majority-class solution: training loss stayed
near 1.05 for three epochs and the model never predicted Against. This is a known BERT fine-tuning
instability (Mosbach et al., 2021). We detect such runs with a dev-only rule (dev F1 < 0.5) and report seed
statistics both with and without them (ASSUMPTIONS #17).

Seed sensitivity is also why the ensemble wins. Averaging two transformers with two
differently-biased classical models smooths out an unlucky seed. Both uncalibrated ensembles (plain
average 0.209, dev-weighted 0.211) fail fewer challenge tests than any single member (best: 0.220).

## 3. Target-aware pair input

Over three seeds, pair input with the claim in English averages 0.726 ± 0.054 gold F1 and 0.247 fail. With the
claim in Bengali it averages 0.774 ± 0.005 and 0.238 fail (two seeds that trained; the third failed to train).
Single input averages 0.718 ± 0.099 and 0.272 fail. Pair input is therefore at least as accurate as single input, and
more robust and more stable. The same direction appears for frozen embeddings: claim-aware
`[u, v, |u-v|, u*v]` features add +0.03 gold F1 to both LaBSE and e5 [baselines.csv].

Giving the claim in the comment's own language ("auto") was the *worst* MuRIL variant with a single seed
(0.674). We do not have a confirmed explanation. One hypothesis: when the claim and the comment share a
script, the model leans on lexical overlap with the claim rather than on stance.

## 4. Validation disagrees with the test sets

Cross-validation on the constructed training data and the gold set often rank models in opposite
orders:

- Adding transliterated copies gives the best CV of any classical model (0.675), but among the worst gold
  scores (0.550, Bengali 0.396).
- TF-IDF + Complement NB has one of the lowest CV scores (0.626) but the best classical gold score (0.732).
- Calibration and dev-tuned weights improved dev F1 and calibration error (dev ECE 0.065 → 0.028), but
  lowered gold F1 by about 0.03.

The gap analysis explains much of this. Out-of-fold F1 is 0.94-0.99 on template rows but only 0.51-0.60 on
translated rows. A manual audit found about 15 % label noise in translated rows, and translated text is twice as long as
real comments. CV on this data measures how well a model fits translationese, not how it handles short, explicit
comments. This is why selection uses the test-only gold and challenge sets once, with a fixed rule
(`gold F1 - 0.5 x fail`) and every candidate's score published.

## 5. What does not work

- **Zero-shot NLI** (mDeBERTa-XNLI, no training): gold 0.413. Generic entailment does not transfer to
  stance in Hindi/Bengali comments without fine-tuning. The fine-tuned version reaches 0.700.
- **A 1.5B local LLM** (Qwen2.5-1.5B-Instruct, constrained label scoring): 0.481 zero-shot. Adding four
  retrieved examples *lowered* it to 0.414 and made it 27x slower (9.7 s per comment). The longer prompts overflowed
  the 4 GB GPU. Against recall is only about 0.2: the model rarely labels a comment as denial.
- **mBERT** (0.642) is clearly behind the Indic-aware and larger multilingual encoders.
- **Rule-based negation-flip augmentation**: only 14 of 64 automatic flips were correct on manual review
  (ASSUMPTIONS #15).
- **Full int8 quantisation** of MuRIL collapsed it (gold 0.74 → 0.33). The cause is activation outliers in the
  feed-forward output layers. Quantising only the attention layers keeps the probabilities within 0.02 of fp32
  (ASSUMPTIONS #18).

## 6. Augmentation ablations

Each variant retrains TF-IDF + Complement NB (5-fold CV) and MuRIL with single input (2 epochs, 1 seed) on a modified
training set [ablations.csv]. The rule "keep only if CV (dev for the transformer) and the challenge failure rate both
improve" flags no_codemixed, plus_char_noise, plus_drop_matras, plus_negation_flip (sparse) and
plus_synthetic_x3 (sparse). **None of these gains is above noise, so none was folded into the final model**
(ASSUMPTIONS #19):

- Transformer ablations use one seed, and seed-to-seed spread of gold F1 is 0.05-0.10 (section 2). Every transformer
  delta here (-0.05 to +0.04 gold) is inside that spread.
- The sparse plus_negation_flip CV gain is +0.0002, while the CV standard deviation is 0.006.
- The sparse plus_synthetic_x3 CV gain (+0.074) is mechanical. Template rows are almost trivially learnable
  (out-of-fold F1 0.94-0.99), so adding more of them raises CV regardless of real-world value.

One effect is large and clear: **removing the template data collapses gold F1** (TF-IDF 0.732 → 0.419, MuRIL
0.717 → 0.504). The templates are the only short, explicit, comment-like text in training. Caveat: the
templates and the gold set were written by the same author, so part of this effect may be shared style
rather than real-world benefit. Real annotated comments are needed to tell the two apart.

Character-noise augmentations (dropped matras, repeated characters, typos) did not help either model
beyond noise. The Romanized and code-mixed copies are already a strong form of this.

## 7. Where the final model fails

Accuracy by gold category [gold_final_avg_all.csv joined with the gold categories]:

| Category | Accuracy | n |
|---|---|---|
| implicit | 0.43 | 30 |
| negation | 0.68 | 22 |
| different target | 0.69 | 16 |
| sarcasm | 0.70 | 20 |
| romanized | 0.72 | 32 |
| explicit | 0.87 | 46 |
| codemixed | 0.91 | 22 |
| off-topic | 1.00 | 16 |

The dominant confusion is **Favour predicted as Against** (21 of 110 Favour comments) and Favour predicted as
None (26). Favour recall is only 0.57, while Against recall is 0.90. About half of the Favour→Against errors
are negation (7) or sarcasm (4). These are comments that *quote* the denial in order to reject it ("it is wrong to say climate change
is a lie"). The word for "lie" (झूठ / মিথ্যা) is the strongest Against cue in training, and the models
have not learned that negation or reported speech reverses it. Implicit Favour comments, which describe effects such as
heat or glacier loss without saying "climate change is serious", are often read as None. The training data
has few such cases, because the English sources state stance explicitly.

## 8. Limitations

- The gold set was written by the project author, so it shares the author's idea of what comments look like.
  It is a proxy for the hidden test, not a sample of it.
- With 306 items, the 95 % CI of macro F1 is about ±0.05. Differences smaller than that are not evidence.
- Most configurations have one seed; seed variance is large (section 2).
- Training data is translated from English and carries US/UK framings and inherited label noise.
- The ensemble is about 1.9 GB and costs about 200 ms per comment on CPU, mostly the bge-m3 encoder. The efficient
  model (MuRIL with int8 attention layers, 14 ms per comment) scores 0.715 gold F1 and 0.223 fail
  [final.json, accuracy_vs_cost]. TF-IDF + Complement NB costs 0.2 ms per comment for 0.732 / 0.287.

## 9. What would help most

1. **Real annotated comments**, even 500. The constructed data is the bottleneck, as the CV vs gold
   disagreement shows. `scripts/collect_youtube.py` and the annotation guideline are ready.
2. **Negation and reported-speech training examples** written or reviewed by hand. This is the largest
   error group, and automatic flipping failed.
3. **More seeds per configuration**, with seed averaging or checkpoint averaging inside the final model.
4. IndicTrans2 instead of NLLB for translation, once access is granted.
