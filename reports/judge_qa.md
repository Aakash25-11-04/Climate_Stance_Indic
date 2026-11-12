# Likely judge questions and answers

All numbers are from `results/`; interpretation in `reports/discussion.md`.

**Q1. You had no training data. Why should we trust a model trained on translations?**
We don't trust it blindly; we built two test-only resources that look like the hidden test, and we
select models on them. The 306-comment gold set is hand-written in 14 categories, and the
challenge suite has 1,084 tests. Our gap analysis shows translated text is actually *harder* than real
comments: out-of-fold macro F1 on translated rows is about 0.5-0.6, while the same models score
0.63-0.77 on the gold set. Translated news and tweets carry stance implicitly and inherit about 15 % label
noise, which we found in a manual audit of 50 rows.

**Q2. Isn't your gold set biased, since you wrote it yourself?**
Yes, partly, and we say so. It was written before any model was trained, it is deduplicated against
training data (0 exact or near duplicates), and its categories were fixed in advance from the task
description. The lexicon baseline was written by the same author, so its gold score is flagged as
optimistic. `MANUAL_TASKS.md` and `scripts/collect_youtube.py` let a human annotate 150 real YouTube
comments; `scripts/evaluate.py --agreement` then reports Cohen's kappa between human and model.

**Q3. Is the transformer actually better than TF-IDF?**
Not significantly on accuracy, and clearly on robustness. The final ensemble scores 0.766 vs 0.732 gold F1; the paired
bootstrap 95 % CI of the difference is [-0.019, +0.094], so we do not claim an accuracy win. It fails 21 % of
challenge tests against 29 %, with the biggest gains on negation (0.31 vs 0.59) and neutral topic text (0.13 vs 0.45). We test this with a paired bootstrap on the gold set because
single-seed differences of a few points are within noise on 306 items (95 % CI about ±0.05). The
baseline to beat is TF-IDF with Complement Naive Bayes, which is unusually strong on gold
(0.732) despite a lower CV score. Its per-class priors suit balanced, short, explicit comments.

**Q4. What is your innovation?**
(1) Target-aware NLI-style pair input `[claim] [SEP] [comment]`, with the claim given in English,
Hindi, Bengali or the comment's language. We measured each: over three seeds, a fixed claim language is as accurate as single input and more robust and stable
(claim-bn 0.774 ± 0.005, two seeds; claim-en 0.726 ± 0.054; single 0.718 ± 0.099); the comment's-language claim was worst.
(2) A leakage-controlled pipeline that turns English stance data into Indic training data with
quality-filtered translation, rule-based Romanization (schwa deletion, Bengali inherent-o, casual
spelling variants) and code-mixing. (3) A CheckList-style suite for Hindi/Bengali stance,
including invariance and negation-flip directional tests. (4) A selection rule that weighs robustness
(`gold F1 - 0.5 x challenge failure rate`) rather than validation F1.

**Q5. Why MuRIL?**
It is pre-trained on transliterated (Romanized) Indic text and on translation pairs, which matches the
hidden test's Hinglish/Banglish. We also compared IndicBERT v2, XLM-R, mBERT and BanglaBERT
(Bengali only): MuRIL 0.783, XLM-R 0.776, IndicBERT v2 0.774 (most robust, fail 0.210), mBERT 0.642 (1 seed each).
The top three are tied within seed noise (MuRIL single input ranged 0.604-0.783 over 3 seeds), so MuRIL is a
reasonable default, not a proven winner. BanglaBERT scored 0.797 on the Bengali half only.

**Q6. Why did you freeze the embedding layer?**
On a 4 GB GPU, MuRIL's 197k-token embedding matrix plus its AdamW states does not fit; Windows
silently spills into shared memory and training slowed to 8 s per step. Freezing only the
embeddings keeps all transformer layers trainable, cuts a run to 15 minutes, and is a standard trade-off for
classification fine-tuning. A Colab notebook trains without this restriction.

**Q7. How do you handle sarcasm and negation?**
We never remove negators or stopwords; negation-flip directional tests check that "X is a hoax" and
"X is not a hoax" get opposite labels. Sarcasm remains one of the hardest categories:
final-model challenge failure rates are 0.40 for sarcasm, 0.20 for negation MFTs and 0.31 for negation-flip tests
(baseline: 0.57, 0.33, 0.59). We tried rule-based negation-flip augmentation,
but on review only 14 of 64 automatic flips were correct, so most of it was discarded rather than
injecting noise.

**Q8. Did synthetic data help?**
Yes, strongly: removing templates drops gold F1 from 0.732 to 0.419 (TF-IDF) and 0.717 to 0.504 (MuRIL). Tripling them
looks better again (0.756), but within noise for the transformer. Caveat: templates and gold share an author, so some
of this may be shared style. Template rows are almost
trivially learnable (out-of-fold F1 0.94-0.99), which warns that templates teach surface patterns.

**Q9. Why not use a large LLM?**
No paid APIs, and a 4 GB GPU. We ran Qwen2.5-1.5B-Instruct locally, zero-shot and few-shot with kNN
retrieval and constrained label scoring: gold F1 0.481 zero-shot and 0.414 with 4 shots, far below the fine-tuned
encoders (about 0.77). Few-shot was also 27x slower because long prompts overflow 4 GB VRAM. It is also used for one-sentence
rationales, which are an aid, not ground truth.

**Q10. How is the final model chosen, and is that leakage?**
All fitting (weights, thresholds, temperature, stacking) uses training folds or dev only. The gold set
and challenge suite are used once, to compare a small, pre-declared set of candidates. Every
candidate's score is published in `results/final_candidates.csv`, so the selection is transparent.

**Q11. How fast is it, and can it run without a GPU?**
On CPU: TF-IDF 0.2 ms per comment (gold 0.732); MuRIL int8 14 ms (0.715); the ensemble 197 ms (0.766).
`scripts/predict.py --model efficient --device cpu` runs the int8 MuRIL. Only attention layers are quantised:
quantising every layer collapsed it to 0.33 because of activation outliers in the feed-forward outputs.

**Q12. What fails, and what would you do with more time?**
Implicit stance (0.43 accuracy: effects described without saying climate change is serious), negation of denial,
where "it is wrong to say it is a lie" is read as Against, comments about a different target, and sarcasm. Next steps: label real comments (the pipeline is
ready), switch to IndicTrans2 for translation, run 3 seeds for every configuration, and run QLoRA on
a larger LLM.

**Q13. How do I reproduce your numbers?**
`.\run_all.ps1` reproduces the pipeline end to end. Every script has a YAML config and fixed seed, and every run is a
row in `results/experiments.csv` with its git commit.

**Q14. Ethics?**
No personal data is stored; the optional YouTube collector hashes usernames. The model is for
aggregate analysis of public discourse, not for judging individuals. It may carry US/UK framings of
climate debate from its English source data. See the model card's limitations section.
