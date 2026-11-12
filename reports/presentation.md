# Presentation outline (14 slides, about 12 minutes plus a 5-minute demo)

Every number comes from `results/` (see `reports/discussion.md` for interpretation). Figures live in `results/figures/`.

1. **Title.** Stance toward "Climate change and global warming is a serious concern" in Hindi and
   Bengali social media. One line: no training data, a 4 GB laptop GPU, and every number reproducible.

2. **The task and why it is hard.** Three labels and macro F1. Show four example comments: Romanized,
   code-mixed, sarcastic, and topic-related neutral. The hidden test is noisy, so robustness matters more than
   validation score.

3. **Constraint: no in-domain data.** The four public sources and the label-mapping table
   (README). Note that Dhoroni's "Negative" means bad environmental news, not denial, and why that
   matters.

4. **Building the dataset.** Pipeline diagram: English stance data, NLLB translation, back-translation
   filtered by LaBSE, Romanized and code-mixed copies, templates (capped), deduplication including gold,
   group-wise split. Key counts: 19,278 train / 1,402 dev; 93 % of translations kept; 0 leakage.

5. **Test sets that look like the hidden test.** 306 hand-written gold comments in 14 categories, and a
   CheckList suite of 1,084 tests: functionality, invariance and negation-flip. Both are used only for testing.

6. **EDA insights that changed decisions.** Train text is twice as long as real comments (median 107 vs
   62 characters); Against is only 11.5 % of train; about 15 % label noise was found in a manual audit;
   template rows are trivially easy (out-of-fold F1 0.94-0.99). Figure: `source_type_vs_gold.png`.

7. **Baselines.** Table: majority, lexicon, TF-IDF LR/SVM/CNB, NB-SVM, embeddings, zero-shot NLI and the stacked hybrid.
   Baseline to beat: TF-IDF + Complement NB, gold 0.732 (CV 0.626). Message: CV and gold disagree, and we say why:
   transliterated copies give the best CV (0.675) but gold 0.550. Zero-shot NLI 0.413; best embeddings bge-m3 0.719.

8. **Key innovation: target-aware pair input.** `[claim] [SEP] [comment]` with the claim in
   en/hi/bn/auto, against single input. MuRIL, gold F1 / fail: single 0.783 / 0.262, claim-bn 0.777 / 0.245, claim-en
   0.737 / 0.220, claim-hi 0.728 / 0.246, claim-auto 0.674 / 0.291 (1 seed). Over 3 seeds: single 0.718 ± 0.099,
   claim-en 0.726 ± 0.054, claim-bn 0.774 ± 0.005 (2 seeds; 1 failed to train).
   Lesson: a pair input with a fixed claim language is as accurate as single input and more robust and stable;
   claim-aware features also add +0.03 to frozen LaBSE/e5 embeddings. The comment's-language claim (auto) was worst.

9. **Backbones and loss.** MuRIL vs IndicBERT v2 vs XLM-R vs mBERT vs BanglaBERT vs fine-tuned NLI;
   weighted cross-entropy vs focal loss. Gold F1 (1 seed): MuRIL 0.783, XLM-R 0.776, IndicBERT v2 0.774 (lowest fail
   0.210), fine-tuned mDeBERTa-XNLI 0.700, mBERT 0.642; BanglaBERT 0.797 on the Bengali half only. Focal 0.732 vs
   weighted CE 0.674 (claim-auto). The 3-seed variance (single input: 0.604-0.783 gold) shows the top three are tied.

10. **What helped and what didn't (ablations).** Table of the keep/drop rule (CV and challenge must both
    improve). Five variants pass the rule, but every gain is within seed or CV noise, so none was adopted
    (ASSUMPTIONS #19). Removing template data collapses gold (TF-IDF 0.732 → 0.419). Honest note: 50 of 64 negation flips were wrong and discarded.

11. **Local LLM.** Qwen2.5-1.5B, zero-shot vs kNN few-shot, using constrained label scoring. Gold F1 0.481 zero-shot
    (0.36 s/comment), 0.414 with 4 retrieved shots (9.7 s/comment, VRAM overflow), vs about 0.77 for the encoders. Rationales as an explanation aid.

12. **Final system and robustness.** Ensemble plus calibration; selection rule `gold F1 - 0.5 x challenge
    fail`. Heatmap `robustness_heatmap.png`. Final = average of MuRIL claim-en, MuRIL focal,
    TF-IDF CNB, bge-m3 LR: gold 0.766 [0.715, 0.810], hi 0.800, bn 0.732, fail 0.209 (baseline 0.287). +0.034 over the
    baseline is not significant (paired bootstrap CI [-0.019, 0.094]); the robustness gain is. Weakest: implicit (0.43
    accuracy), negation of denial, sarcasm (fail 0.40).

13. **Efficiency and explainability.** Accuracy vs cost on CPU: TF-IDF 0.2 ms / 0.732; MuRIL int8 (attention only)
    14 ms / 0.715; MuRIL fp32 17 ms / 0.737; ensemble 197 ms / 0.766. Full int8 collapsed MuRIL to 0.33 (outliers). Integrated Gradients and LIME examples;
    error taxonomy `error_taxonomy.png`, showing what the final model fixed or broke relative to the baseline.

14. **Limitations, ethics, next steps.** Translationese, author-written gold set, single seeds for most runs, US/UK
    framings in the sources. Next: real annotated comments (pipeline ready), IndicTrans2, larger LLMs.
    Reproduce with `.\run_all.ps1`.

**Demo** (see `reports/demo_script.md`) after slide 12 or at the end.
