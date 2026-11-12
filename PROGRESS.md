# PROGRESS

Resume point for any new session: read this file top to bottom, then `results/experiments.csv`.
Run everything with `PYTHONUTF8=1` and the interpreter `.venv/Scripts/python.exe`.

## Status board

| Phase | Status | Key output |
|---|---|---|
| 0 Scaffold + hardware | done | HARDWARE.md, .venv, src/stance, configs/, run_all.ps1, Makefile, tests/ (10 pass) |
| 1 Dataset construction | done (v2) | data/processed/{train,dev,train_en}.csv, data/gold/test_handcrafted.csv, docs/datasheet.md |
| 2 EDA | done | scripts/eda.py, notebooks/eda.ipynb (executed), results/figures/ |
| 3 Preprocessing + baselines | done (SetFit not run, see README "Deviations") | results/baselines.csv |
| 4 Transformers | done (11 configs, 3 seeds for 3 configs) | results/transformers.csv, results/tables.md |
| 5 Local LLM | done | results/llm.csv, results/llm_rationales.csv |
| 6 Augmentation ablations | done (nothing adopted, ASSUMPTIONS #19) | results/ablations.csv |
| 7 Challenge suite | done | data/challenge/, results/robustness.csv, results/figures/robustness_heatmap.png |
| 8 Ensemble / calibration / efficiency | done | models/final/MODEL.json (avg of 4), models/efficient/MODEL.json (int8 MuRIL), results/final.json |
| 9 Explainability + error analysis | done | results/explanations/, results/error_analysis.json, results/significance.csv |
| 10 Deliverables | done | predict.py, app.py, upload_to_hf.py (not run, by instruction), model card, README |
| 11 Reports | done | reports/: literature_review, methodology_results, discussion, presentation, demo_script, judge_qa |

## Session log

### 2026-11-10, session 1 (Phase 0, Phase 1 code, Phase 3/4 code)
- Hardware: i5-12500H, 16 GB RAM (~3.5 GB free), RTX 3050 4 GB -> Tier C+ (HARDWARE.md).
- `.venv` (Python 3.11, torch 2.14.1+cu126 reused via system-site-packages), pinned requirements.txt, `pip install -e .`.
- Loaders for tweet_eval/stance_climate, GWSD, ClimateStance, Dhoroni; label-mapping table in loaders.py.
- IndicTrans2 gated -> NLLB-200-distilled-600M fallback (ASSUMPTIONS #3); pure-Python IndicProcessor re-implementation kept for when a token exists.
- Romanizer, code-mixer, template generator, MinHash dedup, group-wise split, 308-item gold set, challenge suite (13 categories).
- Sparse/dense/NLI baselines + CV framework + stacking; transformer trainer (single/pair, claim en/hi/bn/auto, weighted CE/focal); predict.py; smoke-tested everything on a 50-row build.
- Full dataset build started (background).

### 2026-11-11, session 2
- Build finished after 300 min (network dropped mid-way; HF cache carried it): train 19,488 / dev 1,402 / gold 306.
  Translation kept 11,888/12,778 (93 %); Dhoroni bn->hi kept 1,347/1,461; dedup dropped 596 exact + 26 near; 0 leakage.
  Synthetic share only 8.9 % (cap 40 % not binding). Three train rows near-duplicating challenge tests were removed.
- Wrote scripts for phases 5-11 (run_llm, run_ablations, run_robustness, build_final, explain, error_analysis,
  write_model_card, prepare_official, collect_youtube), docs/datasheet.md, reports/literature_review.md, tests (10 pass), Colab notebook.
- EDA run: figures in results/figures. 50-row label audit (results/eda_label_audit_sample.csv) found ~8/50 questionable
  labels among translated rows (spam tweets labelled Favour, a sarcastic tweet labelled Favour, "mass hysteria" labelled None),
  i.e. roughly 15 % inherited label noise, plus two generator bugs: candra-vowel characters leaking into Romanized text
  ("rikaॉrd") and code-mix substitution inside words ("peopleों"). Both fixed in code; **dataset must be rebuilt**
  (`python scripts/build_dataset.py --config configs/dataset.yaml`, ~10 min from cache) before the full Phase 3/4 runs.
- Running now (background, logs in results/): MVS baselines (`results/baselines_mvs.log`) and MuRIL pair (`results/train_muril_pair.log`).

### 2026-11-11, session 3 (MVS done, Phase 3/4 running)
- MuRIL run was pathologically slow (8 s/step): weights + AdamW states of the 197k-vocab model exceeded 4 GB and
  spilled into shared memory. Fix: `model.freeze_embeddings: true` (ASSUMPTIONS #12) -> 0.25 s/step, 5 min/epoch, 2.7 GB.
- Dataset rebuilt (v2, fixed romanizer + whole-word code-mix): train 19,278 / dev 1,402 / gold 306; 0 leakage.
  `scripts/build_challenge.py` now also removes train rows that near-duplicate challenge tests (3 rows).
- EDA rerun, `notebooks/eda.ipynb` executed (kernel `stance-venv`).
- **MVS committed** (367140e): MuRIL pair (claim auto) dev 0.654 / gold 0.674 / challenge fail 0.291; predict.py OK on CPU.
- Results so far on v2 data (results/baselines.csv, results/predictions/metrics_*.json):
  tfidf_lr CV 0.666 gold 0.635 fail 0.320 | **tfidf_cnb CV 0.627 gold 0.732 fail 0.287** | nbsvm CV 0.653 gold 0.612 |
  lexicon gold 0.575 | muril pair e6/lr3e-5 gold 0.687 fail 0.275 | **muril single gold 0.783 fail 0.262**.
- IMPORTANT operational note: background jobs started from an interactive shell die when that shell closes. Long jobs are now
  launched detached with PowerShell `Start-Process cmd /c "... > results\<log> 2>&1"`; check logs, do not relaunch blindly.
- Running detached now: `results/transformers_matrix2.log` (run_transformers --skip_existing, GPU) and
  `results/baselines_sparse2.log` (tfidf_lr_translit, nbsvm_translit, tfidf_svm; CPU, resume mode).

### 2026-11-11, session 4 (queues for all remaining phases)
- Killed the transliterated-copy sparse job: it had run 5 h on one model and exhausted memory (1 GB free virtual), which
  also slowed the GPU job ~15x. Its nested grid search is removed (ASSUMPTIONS #14).
- MuRIL results (1 seed): single 0.783 gold / 0.262 fail; pair claim-bn 0.777 / 0.245; pair claim-en 0.737 / **0.220**;
  claim-hi 0.728; claim-auto 0.674; claim-auto focal 0.732 / 0.271. Paired bootstrap (`scripts/significance.py`): **no MuRIL
  config is significantly better than tfidf_cnb (gold 0.732) yet**; CIs are about +-0.05 on 306 items.
- Gap analysis (`results/tables.md`): OOF F1 on template rows 0.94-0.99 (too easy), translated rows 0.51-0.60.
- Negation flips: 64 candidates hand-reviewed, 14 kept (`data/augmentation/negation_flips_reviewed.csv`, ASSUMPTIONS #15).
- New: `scripts/job_queue.py`, `significance.py`, `make_tables.py`, `write_report.py`, `prepare_space.py`,
  `src/stance/evaluation/select.py` (auto selection: baseline to beat by gold; best transformer by DEV), Indic fonts in figures,
  `reports/methodology.md`. Demo (`app.py`) smoke-tested (classify + server HTTP 200). Upload script has `--dry_run`.

### 2026-11-11/12, session 5 (all queues finished; final system)
- Transformer matrix done: XLM-R 0.776, IndicBERT v2 0.774 (fail 0.210), mDeBERTa-XNLI 0.700, mBERT 0.642; BanglaBERT 0.797
  on Bengali only, so it is excluded from whole-system selection (ASSUMPTIONS #16).
- All baselines on v2: best gold tfidf_cnb 0.732; bge-m3 0.719; target-aware features +0.03 for LaBSE/e5; zero-shot NLI 0.413.
- Seeds: MuRIL single input 0.604/0.783/0.769 gold; one claim-bn seed failed to train, detected on dev (ASSUMPTIONS #17).
- LLM (Qwen2.5-1.5B): 0.481 zero-shot, 0.414 four-shot (9.7 s/comment).
- Fixed: sparse `predict(texts, languages)` signature (crashed ablations and the first final build); int8
  quantisation collapse (attention-only now, ASSUMPTIONS #18); the efficient model is chosen by score; ensemble latency
  is measured honestly; `.gitignore` had swallowed `src/stance/models` (now tracked).
- Final: avg_all, gold 0.766 (hi 0.800, bn 0.732), fail 0.209; +0.034 over the baseline, not significant.
- Ablations: rule flags five variants, all within noise, so none adopted (ASSUMPTIONS #19).
- Wrote reports/discussion.md; filled presentation, demo script and judge Q&A from results; demo input 5 is a
  known, demonstrated failure (negation of denial).
- `run_all.ps1 -Quick` passes end to end in a clean git worktree of the committed repo (dataset 5.3 min, then challenge,
  baselines, MuRIL, final model, predict.py; exit 0). Fixed on the way: with redirected output, PowerShell 5.1 turned tqdm's
  stderr into a terminating error. `-Quick` now refuses to run where trained models exist.

## Next steps (manual, for the user; see MANUAL_TASKS.md)
1. Push to GitHub; `huggingface-cli login`, then `python scripts/upload_to_hf.py --dry_run`, then without `--dry_run`.
2. Optional: request IndicTrans2 access and rebuild; annotate real YouTube comments (`scripts/collect_youtube.py`).
3. Optional with more compute: more seeds per configuration; train without frozen embeddings on Colab.
