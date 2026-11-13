# Hardware report

Detected automatically on 2026-11-10 (Phase 0).

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language, build 10.0.26300 |
| CPU | 12th Gen Intel Core i5-12500H (12 physical cores, 16 logical) |
| RAM | 16 GB total (**only ~3.5 GB free at detection time**, other processes hold the rest) |
| GPU | NVIDIA GeForce RTX 3050 4GB Laptop GPU, **4 GB VRAM**, driver 555.97, CUDA 12.5 |
| iGPU | Intel Iris Xe (not used) |
| Disk | 211 GB free on C: |
| Python | 3.11.16 (conda env `stance`, has `torch 2.14.1+cu126`, CUDA available). System Python 3.14 is too new for the ML stack and is not used. |
| Apple MPS | n/a |

## Compute tier decision

The prompt defines Tier B as 6-12 GB VRAM and Tier C as CPU-only. A 4 GB GPU sits
between them. Decision: **Tier C plan, GPU-accelerated** ("Tier C+"):

- Everything that the Tier C plan allows runs here, but on the GPU, which makes
  base-size encoder fine-tuning (MuRIL / IndicBERT / XLM-R base, fp16, batch 16,
  max_length 128) feasible in minutes rather than hours.
- Base-size encoders only (no XLM-R large). 1 seed by default, 3 seeds for the
  final configuration only if time allows (the limitation is stated in the README).
- IndicTrans2 **distilled 200M** for translation (fits easily in 4 GB).
- Embedding models: LaBSE, multilingual-e5-base (bge-m3 only if it fits and is fast enough).
- Local LLM: Qwen2.5-1.5B-Instruct (fp16 ~3.1 GB, or 4-bit) only if a generation run
  finishes in under ~2 hours; otherwise template-based synthetic generation.
  The 3B/7B models of Tier A/B do not fit in 4 GB without heavy offloading.
- No QLoRA (would not fit alongside activations in 4 GB). SetFit only if < 1 hour.
- Also ship `notebooks/train_on_colab.ipynb` for heavier runs on a free Colab GPU.

Low free system RAM (~3.5 GB) is the main practical constraint: models are loaded one
at a time, embeddings are cached to disk, and dataloaders use `num_workers=0`.
