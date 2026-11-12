"""Figure A of PROJECT_EXPLAINED.md: gold macro F1 of the main models (values from results/)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#a9a8a0", "#1f1f1e", "#6b6a63", "#e4e3dd"
rows = [  # (label, gold macro F1, colour)
    ("Final committee (4 models)", 0.766, BLUE),
    ("MuRIL, single input*", 0.783, GRAY), ("XLM-R", 0.776, GRAY), ("IndicBERT v2", 0.774, GRAY),
    ("MuRIL, pair (English claim)", 0.737, GRAY), ("TF-IDF + Naive Bayes (baseline)", 0.732, ORANGE),
    ("MuRIL, focal loss", 0.732, GRAY), ("BGE-M3 embeddings", 0.719, GRAY), ("mDeBERTa-XNLI, fine-tuned", 0.700, GRAY),
    ("e5 claim-aware embeddings", 0.671, GRAY), ("mBERT", 0.642, GRAY), ("TF-IDF + logistic regression", 0.635, GRAY),
    ("Lexicon (word lists)", 0.575, GRAY), ("Qwen2.5-1.5B LLM, zero-shot", 0.481, GRAY),
    ("Zero-shot NLI (no training)", 0.413, GRAY), ("Majority guess", 0.176, GRAY)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = range(len(rows))
ax.barh(list(y), [r[1] for r in rows], color=[r[2] for r in rows], height=0.7, zorder=2)
for i, (_, v, _) in enumerate(rows):
    ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0, 0.9); ax.set_xlabel("Gold macro F1 on 306 hand-written comments (higher is better)")
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.text(0.01, 0.012, "* best of 3 random seeds (3-seed average 0.718). Single models are one seed each; differences under ~0.05 can be luck.",
         fontsize=7.5, color=MUTED)
fig.savefig("reports/explainer_fig/model_ranking.png", dpi=200)
