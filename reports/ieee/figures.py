"""Figures for the IEEE-format report (3.5 in single-column width, print/grayscale-safe)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "savefig.dpi": 300})

# ---------------------------------------------------------------- Fig. 1 pipeline
fig, ax = plt.subplots(figsize=(3.5, 3.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 11.2); ax.axis("off")


def box(x, y, w, h, text, fill="#ffffff", edge=MUTED, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fill, ec=edge, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, mutation_scale=7))


L = 0.2; W = 5.6
box(L, 9.9, W, 1.0, "English stance data\nSemEval-16 T6, GWSD, ClimateStance", "#eef4fc")
box(L, 8.4, W, 1.0, "NLLB-200 en→hi/bn + back-translation\nkept if LaBSE cos ≥ 0.75 (93 %)")
box(L, 6.9, W, 1.0, "Romanized + code-mixed copies;\ntemplates (≤ 40 %, actual 8.9 %)")
box(L, 5.4, W, 1.0, "MinHash dedup (incl. gold), group-wise\nsplit, challenge-leakage removal")
box(L, 3.9, W, 1.0, "Train 19,278 / Dev 1,402", "#eef4fc", bold=True)
box(L, 2.1, W, 1.3, "Baselines · fine-tuned encoders ·\nlocal LLM · ensembles + calibration\n(fit on train folds / dev only)")
box(L, 0.2, W, 1.3, "Select by gold F1 − 0.5 × fail\n→ final: average of 4 models", "#eef4fc", bold=True)
for y in (9.9, 8.4, 6.9, 5.4, 3.9):
    arrow(L + W / 2, y, L + W / 2, y - 0.5)
arrow(L + W / 2, 2.1, L + W / 2, 1.5)
box(6.3, 9.9, 3.5, 1.0, "Dhoroni (bn news)\n+ bn→hi", "#eef4fc")
arrow(6.3, 10.1, L + W, 8.95)
box(6.3, 4.6, 3.5, 2.3, "TEST-ONLY\nGold: 306 hand-\nwritten comments\nChallenge: 1,084\nCheckList tests", "#fdf0ea", edge=ORANGE)
arrow(8.05, 4.6, L + W, 1.0)
ax.text(8.95, 2.4, "used only to\ncompare\ncandidates", ha="center", fontsize=6.2, color=MUTED, style="italic")
fig.tight_layout(pad=0.1)
fig.savefig(OUT / "fig1_pipeline.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2 accuracy vs robustness
rows = []
b = pd.read_csv(ROOT / "results/baselines.csv")
for r in b.itertuples():
    rows.append(("baseline", r.model, r.gold_macro_f1, r.challenge_fail_rate))
for p in sorted((ROOT / "results/predictions").glob("metrics_*_s4*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if "history" not in d or "ablation" in d["run"] or len(d["gold"].get("per_language", {})) < 2:
        continue
    rows.append(("transformer", d["run"], d["gold"]["macro_f1"], d["challenge"]["overall_fail_rate"]))
llm = pd.read_csv(ROOT / "results/llm.csv")
for r in llm.itertuples():
    rows.append(("llm", r.model, r.gold_macro_f1, r.challenge_fail_rate))
fc = pd.read_csv(ROOT / "results/final_candidates.csv")
fin = fc[fc.candidate == "avg_all"].iloc[0]
t = pd.DataFrame(rows, columns=["kind", "name", "gold", "fail"])
t = t[t.gold > 0.3]  # drop majority / random / failed-to-train run (shown in tables)
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
marker = {"baseline": "s", "transformer": "o", "llm": "^"}
label = {"baseline": "Baselines", "transformer": "Fine-tuned encoders", "llm": "Local LLM"}
for k in ("baseline", "transformer", "llm"):
    s = t[t.kind == k]
    ax.scatter(s.fail, s.gold, s=16, marker=marker[k], facecolor="#bdbcb5" if k != "transformer" else "#8a8980",
               edgecolor="white", lw=0.5, label=label[k], zorder=2)
base = t[t.name == "tfidf_cnb"].iloc[0]
ax.scatter([base.fail], [base.gold], s=40, marker="s", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
ax.annotate("Baseline to beat\n(TF-IDF + CNB)", (base.fail, base.gold), xytext=(0.315, 0.79), fontsize=6.5, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
ax.scatter([fin.challenge_fail_rate], [fin.gold_macro_f1], s=55, marker="*", color=BLUE, edgecolor="white", lw=0.6, zorder=3)
ax.annotate("Final ensemble", (fin.challenge_fail_rate, fin.gold_macro_f1), xytext=(0.215, 0.82), fontsize=6.5, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
ax.set_xlabel("Challenge-suite failure rate (lower is better)")
ax.set_ylabel("Gold macro F1")
ax.legend(loc="lower left", fontsize=6.3, frameon=False, handletextpad=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig2_accuracy_robustness.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 failure rate per category
rb = pd.read_csv(ROOT / "results/robustness.csv").set_index("category")
cats = ["negation", "dir_negation", "sarcasm", "implicit", "different_target", "conspiracy", "neutral_topic",
        "romanized", "codemixed", "typos", "emoji", "short", "long", "offtopic"]
nice = {"dir_negation": "negation flip (DIR)", "different_target": "different target", "neutral_topic": "neutral topic",
        "codemixed": "code-mixed", "offtopic": "off-topic"}
f = rb.loc[cats, "final_avg_all"]; bl = rb.loc[cats, "final_tfidf_cnb"]
fig, ax = plt.subplots(figsize=(3.5, 3.1))
y = range(len(cats)); h = 0.38
ax.barh([i - h / 2 - 0.02 for i in y], f, height=h, color=BLUE, label="Final ensemble", zorder=2)
ax.barh([i + h / 2 + 0.02 for i in y], bl, height=h, color=ORANGE, hatch="////", edgecolor="white", lw=0, label="Baseline (TF-IDF + CNB)", zorder=2)
for i, v in enumerate(f):
    if v == 0:
        ax.text(0.005, i - h / 2 - 0.02, "0", va="center", fontsize=6.3, color=INK)
ax.set_yticks(list(y)); ax.set_yticklabels([nice.get(c, c) for c in cats]); ax.invert_yaxis()
ax.set_xlabel("Failure rate (lower is better)")
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", fontsize=6.3, frameon=False)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig3_category_failure.png"); plt.close(fig)
print("ok", len(t), "points")
