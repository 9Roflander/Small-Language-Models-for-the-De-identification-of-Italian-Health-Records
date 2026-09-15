"""
Generate all figures for the thesis from the project's actual result JSONs.
Outputs PDF (vector, for LaTeX \includegraphics) into ./figures/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

C_OURS_3B   = "#1f5fa8"   # blue    - Llama-3.2-3B (ours)
C_OURS_1B   = "#2e9e6b"   # green   - Gemma-3 1B (ours)
C_PAPER     = "#a3a3a3"   # grey    - zero-shot baselines
C_PAPER_BEST= "#6b6b6b"   # darker grey - gemma3:12b
C_V3        = "#c98a3e"   # amber   - v3
C_V4        = "#1f5fa8"   # blue    - v4
C_ACCENT    = "#b23b3b"   # red accent

def load(fn):
    with open(fn) as f:
        return json.load(f)

v3 = load("cv_v3_results.json")
v4 = load("cv_v4_results.json")
g1b = load("cv_gemma1b_results.json")

CATS = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]
CATS_SHORT = ["NOME", "ETÀ", "DATA", "LUOGO"]

# =====================================================================
# Figure 1 — headline leaderboard: paper zero-shot baselines vs ours
# =====================================================================
models = [
    ("phi4:14b\n(zero-shot)", 0.39, C_PAPER),
    ("llama3.2:3b\n(zero-shot)", 0.41, C_PAPER),
    ("mistral:7b\n(zero-shot)", 0.43, C_PAPER),
    ("gemma3:4b\n(zero-shot)", 0.51, C_PAPER),
    ("gemma3:12b\n(zero-shot, paper best)", 0.620, C_PAPER_BEST),
    ("Gemma-3 1B\n(ours, fine-tuned)", g1b["aggregate"]["macro_f1_mean"], C_OURS_1B),
    ("Llama-3.2-3B\n(ours, fine-tuned v4)", v4["aggregate"]["macro_f1_mean"], C_OURS_3B),
]
models_sorted = sorted(models, key=lambda x: x[1])
labels = [m[0] for m in models_sorted]
vals = [m[1] for m in models_sorted]
colors = [m[2] for m in models_sorted]

errs = [0]*5 + [g1b["aggregate"]["macro_f1_std"], v4["aggregate"]["macro_f1_std"]]
errs_sorted = []
for lab, v, c in models_sorted:
    if "Gemma-3 1B" in lab:
        errs_sorted.append(g1b["aggregate"]["macro_f1_std"])
    elif "Llama-3.2-3B" in lab:
        errs_sorted.append(v4["aggregate"]["macro_f1_std"])
    else:
        errs_sorted.append(0)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = np.arange(len(labels))
bars = ax.barh(y, vals, xerr=errs_sorted, color=colors, height=0.62,
                error_kw=dict(ecolor="#333333", capsize=3, lw=1))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Macro F1 (deterministic placeholder-count metric)")
ax.set_xlim(0, 0.78)
ax.axvline(0.620, color=C_ACCENT, linestyle="--", linewidth=1, alpha=0.8)
ax.text(0.622, 6.65, "paper best (0.620)", color=C_ACCENT, fontsize=8.5, va="top")
for yi, v in zip(y, vals):
    ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=9)
ax.set_title("Macro-F1 across zero-shot baselines and fine-tuned models\n(5-fold CV mean $\\pm$ std where applicable)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_leaderboard.pdf")
plt.close(fig)

# =====================================================================
# Figure 2 — per-category F1: paper best vs Llama v4 vs Gemma-1B
# =====================================================================
paper_best = {"NOME": 0.81, "ETÀ": 0.14, "DATA": 0.65, "LUOGO/INDIRIZZO": 0.88}
llama_v4 = {c: v4["aggregate"]["per_category"][c]["f1_mean"] for c in CATS}
llama_v4_std = {c: v4["aggregate"]["per_category"][c]["f1_std"] for c in CATS}
gemma1b_f1 = {c: g1b["aggregate"]["per_category"][c]["f1_mean"] for c in CATS}
gemma1b_std = {c: g1b["aggregate"]["per_category"][c]["f1_std"] for c in CATS}

x = np.arange(len(CATS))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(x - w, [paper_best[c] for c in CATS], width=w, color=C_PAPER_BEST, label="gemma3:12b (paper best, zero-shot)")
ax.bar(x,     [llama_v4[c] for c in CATS], width=w, yerr=[llama_v4_std[c] for c in CATS],
       color=C_OURS_3B, label="Llama-3.2-3B v4 (ours)", error_kw=dict(ecolor="#333", capsize=3, lw=1))
ax.bar(x + w, [gemma1b_f1[c] for c in CATS], width=w, yerr=[gemma1b_std[c] for c in CATS],
       color=C_OURS_1B, label="Gemma-3 1B (ours)", error_kw=dict(ecolor="#333", capsize=3, lw=1))
ax.set_xticks(x)
ax.set_xticklabels(CATS_SHORT)
ax.set_ylabel("F1")
ax.set_ylim(0, 1.02)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=9)
ax.set_title("Per-category F1: fine-tuned models vs. best zero-shot baseline", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_per_category.pdf")
plt.close(fig)

# =====================================================================
# Figure 3 — per-fold macro F1, v3 vs v4 (variance reduction)
# =====================================================================
folds = [1, 2, 3, 4, 5]
v3_folds = v3["aggregate"]["fold_macros"]
v4_folds = v4["aggregate"]["fold_macros"]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(folds, v3_folds, marker="o", color=C_V3, label=f"v3 (878 synthetic; $\\mu={np.mean(v3_folds):.3f}$, $\\sigma={np.std(v3_folds):.3f}$)", linewidth=2)
ax.plot(folds, v4_folds, marker="s", color=C_V4, label=f"v4 (1801 synthetic; $\\mu={np.mean(v4_folds):.3f}$, $\\sigma={np.std(v4_folds):.3f}$)", linewidth=2)
ax.axhline(0.620, color=C_ACCENT, linestyle="--", linewidth=1, alpha=0.8)
ax.text(1.02, 0.628, "paper best (0.620)", color=C_ACCENT, fontsize=8.5)
ax.set_xticks(folds)
ax.set_xlabel("Cross-validation fold")
ax.set_ylabel("Macro F1")
ax.set_ylim(0, 0.85)
ax.legend(loc="lower center", frameon=False, fontsize=9)
ax.set_title("Per-fold macro F1: doubling style-anchored synthetic data\nhalves cross-fold variance (v3 $\\to$ v4)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_fold_variance.pdf")
plt.close(fig)

# =====================================================================
# Figure 4 — training data composition (v4)
# =====================================================================
sources = ["Gemini synth.\n(with names)", "Gemini synth.\n(name-free)", "Gold standard\n(5$\\times$ oversampled)", "CRF negatives\n(no-PII)"]
counts = [1201, 475, 320, 80]
colors4 = ["#1f5fa8", "#5b9bd5", "#2e9e6b", "#c98a3e"]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
bars = ax.bar(sources, counts, color=colors4, width=0.6)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, c + 25, str(c), ha="center", fontsize=9.5)
ax.set_ylabel("Training records (per fold)")
ax.set_title("Composition of the v4 training set (2 076 records/fold)", fontsize=11)
ax.set_ylim(0, 1350)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_data_composition.pdf")
plt.close(fig)

# =====================================================================
# Figure 5 — iteration timeline (macro F1 across project history)
# =====================================================================
iters = [
    ("JSON\nextraction", 0.13, "single-split"),
    ("No-gold\ncontrol (C2)", 0.18, "all-80 eval"),
    ("Redaction v2\n(Gemini synth)", 0.635, "held-out 16"),
    ("Redaction v3\n(+ no-name)", 0.692, "held-out 16"),
    ("v3, 5-fold CV", 0.535, "CV mean"),
    ("v4, 5-fold CV", 0.649, "CV mean"),
]
xl = [i[0] for i in iters]
yv = [i[1] for i in iters]
fig, ax = plt.subplots(figsize=(7.4, 4.3))
xs = np.arange(len(iters))
colors5 = [C_ACCENT, C_ACCENT, C_V3, C_V3, "#c98a3e", C_OURS_3B]
ax.plot(xs, yv, color="#999999", linewidth=1.4, zorder=1)
ax.scatter(xs, yv, color=colors5, s=90, zorder=2, edgecolor="white", linewidth=1)
for xi, yi, note in zip(xs, yv, [i[2] for i in iters]):
    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5)
    ax.annotate(note, (xi, yi), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7.5, color="#666666", style="italic")
ax.axhline(0.41, color="#999999", linestyle=":", linewidth=1)
ax.text(-0.35, 0.425, "llama3.2:3b zero-shot (0.41)", fontsize=7.5, color="#666666")
ax.axhline(0.620, color=C_ACCENT, linestyle="--", linewidth=1)
ax.text(-0.35, 0.635, "gemma3:12b zero-shot (0.62)", fontsize=7.5, color=C_ACCENT)
ax.set_xticks(xs)
ax.set_xticklabels(xl, fontsize=8.5)
ax.set_ylabel("Macro F1")
ax.set_ylim(0, 0.85)
ax.set_title("Macro-F1 across the project's development timeline", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_iteration_timeline.pdf")
plt.close(fig)

# =====================================================================
# Figure 6 — precision vs recall per category (v4)
# =====================================================================
fig, ax = plt.subplots(figsize=(6.4, 5.4))
markers = ["o", "s", "^", "D"]
for i, c in enumerate(CATS):
    p = v4["aggregate"]["per_category"][c]["p_mean"]
    r = v4["aggregate"]["per_category"][c]["r_mean"]
    ax.scatter(r, p, s=170, marker=markers[i], color=[C_OURS_3B, C_OURS_1B, "#c98a3e", C_ACCENT][i],
               label=CATS_SHORT[i], edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(CATS_SHORT[i], (r, p), textcoords="offset points", xytext=(9, 4), fontsize=9.5)
# iso-F1 curves
for f in [0.2, 0.4, 0.6, 0.8]:
    rr = np.linspace(f/2, 1, 100)
    pp = (f * rr) / (2 * rr - f)
    valid = (pp > 0) & (pp <= 1.05)
    ax.plot(rr[valid], pp[valid], color="#cccccc", linewidth=0.8, zorder=1)
    idx = int(len(rr[valid]) * 0.75) if valid.any() else None
    if idx:
        ax.text(rr[valid][idx], pp[valid][idx], f"F1={f}", fontsize=7, color="#999999")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–recall operating point per category\n(Llama-3.2-3B v4, 5-fold CV mean)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_precision_recall.pdf")
plt.close(fig)

# =====================================================================
# Figure 7 — architecture comparison: Llama-3B vs Gemma-1B, per category
# =====================================================================
fig, ax = plt.subplots(figsize=(7.0, 4.2))
w = 0.35
ax.bar(x - w/2, [llama_v4[c] for c in CATS], width=w, yerr=[llama_v4_std[c] for c in CATS],
       color=C_OURS_3B, label="Llama-3.2-3B (3B params)", error_kw=dict(ecolor="#333", capsize=3, lw=1))
ax.bar(x + w/2, [gemma1b_f1[c] for c in CATS], width=w, yerr=[gemma1b_std[c] for c in CATS],
       color=C_OURS_1B, label="Gemma-3 1B (1B params)", error_kw=dict(ecolor="#333", capsize=3, lw=1))
ax.set_xticks(x)
ax.set_xticklabels(CATS_SHORT)
ax.set_ylabel("F1")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.set_title("Cross-architecture comparison under the identical v4 pipeline", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_architecture_comparison.pdf")
plt.close(fig)

print("All figures written to", os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    print(" -", f)
