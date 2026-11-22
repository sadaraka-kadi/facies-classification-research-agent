# Run:  python well02_all_curves.py   (from E:\IGS2026_AgenticAI_Kit, .venv active)
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = sys.argv[1] if len(sys.argv) > 1 else r"data\train_data.csv"
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"E:\talk\well02_all_curves.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
CURVES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
W2 = "WELL_02"
df = pd.read_csv(DATA)
real = df[df["Well Name"] != W2].copy()
w2 = df[df["Well Name"] == W2].copy()
INK, OLIVE = "#1B3A5C", "#bcbd22"

fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharey="row")
print(f"{'curve':10s} {'realF9 median z':>16s} {'WELL_02 pooled-z mean':>22s} {'gap':>6s}")
for j, c in enumerate(CURVES):
    r = real.copy()
    r["z"] = r.groupby("Well Name")[c].transform(lambda s: (s - s.mean()) / s.std())
    f9 = r.loc[r["Facies"] == 9, "z"]
    sd = w2[c].std()
    before = ((w2[c] - w2[c].mean()) / sd) if sd > 0 else pd.Series(0.0, index=w2.index)
    after = (w2[c] - real[c].mean()) / real[c].std()
    print(f"{c:10s} {f9.median():16.2f} {after.mean():22.2f} {after.mean()-f9.median():6.2f}")
    lo, hi = np.percentile(np.r_[f9, after], [0.5, 99.5]); lo = min(lo, -2.5); hi = max(hi, 3)
    bins = np.linspace(lo, hi, 36)
    for i, (vals, lab) in enumerate([(before, "before"), (after, "after")]):
        ax = axes[i, j]
        ax.hist(f9, bins=bins, color=OLIVE, alpha=.85)
        ax.hist(vals, bins=bins, color="none", edgecolor=INK, hatch="///", lw=1)
        ax.axvline(f9.median(), color=OLIVE, ls="--", lw=1.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if i == 0: ax.set_title(c, fontsize=12, color=INK, fontweight="bold")
        if j == 0: ax.set_ylabel(f"{lab}: rows", fontsize=11, color=INK)
        if i == 1: ax.set_xlabel("z-score", fontsize=10, color=INK)
fig.suptitle("Facies 9 (olive) vs WELL_02 (hatched): top = z-scored against itself, bottom = other wells' stats",
             fontsize=13, color=INK, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT, dpi=170, facecolor="white")
print("saved", OUT)