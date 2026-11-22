import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

DATA = r"E:\IGS2026_AgenticAI_Kit\data\train_data.csv"
OUT_DIR = r"E:\IGS2026_AgenticAI_Kit\results\figures"

CURVES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
WINDOW = None 
CLIP = False
SHADE = 0.40

CURVE_STYLE = {
    "GR":        ("black",   "-",                1.6),
    "ILD_log10": ("#8B0000", "--",               1.6),
    "DeltaPHI":  ("#00008B", "-.",               1.6),
    "PHIND":     ("#004D00", ":",                2.0),
    "PE":        ("#4B0082", (0, (5, 1, 1, 1, 1, 1)), 1.6),
}

NAMES = {1: "nonmarine sandstone", 
         2: "nonmarine coarse siltstone",
         3: "nonmarine fine siltstone", 
         4: "marine siltstone and shale",
         5: "mudstone", 
         6: "wackestone", 
         7: "dolomite",
         8: "packstone-grainstone", 
         9: "phylloid-algal bafflestone"}
cmap = plt.get_cmap("tab10")

def pick_window(w, n):
    best_start, best_count = 0, -1
    for s in range(0, max(1, len(w) - n + 1), 10):
        c = w["Facies"].iloc[s:s + n].nunique()
        if c > best_count:
            best_start, best_count = s, c
    return w.iloc[best_start:best_start + n]


def facies_runs(f):
    """Yield (first_index, last_index, facies) for each stretch of one facies."""
    start = 0
    for k in range(1, len(f) + 1):
        if k == len(f) or f[k] != f[start]:
            yield start, k - 1, int(f[start])
            start = k


def depth_edges(d):
    """Boundaries halfway between samples, so each shaded band covers its own sample."""
    if len(d) < 2:
        return np.array([d[0] - 0.25, d[0] + 0.25])
    mid = (d[:-1] + d[1:]) / 2
    return np.concatenate([[d[0] - (d[1] - d[0]) / 2], mid, [d[-1] + (d[-1] - d[-2]) / 2]])


def draw(w, well):
    d = w["Depth"].to_numpy()
    f = w["Facies"].to_numpy()
    edges = depth_edges(d)
    runs = list(facies_runs(f))

    fig, axes = plt.subplots(1, len(CURVES), figsize=(12, 7), sharey=True)
    for ax, c in zip(axes, CURVES):
        for i, j, fac in runs:
            ax.axhspan(edges[i], edges[j + 1], facecolor=cmap(fac - 1),
                       alpha=SHADE, linewidth=0, zorder=0)
        color, ls, lw = CURVE_STYLE[c]
        ax.plot(w[c], d, color=color, linestyle=ls, linewidth=lw, zorder=3)
        if CLIP:
            lo, hi = np.percentile(w[c], [1, 99])
            pad = 0.05 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
        ax.locator_params(axis="x", nbins=3)
        ax.set_title(c, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(edges[-1], edges[0])   # deepest at the bottom
    axes[0].set_ylabel("Depth (ft)")

    fig.suptitle(well, fontsize=13)
    fig.legend([Patch(facecolor=cmap(k - 1), alpha=SHADE) for k in NAMES],
               [f"{k}  {NAMES[k]}" for k in NAMES],
               loc="center left", bbox_to_anchor=(0.785, 0.5),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.78, 0.95))
    return fig


df = pd.read_csv(DATA)
os.makedirs(OUT_DIR, exist_ok=True)

with PdfPages(os.path.join(OUT_DIR, "all_wells.pdf")) as pdf:
    for well in sorted(df["Well Name"].unique()):
        w = df[df["Well Name"] == well].sort_values("Depth").reset_index(drop=True)
        if WINDOW and len(w) > WINDOW:
            w = pick_window(w, WINDOW)
        fig = draw(w, well)
        fig.savefig(os.path.join(OUT_DIR, f"strip_{well}.png"), dpi=200)
        pdf.savefig(fig)
        plt.close(fig)
        print(well, len(w), "rows, depth", w["Depth"].min(), "to", w["Depth"].max(),
              "facies", sorted(int(k) for k in w["Facies"].unique()))

print("saved to", OUT_DIR)