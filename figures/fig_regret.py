"""Fig. 3 (distribution of window regret) in the style of the micro F1 chart, without grid:
seaborn white theme, default tab10 colors, sans-serif text, legend inside the right panel."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="white", context="paper", palette="tab10", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})

import os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "pilot", "results")
ESTS = ("dist", "ordermean", "fixorder", "mean")
L = json.load(open(f"{RES}/london_raw.json"))
lon = {e: [] for e in ESTS}
for key, r, _ in L["regs"]:
    if abs(key[0] - 0.5) < 1e-9:
        for e in ESTS:
            lon[e] += r[e]
S = json.load(open(f"{RES}/sumo_ms_raw.json"))
sumo = {e: [] for e in ESTS}
for a, o, r in S:
    if r:
        for e in ESTS:
            sumo[e] += r[e]["win"] if isinstance(r[e], dict) else r[e]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
grid = np.linspace(0, 60, 601)
for ax, data, title in ((axs[0], lon, "London model, single stop (900 windows)"), (axs[1], sumo, "SUMO, two stops (900 windows)")):
    for e in ESTS:
        v = np.array(data[e]) * 100
        ax.plot(grid, [(v <= g).mean() * 100 for g in grid], lw=1.5, label=e)
    ax.axvline(1, color="black", lw=0.6, ls=":")
    ax.set_title(title); ax.set_xlabel("Largest regret in the window (% of $R$)")
    ax.set_xlim(0, 60); ax.set_ylim(0, 102)
axs[0].set_ylabel("Windows with regret $\\leq x$ (%)")
axs[0].text(1.8, 4, "stability threshold", fontsize=6.5)
axs[1].tick_params(labelleft=True)   # y values on the right panel too
axs[1].legend(loc="lower right", fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_regret.png"), dpi=200, bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_regret.pdf"), bbox_inches="tight")
print({e: len(lon[e]) for e in ESTS}, {e: len(sumo[e]) for e in ESTS})
