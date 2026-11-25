"""Fig. 2 (stable windows by setting and estimator) redrawn in the style of the micro F1 chart, without grid:
seaborn white theme, default tab10 colors, sans-serif text, bar_label values, tilted category labels,
title on top and legend below."""
import os
import numpy as np, matplotlib
HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
sns.set_theme(style="white", context="paper", palette="tab10", rc={"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

settings = ["Synthetic\n(cv 0.3)", "London\nsingle stop", "SUMO\nsingle stop", "London\ntwo stops", "SUMO\ntwo stops"]
share = {"dist": [94, 94, 86, 91, 89], "ordermean": [83, 78, 28, 69, 16], "fixorder": [39, 28, 9, 11, 3], "mean": [38, 28, 10, 10, 3]}
share0 = {"dist": [94, 94, 86, 77, 88], "ordermean": [83, 78, 28, 60, 16], "fixorder": [39, 28, 9, 6, 2], "mean": [38, 28, 10, 5, 2]}

fig, ax = plt.subplots(figsize=(7.16, 2.1))
x, w = np.arange(len(settings)), 0.2
for j, (e, vals) in enumerate(share.items()):
    xs = x + (j - 1.5) * w
    bars = ax.bar(xs, vals, w, label=e)
    ax.bar_label(bars, fmt="%d", fontsize=6.5, padding=1)
    for xi, v, v0 in zip(xs, vals, share0[e]):
        if v0 != v:   # stability under the nominal reward R0, when it differs from the announced reward
            ax.plot([xi - w * 0.45, xi + w * 0.45], [v0, v0], color="black", lw=1.4)
ax.set_xticks(x, settings)
ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Stable windows (%)")
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color="black", lw=1.4)); labels.append("under $R_0$ (if different)")
ax.legend(handles, labels, ncol=5, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.savefig(os.path.join(HERE, "fig_stability.png"), dpi=200, bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_stability.pdf"), bbox_inches="tight")
