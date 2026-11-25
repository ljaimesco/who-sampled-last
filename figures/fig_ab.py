"""Fig. 1: arrival-time densities in the A/B example (lognormal travel times, means 100 and 115 s, cv 0.30;
X last sampled 200 s before departure). Single-column figure, seaborn white theme, tab10 colors."""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="white", context="paper", palette="tab10",
              rc={"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})

def lognorm(mean, cv):
    s2 = np.log(1 + cv * cv); return stats.lognorm(s=np.sqrt(s2), scale=mean / np.sqrt(1 + cv * cv))
A, B = lognorm(100, 0.3), lognorm(115, 0.3)
t = np.linspace(20, 260, 600)
pal = sns.color_palette("tab10")

fig, ax = plt.subplots(figsize=(3.45, 2.0))          # single-column width
for d, name, c, ls in ((A, "A (mean 100 s)", pal[0], "-"), (B, "B (mean 115 s)", pal[1], "--")):
    ax.plot(t, d.pdf(t) * 1000, color=c, lw=1.5, ls=ls, label=name)
    ax.axvline(d.mean(), color=c, lw=0.8, ls=":")
ax.set_xlim(20, 260); ax.set_ylim(0, 16); ax.set_yticks([0, 5, 10, 15])
ax.set_xlabel("Arrival time at X after departure (s)"); ax.set_ylabel("Density ($\\times10^{-3}$)")
ax.legend(loc="upper right", fontsize=6.5, frameon=False)
ax.text(0.98, 0.66, "B arrives first: $P=0.37$\nB's value at X:\n11.8 with mean times\n50.5 expected",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.5, linespacing=1.3)
fig.savefig(os.path.join(HERE, "fig_ab.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_ab.png"), dpi=220, bbox_inches="tight")
