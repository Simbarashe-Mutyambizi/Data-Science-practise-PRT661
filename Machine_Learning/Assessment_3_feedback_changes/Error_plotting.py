# %%
"""
Error_plotting.py  -  PRT661 Group Dan3, Assessment 4

Shows how prediction error changes as the true prevalence increases, on the
hold-out test set of the best model:

  (a) absolute error in percentage points (pp)  - grows with prevalence
  (b) relative error (% of the true value)      - shrinks with prevalence

The same model can look "least accurate" for high-prevalence groups in pp and
"least accurate" for low-prevalence groups in relative terms, so both are shown.

Input : outputs/test_predictions.csv (from Model_training_evaluation.py)
Output: outputs/figures/error_vs_prevalence.png
        outputs/error_vs_prevalence_bins.csv
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
N_BINS = 12  # equal-count bins along true prevalence

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#9ec5f4", "#eb6834"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# %%
errors = pd.read_csv(OUT_DIR / "test_predictions.csv")
best_name = json.load(open(OUT_DIR / "best_model.json"))["best_model"]
print(f"{best_name}: {len(errors):,} test predictions")

# Equal-count bins along the true prevalence, summarised by median and IQR
errors["bin"] = pd.qcut(errors["Actual"], q=N_BINS, duplicates="drop")
binned = errors.groupby("bin", observed=True).agg(
    n=("Actual", "size"),
    actual_median=("Actual", "median"),
    abs_err_median=("Abs_Error", "median"),
    abs_err_q1=("Abs_Error", lambda s: s.quantile(0.25)),
    abs_err_q3=("Abs_Error", lambda s: s.quantile(0.75)),
    rel_err_median=("Rel_Error_pct", "median"),
    rel_err_q1=("Rel_Error_pct", lambda s: s.quantile(0.25)),
    rel_err_q3=("Rel_Error_pct", lambda s: s.quantile(0.75)),
).reset_index()
binned["bin"] = binned["bin"].astype(str)
binned.round(3).to_csv(OUT_DIR / "error_vs_prevalence_bins.csv", index=False)
print(binned.round(2).to_string(index=False))

# %%
# Log axes labelled 0.01, 0.1, 1, 10, 100 instead of 10^-2 ...
plain_numbers = FuncFormatter(lambda v, _: f"{v:g}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
panels = [
    ("Abs_Error", "abs_err", "Absolute error (percentage points)",
     "(a) Absolute error grows with prevalence"),
    ("Rel_Error_pct", "rel_err", "Relative error (% of true value)",
     "(b) Relative error shrinks with prevalence"),
]
for ax, (col, key, ylabel, title) in zip(axes, panels):
    ax.scatter(errors["Actual"], errors[col], s=9, color=BLUE_LIGHT, alpha=0.6,
               linewidths=0, label="Individual test predictions")
    ax.fill_between(binned["actual_median"], binned[f"{key}_q1"], binned[f"{key}_q3"],
                    color=BLUE, alpha=0.18, linewidth=0, label="Middle 50% (IQR) per bin")
    ax.plot(binned["actual_median"], binned[f"{key}_median"], color=BLUE, lw=2,
            marker="o", ms=5, label="Median per bin")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plain_numbers)
    ax.yaxis.set_major_formatter(plain_numbers)
    ax.set_xlabel("Actual prevalence (%) - log scale")
    ax.set_ylabel(ylabel + " - log scale")
    ax.set_title(title, loc="left")

# Reference lines on the relative panel
for ref in (10, 50, 100):
    axes[1].axhline(ref, color=INK_2, lw=0.8, ls=":")
    axes[1].annotate(f"{ref}%", (axes[1].get_xlim()[1], ref), xytext=(-4, 3),
                     textcoords="offset points", ha="right", fontsize=8, color=INK_2)

axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
fig.suptitle(f"Prediction error vs true prevalence - {best_name}, hold-out test set "
             f"(n = {len(errors):,}, {N_BINS} equal-count bins)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "error_vs_prevalence.png")
plt.show()

low = binned.iloc[0]
high = binned.iloc[-1]
print(
    f"\nLowest bin  (median actual {low.actual_median:.2f}%): "
    f"median error {low.abs_err_median:.2f} pp = {low.rel_err_median:.0f}% of the true value"
)
print(
    f"Highest bin (median actual {high.actual_median:.1f}%): "
    f"median error {high.abs_err_median:.2f} pp = {high.rel_err_median:.0f}% of the true value"
)
# %%
