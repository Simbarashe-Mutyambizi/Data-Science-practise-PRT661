# %%
"""
Demographic_error.py  -  PRT661 Group Dan3, Assessment 4

Error analysis across demographic groups and conditions.

The best model from Model_training_evaluation.py is retrained with grouped
5-fold cross-validation over the FULL cleaned dataset. Every row therefore gets
an out-of-fold prediction from a model that never saw its demographic group,
so every subgroup is evaluated on all of its rows, not only the ~20% that fell
in the hold-out test set.

For each grouping (LTHC, age group, sex, region, subregion, years in
Australia, English proficiency, source table) it reports:
  n             rows in the subgroup
  Mean actual   mean true prevalence (%)
  MAE (pp)      mean absolute error in percentage points
  Bias (pp)     mean (predicted - actual); + = over-predicts, - = under-predicts
  Relative MAE  MAE / mean actual, so conditions with different base rates compare fairly
95% confidence intervals use a cluster bootstrap (resampling whole demographic
groups, because the conditions of one group are not independent).

Input : Cleaned_data_A4.csv, outputs/best_model_pipeline.pkl, outputs/best_model.json
Output: outputs/demographic_error_all.csv        (numbers + CIs)
        outputs/demographic_error_tables.xlsx    (one formatted sheet per grouping)
        outputs/out_of_fold_predictions.csv
        outputs/figures/demographic_error.png
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

RANDOM_STATE = 42
N_FOLDS = 5
N_BOOT = 1000
MIN_N = 30  # subgroups smaller than this are flagged as unreliable

DATA_FILE = Path("Cleaned_data_A4.csv")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Age-specific percentage of population reporting LTHC(s)"
GROUP = "demographic_group"
NOT_COLLECTED = "Not collected"

# Grouping label -> (column, which table the variable comes from)
DIMENSIONS = {
    "Source table": ("source_table", None),
    "Condition (LTHC)": ("Long-term health condition (LTHC)", None),
    "Age group": ("Age group", None),
    "Sex": ("Sex", None),
    "Region of birth": ("Region_class", "S10"),
    "Subregion of birth": ("Subregion_class", "S10"),
    "Years in Australia": ("Years spent in Australia", "S10"),
    "English proficiency": ("Proficiency in spoken English", "S13"),
}
FIGURE_DIMENSIONS = [d for d in DIMENSIONS if d != "Subregion of birth"]  # 33 subregions: table only
SORT_BY_ERROR = {"Condition (LTHC)", "Region of birth", "Subregion of birth"}
SOURCE_LABELS = {"S10": "S10 - country of birth rows", "S13": "S13 - language at home rows"}

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED, NEUTRAL, DARK = "#2a78d6", "#e34948", "#a8a7a2", "#3d3d3a"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# %%
# 1. Retrain the best model with grouped CV to get an out-of-fold prediction for every row
meta = json.load(open(OUT_DIR / "best_model.json"))
best_name, features = meta["best_model"], meta["features"]
model = clone(joblib.load(OUT_DIR / "best_model_pipeline.pkl"))  # same settings, unfitted

df = pd.read_csv(DATA_FILE)
oof = cross_val_predict(
    model, df[features], df[TARGET], groups=df[GROUP],
    cv=GroupKFold(n_splits=N_FOLDS), n_jobs=-1,
)

df["Actual"] = df[TARGET]
df["Predicted"] = oof
df["Error"] = df["Predicted"] - df["Actual"]   # + = over-prediction
df["Abs_Error"] = df["Error"].abs()
df.to_csv(OUT_DIR / "out_of_fold_predictions.csv", index=False)

overall_mae = df["Abs_Error"].mean()
overall_rel = df["Abs_Error"].sum() / df["Actual"].sum() * 100
print(f"{best_name}, out-of-fold over all {len(df):,} rows:")
print(f"  MAE {overall_mae:.3f} pp | RMSE {np.sqrt(mean_squared_error(df['Actual'], oof)):.3f} pp | "
      f"R2 {r2_score(df['Actual'], oof):.4f} | bias {df['Error'].mean():+.3f} pp | "
      f"relative MAE {overall_rel:.1f}%")


# %%
# 2. Subgroup summaries with cluster-bootstrap confidence intervals
def bootstrap_ci(sub, rng):
    """95% CIs for MAE, bias and relative MAE, resampling whole demographic groups."""
    per_group = sub.groupby(GROUP).agg(
        abs_sum=("Abs_Error", "sum"), err_sum=("Error", "sum"),
        act_sum=("Actual", "sum"), n=("Actual", "size"),
    ).to_numpy()
    draws = rng.integers(0, len(per_group), size=(N_BOOT, len(per_group)))
    totals = per_group[draws].sum(axis=1)
    mae = totals[:, 0] / totals[:, 3]
    bias = totals[:, 1] / totals[:, 3]
    rel = totals[:, 0] / totals[:, 2] * 100
    lo, hi = 2.5, 97.5
    return {
        "MAE_lo": np.percentile(mae, lo), "MAE_hi": np.percentile(mae, hi),
        "Bias_lo": np.percentile(bias, lo), "Bias_hi": np.percentile(bias, hi),
        "RelMAE_lo": np.percentile(rel, lo), "RelMAE_hi": np.percentile(rel, hi),
    }


def category_label(value, dimension):
    column, table = DIMENSIONS[dimension]
    if column == "source_table":
        return SOURCE_LABELS.get(value, value)
    if value == NOT_COLLECTED:
        other = "S13" if table == "S10" else "S10"
        return f"Not collected ({other} rows)"
    return value


rng = np.random.default_rng(RANDOM_STATE)
rows = []
for dimension, (column, _) in DIMENSIONS.items():
    for value, sub in df.groupby(column):
        rows.append({
            "Group": dimension,
            "Category": category_label(value, dimension),
            "Collected": value != NOT_COLLECTED,
            "n": len(sub),
            "Demographic groups": sub[GROUP].nunique(),
            "Mean actual (%)": sub["Actual"].mean(),
            "MAE (pp)": sub["Abs_Error"].mean(),
            "Bias (pp)": sub["Error"].mean(),
            "Relative MAE (%)": sub["Abs_Error"].sum() / sub["Actual"].sum() * 100,
            **bootstrap_ci(sub, rng),
            "Low n": len(sub) < MIN_N,
        })

results = pd.DataFrame(rows)


def order_within(dim_rows, dimension):
    if dimension in SORT_BY_ERROR:
        dim_rows = dim_rows.sort_values("Relative MAE (%)", ascending=False)
    return dim_rows.sort_values("Collected", ascending=False, kind="stable")  # 'Not collected' last


results = pd.concat(
    [order_within(results[results["Group"] == d], d) for d in DIMENSIONS], ignore_index=True
)
results.round(4).to_csv(OUT_DIR / "demographic_error_all.csv", index=False)

# %%
# 3. Formatted tables (same layout as the report table), one sheet per grouping
def formatted(dim_rows):
    return pd.DataFrame({
        "Category": dim_rows["Category"] + np.where(dim_rows["Low n"], " *", ""),
        "n": dim_rows["n"],
        "Mean actual": dim_rows["Mean actual (%)"].map(lambda v: f"{v:.1f}%"),
        "MAE (pp)": dim_rows["MAE (pp)"].map(lambda v: f"{v:.2f}"),
        "Bias (pp)": dim_rows["Bias (pp)"].map(lambda v: f"{v:+.2f}"),
        "Relative MAE": dim_rows["Relative MAE (%)"].map(lambda v: f"{v:.0f}%"),
        "Bias 95% CI (pp)": [f"{lo:+.2f} to {hi:+.2f}"
                             for lo, hi in zip(dim_rows["Bias_lo"], dim_rows["Bias_hi"])],
    })


with pd.ExcelWriter(OUT_DIR / "demographic_error_tables.xlsx") as writer:
    for dimension in DIMENSIONS:
        table = formatted(results[results["Group"] == dimension])
        table.to_excel(writer, sheet_name=dimension[:31], index=False)
        print(f"\n{dimension.upper()}")
        print(table.to_string(index=False))
print(f"\n* = fewer than {MIN_N} rows; treat as unreliable.")

# %%
# 4. Figure: relative MAE and bias by subgroup (forest-plot style)
# 'Not collected' rows are left out of the figure: they are the same rows as the
# 'Source table' block at the top.
plot_rows = []
for dimension in FIGURE_DIMENSIONS:
    dim_rows = results[(results["Group"] == dimension) & results["Collected"]]
    plot_rows.append({"header": dimension})
    plot_rows += [{"header": None, **r} for r in dim_rows.to_dict("records")]

n_rows = len(plot_rows)
y = np.arange(n_rows)[::-1]
fig, (ax_rel, ax_bias) = plt.subplots(
    1, 2, figsize=(11, 0.24 * n_rows + 1.6), sharey=True,
    gridspec_kw={"width_ratios": [1.15, 1]},
)

labels = []
for yi, r in zip(y, plot_rows):
    if r["header"]:
        labels.append(r["header"])
        for ax in (ax_rel, ax_bias):
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f0efec", zorder=0, lw=0)
        continue
    labels.append(f"{r['Category']}  (n={r['n']:,})")
    hollow = r["Low n"]

    # Relative MAE with CI
    ax_rel.plot([r["RelMAE_lo"], r["RelMAE_hi"]], [yi, yi], color=DARK, lw=2, solid_capstyle="round")
    ax_rel.plot(r["Relative MAE (%)"], yi, "o", ms=6.5, color=DARK,
                mfc="white" if hollow else DARK, mew=1.6)

    # Bias with CI, coloured only when the CI excludes zero
    if r["Bias_lo"] > 0:
        colour = RED
    elif r["Bias_hi"] < 0:
        colour = BLUE
    else:
        colour = NEUTRAL
    ax_bias.plot([r["Bias_lo"], r["Bias_hi"]], [yi, yi], color=colour, lw=2, solid_capstyle="round")
    ax_bias.plot(r["Bias (pp)"], yi, "o", ms=6.5, color=colour,
                 mfc="white" if hollow else colour, mew=1.6)

ax_rel.set_yticks(y, labels)
for tick, r in zip(ax_rel.get_yticklabels(), plot_rows):
    if r["header"]:
        tick.set_fontweight("bold")
        tick.set_color(INK)
ax_rel.set_ylim(-0.7, n_rows - 0.3)

ax_rel.axvline(overall_rel, color=INK_2, ls="--", lw=1)
ax_rel.annotate(f"overall {overall_rel:.0f}%", (overall_rel, n_rows - 0.3), xytext=(4, -10),
                textcoords="offset points", fontsize=8, color=INK_2)
ax_rel.set_xlim(left=0)
ax_rel.set_xlabel("Relative MAE (% of mean true prevalence), 95% CI")
ax_rel.set_title("(a) How far off, relative to the group's prevalence", loc="left")

ax_bias.axvline(0, color=INK_2, lw=1)
ax_bias.set_xlabel("Bias, predicted - actual (percentage points), 95% CI")
ax_bias.set_title("(b) Which direction", loc="left")
xmax = np.nanmax(np.abs([r.get(k, 0) for r in plot_rows if not r["header"]
                          for k in ("Bias_lo", "Bias_hi")])) * 1.1
ax_bias.set_xlim(-xmax, xmax)
# Direction labels sit in the first (shaded header) row, either side of zero
ax_bias.annotate("<- under-predicts", (0, y[0]), xytext=(-6, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=8.5, color=BLUE, fontweight="bold")
ax_bias.annotate("over-predicts ->", (0, y[0]), xytext=(6, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=8.5, color=RED, fontweight="bold")

for ax in (ax_rel, ax_bias):
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

fig.suptitle(
    f"Prediction error by subgroup - {best_name}, out-of-fold predictions "
    f"(grouped {N_FOLDS}-fold CV, {len(df):,} rows)",
    x=0.01, ha="left", fontweight="bold",
)
fig.text(
    0.01, -0.01,
    f"Hollow markers: fewer than {MIN_N} rows. Bias is grey when its 95% CI includes zero. "
    "CIs from a cluster bootstrap over demographic groups (1,000 resamples).",
    fontsize=8, color=INK_2,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "demographic_error.png")
plt.show()
print(f"\nSaved {FIG_DIR / 'demographic_error.png'}")
# %%
