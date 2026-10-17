# %% =========================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from scipy import sparse

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

OUTPUT_DIR = Path("model_evaluation_results")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# --- Visual style: one accent colour, everything else neutral ---------
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text / axes
GRID = "#e6e5e1"         # recessive gridlines
NEUTRAL = "#b9b8b0"      # non-highlighted bars / "training" series
ACCENT = "#2a78d6"       # best model / "testing" series
WARN = "#eb6834"         # out-of-range predictions

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "xtick.labelsize": 12,
    "ytick.labelsize":12,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(FIGURE_DIR / name)


# %% =========================================================
# 2. LOAD PROCESSED DATA
# ============================================================

X_train = sparse.load_npz("X_train_processed.npz")
X_test = sparse.load_npz("X_test_processed.npz")

y_train = np.asarray(pd.read_csv("y_train.csv").squeeze("columns")).ravel()
y_test = np.asarray(pd.read_csv("y_test.csv").squeeze("columns")).ravel()

print("DATASET SUMMARY")
print("=" * 55)
print(f"Training features: {X_train.shape}")
print(f"Testing features:  {X_test.shape}")
print(f"Training targets:  {y_train.shape}")
print(f"Testing targets:   {y_test.shape}")

if X_train.shape[0] != len(y_train):
    raise ValueError("Training features and targets do not match.")
if X_test.shape[0] != len(y_test):
    raise ValueError("Testing features and targets do not match.")
if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Training and testing feature counts differ.")

# %% =========================================================
# 3. LOAD TRAINED MODELS
# ============================================================

models = {
    "Linear Regression": joblib.load("linear_regression_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "XGBoost": joblib.load("xgboost_regression_model.pkl"),
}

print("\nModels loaded successfully:")
for name in models:
    print(f" - {name}")

# %% =========================================================
# 4. GENERATE PREDICTIONS
# ============================================================

train_predictions = {}
test_predictions = {}

for name, model in models.items():
    train_predictions[name] = np.asarray(model.predict(X_train)).ravel()
    test_predictions[name] = np.asarray(model.predict(X_test)).ravel()

print("\nPredictions generated for all models.")

# %% =========================================================
# 5. EVALUATE TEST PERFORMANCE
# ============================================================

results = []
for name, prediction in test_predictions.items():
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
        "R2": r2_score(y_test, prediction),
    })

results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df["RMSE"].idxmin(), "Model"]

display_results = results_df.copy()
display_results[["MAE", "RMSE", "R2"]] = (
    display_results[["MAE", "RMSE", "R2"]].round(3)
)

print("\nTEST SET MODEL COMPARISON")
print("=" * 75)
print(display_results.to_string(index=False))
print(f"\nLowest RMSE: {best_model_name}")

# %% =========================================================
# 6. TRAINING VS TESTING PERFORMANCE
# ============================================================

overfitting_results = []
for name in models:
    overfitting_results.append({
        "Model": name,
        "Train MAE": mean_absolute_error(y_train, train_predictions[name]),
        "Test MAE": mean_absolute_error(y_test, test_predictions[name]),
        "Train R2": r2_score(y_train, train_predictions[name]),
        "Test R2": r2_score(y_test, test_predictions[name]),
    })

overfitting_df = pd.DataFrame(overfitting_results)
overfitting_df["R2 Difference"] = (
    overfitting_df["Train R2"] - overfitting_df["Test R2"]
)

print("\nTRAINING VS TESTING")
print("=" * 75)
print(overfitting_df.round(4).to_string(index=False))

# %% =========================================================
# 7. PREDICTION RANGE CHECK (table used by Figure 3)
# ============================================================

range_results = []
for name, prediction in test_predictions.items():
    range_results.append({
        "Model": name,
        "Negative predictions": int((prediction < 0).sum()),
        "Above 100%": int((prediction > 100).sum()),
        "Within 0–100%": int(((prediction >= 0) & (prediction <= 100)).sum()),
    })

range_df = pd.DataFrame(range_results)

print("\nPREDICTION RANGE VALIDATION")
print("=" * 65)
print(range_df.to_string(index=False))

# %% =========================================================
# 8. SHARED PLOTTING HELPERS
# ============================================================

model_order = list(models.keys())
SHORT = {                                  # compact labels save width
    "Linear Regression": "Linear Reg.",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


def hbar_panel(ax, values, title, fmt, best="min", xlim=None,
               show_labels=True, colors=None):
    """Horizontal bar panel; best bar in accent colour, rest neutral."""
    values = np.asarray(values, dtype=float)
    best_idx = values.argmin() if best == "min" else values.argmax()
    if colors is None:
        colors = [ACCENT if i == best_idx else NEUTRAL
                  for i in range(len(values))]
    y = np.arange(len(values))
    ax.barh(y, values, height=0.62, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in model_order] if show_labels
                       else [])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlim(*(xlim or (0, values.max() * 1.22)))
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {fmt.format(v)}", va="center", ha="left",
                fontsize=15, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


# %% =========================================================
# 9. FIGURE 1 — TEST PERFORMANCE (MAE | RMSE | R²) IN ONE ROW
# ============================================================

plot_results = results_df.set_index("Model").loc[model_order]

fig, axes = plt.subplots(1, 3, figsize=(11, 2.9),
                         gridspec_kw={"wspace": 0.12})

hbar_panel(axes[0], plot_results["MAE"], "MAE  ↓ lower is better",
           "{:.2f}", best="min")
hbar_panel(axes[1], plot_results["RMSE"], "RMSE  ↓ lower is better",
           "{:.2f}", best="min", show_labels=False)
hbar_panel(axes[2], plot_results["R2"], "R²  ↑ higher is better",
           "{:.3f}", best="max", show_labels=False, xlim=(0, 1.18))

fig.suptitle("Test-set performance by model (best highlighted)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
save(fig, "01_model_performance_comparison.png")
plt.show()

# %% =========================================================
# 10. FIGURE 2 — BEST-MODEL DIAGNOSTICS (3 PANELS, ONE ROW)
# ============================================================

best_predictions = test_predictions[best_model_name]
residuals = y_test - best_predictions            # actual − predicted

lo = min(y_test.min(), best_predictions.min())
hi = max(y_test.max(), best_predictions.max())
pad = 0.02 * (hi - lo)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5),
                         gridspec_kw={"width_ratios": [1, 1, 1],
                                      "wspace": 0.3})

# (a) Actual vs predicted
ax = axes[0]
ax.grid(axis="both")
ax.scatter(y_test, best_predictions, s=9, alpha=0.35, color=ACCENT,
           edgecolors="none", rasterized=True)
ax.plot([lo, hi], [lo, hi], ls="--", lw=1.4, color=INK_2,
        label="Perfect prediction")
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(lo - pad, hi + pad)
ax.set_title("Actual vs predicted")
ax.set_xlabel("Actual prevalence (%)")
ax.set_ylabel("Predicted prevalence (%)")
ax.legend(loc="upper left", handlelength=1.6)
r2_best = r2_score(y_test, best_predictions)
ax.text(0.97, 0.05, f"R² = {r2_best:.3f}", transform=ax.transAxes,
        ha="right", fontsize=15, fontweight="bold")

# (b) Residuals vs predicted
ax = axes[1]
ax.grid(axis="y")
ax.scatter(best_predictions, residuals, s=9, alpha=0.35, color=ACCENT,
           edgecolors="none", rasterized=True)
ax.axhline(0, ls="--", lw=1.4, color=INK_2)
ax.set_title("Residuals vs predicted")
ax.set_xlabel("Predicted prevalence (%)")
ax.set_ylabel("Residual (actual − predicted)")

# (c) Residual distribution
ax = axes[2]
ax.grid(axis="y")
ax.hist(residuals, bins=40, color=ACCENT, edgecolor="white", linewidth=0.6)
ax.axvline(0, ls="--", lw=1.4, color=INK_2)
ax.set_title("Distribution of residuals")
ax.set_xlabel("Residual (percentage points)")
ax.set_ylabel("Observations")

fig.suptitle(f"Residual diagnostics — {best_model_name}",
             x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
save(fig, "02_best_model_diagnostics.png")
plt.show()

# %% =========================================================
# 11. FIGURE 3 — GENERALISATION & RANGE CHECK (3 PANELS, ONE ROW)
# ============================================================

ov = overfitting_df.set_index("Model").loc[model_order]
rng = range_df.set_index("Model").loc[model_order]
y = np.arange(len(model_order))
h = 0.36

fig, axes = plt.subplots(1, 3, figsize=(11, 2.9),
                         gridspec_kw={"wspace": 0.12})


def train_test_panel(ax, train, test, title, fmt, xlim, show_labels=True):
    ax.barh(y - h / 2, train, h, color=NEUTRAL, label="Training")
    ax.barh(y + h / 2, test, h, color=ACCENT, label="Testing")
    for yi, a, b in zip(y, train, test):
        ax.text(a, yi - h / 2, f" {fmt.format(a)}", va="center", fontsize=12)
        ax.text(b, yi + h / 2, f" {fmt.format(b)}", va="center", fontsize=12)
    ax.set_yticks(y)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticklabels([SHORT[m] for m in model_order]
                       if show_labels else [])


train_test_panel(axes[0], ov["Train MAE"], ov["Test MAE"],
                 "MAE: train vs test", "{:.2f}",
                 (0, max(ov["Train MAE"].max(), ov["Test MAE"].max()) * 1.22))
train_test_panel(axes[1], ov["Train R2"], ov["Test R2"],
                 "R²: train vs test", "{:.3f}", (0, 1.22),
                 show_labels=False)
axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axes[1].set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])

# Range check: share of test predictions outside 0–100 %
n_test = len(y_test)
outside = (rng["Negative predictions"] + rng["Above 100%"]) / n_test * 100
ax = axes[2]
ax.barh(y, outside, height=0.62,
        color=[WARN if v > 0 else NEUTRAL for v in outside])
for yi, v, neg, hi_ in zip(y, outside, rng["Negative predictions"],
                           rng["Above 100%"]):
    label = f" {v:.1f}%  ({neg + hi_} of {n_test:,})" if v > 0 else " none"
    ax.text(v, yi, label, va="center", fontsize=8.5)
ax.set_yticks(y)
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlim(0, max(outside.max(), 1) * 1.7)
ax.set_title("Predictions outside 0–100%")
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2,
           bbox_to_anchor=(1.0, 1.06), handlelength=1.2)

fig.suptitle("Generalisation and prediction-range checks",
             x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
save(fig, "03_generalisation_and_range_checks.png")
plt.show()

# %% =========================================================
# 12. EXPORT TABLES
# ============================================================

results_df.to_csv(TABLE_DIR / "Model_evaluation.csv", index=False)
overfitting_df.to_csv(TABLE_DIR / "Overfitting_analysis.csv", index=False)
range_df.to_csv(TABLE_DIR / "Prediction_range_validation.csv", index=False)

pd.DataFrame({
    "Actual_Prevalence": y_test,
    "Predicted_Prevalence": best_predictions,
    "Residual": residuals,
    "Absolute_Error": np.abs(residuals),
}).to_csv(TABLE_DIR / "Final_predictions.csv", index=False)

all_predictions = pd.DataFrame({"Actual_Prevalence": y_test})
for name, prediction in test_predictions.items():
    all_predictions[f"{name}_Prediction"] = prediction
    all_predictions[f"{name}_Residual"] = y_test - prediction
all_predictions.to_csv(TABLE_DIR / "All_model_predictions.csv", index=False)

# %% =========================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION SUMMARY")
print("=" * 70)
print(display_results.to_string(index=False))
print(f"\nModel selected by lowest test RMSE: {best_model_name}")
print("\nOutput folders:")
print(f"Figures: {FIGURE_DIR.resolve()}")
print(f"Tables:  {TABLE_DIR.resolve()}")
print("\nEvaluation completed successfully.")
