
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

# Create an organized output directory
OUTPUT_DIR = Path("model_evaluation_results")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# %% =========================================================
# 2. LOAD PROCESSED DATA
# ============================================================

X_train = sparse.load_npz("X_train_processed.npz")
X_test = sparse.load_npz("X_test_processed.npz")

y_train = pd.read_csv("y_train.csv").squeeze("columns")
y_test = pd.read_csv("y_test.csv").squeeze("columns")

y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()

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
    "Linear Regression": joblib.load(
        "linear_regression_model.pkl"
    ),
    "Decision Tree": joblib.load(
        "decision_tree_model.pkl"
    ),
    "Random Forest": joblib.load(
        "random_forest_model.pkl"
    ),
    "XGBoost": joblib.load(
        "xgboost_regression_model.pkl"
    )
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
    train_predictions[name] = np.asarray(
        model.predict(X_train)
    ).ravel()

    test_predictions[name] = np.asarray(
        model.predict(X_test)
    ).ravel()

print("\nPredictions generated for all models.")

# %% =========================================================
# 5. EVALUATE TEST PERFORMANCE
# ============================================================

results = []

for name, prediction in test_predictions.items():
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, prediction)
        ),
        "R2": r2_score(y_test, prediction)
    })

results_df = pd.DataFrame(results)

# Identify the model with the lowest test RMSE
best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(), "Model"
]

# Rounded version for display
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
    train_prediction = train_predictions[name]
    test_prediction = test_predictions[name]

    train_r2 = r2_score(y_train, train_prediction)
    test_r2 = r2_score(y_test, test_prediction)

    train_mae = mean_absolute_error(
        y_train, train_prediction
    )
    test_mae = mean_absolute_error(
        y_test, test_prediction
    )

    overfitting_results.append({
        "Model": name,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "R2 Difference": train_r2 - test_r2
    })

overfitting_df = pd.DataFrame(overfitting_results)

print("\nTRAINING VS TESTING")
print("=" * 75)
print(overfitting_df.round(4).to_string(index=False))

# %% =========================================================
# 7. FIGURE 1 — MODEL PERFORMANCE COMPARISON
# ============================================================

# A consistent model order across all plots
model_order = list(models.keys())

plot_results = (
    results_df
    .set_index("Model")
    .loc[model_order]
    .reset_index()
)

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5.8)
)

fig.suptitle(
    "Regression Model Performance Comparison",
    fontsize=17,
    fontweight="bold",
    y=1.03
)

metrics = [
    ("MAE", "Mean Absolute Error", "Lower is better"),
    ("RMSE", "Root Mean Squared Error", "Lower is better"),
    ("R2", "Coefficient of Determination (R²)", "Higher is better")
]

for ax, (metric, title, note) in zip(axes, metrics):
    values = plot_results[metric]

    bars = ax.bar(
        plot_results["Model"],
        values
    )

    # Highlight the best value for the relevant metric
    if metric in ["MAE", "RMSE"]:
        best_idx = values.idxmin()
    else:
        best_idx = values.idxmax()

    bars[best_idx].set_hatch("//")

    ax.set_title(title, pad=12)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.set_axisbelow(True)

    value_range = values.max() - values.min()
    padding = max(value_range * 0.12, abs(values.max()) * 0.04, 0.01)
    ax.set_ylim(
        min(0, values.min() - padding),
        values.max() + padding * 2.5
    )

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.text(
        0.5, -0.28, note,
        transform=ax.transAxes,
        ha="center",
        fontsize=9
    )

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "01_model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% =========================================================
# 8. FIGURE 2 — ACTUAL VS PREDICTED
# ============================================================

best_predictions = test_predictions[best_model_name]

min_value = min(
    float(np.min(y_test)),
    float(np.min(best_predictions))
)

max_value = max(
    float(np.max(y_test)),
    float(np.max(best_predictions))
)

fig, ax = plt.subplots(figsize=(8.5, 8))

ax.scatter(
    y_test,
    best_predictions,
    alpha=0.45,
    s=28,
    edgecolors="none",
    label="Test observations"
)

# Perfect-prediction reference line
ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

ax.set_title(
    f"Actual vs Predicted Prevalence\n{best_model_name}",
    pad=14
)

ax.set_xlabel("Actual Prevalence (%)")
ax.set_ylabel("Predicted Prevalence (%)")

ax.legend(frameon=False)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "02_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% =========================================================
# 9. FIGURE 3 — RESIDUAL ANALYSIS
# ============================================================

# Residual = actual - predicted
residuals = y_test - best_predictions

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 5.8)
)

fig.suptitle(
    f"Residual Analysis — {best_model_name}",
    fontsize=16,
    fontweight="bold",
    y=1.02
)

# Residuals versus predictions
axes[0].scatter(
    best_predictions,
    residuals,
    alpha=0.45,
    s=28,
    edgecolors="none"
)

axes[0].axhline(
    0,
    linestyle="--",
    linewidth=2
)

axes[0].set_title("Residuals vs Predicted Values")
axes[0].set_xlabel("Predicted Prevalence (%)")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].set_axisbelow(True)

# Residual distribution
axes[1].hist(
    residuals,
    bins=35,
    edgecolor="black",
    linewidth=0.5
)

axes[1].axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero error"
)

axes[1].set_title("Distribution of Residuals")
axes[1].set_xlabel("Residual (Percentage Points)")
axes[1].set_ylabel("Number of Observations")
axes[1].legend(frameon=False)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "03_residual_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% =========================================================
# 10. FIGURE 4 — TRAINING VS TESTING
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.5)
)

fig.suptitle(
    "Training vs Testing Performance",
    fontsize=16,
    fontweight="bold",
    y=1.02
)

x = np.arange(len(model_order))
width = 0.36

ordered_overfitting = (
    overfitting_df
    .set_index("Model")
    .loc[model_order]
)

# MAE comparison
axes[0].bar(
    x - width / 2,
    ordered_overfitting["Train MAE"],
    width,
    label="Training MAE"
)

axes[0].bar(
    x + width / 2,
    ordered_overfitting["Test MAE"],
    width,
    label="Testing MAE"
)

axes[0].set_title("MAE: Training vs Testing")
axes[0].set_ylabel("Mean Absolute Error")
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_order, rotation=25, ha="right")
axes[0].legend(frameon=False)

# R² comparison
axes[1].bar(
    x - width / 2,
    ordered_overfitting["Train R2"],
    width,
    label="Training R²"
)

axes[1].bar(
    x + width / 2,
    ordered_overfitting["Test R2"],
    width,
    label="Testing R²"
)

axes[1].set_title("R²: Training vs Testing")
axes[1].set_ylabel("R²")
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_order, rotation=25, ha="right")
axes[1].legend(frameon=False)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "04_training_vs_testing.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% =========================================================
# 11. FIGURE 5 — PREDICTION RANGE CHECK
# ============================================================

range_results = []

for name, prediction in test_predictions.items():
    range_results.append({
        "Model": name,
        "Negative predictions": int((prediction < 0).sum()),
        "Above 100%": int((prediction > 100).sum()),
        "Within 0–100%": int(
            ((prediction >= 0) & (prediction <= 100)).sum()
        )
    })

range_df = pd.DataFrame(range_results)

print("\nPREDICTION RANGE VALIDATION")
print("=" * 65)
print(range_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(range_df))
width = 0.24

ax.bar(
    x - width,
    range_df["Negative predictions"],
    width,
    label="Below 0%"
)

ax.bar(
    x,
    range_df["Within 0–100%"],
    width,
    label="Within 0–100%"
)

ax.bar(
    x + width,
    range_df["Above 100%"],
    width,
    label="Above 100%"
)

ax.set_title("Prediction Range Validation by Model", pad=12)
ax.set_ylabel("Number of Test Predictions")
ax.set_xticks(x)
ax.set_xticklabels(range_df["Model"], rotation=25, ha="right")
ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "05_prediction_range_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% =========================================================
# 12. EXPORT TABLES
# ============================================================

# Full-precision metrics
results_df.to_csv(
    TABLE_DIR / "Model_evaluation.csv",
    index=False
)

overfitting_df.to_csv(
    TABLE_DIR / "Overfitting_analysis.csv",
    index=False
)

range_df.to_csv(
    TABLE_DIR / "Prediction_range_validation.csv",
    index=False
)

final_predictions = pd.DataFrame({
    "Actual_Prevalence": y_test,
    "Predicted_Prevalence": best_predictions,
    "Residual": residuals,
    "Absolute_Error": np.abs(residuals)
})

final_predictions.to_csv(
    TABLE_DIR / "Final_predictions.csv",
    index=False
)

# Save predictions from all four models for detailed analysis
all_predictions = pd.DataFrame({
    "Actual_Prevalence": y_test
})

for name, prediction in test_predictions.items():
    all_predictions[f"{name}_Prediction"] = prediction
    all_predictions[f"{name}_Residual"] = y_test - prediction

all_predictions.to_csv(
    TABLE_DIR / "All_model_predictions.csv",
    index=False
)

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
# %%
