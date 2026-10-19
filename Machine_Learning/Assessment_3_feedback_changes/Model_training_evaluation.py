# %%
"""
Model_training_evaluation.py  -  PRT661 Group Dan3, Assessment 4

1. Grouped hold-out split (80/20) on demographic_group.
2. Grouped 5-fold cross-validation on the training set for four models:
   Linear Regression, Decision Tree, Random Forest, XGBoost (reg:gamma),
   plus a naive baseline for reference. Reports mean +/- SD across folds.
3. Picks the model with the lowest mean CV RMSE, refits it on the full
   training set and evaluates once on the untouched test set.
4. Feature importance (permutation + model-native) and error analysis
   on the best model.

XGBoost uses objective="reg:gamma" (log link), so every prediction is > 0.
That is valid because the published data has no 0% cells (min 0.0069%).

Input : Cleaned_data_A4.csv (from Data_Cleaning.py)
Output: outputs/  (tables, best_model_pipeline.pkl, best_model.json)
        outputs/figures/
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20

DATA_FILE = Path("Cleaned_data_A4.csv")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

TARGET = "Age-specific percentage of population reporting LTHC(s)"
GROUP = "demographic_group"
LTHC = "Long-term health condition (LTHC)"
AGE = "Age group"
SEX = "Sex"
FEATURES = [
    "Country of birth of person",
    "Years spent in Australia",
    "Language used at home",
    "Proficiency in spoken English",
    "Region_class",
    "Subregion_class",
    AGE,
    SEX,
    LTHC,
]
BASELINE_NAME = "Baseline (mean by LTHC x age x sex)"

# Plot style
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#a8a7a2"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def build_models():
    """Four candidate models, each wrapped with the same one-hot preprocessor."""
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost (gamma)": XGBRegressor(
            objective="reg:gamma",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline([("prep", clone(preprocessor)), ("model", est)])
        for name, est in estimators.items()
    }


def score(y_true, y_pred):
    return {
        "MAE (pp)": mean_absolute_error(y_true, y_pred),
        "RMSE (pp)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Negative predictions": int((y_pred < 0).sum()),
    }


def baseline_predict(train, test):
    """Mean training prevalence for the same LTHC x age x sex (falls back to LTHC, then overall)."""
    keys = [LTHC, AGE, SEX]
    by_cell = train.groupby(keys)[TARGET].mean().rename("cell")
    by_lthc = train.groupby(LTHC)[TARGET].mean().rename("lthc")
    out = test[keys].join(by_cell, on=keys).join(by_lthc, on=LTHC)
    return out["cell"].fillna(out["lthc"]).fillna(train[TARGET].mean()).to_numpy()


# %%
# 1. Load data and make the grouped hold-out split
df = pd.read_csv(DATA_FILE)
print(f"Rows: {len(df):,}   demographic groups: {df[GROUP].nunique():,}")

splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(df, groups=df[GROUP]))
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

assert set(train_df[GROUP]).isdisjoint(test_df[GROUP]), "Group leakage between train and test"
print(f"Train: {len(train_df):,} rows   Test: {len(test_df):,} rows (no shared groups)")

X_train, y_train, g_train = train_df[FEATURES], train_df[TARGET], train_df[GROUP]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# %%
# 2. Grouped 5-fold cross-validation on the training set
models = build_models()
cv = GroupKFold(n_splits=N_FOLDS)
fold_rows = []

for fold, (tr, va) in enumerate(cv.split(X_train, y_train, groups=g_train), start=1):
    for name, pipeline in models.items():
        fitted = clone(pipeline).fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = fitted.predict(X_train.iloc[va])
        fold_rows.append({"Fold": fold, "Model": name, **score(y_train.iloc[va], pred)})

    pred = baseline_predict(train_df.iloc[tr], train_df.iloc[va])
    fold_rows.append({"Fold": fold, "Model": BASELINE_NAME, **score(y_train.iloc[va], pred)})
    print(f"Fold {fold} done")

cv_folds = pd.DataFrame(fold_rows)
cv_folds.to_csv(OUT_DIR / "cv_fold_results.csv", index=False)

metric_cols = ["MAE (pp)", "RMSE (pp)", "R2"]
cv_summary = cv_folds.groupby("Model", sort=False).agg(
    **{f"{m} mean": (m, "mean") for m in metric_cols},
    **{f"{m} SD": (m, "std") for m in metric_cols},
    **{"Negative predictions (total)": ("Negative predictions", "sum")},
)
cv_summary = cv_summary.sort_values("RMSE (pp) mean")
cv_summary.round(4).to_csv(OUT_DIR / "cv_summary.csv")

print(f"\nGROUPED {N_FOLDS}-FOLD CROSS-VALIDATION (training set, mean +/- SD)")
for model_name, r in cv_summary.iterrows():
    print(
        f"  {model_name:<38} MAE {r['MAE (pp) mean']:.3f} +/- {r['MAE (pp) SD']:.3f} pp | "
        f"RMSE {r['RMSE (pp) mean']:.3f} +/- {r['RMSE (pp) SD']:.3f} pp | "
        f"R2 {r['R2 mean']:.4f} +/- {r['R2 SD']:.4f} | "
        f"negatives {int(r['Negative predictions (total)'])}"
    )

# %%
# 3. Choose the best model on CV RMSE (the baseline is a reference, not a candidate)
candidates = cv_summary.drop(index=BASELINE_NAME)
best_name = candidates["RMSE (pp) mean"].idxmin()
print(f"\nBest model on mean CV RMSE: {best_name}")

# CV comparison figure: one panel per metric, mean with +/- 1 SD bars
order = list(cv_summary.index)[::-1]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, metric in zip(axes, metric_cols):
    means = cv_summary.loc[order, f"{metric} mean"]
    sds = cv_summary.loc[order, f"{metric} SD"]
    for i, name in enumerate(order):
        colour = BLUE if name == best_name else (MUTED if name == BASELINE_NAME else INK_2)
        ax.errorbar(means[name], i, xerr=sds[name], fmt="o", ms=8, color=colour,
                    ecolor=colour, elinewidth=2, capsize=0)
        ax.annotate(f"{means[name]:.3f}", (means[name], i), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_title("R² (higher is better)" if metric == "R2" else f"{metric} (lower is better)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order)), order)
axes[0].set_ylim(-0.6, len(order) - 0.4)
fig.suptitle(f"Grouped {N_FOLDS}-fold cross-validation (mean +/- 1 SD across folds)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "cv_model_comparison.png")
plt.show()

# %%
# 4. Refit the best model on the full training set; evaluate once on the test set
best_model = clone(models[best_name]).fit(X_train, y_train)
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)
baseline_test_pred = baseline_predict(train_df, test_df)

test_metrics = pd.DataFrame([
    {"Model": best_name, "Set": "Train", **score(y_train, train_pred)},
    {"Model": best_name, "Set": "Test", **score(y_test, test_pred)},
    {"Model": BASELINE_NAME, "Set": "Test", **score(y_test, baseline_test_pred)},
])
test_metrics.round(4).to_csv(OUT_DIR / "test_metrics.csv", index=False)
print("\nHOLD-OUT TEST SET")
print(test_metrics.round(4).to_string(index=False))

joblib.dump(best_model, OUT_DIR / "best_model_pipeline.pkl")
with open(OUT_DIR / "best_model.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "selection": f"lowest mean RMSE in grouped {N_FOLDS}-fold CV",
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in best_model.named_steps["model"].get_params().items()},
        "test_metrics": test_metrics.iloc[1][metric_cols].round(4).to_dict(),
    }, f, indent=2)
print(f"Saved {OUT_DIR / 'best_model_pipeline.pkl'} (preprocessor + model; predict() takes raw columns)")

# %%
# 5. Feature importance on the best model
# 5a. Permutation importance on the test set, per original column:
#     how much test MAE rises (pp) when that column is shuffled.
#     Region/Subregion are derived from country, so these share importance.
perm = permutation_importance(
    best_model, X_test, y_test, scoring="neg_mean_absolute_error",
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
)
perm_df = pd.DataFrame({
    "Feature": FEATURES,
    "MAE increase (pp) mean": perm.importances_mean,
    "MAE increase (pp) SD": perm.importances_std,
}).sort_values("MAE increase (pp) mean", ascending=False)
perm_df.round(4).to_csv(OUT_DIR / "permutation_importance.csv", index=False)
print("\nPERMUTATION IMPORTANCE (test MAE increase when shuffled)")
print(perm_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.2))
p = perm_df.iloc[::-1]
ax.barh(p["Feature"], p["MAE increase (pp) mean"], xerr=p["MAE increase (pp) SD"],
        color=BLUE, height=0.6, error_kw={"ecolor": INK_2, "elinewidth": 1})
ax.set_xlabel("Increase in test MAE when feature is shuffled (percentage points)")
ax.set_title(f"Permutation importance - {best_name}", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "permutation_importance.png")
plt.show()

# 5b. Model-native importance (tree models), per one-hot feature and summed per column
estimator = best_model.named_steps["model"]
encoder = best_model.named_steps["prep"].named_transformers_["cat"]
if hasattr(estimator, "feature_importances_"):
    onehot_names, source_cols = [], []
    for col, cats in zip(FEATURES, encoder.categories_):
        onehot_names += [f"{col} = {c}" for c in cats]
        source_cols += [col] * len(cats)
    native = pd.DataFrame({
        "Feature": onehot_names,
        "Column": source_cols,
        "Importance": estimator.feature_importances_,
    }).sort_values("Importance", ascending=False)
    native.to_csv(OUT_DIR / "native_importance_onehot.csv", index=False)
    by_col = native.groupby("Column")["Importance"].sum().sort_values(ascending=False)
    by_col.to_csv(OUT_DIR / "native_importance_by_column.csv")
    print("\nMODEL-NATIVE IMPORTANCE, summed per column")
    print(by_col.round(4).to_string())

    top = native.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["Feature"], top["Importance"], color=BLUE, height=0.6)
    ax.set_xlabel("Importance (share of total)")
    ax.set_title(f"Top 15 one-hot features - {best_name}", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "native_importance_top15.png")
    plt.show()

# %%
# 6. Error analysis on the best model (test set)
errors = test_df.copy()
errors["Actual"] = y_test.to_numpy()
errors["Predicted"] = test_pred
errors["Error"] = errors["Predicted"] - errors["Actual"]          # + = over-prediction
errors["Abs_Error"] = errors["Error"].abs()
errors["Rel_Error_pct"] = errors["Abs_Error"] / errors["Actual"] * 100
errors.to_csv(OUT_DIR / "test_predictions.csv", index=False)

print(f"\nNegative predictions: {(errors['Predicted'] < 0).sum()}")
print(f"Predictions below 0.5%: {(errors['Predicted'] < 0.5).sum()} "
      f"(actual values below 0.5%: {(errors['Actual'] < 0.5).sum()})")

bands = pd.cut(errors["Actual"], bins=[0, 5, 10, 20, 50, 100],
               labels=["0-5%", "5-10%", "10-20%", "20-50%", "50-100%"])
band_table = errors.groupby(bands, observed=True).agg(
    n=("Actual", "size"),
    mean_actual=("Actual", "mean"),
    MAE_pp=("Abs_Error", "mean"),
    median_abs_error_pp=("Abs_Error", "median"),
    max_abs_error_pp=("Abs_Error", "max"),
    bias_pp=("Error", "mean"),
    median_relative_error_pct=("Rel_Error_pct", "median"),
)
band_table["relative_MAE_pct"] = band_table["MAE_pp"] / band_table["mean_actual"] * 100
band_table.index.name = "Actual prevalence band"
band_table.round(3).to_csv(OUT_DIR / "error_by_band.csv")
print("\nERROR BY PREVALENCE BAND (test set)")
print(band_table.round(3).to_string())

largest = errors.sort_values("Abs_Error", ascending=False).head(20)
largest[["source_table"] + FEATURES + ["Actual", "Predicted", "Error", "Rel_Error_pct"]] \
    .round(3).to_csv(OUT_DIR / "largest_errors.csv", index=False)

over = (errors["Error"] > 0).mean() * 100
print(f"\nOver-predictions: {over:.1f}%   Under-predictions: {100 - over:.1f}%")

# Actual vs predicted (linear and log scale)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, log in zip(axes, [False, True]):
    ax.scatter(errors["Actual"], errors["Predicted"], s=10, alpha=0.35, color=BLUE, linewidths=0)
    lo = 0.005 if log else 0
    hi = max(errors["Actual"].max(), errors["Predicted"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", color=INK_2, lw=1, label="Perfect prediction")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Actual prevalence (%)" + (" - log scale" if log else ""))
    ax.set_ylabel("Predicted prevalence (%)")
    ax.legend(frameon=False, loc="upper left")
axes[0].set_title(f"Actual vs predicted - {best_name}", loc="left")
axes[1].set_title("Same, log scale (shows small values)", loc="left")
fig.tight_layout()
fig.savefig(FIG_DIR / "actual_vs_predicted.png")
plt.show()

# Residuals vs predicted
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(errors["Predicted"], errors["Error"], s=10, alpha=0.35, color=BLUE, linewidths=0)
ax.axhline(0, ls="--", color=INK_2, lw=1)
ax.set_xlabel("Predicted prevalence (%)")
ax.set_ylabel("Error, predicted - actual (pp)")
ax.set_title(f"Residuals - {best_name}", loc="left")
fig.tight_layout()
fig.savefig(FIG_DIR / "residuals.png")
plt.show()

print(f"\nAll outputs saved to {OUT_DIR.resolve()}")
# %%
