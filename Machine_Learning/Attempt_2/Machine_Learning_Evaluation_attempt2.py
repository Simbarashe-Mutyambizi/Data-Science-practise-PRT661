#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy import sparse
import joblib
# %%
X_train = sparse.load_npz("X_train_processed.npz")
X_test = sparse.load_npz("X_test_processed.npz")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
# %%
linear_model = joblib.load(
    "linear_regression_model.pkl"
)

decision_tree_model = joblib.load(
    "decision_tree_model.pkl"
)

random_forest_model = joblib.load(
    "random_forest_model.pkl"
)

xgb_model = joblib.load(
    "xgboost_regression_model.pkl"
)

models = {
    "Linear Regression": linear_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgb_model
}

print("Models loaded successfully.")
# %%
train_predictions = {}
test_predictions = {}

for name, model in models.items():

    train_predictions[name] = model.predict(X_train)
    test_predictions[name] = model.predict(X_test)

print("Train and test predictions generated.")
# %%
results = []

for name in models:

    mae = mean_absolute_error(
        y_test,
        test_predictions[name]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions[name]
        )
    )

    r2 = r2_score(
        y_test,
        test_predictions[name]
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    })

results_df = pd.DataFrame(results)

results_df
# %%
results_df[
    ["MAE", "RMSE", "R²"]
] = results_df[
    ["MAE", "RMSE", "R²"]
].round(3)

results_df
# %%
best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

print("Best model based on RMSE:")
print(best_model_name)
# %%
best_result = results_df[
    results_df["Model"] == best_model_name
]

print(best_result.to_string(index=False))
# %%
overfitting_results = []

for name in models:

    train_r2 = r2_score(
        y_train,
        train_predictions[name]
    )

    test_r2 = r2_score(
        y_test,
        test_predictions[name]
    )

    train_mae = mean_absolute_error(
        y_train,
        train_predictions[name]
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions[name]
    )

    overfitting_results.append({
        "Model": name,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R²": train_r2,
        "Test R²": test_r2,
        "R² Difference": train_r2 - test_r2
    })

overfitting_df = pd.DataFrame(
    overfitting_results
)

overfitting_df.round(4)
# %%
plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("Model Comparison — Mean Absolute Error")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.title("Model Comparison — RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R²"]
)

plt.title("Model Comparison — R²")
plt.xlabel("Model")
plt.ylabel("R²")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
# %%
best_predictions = test_predictions[best_model_name]

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.4
)

min_value = min(
    y_test.min(),
    best_predictions.min()
)

max_value = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(
    f"Actual vs Predicted — {best_model_name}"
)

plt.xlabel("Actual Prevalence (%)")
plt.ylabel("Predicted Prevalence (%)")

plt.tight_layout()
plt.show()
# %%
residuals = (
    y_test.values -
    best_predictions
)

plt.figure(figsize=(10, 6))

plt.scatter(
    best_predictions,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    f"Residual Plot — {best_model_name}"
)

plt.xlabel("Predicted Prevalence (%)")
plt.ylabel("Residual")

plt.tight_layout()
plt.show()
# %%
for name, prediction in test_predictions.items():

    negative = (prediction < 0).sum()
    above_100 = (prediction > 100).sum()

    print(f"\n{name}")
    print("Negative predictions:", negative)
    print("Predictions above 100%:", above_100)
# %%
final_predictions = pd.DataFrame({
    "Actual Prevalence": y_test.values,
    "Predicted Prevalence": best_predictions,
    "Residual": residuals
})

final_predictions.head(20)
# %%
results_df.to_csv(
    "phase5_model_evaluation.csv",
    index=False
)

overfitting_df.to_csv(
    "phase5_overfitting_analysis.csv",
    index=False
)

final_predictions.to_csv(
    "phase5_final_predictions.csv",
    index=False
)

print("Phase 5 results saved successfully.")
# %%
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nBest model based on RMSE:")
print(best_model_name)
# %%
