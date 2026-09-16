# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load processed test data
X_test = sparse.load_npz("X_test_processed.npz")
y_test = pd.read_csv("y_test.csv").squeeze()

# Load final model
model = joblib.load(r"Models\xgboost_regression_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

print("Predictions generated successfully.")
print("Number of test observations:", len(y_test))

# %%
model = joblib.load(r"Models\xgboost_regression_model.pkl")
# %%
error_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Prediction error
error_df["Error"] = error_df["Predicted"] - error_df["Actual"]

# Absolute error
error_df["Absolute_Error"] = abs(error_df["Error"])

# Squared error
error_df["Squared_Error"] = error_df["Error"] ** 2

print(error_df.head(10))

# %%

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("ERROR ANALYSIS")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
# %%
largest_errors = error_df.sort_values(
    "Absolute_Error",
    ascending=False
)

print("10 largest prediction errors:")
print(largest_errors.head(10))
# %%
largest_errors.head(20).to_csv(
    "largest_prediction_errors.csv",
    index=False
)
# %%
plt.figure(figsize=(8, 6))

plt.scatter(
    error_df["Actual"],
    error_df["Predicted"],
    alpha=0.5
)

# Perfect prediction line
min_value = min(error_df["Actual"].min(), error_df["Predicted"].min())
max_value = max(error_df["Actual"].max(), error_df["Predicted"].max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Prevalence (%)")
plt.ylabel("Predicted Prevalence (%)")
plt.title("Actual vs Predicted LTHC Prevalence")

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(8, 6))

plt.scatter(
    error_df["Predicted"],
    error_df["Error"],
    alpha=0.5
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Prevalence (%)")
plt.ylabel("Residual / Error (%)")
plt.title("Residuals vs Predicted Prevalence")

plt.tight_layout()
plt.show()
# %%
error_df["Actual_Range"] = pd.cut(
    error_df["Actual"],
    bins=[-np.inf, 5, 10, 20, 50, 100],
    labels=[
        "0–5%",
        "5–10%",
        "10–20%",
        "20–50%",
        "50–100%"
    ]
)

range_error = error_df.groupby(
    "Actual_Range",
    observed=True
)["Absolute_Error"].agg(
    ["count", "mean", "median", "max"]
)

print(range_error)
# %%
over_predictions = (error_df["Error"] > 0).sum()
under_predictions = (error_df["Error"] < 0).sum()
accurate_predictions = (error_df["Error"] == 0).sum()

print("Overpredictions :", over_predictions)
print("Underpredictions:", under_predictions)
print("Exact predictions:", accurate_predictions)
total = len(error_df)

print(f"Overprediction percentage: {over_predictions / total * 100:.2f}%")
print(f"Underprediction percentage: {under_predictions / total * 100:.2f}%")
# %%
