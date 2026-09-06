#%%
import pandas as pd
import numpy as np

from scipy import sparse

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import joblib
# %%
X_train = sparse.load_npz("X_train_processed.npz")
X_test = sparse.load_npz("X_test_processed.npz")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
# %%

print("Training target statistics:")
print(y_train.describe())

print("\nTesting target statistics:")
print(y_test.describe())
# %%
print("Training target statistics:")
print(y_train.describe())

print("\nTesting target statistics:")
print(y_test.describe())
# %%

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

print("Linear Regression training complete.")
# %%
decision_tree_model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=10,
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

print("Decision Tree training complete.")
# %%
random_forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training complete.")
# %%
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost training complete.")
# %%
models = {
    "Linear Regression": linear_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgb_model
}

print("Models trained:")
for name in models:
    print("-", name)
# %%
predictions = {}

for name, model in models.items():
    predictions[name] = model.predict(X_test)

print("Predictions generated successfully.")
# %%
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Linear Regression": predictions["Linear Regression"],
    "Decision Tree": predictions["Decision Tree"],
    "Random Forest": predictions["Random Forest"],
    "XGBoost": predictions["XGBoost"]
})

comparison.head(20)
# %%
for name, pred in predictions.items():
    print(
        name,
        "Minimum:", round(pred.min(), 2),
        "Maximum:", round(pred.max(), 2)
    )
# %%
comparison.to_csv(
    "phase4_model_predictions.csv",
    index=False
)

print("Predictions saved successfully.")
# %%
joblib.dump(
    linear_model,
    "linear_regression_model.pkl"
)

joblib.dump(
    decision_tree_model,
    "decision_tree_model.pkl"
)

joblib.dump(
    random_forest_model,
    "random_forest_model.pkl"
)

joblib.dump(
    xgb_model,
    "xgboost_regression_model.pkl"
)

print("All models saved successfully.")
# %%
model_names = list(models.keys())

print("Saved models:")
for name in model_names:
    print("-", name)
# %%
