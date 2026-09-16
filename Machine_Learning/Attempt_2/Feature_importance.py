# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# %%

X_train = sparse.load_npz("X_train_processed.npz")
X_test = sparse.load_npz("X_test_processed.npz")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

feature_names = np.load(
    "feature_names.npy",
    allow_pickle=True
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Number of feature names:", len(feature_names))

# %%

if X_train.shape[1] != len(feature_names):
    raise ValueError(
        "Number of feature names does not match "
        "number of processed features."
    )

print("Feature names successfully matched.")

# %%

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": np.abs(linear_model.coef_),
    "Coefficient": linear_model.coef_
})

linear_importance = linear_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 15 Linear Regression Features:")
print(linear_importance.head(15))

# %%

dt_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": dt_model.feature_importances_
})

dt_importance = dt_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 15 Decision Tree Features:")
print(dt_importance.head(15))

# %%

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 15 Random Forest Features:")
print(rf_importance.head(15))

# %%

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 15 XGBoost Features:")
print(xgb_importance.head(15))

# %%
def plot_feature_importance(
    importance_df,
    title
):

    top_features = (
        importance_df
        .head(15)
        .sort_values("Importance")
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)

    plt.tight_layout()
    plt.show()

# %%

plot_feature_importance(
    linear_importance,
    "Top 15 Features - Linear Regression"
)

plot_feature_importance(
    dt_importance,
    "Top 15 Features - Decision Tree"
)

plot_feature_importance(
    rf_importance,
    "Top 15 Features - Random Forest"
)

plot_feature_importance(
    xgb_importance,
    "Top 15 Features - XGBoost"
)
# %%

linear_importance.to_csv(
    "linear_feature_importance.csv",
    index=False
)

dt_importance.to_csv(
    "decision_tree_feature_importance.csv",
    index=False
)

rf_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

xgb_importance.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)

print("\nFeature importance results saved successfully.")
# %%
