#%%
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
import joblib

#%%
X_train = pd.read_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\X_train_processed.csv")
X_test = pd.read_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\X_test_processed.csv")

y_train = pd.read_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\y_train.csv").squeeze()
y_test = pd.read_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\y_test.csv").squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#%%
print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_test:", X_test.isnull().sum().sum())
print("Missing values in y_train:", y_train.isnull().sum())
print("Missing values in y_test:", y_test.isnull().sum())

# %%
print("Number of LTHC classes:", y_train.nunique())

print("\nLTHC classes:")
print(y_train.unique())
# %%
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train, y_train)

print("Logistic Regression training complete.")
# %%
decision_tree_model = DecisionTreeClassifier(
    max_depth=20,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(X_train, y_train)

print("Decision Tree training complete.")
# %%
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

print("Random Forest training complete.")

#%%
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Number of classes:", len(label_encoder.classes_))

print("\nClass mapping:")

for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i} → {class_name}")
# %%
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train_encoded
)

print("XGBoost training complete.")
# %%
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgb_model
}

print("Models trained:")
for name in models:
    print("-", name)
# %%
predictions = {}

# Logistic Regression
predictions["Logistic Regression"] = logistic_model.predict(X_test)

# Decision Tree
predictions["Decision Tree"] = decision_tree_model.predict(X_test)

# Random Forest
predictions["Random Forest"] = random_forest_model.predict(X_test)

# XGBoost
xgb_predictions_encoded = xgb_model.predict(X_test)

predictions["XGBoost"] = label_encoder.inverse_transform(
    xgb_predictions_encoded.astype(int)
)

print("Predictions generated for all models.")

# %%
comparison = pd.DataFrame({
    "Actual": y_test.iloc[:20].values,
    "Logistic Regression": predictions["Logistic Regression"][:20],
    "Decision Tree": predictions["Decision Tree"][:20],
    "Random Forest": predictions["Random Forest"][:20],
    "XGBoost": predictions["XGBoost"][:20]
})

comparison
# %%
prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Logistic Regression": predictions["Logistic Regression"],
    "Decision Tree": predictions["Decision Tree"],
    "Random Forest": predictions["Random Forest"],
    "XGBoost": predictions["XGBoost"]
})

prediction_df.to_csv(
    "C:\Personal\Masters\Masters_work\Study\Y2_S1\PRT661\Project\Data-Science-practise-PRT661\Predictions\phase4_model_predictions.csv",
    index=False
)

print("Predictions saved to phase4_model_predictions.csv")
# %%
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(decision_tree_model, "decision_tree_model.pkl")
joblib.dump(random_forest_model, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(label_encoder, "lthc_label_encoder.pkl")

print("Models saved successfully.")