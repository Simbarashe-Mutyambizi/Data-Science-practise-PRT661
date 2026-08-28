#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from IPython.display import display


#%%
df_clean = pd.read_csv("C:\Personal\Masters\Masters_work\Study\Y2_S1\PRT661\Project\Data-Science-practise-PRT661\development folder\LTHC_prepped_cleaned.csv")

print("Dataset shape:", df_clean.shape)

#%%
target = "Long-term health condition (LTHC)"

features = [
    "Country of birth of person",
    "Years spent in Australia",
    "Age group",
    "Sex",
    "Language used at home",
    "Proficiency in spoken English",
    "Region_class",
    "Subregion_class"
]

X = df_clean[features]
y = df_clean[target]

print("Features:")
for feature in features:
    print("-", feature)

print("\nTarget:", target)

#%%
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX:")
display(X.head())

print("\ny:")
display(y.head())
# %%
class_counts = y.value_counts()

print(class_counts)

plt.figure(figsize=(12, 6))
class_counts.plot(kind="bar")

plt.title("LTHC Class Distribution")
plt.xlabel("LTHC")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#%%
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)
# %%

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

#%%
X_train_encoded = preprocessor.fit_transform(X_train)

X_test_encoded = preprocessor.transform(X_test)

print("Encoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)

# %%
encoded_columns = preprocessor.get_feature_names_out()

print("Number of encoded features:", len(encoded_columns))

print("\nFirst 30 encoded features:")
print(encoded_columns[:30])
# %%
class_distribution = y_train.value_counts(normalize=True) * 100

print(class_distribution.round(2))
# %%
X_train_processed = pd.DataFrame(
    X_train_encoded,
    columns=encoded_columns,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_encoded,
    columns=encoded_columns,
    index=X_test.index
)

print("Processed training data:")
display(X_train_processed.head())
# %%
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\nMissing values in training:",
      X_train_processed.isnull().sum().sum())

print("Missing values in testing:",
      X_test_processed.isnull().sum().sum())

# %%
X_train_processed.to_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\X_train_processed.csv", index=False)
X_test_processed.to_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\X_test_processed.csv", index=False)

y_train.to_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\y_train.csv", index=False)
y_test.to_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Training_csv\\y_test.csv", index=False)

print("Phase 3 datasets saved successfully.")
# %%
