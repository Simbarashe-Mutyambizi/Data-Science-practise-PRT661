#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

#%%
df = pd.read_csv("C:\\Personal\\Masters\\Masters_work\\Study\\Y2_S1\\PRT661\\Project\\Data-Science-practise-PRT661\\Machine_Learning\\Attempt_2\\LTHC_phase2_cleaned.csv")

print("Dataset shape:", df.shape)
df.head()
# %%
target = "Age-specific percentage of population reporting LTHC(s)"

print("Target:", target)
print("\nTarget statistics:")
print(df[target].describe())
# %%
features = [
    "Country of birth of person",
    "Years spent in Australia",
    "Age group",
    "Sex",
    "Language used at home",
    "Proficiency in spoken English",
    "Region_class",
    "Subregion_class",
    "Long-term health condition (LTHC)" 
]

X = df[features]
y = df[target]

print("Features:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print(target)
# %%
leakage_columns = [
    "Number of people reporting LTHC(s)",
    "Population"
]

print("Potential leakage columns excluded:")
for col in leakage_columns:
    print("-", col)

print("\nFeatures being used:")
print(X.columns.tolist())
# %%
group_columns = [
    "Country of birth of person",
    "Years spent in Australia",
    "Age group",
    "Sex",
    "Language used at home",
    "Proficiency in spoken English",
    "Region_class",
    "Subregion_class"
]

df["demographic_group"] = (
    df[group_columns]
    .astype(str)
    .agg("|".join, axis=1)
)

print(
    "Number of demographic groups:",
    df["demographic_group"].nunique()
)
# %%
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=df["demographic_group"]
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# %%
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=df["demographic_group"]
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
#%%
train_groups = set(
    df.iloc[train_idx]["demographic_group"]
)

test_groups = set(
    df.iloc[test_idx]["demographic_group"]
)

overlap = train_groups.intersection(test_groups)

print("Training groups:", len(train_groups))
print("Testing groups:", len(test_groups))
print("Overlapping groups:", len(overlap))
# %%
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("Categorical features:")

for col in categorical_features:
    print("-", col)
# %%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        )
    ]
)
# %%
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
# %%
print("Training target:")
print(y_train.describe())

print("\nTesting target:")
print(y_test.describe())
# %%
sparse.save_npz(
    "X_train_processed.npz",
    X_train_processed
)

sparse.save_npz(
    "X_test_processed.npz",
    X_test_processed
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

y_test.to_csv(
    "y_test.csv",
    index=False
)

print("Phase 3 datasets saved successfully.")
# %%
import joblib

joblib.dump(
    preprocessor,
    "lthc_preprocessor.pkl"
)

print("Preprocessing pipeline saved successfully.")
# %%
