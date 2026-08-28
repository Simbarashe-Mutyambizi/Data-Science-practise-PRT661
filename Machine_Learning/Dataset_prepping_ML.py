# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("C:\Personal\Masters\Masters_work\Study\Y2_S1\PRT661\Project\Data-Science-practise-PRT661\development folder\Cleaned_data_final.csv")
print(f"Dataset shape: {df.shape}")
# %%
df_clean = df.copy()
if "Unnamed: 0" in df_clean.columns:
    df_clean = df_clean.drop(columns=["Unnamed: 0"])

print(df_clean.shape)

# %%

df_clean.columns = (
    df_clean.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print(df_clean.columns.tolist())

# %%

categorical_columns = df_clean.select_dtypes(include="object").columns

for col in categorical_columns:
    df_clean[col] = df_clean[col].str.strip()

#%%

missing = df_clean.isnull().sum()

missing[missing > 0]

# %%

zero_columns = [
    "Number of people reporting LTHC(s)",
    "Population",
    "Age-specific percentage of population reporting LTHC(s)"
]

zero_mask = (df_clean[zero_columns] == 0).all(axis=1)

print("Rows where all three values are zero:", zero_mask.sum())
print("Percentage:", round(zero_mask.mean() * 100, 2), "%")

#%%

df_clean.loc[zero_mask].head(20)
df_clean.loc[~zero_mask].head(20)
#%%

target = "Long-term health condition (LTHC)"

leakage_columns = [
    "Number of people reporting LTHC(s)",
    "Age-specific percentage of population reporting LTHC(s)"
]

for col in leakage_columns:
    print(f"\n{col}")
    print(df_clean.groupby(target)[col].describe())

df_clean = df_clean.drop(columns=leakage_columns)

print("Shape after removing leakage columns:", df_clean.shape)

#%%
print("Missing target values:", df_clean[target].isnull().sum())

print("\nLTHC distribution:")
print(df_clean[target].value_counts())
# %%
df_clean = df_clean.dropna(subset=[target])

# %%

duplicates = df_clean.duplicated().sum()
print("Duplicate rows:", duplicates)
df_clean = df_clean.drop_duplicates()
print("Shape after removing duplicates:", df_clean.shape)
#%%
print("Final dataset shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nColumns:")
for col in df_clean.columns:
    print("-", col)

#%%
df_clean.to_csv("C:\Personal\Masters\Masters_work\Study\Y2_S1\PRT661\Project\Data-Science-practise-PRT661\development folder\LTHC_prepped_cleaned.csv", index=False)
print("Phase 2 dataset saved successfully.")
# %%
