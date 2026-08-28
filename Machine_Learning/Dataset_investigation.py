# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("C:\Personal\Masters\Masters_work\Study\Y2_S1\PRT661\Project\Data-Science-practise-PRT661\development folder\Cleaned_data_final.csv")
print(f"Dataset shape: {df.shape}")
df.head()

df.info()
# %%
print("Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
df.describe(include="all").T
# %%
missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": (missing / len(df) * 100).round(2)
})

missing_df[missing_df["Missing Values"] > 0].sort_values(
    "Percentage", ascending=False
)

#%%

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")
print(f"Duplicate percentage: {duplicates / len(df) * 100:.2f}%")
# %%
target = "Long-term health condition (LTHC)"

print(f"Number of unique LTHC classes: {df[target].nunique()}")
print("\nLTHC classes:")

for value in df[target].dropna().unique():
    print(value)

lthc_counts = df[target].value_counts()

print(lthc_counts)

plt.figure(figsize=(12, 6))

lthc_counts.plot(kind="bar")

plt.title("Distribution of Long-term Health Conditions")
plt.xlabel("LTHC")
plt.ylabel("Number of Records")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

lthc_distribution = pd.DataFrame({
    "Count": lthc_counts,
    "Percentage": (lthc_counts / len(df) * 100).round(2)
})

lthc_distribution
# %%
zero_counts = (df == 0).sum()

zero_df = pd.DataFrame({
    "Zero Values": zero_counts,
    "Percentage": (zero_counts / len(df) * 100).round(2)
})

zero_df[zero_df["Zero Values"] > 0].sort_values(
    "Percentage", ascending=False
)
# %%
categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    print(f"\n{'='*60}")
    print(f"{col}")
    print(f"Unique values: {df[col].nunique()}")
    print(df[col].value_counts().head(10))
# %%
if "Number of people reporting LTHC(s)" in df.columns:
    print(
        df.groupby(target)["Number of people reporting LTHC(s)"]
        .describe()
    )
if "Age-specific percentage of population reporting LTHC(s)" in df.columns:
    print(
        df.groupby(target)["Age-specific percentage of population reporting LTHC(s)"]
        .describe()
    )
# %%

df[
    [
        "Number of people reporting LTHC(s)",
        "Population",
        "Age-specific percentage of population reporting LTHC(s)"
    ]
].head(20)

# %%
    