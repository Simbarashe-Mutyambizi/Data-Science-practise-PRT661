#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv(r"C:\Personal\Masters\Masters_work\Study\Y2_S1\PRT661\Project\Data-Science-practise-PRT661\development folder\Cleaned_data_final.csv")

print("Dataset shape:", df.shape)
df.head()
# %%
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("Dataset shape:", df.shape)
# %%
print("Columns:")

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

df.info()
# %%
missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": (missing / len(df) * 100).round(2)
})

missing_df[missing_df["Missing Values"] > 0].sort_values(
    "Percentage",
    ascending=False
)
# %%
duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

print(
    "Duplicate percentage:",
    round(duplicates / len(df) * 100, 2),
    "%"
)
# %%
categorical_columns = df.select_dtypes(
    include="object"
).columns

for col in categorical_columns:
    print("\n" + "=" * 60)
    print(col)
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts().head(10))
# %%
numerical_columns = df.select_dtypes(
    exclude="object"
).columns

df[numerical_columns].describe().T
# %%
zero_population = df["Population"] == 0

print("Zero population rows:", zero_population.sum())

print(
    "Percentage:",
    round(zero_population.mean() * 100, 2),
    "%"
)
# %%
df.loc[zero_population].head(20)
# %%
df["calculated_prevalence"] = np.where(
    df["Population"] > 0,
    (
        df["Number of people reporting LTHC(s)"]
        / df["Population"]
    ) * 100,
    0
)

difference = (
    df["calculated_prevalence"]
    - df["Age-specific percentage of population reporting LTHC(s)"]
).abs()

print("Maximum difference:", difference.max())
print("Mean difference:", difference.mean())


# %%
target = "Age-specific percentage of population reporting LTHC(s)"

leakage_columns = [
    "Number of people reporting LTHC(s)",
    "Population"
]

print("Target:")
print(target)

print("\nPotential leakage variables:")

for col in leakage_columns:
    print("-", col)
# %%
lthc_counts = df["Long-term health condition (LTHC)"].value_counts()

print("Number of LTHC categories:", len(lthc_counts))

print("\nLTHC distribution:")
print(lthc_counts)
# %%
plt.figure(figsize=(12, 6))

lthc_counts.plot(kind="bar")

plt.title("Distribution of Long-term Health Conditions")
plt.xlabel("LTHC")
plt.ylabel("Number of Records")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(10, 6))

df[target].hist(bins=50)

plt.title("Distribution of LTHC Prevalence")
plt.xlabel("Age-specific percentage")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()
# %%
print(df["Age group"].value_counts())
# %%
lthc_prevalence = (
    df.groupby("Long-term health condition (LTHC)")[target]
    .mean()
    .sort_values(ascending=False)
)

print(lthc_prevalence)
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
    "Unique demographic groups:",
    df["demographic_group"].nunique()
)
# %%
group_sizes = df["demographic_group"].value_counts()

print("\nDemographic group size distribution:")
print(group_sizes.describe())
# %%
lthc_per_group = (
    df.groupby("demographic_group")[
        "Long-term health condition (LTHC)"
    ].nunique()
)

print(lthc_per_group.describe())
# %%
print(
    "Groups containing more than one LTHC:",
    (lthc_per_group > 1).sum()
)
# %%
df = df.drop(
    columns=[
        "calculated_prevalence",
        "demographic_group"
    ],
    errors="ignore"
)
# %%
df = df[df["Population"] > 0].copy()

print("Final Phase 2 shape:", df.shape)
# %%
df.to_csv(
    "LTHC_phase2_cleaned.csv",
    index=False
)

print("Phase 2 dataset saved successfully.")
# %%
