# %%
"""
Data_Cleaning.py  -  PRT661 Group Dan3, Assessment 4

Builds the modelling dataset from the AIHW 2021 CALD chronic-conditions workbook.

What changed from Assessment 3 (marker feedback):
  * NO country <-> language imputation. Table S10 (country of birth, years in
    Australia) and Table S13 (language used at home, English proficiency) are
    kept as separate sources. Columns a table never collected are set to
    'Not collected' instead of being guessed by an AI-generated mapping.
  * A 'source_table' column records which table each row came from.
  * 'Persons' rows are kept (Sex = Persons / Male / Female). Persons is the
    population-weighted average of the Male and Female rows, so the grouping
    key leaves out Sex: a cell's Persons, Male and Female rows always land in
    the same train/test fold, which prevents leakage.
    For descriptive charts, sum Population over Persons rows only - adding
    Male and Female on top counts people twice.
  * Cells AIHW suppressed ('n.p') are dropped, not converted to 0.
    The published data contains no 0% cells.

Input : aihw-phe-320-age-specific-percentages-chronic-conditions-cald-2021.xlsx
Output: Cleaned_data_A4.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from region_mapping import COUNTRY_TO_REGION, COUNTRY_TO_SUBREGION

RAW_FILE = Path("aihw-phe-320-age-specific-percentages-chronic-conditions-cald-2021.xlsx")
OUTPUT_FILE = Path("Cleaned_data_A4.csv")

# Persons is kept: removing it would cut the data from 16,184 to 6,278 rows and
# drop every country / language that is only published for Persons
# (168 -> 67 countries, 147 -> 61 languages). The grouping below keeps a cell's
# Persons, Male and Female rows together, so keeping Persons does not leak.
REMOVE_PERSONS = False

NOT_COLLECTED = "Not collected"

COUNTRY = "Country of birth of person"
YEARS = "Years spent in Australia"
LANGUAGE = "Language used at home"
PROFICIENCY = "Proficiency in spoken English"
AGE = "Age group"
SEX = "Sex"
LTHC = "Long-term health condition (LTHC)"
NUMBER = "Number of people reporting LTHC(s)"
POPULATION = "Population"
TARGET = "Age-specific percentage of population reporting LTHC(s)"

CATEGORICAL_COLUMNS = [
    COUNTRY, YEARS, LANGUAGE, PROFICIENCY,
    "Region_class", "Subregion_class", AGE, SEX, LTHC,
]
NUMERIC_COLUMNS = [NUMBER, POPULATION, TARGET]

# %%
# 1. Load the two source tables separately and tag them
s10 = pd.read_excel(RAW_FILE, sheet_name="Table S10", skiprows=3)
s13 = pd.read_excel(RAW_FILE, sheet_name="Table S13", skiprows=3)

s10["source_table"] = "S10"
s13["source_table"] = "S13"

print(f"Table S10 (country of birth): {len(s10):,} rows")
print(f"Table S13 (language at home): {len(s13):,} rows")

# %%
# 2. Region and subregion come from country of birth, so they only exist for S10.
#    This is a geographic lookup (ABS SACC groups), not a language guess.
unmapped = sorted(set(s10[COUNTRY]) - set(COUNTRY_TO_REGION))
if unmapped:
    raise ValueError(f"Countries with no region mapping: {unmapped}")

s10["Region_class"] = s10[COUNTRY].map(COUNTRY_TO_REGION)
s10["Subregion_class"] = s10[COUNTRY].map(COUNTRY_TO_SUBREGION)

# %%
# 3. Stack the tables. Columns a table never collected become NaN here.
df = pd.concat([s10, s13], ignore_index=True, sort=False)
print(f"\nCombined: {len(df):,} rows")

# %%
# 4. Drop suppressed cells ('n.p'). In both tables the count, population and
#    percentage are always suppressed together, so these rows hold no data.
for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

suppressed = df[TARGET].isna()
print("\nSuppressed (n.p) rows dropped:")
print(df.loc[suppressed, "source_table"].value_counts().to_string())
df = df.loc[~suppressed].copy()

# %%
# 5. Remove 'Persons' (it is the weighted average of Male and Female)
if REMOVE_PERSONS:
    persons = df[SEX] == "Persons"
    print(f"\n'Persons' rows removed: {persons.sum():,}")
    df = df.loc[~persons].copy()

# %%
# 6. Every categorical gap is a variable the source table did not collect
df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna(NOT_COLLECTED)

# One ID per demographic cell (everything except condition and sex).
# Used for the grouped train/test split and grouped cross-validation, so all
# conditions and both sexes of one cell always land on the same side of a split.
df["demographic_group"] = (
    df[["source_table", COUNTRY, YEARS, LANGUAGE, PROFICIENCY, AGE]]
    .astype(str)
    .agg("|".join, axis=1)
)

# %%
# 7. Checks
assert df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS].isna().sum().sum() == 0, "Nulls remain"
if REMOVE_PERSONS:
    assert (df[SEX] != "Persons").all()
assert (df[TARGET] > 0).all() and (df[TARGET] <= 100).all(), "Prevalence outside (0, 100]"

is_s10 = df["source_table"] == "S10"
assert (df.loc[is_s10, [LANGUAGE, PROFICIENCY]] == NOT_COLLECTED).all().all()
assert (df.loc[~is_s10, [COUNTRY, YEARS, "Region_class", "Subregion_class"]] == NOT_COLLECTED).all().all()
assert (df.loc[is_s10, COUNTRY] != NOT_COLLECTED).all()
assert (df.loc[~is_s10, LANGUAGE] != NOT_COLLECTED).all()

# Published percentage should equal count / population
recalculated = df[NUMBER] / df[POPULATION] * 100
print(f"\nMax |published % - count/population|: {(recalculated - df[TARGET]).abs().max():.4f} pp")

# %%
# 8. Summary and save
print("\nFinal dataset")
print(f"  rows                 : {len(df):,}")
print(f"  by source table      : {df['source_table'].value_counts().to_dict()}")
print(f"  by sex               : {df[SEX].value_counts().to_dict()}")
print(f"  demographic groups   : {df['demographic_group'].nunique():,}")
print(f"  countries (S10)      : {df.loc[is_s10, COUNTRY].nunique()}")
print(f"  languages (S13)      : {df.loc[~is_s10, LANGUAGE].nunique()}")
print(f"  prevalence range     : {df[TARGET].min():.4f}% to {df[TARGET].max():.2f}%")

column_order = ["source_table"] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + ["demographic_group"]
df[column_order].to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {OUTPUT_FILE}")
# %%
