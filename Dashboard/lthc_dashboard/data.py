"""Loading the cleaned AIHW data and the weighted-prevalence method.

Each row of the dataset is a census cell (a demographic group x one long-term
health condition), not a person. Two AIHW tables were combined during cleaning:

* Table S10 (country of birth): country and years in Australia are real values,
  language was imputed from country, English proficiency is "Not specified".
* Table S13 (language used at home): language and English proficiency are real
  values, country was imputed from language, years in Australia is "Not specified".

Every row belongs to exactly one of the two tables, which is recorded in the
``source`` column added here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    AGE,
    CASES,
    COUNTRY,
    DATA_FILE,
    GROUP_COLUMNS,
    LANGUAGE,
    LTHC,
    MODEL_FEATURES,
    NOT_SPECIFIED,
    POPULATION,
    REGION,
    SEX,
    SOURCE_COUNTRY,
    SOURCE_LANGUAGE,
    SPLIT_RANDOM_STATE,
    SPLIT_TEST_SIZE,
    SUBREGION,
    TARGET,
    YEARS,
)

PREVALENCE_COLUMNS = ["cases", "population", "prevalence", "cells"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Read the cleaned dataset and add ``source``, ``group_key`` and ``split``."""
    df = pd.read_csv(path)
    df["group_key"] = df[GROUP_COLUMNS].astype(str).agg("|".join, axis=1)
    # The split is recreated before any text normalisation so the group labels
    # are byte-identical to the ones used in Dataset_prepping_Attempt2.ipynb.
    df["split"] = assign_split(df)
    for column in (AGE, YEARS):
        # The Excel export used in the bivariate notebook showed a mis-encoded
        # en dash; normalise defensively (no-op on the CSV).
        df[column] = df[column].astype(str).str.replace("â€“", "–", regex=False)
    df["source"] = np.where(df[YEARS] != NOT_SPECIFIED, SOURCE_COUNTRY, SOURCE_LANGUAGE)
    return df


def assign_split(df: pd.DataFrame) -> np.ndarray:
    """Recreate the group-aware train/test split used to train the models.

    ``GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`` over the
    demographic group gives exactly the 13,060 training and 3,124 test rows
    saved as y_train.csv / y_test.csv on the Machine_Learning branch.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df["group_key"]))
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"
    return split


# ---------------------------------------------------------------------------
# Weighted prevalence
# ---------------------------------------------------------------------------
def weighted_prevalence(rows: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Pool cases and population within each group, then divide.

    prevalence (%) = sum(Number of people reporting LTHC(s)) / sum(Population) x 100

    This is the weighted-prevalence function from the Assessment 2 report: a
    group of 10,000 people counts for more than a volatile group of 1,000.
    Returns one row per group with ``cases``, ``population``, ``prevalence``
    and ``cells`` (number of census cells pooled).
    """
    by = list(by or [])
    if rows.empty:
        return pd.DataFrame(columns=by + PREVALENCE_COLUMNS)
    if by:
        grouped = (
            rows.groupby(by, observed=True, sort=False)
            .agg(cases=(CASES, "sum"), population=(POPULATION, "sum"), cells=(CASES, "size"))
            .reset_index()
        )
    else:
        grouped = pd.DataFrame(
            {
                "cases": [rows[CASES].sum()],
                "population": [rows[POPULATION].sum()],
                "cells": [len(rows)],
            }
        )
    grouped = grouped[grouped["population"] > 0].copy()
    grouped["prevalence"] = grouped["cases"] / grouped["population"] * 100
    return grouped[by + PREVALENCE_COLUMNS].reset_index(drop=True)


def select_rows(
    df: pd.DataFrame,
    *,
    conditions: list[str] | None = None,
    ages: list[str] | None = None,
    sexes: list[str] | None = None,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Filter rows; ``None`` means no filter on that column."""
    mask = pd.Series(True, index=df.index)
    if conditions is not None:
        mask &= df[LTHC].isin(conditions)
    if ages is not None:
        mask &= df[AGE].isin(ages)
    if sexes is not None:
        mask &= df[SEX].isin(sexes)
    if sources is not None:
        mask &= df["source"].isin(sources)
    return df[mask]


# ---------------------------------------------------------------------------
# Lookups used by the prediction form
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lookups:
    country_region: dict[str, str]
    country_subregion: dict[str, str]
    country_language: dict[str, str]  # language paired with a country in Table S10 rows
    language_country: dict[str, str]  # country paired with a language in Table S13 rows
    countries: list[str]  # countries with real (Table S10) rows
    languages: list[str]  # languages with real (Table S13) rows


def build_lookups(df: pd.DataFrame) -> Lookups:
    country_rows = df[df["source"] == SOURCE_COUNTRY]
    language_rows = df[df["source"] == SOURCE_LANGUAGE]
    first = lambda frame, key, value: (  # noqa: E731 - small local helper
        frame.drop_duplicates(key).set_index(key)[value].to_dict()
    )
    return Lookups(
        country_region=first(df, COUNTRY, REGION),
        country_subregion=first(df, COUNTRY, SUBREGION),
        country_language=first(country_rows, COUNTRY, LANGUAGE),
        language_country=first(language_rows, LANGUAGE, COUNTRY),
        countries=sorted(country_rows[COUNTRY].unique(), key=str.casefold),
        languages=sorted(language_rows[LANGUAGE].unique(), key=lambda s: s.lstrip(".").casefold()),
    )


# ---------------------------------------------------------------------------
# Container with the data and fast indexes, loaded once per process
# ---------------------------------------------------------------------------
@dataclass
class DashboardData:
    df: pd.DataFrame
    lookups: Lookups
    cell_index: dict[tuple, int]
    group_index: dict[str, np.ndarray]

    def find_cell(self, record: dict[str, str]) -> pd.Series | None:
        """The published census row for an exact model input, if AIHW published it."""
        key = tuple(record[column] for column in MODEL_FEATURES)
        position = self.cell_index.get(key)
        return None if position is None else self.df.iloc[position]

    def group_rows(self, record: dict[str, str]) -> pd.DataFrame:
        """All published condition rows for the same demographic group."""
        key = "|".join(str(record[column]) for column in GROUP_COLUMNS)
        positions = self.group_index.get(key)
        if positions is None:
            return self.df.iloc[0:0]
        return self.df.iloc[positions]


def build_dashboard_data(df: pd.DataFrame) -> DashboardData:
    keys = list(df[MODEL_FEATURES].itertuples(index=False, name=None))
    cell_index = {key: position for position, key in enumerate(keys)}
    group_index = {key: np.asarray(positions) for key, positions in df.groupby("group_key", sort=False).indices.items()}
    return DashboardData(df=df, lookups=build_lookups(df), cell_index=cell_index, group_index=group_index)


@lru_cache(maxsize=1)
def get_data() -> DashboardData:
    return build_dashboard_data(load_dataset())


def dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    """Headline counts for the overview page."""
    return {
        "rows": len(df),
        "groups": df["group_key"].nunique(),
        "countries": df.loc[df["source"] == SOURCE_COUNTRY, COUNTRY].nunique(),
        "languages": df.loc[df["source"] == SOURCE_LANGUAGE, LANGUAGE].nunique(),
        "conditions": df[LTHC].nunique(),
        "train_rows": int((df["split"] == "train").sum()),
        "test_rows": int((df["split"] == "test").sum()),
    }
