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


def load_dataset(path: Path | str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["group_key"] = df[GROUP_COLUMNS].astype(str).agg("|".join, axis=1)
    df["split"] = assign_split(df)
    for column in (AGE, YEARS):
        df[column] = df[column].astype(str).str.replace("â€“", "–", regex=False)
    df["source"] = np.where(df[YEARS] != NOT_SPECIFIED, SOURCE_COUNTRY, SOURCE_LANGUAGE)
    return df


def assign_split(df: pd.DataFrame) -> np.ndarray:
    splitter = GroupShuffleSplit(n_splits=1, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df["group_key"]))
    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"
    return split


def weighted_prevalence(rows: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
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


@dataclass(frozen=True)
class Lookups:
    country_region: dict[str, str]
    country_subregion: dict[str, str]
    country_language: dict[str, str]
    language_country: dict[str, str]
    countries: list[str]
    languages: list[str]


def build_lookups(df: pd.DataFrame) -> Lookups:
    country_rows = df[df["source"] == SOURCE_COUNTRY]
    language_rows = df[df["source"] == SOURCE_LANGUAGE]

    def first(frame, key, value):
        return frame.drop_duplicates(key).set_index(key)[value].to_dict()

    return Lookups(
        country_region=first(df, COUNTRY, REGION),
        country_subregion=first(df, COUNTRY, SUBREGION),
        country_language=first(country_rows, COUNTRY, LANGUAGE),
        language_country=first(language_rows, LANGUAGE, COUNTRY),
        countries=sorted(country_rows[COUNTRY].unique(), key=str.casefold),
        languages=sorted(language_rows[LANGUAGE].unique(), key=lambda s: s.lstrip(".").casefold()),
    )


@dataclass
class DashboardData:
    df: pd.DataFrame
    lookups: Lookups
    cell_index: dict[tuple, int]
    group_index: dict[str, np.ndarray]

    def find_cell(self, record: dict[str, str]) -> pd.Series | None:
        key = tuple(record[column] for column in MODEL_FEATURES)
        position = self.cell_index.get(key)
        return None if position is None else self.df.iloc[position]

    def group_rows(self, record: dict[str, str]) -> pd.DataFrame:
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
    return {
        "rows": len(df),
        "groups": df["group_key"].nunique(),
        "countries": df.loc[df["source"] == SOURCE_COUNTRY, COUNTRY].nunique(),
        "languages": df.loc[df["source"] == SOURCE_LANGUAGE, LANGUAGE].nunique(),
        "conditions": df[LTHC].nunique(),
        "train_rows": int((df["split"] == "train").sum()),
        "test_rows": int((df["split"] == "test").sum()),
    }
