from __future__ import annotations

import pandas as pd

from .config import (
    AGE,
    ANY_CONDITION,
    COMMON_CONDITIONS,
    LTHC,
    REGION,
    SEX,
    SOURCE_COUNTRY,
    SPECIFIC_CONDITIONS,
    YEARS,
)
from .data import select_rows, weighted_prevalence


def prevalence_by_age(df: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> pd.DataFrame:
    rows = select_rows(df, conditions=conditions, sexes=["Persons"])
    return weighted_prevalence(rows, [LTHC, AGE])


def prevalence_by_sex(df: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> pd.DataFrame:
    rows = select_rows(df, conditions=conditions)
    return weighted_prevalence(rows, [LTHC, SEX])


def prevalence_by_region(df: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> pd.DataFrame:
    rows = select_rows(df, conditions=conditions, sexes=["Persons"])
    return weighted_prevalence(rows, [REGION, LTHC])


def age_sex_table(df: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> pd.DataFrame:
    rows = select_rows(df, conditions=conditions, sexes=["Female", "Male"])
    return weighted_prevalence(rows, [LTHC, SEX, AGE])


def residency_by_age(df: pd.DataFrame, condition: str = ANY_CONDITION) -> pd.DataFrame:
    rows = select_rows(df, conditions=[condition], sexes=["Persons"], sources=[SOURCE_COUNTRY])
    return weighted_prevalence(rows, [AGE, YEARS])


def top_condition(df: pd.DataFrame, by: list[str], conditions: list[str] = SPECIFIC_CONDITIONS) -> pd.DataFrame:
    table = weighted_prevalence(select_rows(df, conditions=conditions), by + [LTHC])
    table = table.sort_values("prevalence", ascending=False)
    return table.groupby(by, sort=False).head(1).sort_values(by).reset_index(drop=True)
