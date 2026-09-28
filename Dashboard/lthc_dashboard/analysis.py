"""Descriptive analysis behind the Overview and Explore pages.

Method rules (from the Assessment 2 report):
* prevalence is weighted: pooled cases / pooled population;
* "Persons" is the Male + Female total, so it is used on its own whenever sex is
  not the comparison, and never pooled with Male/Female rows (no double counting);
* country of birth and years in Australia are only real in Table S10 rows, and
  home language and English proficiency only in Table S13 rows, so comparisons
  by those variables use that table only. Other comparisons pool both tables,
  as in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    AGE,
    AGE_GROUPS,
    ANY_CONDITION,
    COMMON_CONDITIONS,
    COUNTRY,
    ENGLISH,
    ENGLISH_OPTIONS,
    LANGUAGE,
    LTHC,
    NOT_SPECIFIED,
    REGION,
    SEX,
    SEXES,
    SOURCE_COUNTRY,
    SOURCE_LABELS,
    SOURCE_LANGUAGE,
    SPECIFIC_CONDITIONS,
    SUBREGION,
    YEARS,
    YEARS_OPTIONS,
    age_label,
    sex_label,
)
from .data import select_rows, weighted_prevalence

ALL = "all"
BOTH = "both"


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    column: str
    source: str | None = None  # table the variable is real in (None = both)
    order: tuple[str, ...] | None = None  # natural order for ordinal variables


DIMENSIONS: dict[str, Dimension] = {
    d.key: d
    for d in [
        Dimension("age", "Age group", AGE, order=tuple(AGE_GROUPS)),
        Dimension("sex", "Sex", SEX, order=tuple(SEXES)),
        Dimension("region", "Region", REGION),
        Dimension("subregion", "Subregion", SUBREGION),
        Dimension("country", "Country of birth", COUNTRY, source=SOURCE_COUNTRY),
        Dimension("language", "Language used at home", LANGUAGE, source=SOURCE_LANGUAGE),
        Dimension("years", "Years in Australia", YEARS, source=SOURCE_COUNTRY, order=tuple(YEARS_OPTIONS)),
        Dimension(
            "english", "Proficiency in spoken English", ENGLISH, source=SOURCE_LANGUAGE, order=tuple(ENGLISH_OPTIONS)
        ),
    ]
}


def display_value(dimension: Dimension, value: str) -> str:
    if dimension.column == AGE:
        return age_label(value)
    if dimension.column == SEX:
        return sex_label(value)
    return value


# ---------------------------------------------------------------------------
# Overview tables (the report's headline analyses)
# ---------------------------------------------------------------------------
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
    """Highest-prevalence condition within each group (e.g. region x sex)."""
    table = weighted_prevalence(select_rows(df, conditions=conditions), by + [LTHC])
    table = table.sort_values("prevalence", ascending=False)
    return table.groupby(by, sort=False).head(1).sort_values(by).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Explore page
# ---------------------------------------------------------------------------
@dataclass
class Comparison:
    dimension: Dimension
    condition: str
    table: pd.DataFrame  # one row per group, sorted for display
    overall: float | None  # prevalence for the whole filtered selection
    excluded_small: int  # groups hidden by the minimum-population filter
    notes: list[str] = field(default_factory=list)
    rows: pd.DataFrame | None = None  # filtered rows (all conditions), for the profile


def compare_groups(
    df: pd.DataFrame,
    dimension_key: str,
    condition: str,
    *,
    age: str = ALL,
    sex: str = "Persons",
    source: str = BOTH,
    min_population: int = 0,
) -> Comparison:
    dimension = DIMENSIONS[dimension_key]
    notes: list[str] = []

    sexes = list(SEXES) if dimension.column == SEX else [sex]
    ages = None if (dimension.column == AGE or age == ALL) else [age]

    if dimension.source:
        sources = [dimension.source]
        if source not in (BOTH, dimension.source):
            notes.append(
                f"{dimension.label} is only recorded in the {SOURCE_LABELS[dimension.source]}, "
                "so that table is used regardless of the census table filter."
            )
        else:
            notes.append(f"{dimension.label} is only recorded in the {SOURCE_LABELS[dimension.source]}.")
    elif source == BOTH:
        sources = None
        notes.append(
            "Both census tables are pooled, as in the Assessment 2 report. For language-table rows, "
            "region and subregion come from the country paired with the home language."
        )
    else:
        sources = [source]

    if dimension.column != SEX:
        if sex == "Persons":
            notes.append("‘All people’ uses the census Persons totals only, so nobody is counted twice.")
    else:
        notes.append(
            "‘All people’ is the census Persons total (Female + Male, including cells where only the total was published)."
        )

    if ages is None and dimension.column != AGE:
        notes.append(
            "All ages combined gives crude rates: groups with older populations show higher prevalence. "
            "Pick an age group to compare like with like."
        )

    rows = select_rows(df, ages=ages, sexes=sexes, sources=sources)
    rows = rows[rows[dimension.column] != NOT_SPECIFIED]
    condition_rows = rows[rows[LTHC] == condition]

    table = weighted_prevalence(condition_rows, [dimension.column])
    overall_rows = condition_rows if dimension.column != SEX else condition_rows[condition_rows[SEX] == "Persons"]
    overall_table = weighted_prevalence(overall_rows)
    overall = float(overall_table["prevalence"].iloc[0]) if len(overall_table) else None

    excluded = 0
    if min_population and dimension.order is None:
        small = table["population"] < min_population
        excluded = int(small.sum())
        table = table[~small]

    if dimension.order:
        order = {value: position for position, value in enumerate(dimension.order)}
        table = table.assign(_order=table[dimension.column].map(order)).sort_values("_order").drop(columns="_order")
    else:
        table = table.sort_values("prevalence", ascending=False)
    table = table.reset_index(drop=True)
    table.insert(0, "group", [display_value(dimension, value) for value in table[dimension.column]])

    return Comparison(
        dimension=dimension,
        condition=condition,
        table=table,
        overall=overall,
        excluded_small=excluded,
        notes=notes,
        rows=rows,
    )


def condition_profile(
    comparison: Comparison, groups: list[str], conditions: list[str] = SPECIFIC_CONDITIONS
) -> pd.DataFrame:
    """Prevalence of every condition for the listed groups (same filters)."""
    column = comparison.dimension.column
    rows = comparison.rows
    if rows is None or not groups:
        return pd.DataFrame(columns=[column, LTHC, "prevalence"])
    rows = rows[rows[column].isin(groups) & rows[LTHC].isin(conditions)]
    return weighted_prevalence(rows, [column, LTHC])
