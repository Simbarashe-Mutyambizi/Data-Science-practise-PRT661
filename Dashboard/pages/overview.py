"""Overview: headline findings from the exploratory analysis (historical data)."""

from __future__ import annotations

from functools import lru_cache

import dash
import pandas as pd
from dash import dcc, html

from lthc_dashboard import analysis, components as ui, figures
from lthc_dashboard.config import (
    AGE,
    AGE_GROUPS,
    ANY_CONDITION,
    COMMON_CONDITIONS,
    LTHC,
    MODEL_EVALUATION_FILE,
    POPULATION,
    REGION,
    SEX,
    SOURCE_COUNTRY,
    YEARS,
    age_label,
    short_condition,
)
from lthc_dashboard.data import dataset_summary, get_data, select_rows, weighted_prevalence

dash.register_page(__name__, path="/", name="Overview", title="Overview · LTHC dashboard", order=0)

COMBINED_TABLE_ROWS = 120_978  # rows in Cleaned_data_final.csv (Tables S10 + S13 before removing n.p. cells)


def _value(table: pd.DataFrame, **match) -> float:
    mask = pd.Series(True, index=table.index)
    for column, value in match.items():
        mask &= table[column] == value
    return float(table.loc[mask, "prevalence"].iloc[0])


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# Insight text (computed from the data so it always matches the charts)
# ---------------------------------------------------------------------------
def age_insight(df: pd.DataFrame, table: pd.DataFrame) -> str:
    anyc = weighted_prevalence(select_rows(df, conditions=[ANY_CONDITION], sexes=["Persons"]), [AGE])
    a = [_value(anyc, **{AGE: age}) for age in AGE_GROUPS]
    young = table[table[AGE] == AGE_GROUPS[0]].sort_values("prevalence", ascending=False)
    lead = list(young[LTHC].head(2))
    growth = {
        c: _value(table, **{LTHC: c, AGE: AGE_GROUPS[-1]}) - _value(table, **{LTHC: c, AGE: AGE_GROUPS[0]})
        for c in COMMON_CONDITIONS
    }
    slowest = sorted(growth, key=growth.get)[:2]
    text = (
        f"Reporting at least one long-term condition climbs from {a[0]:.1f}% at 0–44 to {a[1]:.1f}% at 45–64 "
        f"and {a[2]:.1f}% at 65 and over. Arthritis rises from "
        f"{_value(table, **{LTHC: 'Arthritis', AGE: AGE_GROUPS[0]}):.1f}% to "
        f"{_value(table, **{LTHC: 'Arthritis', AGE: AGE_GROUPS[-1]}):.1f}% and diabetes from "
        f"{_value(table, **{LTHC: 'Diabetes', AGE: AGE_GROUPS[0]}):.1f}% to "
        f"{_value(table, **{LTHC: 'Diabetes', AGE: AGE_GROUPS[-1]}):.1f}%."
    )
    lead_text = _join(
        [f"{short_condition(c).lower()} ({_value(table, **{LTHC: c, AGE: AGE_GROUPS[0]}):.1f}%)" for c in lead]
    )
    text += f" Before 45 the most common of these five are {lead_text}"
    text += " — and they grow least with age." if set(lead) == set(slowest) else "."
    return text


def sex_insight(table: pd.DataFrame) -> str:
    pivot = table.pivot(index=LTHC, columns=SEX, values="prevalence")
    diff = (pivot["Female"] - pivot["Male"]).sort_values()
    female = [c for c in diff.index[::-1] if diff[c] >= 1.2]
    male = [c for c in diff.index if diff[c] <= -1.2]

    def listing(conditions: list[str]) -> str:
        return _join(
            [
                f"{short_condition(c).lower()} ({pivot.loc[c, 'Female']:.1f}% of females vs {pivot.loc[c, 'Male']:.1f}% of males)"
                for c in conditions
            ]
        )

    parts = []
    if female:
        parts.append(f"Higher among females: {listing(female)}.")
    if male:
        parts.append(f"Higher among males: {listing(male)}.")
    return " ".join(parts) + " All ages, both census tables."


def region_insight(df: pd.DataFrame, table: pd.DataFrame) -> str:
    arthritis = table[table[LTHC] == "Arthritis"].sort_values("prevalence", ascending=False).head(2)
    top_arthritis = _join([f"{r[REGION]} ({r['prevalence']:.1f}%)" for _, r in arthritis.iterrows()])
    leading = table.sort_values("prevalence", ascending=False).groupby(REGION).head(1)
    diabetes_led = sorted(leading.loc[leading[LTHC] == "Diabetes", REGION])
    cells = select_rows(df, sexes=["Persons"], sources=[SOURCE_COUNTRY]).drop_duplicates("group_key")
    share = cells.groupby([REGION, AGE])[POPULATION].sum().unstack()
    older = (share[AGE_GROUPS[-1]] / share.sum(axis=1) * 100).sort_values(ascending=False)
    text = f"Arthritis is highest for people born in {top_arthritis}."
    if diabetes_led:
        text += f" Diabetes leads these five conditions for {_join(diabetes_led)}."
    text += (
        f" These are all-age rates, and age structure differs a lot: {older.iloc[0]:.0f}% of people born in "
        f"{older.index[0]} in the published cells are 65 or over, against {older.iloc[-1]:.0f}% for "
        f"{older.index[-1]}. Use Explore with an age filter to compare like with like."
    )
    return text


def residency_insight(table: pd.DataFrame) -> str:
    pairs = []
    for age in AGE_GROUPS:
        recent = _value(table, **{AGE: age, YEARS: "0–10 years"})
        longer = _value(table, **{AGE: age, YEARS: "More than 10 years"})
        pairs.append(f"{longer:.1f}% vs {recent:.1f}% at {age_label(age)}")
    return (
        "People who have lived in Australia for more than 10 years report more long-term conditions than recent "
        f"arrivals in every age group ({'; '.join(pairs)}). Because the gap holds within each age group, age alone "
        "does not explain it; the healthy-migrant effect fading over time and differences in diagnosis and "
        "reporting are possible reasons. This is an association, not a cause."
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _content() -> html.Div:
    data = get_data()
    df = data.df
    summary = dataset_summary(df)
    evaluation = pd.read_csv(MODEL_EVALUATION_FILE).set_index("Model")
    xgb = evaluation.loc["XGBoost"]

    by_age = analysis.prevalence_by_age(df)
    by_sex = analysis.prevalence_by_sex(df)
    by_region = analysis.prevalence_by_region(df)
    residency = analysis.residency_by_age(df)
    leading = analysis.top_condition(df, [REGION, SEX])

    tiles = html.Div(
        [
            ui.stat_tile(
                "Published census cells analysed",
                f"{summary['rows']:,}",
                f"of {COMBINED_TABLE_ROWS:,} cells in AIHW Tables S10 and S13; the rest were suppressed (n.p.) or empty",
            ),
            ui.stat_tile(
                "Population groups",
                f"{summary['groups']:,}",
                f"{summary['countries']} countries of birth · {summary['languages']} home languages",
            ),
            ui.stat_tile(
                "Conditions tracked", f"{summary['conditions']}", "10 specific conditions and 3 combined categories"
            ),
            ui.stat_tile(
                "Model accuracy on unseen groups",
                f"R² {xgb['R²']:.3f}",
                f"XGBoost · average error {xgb['MAE']:.1f} percentage points",
            ),
        ],
        className="tiles",
    )

    age_table = ui.data_table(
        by_age.assign(condition=by_age[LTHC].map(short_condition), age=by_age[AGE].map(age_label)).sort_values(
            [LTHC, AGE]
        ),
        [
            ("condition", "Condition", None),
            ("age", "Age group", None),
            ("prevalence", "Prevalence", ui.fmt_pct),
            ("cases", "People reporting", ui.fmt_int),
            ("population", "Population", ui.fmt_int),
        ],
    )
    sex_table = ui.data_table(
        by_sex.pivot(index=LTHC, columns=SEX, values="prevalence")
        .reset_index()
        .assign(condition=lambda t: t[LTHC].map(short_condition)),
        [
            ("condition", "Condition", None),
            ("Female", "Female", ui.fmt_pct),
            ("Male", "Male", ui.fmt_pct),
            ("Persons", "All people", ui.fmt_pct),
        ],
    )
    region_table = ui.data_table(
        by_region.pivot(index=REGION, columns=LTHC, values="prevalence").reset_index()[[REGION] + COMMON_CONDITIONS],
        [(REGION, "Region", None)] + [(c, short_condition(c), ui.fmt_pct) for c in COMMON_CONDITIONS],
    )
    residency_table = ui.data_table(
        residency.assign(age=residency[AGE].map(age_label)),
        [
            ("age", "Age group", None),
            (YEARS, "Years in Australia", None),
            ("prevalence", "One or more conditions", ui.fmt_pct),
            ("population", "Population", ui.fmt_int),
        ],
    )
    leading_pivot = (
        leading.assign(label=lambda t: t[LTHC] + " (" + t["prevalence"].map(lambda v: f"{v:.1f}%") + ")")
        .pivot(index=REGION, columns=SEX, values="label")
        .reset_index()
    )

    return html.Div(
        [
            ui.page_header(
                "Long-term health conditions among culturally and linguistically diverse Australians",
                "How the share of people reporting a long-term health condition changes with age, sex, region of birth "
                "and time in Australia — and a model that estimates it for any published population group. "
                "All figures are weighted prevalence: people reporting a condition divided by the population of the group.",
                eyebrow="2021 Census · Australian Institute of Health and Welfare",
            ),
            tiles,
            html.Div(
                [
                    ui.card(
                        "Age is the strongest driver",
                        ui.graph(figures.age_by_condition(by_age)),
                        ui.insight(age_insight(df, by_age)),
                        ui.table_view(age_table),
                        subtitle="Prevalence of the five most common conditions by age group · all people",
                    ),
                    ui.card(
                        "Females and males differ by condition",
                        ui.graph(figures.sex_by_condition(by_sex)),
                        ui.insight(sex_insight(by_sex)),
                        ui.table_view(sex_table),
                        subtitle="Prevalence by sex · ‘All people’ is the census Persons total",
                    ),
                    ui.card(
                        "Region of birth changes which condition leads",
                        ui.graph(figures.region_heatmap(by_region)),
                        ui.insight(region_insight(df, by_region)),
                        ui.table_view(region_table),
                        subtitle="Prevalence (%) by region · all people, all ages, both census tables",
                    ),
                    ui.card(
                        "Longer residence, higher reported prevalence — at every age",
                        ui.graph(figures.residency_by_age(residency)),
                        ui.insight(residency_insight(residency)),
                        ui.table_view(residency_table),
                        subtitle="One or more long-term conditions · country-of-birth table (years in Australia is only recorded there)",
                    ),
                ],
                className="grid-2",
            ),
            ui.card(
                "Leading specific condition by region and sex",
                ui.data_table(
                    leading_pivot,
                    [
                        (REGION, "Region", None),
                        ("Female", "Female", None),
                        ("Male", "Male", None),
                        ("Persons", "All people", None),
                    ],
                ),
                html.P(
                    "Highest-prevalence condition among the ten specific conditions, all ages, both census tables.",
                    className="card-foot",
                ),
                subtitle="Where a region's leading condition differs by sex, programmes may need to differ too",
            ),
            html.Div(
                [
                    dcc.Link(
                        [
                            html.Strong("Explore the data"),
                            html.Span("Compare any condition across regions, countries, languages and more."),
                        ],
                        href="/explore",
                        className="cta",
                    ),
                    dcc.Link(
                        [
                            html.Strong("Estimate prevalence"),
                            html.Span("Pick a population group and get the model's estimate with context."),
                        ],
                        href="/predict",
                        className="cta",
                    ),
                    dcc.Link(
                        [
                            html.Strong("How the model works"),
                            html.Span("Accuracy on unseen groups, what drives predictions, and limitations."),
                        ],
                        href="/model",
                        className="cta",
                    ),
                ],
                className="cta-row",
            ),
        ]
    )


def layout(**_kwargs):
    return _content()
