from __future__ import annotations

from functools import lru_cache

import dash
import pandas as pd
from dash import html

from lthc_dashboard import analysis, components as ui, figures
from lthc_dashboard.config import (
    AGE,
    AGE_GROUPS,
    ANY_CONDITION,
    COMMON_CONDITIONS,
    LTHC,
    MODEL_EVALUATION_FILE,
    REGION,
    SEX,
    YEARS,
    age_label,
    short_condition,
)
from lthc_dashboard.data import dataset_summary, get_data, select_rows, weighted_prevalence

dash.register_page(__name__, path="/", name="Overview", title="Overview · LTHC dashboard", order=0)

COMBINED_TABLE_ROWS = 120_978

LEAD_COLOURS = {
    "Arthritis": "#0f9d8f",
    "Diabetes": "#2c5a93",
    "Mental health condition": "#7b5ea7",
    "Asthma": "#e39b2f",
}
ASIAN_OR_MIDDLE_EAST = ("Asia", "Middle East")


def _value(table: pd.DataFrame, **match) -> float:
    mask = pd.Series(True, index=table.index)
    for column, value in match.items():
        mask &= table[column] == value
    return float(table.loc[mask, "prevalence"].iloc[0])


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def age_finding(any_by_age: pd.Series) -> str:
    a = [float(any_by_age[age]) for age in AGE_GROUPS]
    return f"One or more conditions: {a[0]:.1f}% at 0–44, {a[1]:.1f}% at 45–64 and {a[2]:.1f}% at 65 and over."


def sex_finding(by_sex: pd.DataFrame) -> str:
    pivot = by_sex.pivot(index=LTHC, columns=SEX, values="prevalence")
    diff = (pivot["Female"] - pivot["Male"]).sort_values()
    female = [c for c in diff.index[::-1] if diff[c] >= 1.2]
    male = [c for c in diff.index if diff[c] <= -1.2]

    def listing(conditions: list[str], first: str, second: str) -> str:
        return _join(
            [
                f"{short_condition(c).lower()} ({pivot.loc[c, first]:.1f}% vs {pivot.loc[c, second]:.1f}%)"
                for c in conditions
            ]
        )

    parts = []
    if female:
        parts.append(f"Females: {listing(female, 'Female', 'Male')}.")
    if male:
        parts.append(f"Males: {listing(male, 'Male', 'Female')}.")
    return " ".join(parts)


def region_finding(by_region: pd.DataFrame) -> str:
    arthritis = by_region[by_region[LTHC] == "Arthritis"].sort_values("prevalence", ascending=False).head(2)
    top = _join([f"{r[REGION]} ({r['prevalence']:.1f}%)" for _, r in arthritis.iterrows()])
    leading = by_region.sort_values("prevalence", ascending=False).groupby(REGION).head(1)
    diabetes_led = leading.loc[leading[LTHC] == "Diabetes", REGION].tolist()
    text = f"Arthritis is highest for {top}"
    if diabetes_led:
        asian = sum(any(word in region for word in ASIAN_OR_MIDDLE_EAST) for region in diabetes_led)
        where = "Asian and Middle Eastern regions" if asian == len(diabetes_led) else "regions"
        text += f"; diabetes leads in {len(diabetes_led)} {where}"
    return text + "."


def residency_finding(residency: pd.DataFrame) -> str:
    oldest = AGE_GROUPS[-1]
    longer = _value(residency, **{AGE: oldest, YEARS: "More than 10 years"})
    recent = _value(residency, **{AGE: oldest, YEARS: "0–10 years"})
    every_age = all(
        _value(residency, **{AGE: age, YEARS: "More than 10 years"})
        > _value(residency, **{AGE: age, YEARS: "0–10 years"})
        for age in AGE_GROUPS
    )
    lead = "in every age group" if every_age else "in most age groups"
    return f"More than 10 years in Australia: more conditions {lead} ({longer:.1f}% vs {recent:.1f}% at 65+)."


def _lead_cell(label: str | float) -> html.Span:
    if not isinstance(label, str):
        return html.Span("—", className="cell-muted")
    condition, _, value = label.rpartition(" (")
    return html.Span(
        [
            html.Span(className="dot", style={"background": LEAD_COLOURS.get(condition, "#8a96a8")}),
            html.Span(short_condition(condition)),
            html.Span(value.rstrip(")"), className="val"),
        ],
        className="lead-cell",
    )


def leading_table(leading: pd.DataFrame) -> html.Div:
    pivot = (
        leading.assign(label=lambda t: t[LTHC] + " (" + t["prevalence"].map(lambda v: f"{v:.1f}%") + ")")
        .pivot(index=REGION, columns=SEX, values="label")
        .reset_index()
    )

    def condition_of(label):
        return label.rpartition(" (")[0] if isinstance(label, str) else None

    rows = []
    for _, row in pivot.iterrows():
        differs = condition_of(row.get("Female")) != condition_of(row.get("Male"))
        region = [row[REGION]] + ([html.Span("differs", className="tag tag-coral")] if differs else [])
        rows.append(
            html.Tr(
                [
                    html.Td(region, className="region-cell"),
                    html.Td(_lead_cell(row.get("Female"))),
                    html.Td(_lead_cell(row.get("Male"))),
                ],
                className="differs" if differs else None,
            )
        )
    header = html.Thead(html.Tr([html.Th(h, scope="col") for h in ["Region of birth", "Females", "Males"]]))
    return html.Div(html.Table([header, html.Tbody(rows)], className="data-table lead-table"), className="scroll")


def findings_list(items: list[tuple[str, str, str]]) -> html.Ul:
    return html.Ul(
        [
            html.Li(
                [
                    html.Span(icon, className="finding-icon", **{"aria-hidden": "true"}),
                    html.P([html.B(title), *ui.emphasise_numbers(text)]),
                ]
            )
            for icon, title, text in items
        ],
        className="findings",
    )


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

    any_rows = select_rows(df, conditions=[ANY_CONDITION], sexes=["Persons"])
    any_all = float(weighted_prevalence(any_rows)["prevalence"].iloc[0])
    any_by_age = weighted_prevalence(any_rows, [AGE]).set_index(AGE)["prevalence"]
    oldest = float(any_by_age[AGE_GROUPS[-1]])
    youngest = float(any_by_age[AGE_GROUPS[0]])

    kpis = ui.kpi_row(
        [
            ui.kpi(
                "Published census cells",
                f"{summary['rows']:,}",
                f"of {COMBINED_TABLE_ROWS:,} in AIHW Tables S10 and S13",
                accent="navy",
            ),
            ui.kpi(
                "Population groups",
                f"{summary['groups']:,}",
                f"{summary['countries']} countries of birth, {summary['languages']} languages",
                accent="navy",
            ),
            ui.kpi("Conditions tracked", f"{summary['conditions']}", "10 specific and 3 combined", accent="navy"),
            ui.kpi(
                "Report a long-term condition",
                ui.fmt_pct(any_all),
                f"{ui.one_in(any_all).capitalize()}; all ages, both tables",
                accent="teal",
            ),
            ui.kpi(
                "Aged 65 and over",
                ui.fmt_pct(oldest),
                f"{ui.one_in(oldest).capitalize()}, against {youngest:.1f}% at 0–44",
                accent="coral",
            ),
            ui.kpi(
                "Model accuracy",
                f"R² {xgb['R²']:.3f}",
                f"XGBoost; average error {xgb['MAE']:.1f} points",
                accent="amber",
            ),
        ]
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

    findings = ui.card(
        "Key findings",
        html.Div(
            [
                findings_list(
                    [
                        ("1", "Age", age_finding(any_by_age)),
                        ("2", "Sex", sex_finding(by_sex)),
                        ("3", "Region", region_finding(by_region)),
                        ("4", "Residence", residency_finding(residency)),
                    ]
                ),
                html.P(
                    "From the exploratory analysis in the Assessment 2 report. The Analysis page age-standardises "
                    "comparisons between groups.",
                    className="method-line tall-only",
                ),
            ],
            className="card-body scroll",
        ),
        subtitle="Weighted prevalence (people reporting ÷ population), all ages",
    )

    return ui.screen(
        "overview",
        "Long-term health conditions among CALD Australians",
        "2021 Census (AIHW Tables S10 and S13), weighted prevalence. Hover a chart for counts, or switch a card to "
        "Table for the numbers.",
        kpis,
        html.Div(
            [
                ui.chart_card(
                    "Age is the strongest driver",
                    figures.age_by_condition(by_age),
                    age_table,
                    key="ov-age",
                    subtitle="Five most common conditions by age group, all people",
                ),
                ui.chart_card(
                    "Females and males differ by condition",
                    figures.sex_by_condition(by_sex),
                    sex_table,
                    key="ov-sex",
                    subtitle="Prevalence by sex, all ages; the table adds all people",
                ),
                ui.chart_card(
                    "Region of birth changes which condition leads",
                    figures.region_heatmap(by_region),
                    region_table,
                    key="ov-region",
                    subtitle="Prevalence (%) by region of birth, all people, all ages",
                ),
                ui.chart_card(
                    "Longer residence, higher prevalence at every age",
                    figures.residency_by_age(residency),
                    residency_table,
                    key="ov-years",
                    subtitle="One or more conditions, by years in Australia and age",
                ),
                ui.card(
                    "Leading condition by region and sex",
                    html.Div(leading_table(leading), className="card-body"),
                    subtitle="Highest of the ten specific conditions for females and males, all ages",
                ),
                findings,
            ],
            className="grid-3x2",
        ),
        action=("Estimate prevalence for a group", "/predict"),
    )


def layout(**_kwargs):
    return _content()
