"""Explore: weighted prevalence of any condition across any demographic variable."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dcc, html

from lthc_dashboard import components as ui, figures
from lthc_dashboard.analysis import ALL, BOTH, DIMENSIONS, compare_groups, condition_profile, display_value
from lthc_dashboard.config import (
    AGE_GROUPS,
    COMPOSITE_CONDITIONS,
    SOURCE_COUNTRY,
    SOURCE_LANGUAGE,
    SPECIFIC_CONDITIONS,
    age_label,
    short_condition,
)
from lthc_dashboard.data import get_data
from lthc_dashboard.theme import BLUE

dash.register_page(__name__, path="/explore", name="Explore", title="Explore · LTHC dashboard", order=1)

TOP_N = 20
PROFILE_ROWS = 12

CONDITION_OPTIONS = [{"label": c, "value": c} for c in SPECIFIC_CONDITIONS] + [
    {"label": f"{c} (combined)", "value": c} for c in COMPOSITE_CONDITIONS
]
AGE_OPTIONS = [{"label": "All ages", "value": ALL}] + [{"label": age_label(a), "value": a} for a in AGE_GROUPS]
SEX_OPTIONS = [
    {"label": "All people", "value": "Persons"},
    {"label": "Female", "value": "Female"},
    {"label": "Male", "value": "Male"},
]
SOURCE_OPTIONS = [
    {"label": "Both tables", "value": BOTH},
    {"label": "Country table (S10)", "value": SOURCE_COUNTRY},
    {"label": "Language table (S13)", "value": SOURCE_LANGUAGE},
]
MIN_POP_OPTIONS = [
    {"label": "Any size", "value": 0},
    {"label": "1,000+ people", "value": 1000},
    {"label": "10,000+ people", "value": 10000},
]
SEX_SCOPE = {"Persons": "all people", "Female": "females", "Male": "males"}


def _dropdown(component_id: str, options, value, **kwargs) -> dcc.Dropdown:
    return dcc.Dropdown(id=component_id, options=options, value=value, clearable=False, **kwargs)


def _dynamic_card(prefix: str, foot: str | None = None) -> html.Section:
    return html.Section(
        [
            html.Div(
                [html.H2(id=f"{prefix}-title"), html.P(id=f"{prefix}-subtitle", className="card-subtitle")],
                className="card-head",
            ),
            ui.graph(figures.empty_figure("Loading…"), prefix),
            html.Div(id=f"{prefix}-table"),
            html.P(foot, className="card-foot") if foot else None,
        ],
        className="card",
    )


def layout(**_kwargs):
    return html.Div(
        [
            ui.page_header(
                "Explore the census data",
                "Choose a condition and a variable to compare. Prevalence is weighted: the people reporting the "
                "condition divided by the population of each group, so large groups count for more than small, "
                "volatile ones.",
                eyebrow="Historical analysis",
            ),
            html.Div(
                [
                    ui.field(
                        "Compare by",
                        _dropdown(
                            "explore-dimension",
                            [{"label": d.label, "value": d.key} for d in DIMENSIONS.values()],
                            "region",
                            searchable=False,
                        ),
                        html_for="explore-dimension",
                    ),
                    ui.field(
                        "Condition",
                        _dropdown("explore-condition", CONDITION_OPTIONS, "Diabetes"),
                        html_for="explore-condition",
                    ),
                    ui.field(
                        "Age group",
                        _dropdown("explore-age", AGE_OPTIONS, ALL, searchable=False),
                        html_for="explore-age",
                    ),
                    ui.field(
                        "Sex",
                        _dropdown("explore-sex", SEX_OPTIONS, "Persons", searchable=False),
                        html_for="explore-sex",
                    ),
                    ui.field(
                        "Census table",
                        _dropdown("explore-source", SOURCE_OPTIONS, BOTH, searchable=False),
                        html_for="explore-source",
                    ),
                    ui.field(
                        "Minimum group size",
                        _dropdown("explore-min-pop", MIN_POP_OPTIONS, 1000, searchable=False),
                        html_for="explore-min-pop",
                    ),
                ],
                className="filter-row",
                role="group",
                **{"aria-label": "Filters"},
            ),
            html.Div(id="explore-notes", className="method-notes"),
            dcc.Loading(
                html.Div(
                    [
                        _dynamic_card("explore-ranked"),
                        _dynamic_card(
                            "explore-profile",
                            "Each row is one of the groups above and each column one of the ten specific conditions, "
                            "with the same filters. Darker cells mean higher prevalence; blank cells were not published by AIHW.",
                        ),
                    ],
                    className="stack",
                ),
                color=BLUE,
                delay_show=300,
                overlay_style={"visibility": "visible", "opacity": 0.55},
            ),
        ]
    )


@callback(
    Output("explore-ranked-title", "children"),
    Output("explore-ranked-subtitle", "children"),
    Output("explore-ranked", "figure"),
    Output("explore-ranked", "style"),
    Output("explore-ranked-table", "children"),
    Output("explore-profile-title", "children"),
    Output("explore-profile-subtitle", "children"),
    Output("explore-profile", "figure"),
    Output("explore-profile", "style"),
    Output("explore-notes", "children"),
    Output("explore-age", "disabled"),
    Output("explore-sex", "disabled"),
    Output("explore-source", "disabled"),
    Input("explore-dimension", "value"),
    Input("explore-condition", "value"),
    Input("explore-age", "value"),
    Input("explore-sex", "value"),
    Input("explore-source", "value"),
    Input("explore-min-pop", "value"),
)
def update_explore(dimension_key, condition, age, sex, source, min_population):
    df = get_data().df
    dimension_key = dimension_key or "region"
    condition = condition or "Diabetes"
    age = age or ALL
    sex = sex or "Persons"
    min_population = int(min_population or 0)
    comparison = compare_groups(
        df, dimension_key, condition, age=age, sex=sex, source=source or BOTH, min_population=min_population
    )
    dimension = comparison.dimension
    table = comparison.table
    shown = table if dimension.order else table.head(TOP_N)

    scope_parts = []
    if dimension.key != "age":
        scope_parts.append("all ages" if age == ALL else f"aged {age_label(age)}")
    if dimension.key != "sex":
        scope_parts.append(SEX_SCOPE[sex])
    scope = ", ".join(scope_parts)

    if dimension.order or len(table) <= TOP_N:
        count_text = f"{len(table)} groups"
    else:
        count_text = f"Highest {TOP_N} of {len(table)} groups"
    ranked_title = f"{short_condition(condition)} by {dimension.label.lower()}"
    ranked_subtitle = " · ".join(part for part in [count_text, scope] if part)
    ranked_table = ui.table_view(
        ui.data_table(
            table,
            [
                ("group", dimension.label, None),
                ("prevalence", "Prevalence", ui.fmt_pct),
                ("cases", "People reporting", ui.fmt_int),
                ("population", "Population", ui.fmt_int),
                ("cells", "Census cells", ui.fmt_int),
            ],
            max_rows=200,
        ),
        summary=f"Show all {len(table)} groups as a table",
    )

    profile_groups = list(shown[dimension.column].head(PROFILE_ROWS))
    profile = condition_profile(comparison, profile_groups)
    display = {value: display_value(dimension, value) for value in profile_groups}
    profile_fig = figures.profile_heatmap(profile, dimension.column, profile_groups, display, SPECIFIC_CONDITIONS)
    profile_title = f"Condition profile by {dimension.label.lower()}"
    profile_subtitle = f"Prevalence (%) of the ten specific conditions for the first {len(profile_groups)} groups above"
    if scope:
        profile_subtitle += f" · {scope}"

    ranked_fig = figures.ranked_groups(comparison, TOP_N)

    notes = list(comparison.notes)
    if comparison.excluded_small:
        notes.append(
            f"{comparison.excluded_small} groups with fewer than {min_population:,} people are hidden because "
            "small groups give volatile rates."
        )
    notes_children = html.Ul([html.Li(note) for note in notes]) if notes else None

    return (
        ranked_title,
        ranked_subtitle,
        ranked_fig,
        ui.graph_style(ranked_fig),
        ranked_table,
        profile_title,
        profile_subtitle,
        profile_fig,
        ui.graph_style(profile_fig),
        notes_children,
        dimension.key == "age",
        dimension.key == "sex",
        dimension.source is not None,
    )
