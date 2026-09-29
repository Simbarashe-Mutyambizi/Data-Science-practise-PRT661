from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

from lthc_dashboard import components as ui
from lthc_dashboard.census_analysis import (
    AGE_ORDER,
    ANY,
    ANY_LTHC,
    estimate_profile,
    fmt_int,
    fmt_pct,
    load_data,
    split_views,
)
from lthc_dashboard.theme import (
    BASELINE,
    CORAL,
    FONT_FAMILY,
    GRID,
    INK,
    INK_2,
    MUTED,
    NAVY,
    TEAL,
    TEMPLATE,
)

dash.register_page(__name__, path="/profile", name="Group profile", title="Group profile · LTHC dashboard", order=3)

MIG, LANG = split_views(load_data())
REGIONS = sorted(r for r in MIG["region"].unique() if r != "Not specified")
NOT_SPECIFIED = {"label": "Not specified", "value": ANY}

LEVEL_COLORS = {
    "Higher than average": CORAL,
    "About average": "#b9c3cf",
    "Lower than average": TEAL,
    "Not reported": "#dde3ea",
}
LEVEL_CLASS = {
    "Higher than average": "Elevated",
    "About average": "Typical",
    "Lower than average": "Lower",
    "Not reported": "Insufficient",
}
LEVEL_ICON = {"Higher than average": "▲", "About average": "●", "Lower than average": "▼", "Not reported": "–"}
GRAPH_CFG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtons": [["toImage"]],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={
        "fontFamily": FONT_FAMILY,
        "fontSize": 12.5,
        "padding": "6px 8px",
        "textAlign": "left",
        "border": "none",
        "borderBottom": "1px solid #e3e8ef",
        "color": INK,
    },
    style_header={
        "fontWeight": 650,
        "fontSize": 12.5,
        "backgroundColor": "#f4f6fa",
        "color": NAVY,
        "borderBottom": "1px solid #e3e8ef",
        "whiteSpace": "normal",
    },
    style_as_list_view=True,
)


def opts(values):
    return [NOT_SPECIFIED] + [{"label": v, "value": v} for v in values]


def field(label, component, hint=None):
    return ui.field(label, component, hint)


def step(num, title, *fields):
    return ui.step(num, title, *fields)


one_in = ui.one_in


def describe(age, sex, region, country, years, language, prof):
    who = {"Persons": "People", "Male": "Men", "Female": "Women"}[sex]
    age_txt = "aged 65 or over" if age == "65 and over" else f"aged {age}"
    place = f"born in {country}" if country != ANY else f"born in {region}" if region != ANY else "born overseas"
    bits = [f"{who} {age_txt}, {place}"]
    if years != ANY:
        bits.append("living in Australia for " + ("up to 10 years" if years == "0–10 years" else "more than 10 years"))
    if language != ANY:
        bits.append(f"speaking {language} at home")
    if prof != ANY:
        bits.append(
            "who speak English " + ("very well or well" if prof.startswith("Very") else "not well or not at all")
        )
    return ", ".join(bits)


def mini_bars(estimate, average, scale):

    def bar(label, value, colour):
        width = 0 if pd.isna(value) or not scale else min(100, value / scale * 100)
        return html.Div(
            [
                html.Span(label, className="mb-label"),
                html.Div(
                    html.Div(className="mb-fill", style={"width": f"{width:.1f}%", "background": colour}),
                    className="mb-track",
                ),
                html.Span(fmt_pct(value), className="mb-value"),
            ],
            className="mb-row",
        )

    return html.Div([bar("This group", estimate[0], estimate[1]), bar("Average", average, NAVY)], className="mini-bars")


def condition_card(row, rank, scale=None):
    cls = LEVEL_CLASS[row["level"]]
    rel = row["relative"]
    rel_txt = (
        f"{rel:.1f}× the average"
        if pd.notna(rel) and rel >= 1
        else f"{(1 - rel) * 100:.0f}% below average"
        if pd.notna(rel)
        else ""
    )
    return html.Div(
        [
            html.Div(f"#{rank}", className="rank"),
            html.Div(row["condition"], className="label"),
            html.Div(fmt_pct(row["likelihood"]), className="value"),
            html.Div(one_in(row["likelihood"]).capitalize(), className="sub"),
            html.Div(
                [
                    html.Span(
                        [html.Span(LEVEL_ICON[row["level"]], className="badge-icon"), row["level"]],
                        className=f"badge {cls}",
                    ),
                    html.Span(rel_txt, className="rel"),
                ],
                className="badge-row",
            ),
            mini_bars((row["likelihood"], LEVEL_COLORS[row["level"]]), row["average"], scale),
        ],
        className=f"kpi cond-card {cls}",
    )


def likelihood_figure(spec):
    plot = spec.sort_values("likelihood", ascending=False)
    fig = go.Figure()
    for level, colour in LEVEL_COLORS.items():
        sub = plot[plot["level"] == level]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["likelihood"],
            y=sub["condition"],
            orientation="h",
            name=level,
            marker=dict(color=colour),
            customdata=np.column_stack([sub["average"], sub["relative"]]),
            hovertemplate="<b>%{y}</b><br>Estimated: %{x:.1f}%<br>Average: %{customdata[0]:.1f}%"
            "<br>%{customdata[1]:.2f}× the average<extra></extra>",
        )
    fig.add_scatter(
        x=plot["average"],
        y=plot["condition"],
        mode="markers",
        name="Average (same age and sex)",
        marker=dict(symbol="diamond", size=11, color=NAVY, line=dict(color="#ffffff", width=1.5)),
        hovertemplate="<b>%{y}</b><br>Average: %{x:.1f}%<extra></extra>",
    )
    fig.add_annotation(
        x=1,
        xref="paper",
        xanchor="left",
        xshift=10,
        y=1,
        yref="paper",
        yanchor="bottom",
        text="Estimate",
        showarrow=False,
        font=dict(color=MUTED, size=11.5),
    )
    for condition, value in zip(plot["condition"], plot["likelihood"], strict=True):
        fig.add_annotation(
            x=1,
            xref="paper",
            xanchor="left",
            xshift=10,
            y=condition,
            yref="y",
            text=fmt_pct(value),
            showarrow=False,
            font=dict(color=INK_2, size=12.5),
        )
    x_max = float(np.nanmax(np.concatenate([plot["likelihood"].to_numpy(), plot["average"].to_numpy()]))) * 1.08
    fig.update_layout(
        template=TEMPLATE,
        barmode="overlay",
        bargap=0.3,
        margin=dict(r=70, t=26),
        showlegend=False,
        xaxis=dict(
            title_text="Estimated prevalence (%)",
            ticksuffix="%",
            range=[0, x_max],
            showgrid=True,
            gridcolor=GRID,
            showline=False,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=plot["condition"].tolist()[::-1],
            showgrid=False,
            showline=True,
            linecolor=BASELINE,
            tickfont=dict(color=INK_2, size=12.5),
        ),
    )
    return fig


def relative_figure(spec):
    rel = spec[spec["relative"].notna() & (spec["relative"] > 0)].sort_values("relative")
    if rel.empty:
        return None
    rel = rel.assign(log_rel=np.log2(rel["relative"]))
    lim = max(1.0, float(np.abs(rel["log_rel"]).max()) + 0.6)
    fig = go.Figure()
    fig.add_bar(
        x=rel["log_rel"],
        y=rel["condition"],
        orientation="h",
        marker=dict(color=[LEVEL_COLORS[level] for level in rel["level"]]),
        text=rel["relative"].map("{:.2f}×".format),
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=INK_2, size=12),
        customdata=np.column_stack([rel["relative"], rel["likelihood"], rel["average"]]),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:.2f}× the average<br>Estimated: %{customdata[1]:.1f}%"
        "<br>Average: %{customdata[2]:.1f}%<extra></extra>",
        showlegend=False,
    )
    fig.update_layout(
        template=TEMPLATE,
        bargap=0.3,
        margin=dict(l=8, r=24),
        xaxis=dict(
            title_text="Times the average (log scale)",
            range=[-lim, lim],
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["¼×", "½×", "1× (average)", "2×", "4×"],
            showgrid=True,
            gridcolor=GRID,
            zeroline=True,
            zerolinecolor=NAVY,
            zerolinewidth=2,
            showline=False,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=rel["condition"].tolist(),
            showgrid=False,
            tickfont=dict(color=INK_2, size=12.5),
        ),
    )
    return fig


def card(title, caption, *children, cls=""):
    head = html.Div(
        html.Div([html.H3(title), html.P(caption, className="caption", title=caption)], className="card-heading"),
        className="card-head",
    )
    return html.Div([head, *children], className=f"card {cls}".strip())


def fill_graph(figure):
    return dcc.Graph(
        figure=figure,
        config=GRAPH_CFG,
        responsive=True,
        style={"height": "100%", "width": "100%"},
        className="graph-fill",
    )


form = html.Div(
    [
        html.H2("Describe the population group"),
        html.P("Only age group and sex are required. Leave anything else as ‘Not specified’.", className="caption"),
        step(
            1,
            "Age and sex",
            field("Age group *", dcc.RadioItems(id="p-age", options=AGE_ORDER, value=None, className="segmented")),
            field(
                "Sex *",
                dcc.RadioItems(
                    id="p-sex",
                    value=None,
                    className="segmented",
                    options=[
                        {"label": "Female", "value": "Female"},
                        {"label": "Male", "value": "Male"},
                        {"label": "All people", "value": "Persons"},
                    ],
                ),
            ),
        ),
        step(
            2,
            "Place of birth",
            field("Region of birth", dcc.Dropdown(id="p-region", options=opts(REGIONS), value=ANY, clearable=False)),
            field(
                "Country of birth",
                dcc.Dropdown(id="p-country", options=[NOT_SPECIFIED], value=ANY, clearable=False),
                hint="Choosing a region shortens this list.",
            ),
            field(
                "Years living in Australia",
                dcc.RadioItems(
                    id="p-years",
                    value=ANY,
                    className="segmented",
                    options=[
                        {"label": "Up to 10 years", "value": "0–10 years"},
                        {"label": "More than 10 years", "value": "More than 10 years"},
                        NOT_SPECIFIED,
                    ],
                ),
            ),
        ),
        step(
            3,
            "Language",
            field(
                "Language used at home",
                dcc.Dropdown(id="p-language", options=[NOT_SPECIFIED], value=ANY, clearable=False),
                hint="Languages recorded for the chosen country or region.",
            ),
            field(
                "Spoken English",
                dcc.RadioItems(
                    id="p-prof",
                    value=ANY,
                    className="segmented",
                    options=[
                        {"label": "Very well or well", "value": "Very well or well"},
                        {"label": "Not well or not at all", "value": "Not well or Not at all"},
                        NOT_SPECIFIED,
                    ],
                ),
            ),
        ),
        html.Div(id="profile-form-error", className="form-error"),
        html.Button("Show group profile", id="profile-submit", n_clicks=0, className="btn-primary"),
    ],
    className="card form-card profile-form",
)

placeholder = html.Div(
    [
        ui.disclaimer(),
        html.Div(
            [
                html.Div(html.Img(src="/assets/logo.svg", alt=""), className="placeholder-icon"),
                html.H3("The group's profile will appear here"),
                html.P(
                    "Choose an age group and sex, add any other characteristics of the group, then select "
                    "“Show group profile”."
                ),
            ],
            className="card placeholder",
        ),
    ],
    className="results-grid placeholder-grid",
)

layout = ui.screen(
    "profile",
    "Condition profile for a population group",
    "Census-based estimate for every condition, compared with all overseas-born people of the same age and sex; "
    "small groups borrow strength from their wider region.",
    html.Div(
        [
            form,
            dcc.Loading(
                html.Div(placeholder, id="profile-results", className="results"),
                type="circle",
                color=TEAL,
                parent_className="results-loading",
            ),
        ],
        className="form-layout",
    ),
    action=("Estimate with the model", "/predict"),
    class_name="analysis-app profile-app",
)


@callback(
    Output("p-country", "options"),
    Output("p-country", "value"),
    Input("p-region", "value"),
    State("p-country", "value"),
)
def update_country_options(region, country):
    src = MIG if region == ANY else MIG[MIG["region"] == region]
    countries = sorted(src.loc[src["population"] > 0, "country"].unique())
    return opts(countries), country if country in countries else ANY


@callback(
    Output("p-language", "options"),
    Output("p-language", "value"),
    Input("p-region", "value"),
    Input("p-country", "value"),
    State("p-language", "value"),
)
def update_language_options(region, country, language):
    src = LANG
    if country != ANY:
        src = src[src["country"] == country]
    elif region != ANY:
        src = src[src["region"] == region]
    langs = sorted(src.loc[src["population"] > 0, "language"].unique())
    return opts(langs), language if language in langs else ANY


@callback(
    Output("profile-results", "children"),
    Output("profile-form-error", "children"),
    Input("profile-submit", "n_clicks"),
    State("p-region", "value"),
    State("p-country", "value"),
    State("p-language", "value"),
    State("p-age", "value"),
    State("p-sex", "value"),
    State("p-years", "value"),
    State("p-prof", "value"),
    prevent_initial_call=True,
)
def show_results(_, region, country, language, age, sex, years, prof):
    missing = [name for name, v in [("age group", age), ("sex", sex)] if not v]
    if missing:
        return placeholder, f"Please choose {' and '.join(missing)}."

    res, info = estimate_profile(
        MIG, LANG, age=age, sex=sex, region=region, country=country, years=years, language=language, proficiency=prof
    )
    any_row = res[res["condition"] == ANY_LTHC].iloc[0]
    spec = res[res["condition"] != ANY_LTHC].sort_values("likelihood", ascending=False)

    conf_cls = {"High": "Lower", "Medium": "Typical", "Low": "Elevated"}[info["confidence"]]
    group_n = info.get("language_population") if (language != ANY or prof != ANY) else info["population"]
    headline = html.Div(
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Population group", className="eyebrow-label"),
                        html.Div(describe(age, sex, region, country, years, language, prof), className="profile-text"),
                        html.Div(
                            [
                                html.Span(f"{info['confidence']} confidence", className=f"badge {conf_cls}"),
                                html.Span(
                                    f"based on a census group of {fmt_int(group_n or 0)} people", className="rel"
                                ),
                            ]
                            + [html.Div(n, className="warn") for n in info["notes"]],
                            className="badge-row",
                        ),
                    ],
                    className="headline-left",
                ),
                html.Div(
                    [
                        html.Div("Report at least one long-term condition", className="eyebrow-label"),
                        html.Div(fmt_pct(any_row["likelihood"], 0), className="big"),
                        html.Div(
                            f"{one_in(any_row['likelihood']).capitalize()}; average for the same age and sex "
                            f"{fmt_pct(any_row['average'], 0)}",
                            className="sub",
                        ),
                    ],
                    className="headline-right",
                ),
            ],
            className="headline-grid",
        ),
        className="card headline-card profile-headline",
    )

    levels_present = [level for level in LEVEL_COLORS if level in set(spec["level"])]
    rel_fig = relative_figure(spec)
    likelihood_card = card(
        "Estimated prevalence of each condition",
        "Bars: estimated share of the group reporting each condition. Diamonds: average for all overseas-born "
        "people of the same age group and sex.",
        ui.legend(
            [(level, LEVEL_COLORS[level], "square") for level in levels_present]
            + [("Average, same age and sex", NAVY, "diamond")]
        ),
        html.Div(fill_graph(likelihood_figure(spec)), className="card-body"),
    )
    relative_card = card(
        "Compared with the average",
        "How many times more (right) or less (left) common each condition is than among overseas-born people of "
        "the same age and sex.",
        ui.legend([(level, LEVEL_COLORS[level], "square") for level in levels_present]),
        html.Div(
            fill_graph(rel_fig) if rel_fig is not None else html.P("Not enough data to compare.", className="caption"),
            className="card-body",
        ),
    )

    tbl = res.sort_values("likelihood", ascending=False)
    rows = pd.DataFrame(
        {
            "Condition": tbl["condition"],
            "Estimate (%)": tbl["likelihood"].round(1),
            "Average (%)": tbl["average"].round(1),
            "× average": tbl["relative"].round(2),
            "Compared with average": tbl["level"],
            "Roughly": tbl["likelihood"].map(one_in),
            "Language effect": tbl["language_effect"].round(2),
            "Data source": np.where(tbl["own_data"], "This group", "Wider region"),
        }
    )
    cols = [
        {"name": c, "id": c, "type": "numeric"} if rows[c].dtype.kind in "fi" else {"name": c, "id": c}
        for c in rows.columns
    ]
    table = card(
        "Full results",
        "Sort by any column or export a CSV. ‘Data source’: this exact group, or the wider region where the census "
        "did not publish the group.",
        html.Div(
            dash_table.DataTable(
                data=rows.to_dict("records"),
                columns=cols,
                sort_action="native",
                page_size=12,
                export_format="csv",
                export_headers="display",
                export_columns="all",
                hidden_columns=["Compared with average", "Roughly", "Language effect", "Data source"],
                **TABLE_STYLE,
                style_cell_conditional=[
                    {"if": {"column_id": "Condition"}, "whiteSpace": "normal", "minWidth": 120, "maxWidth": 160}
                ],
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": '{Compared with average} = "Higher than average"',
                            "column_id": "Compared with average",
                        },
                        "color": "#c2461f",
                        "fontWeight": 700,
                    },
                    {
                        "if": {
                            "filter_query": '{Compared with average} = "Lower than average"',
                            "column_id": "Compared with average",
                        },
                        "color": "#0b7a70",
                        "fontWeight": 700,
                    },
                ],
            ),
            className="card-body scroll",
        ),
        html.P(
            "How it works: the census rate for the group's birthplace, years in Australia, age and sex, with small "
            "groups blended towards their sub-region and region (empirical-Bayes smoothing), then adjusted for the "
            "chosen language or English level.",
            className="card-foot",
        ),
        cls="table-card",
    )

    leaders = spec.head(3)
    scale = float(np.nanmax(leaders[["likelihood", "average"]].to_numpy(dtype=float))) * 1.1 if len(leaders) else None
    top = [condition_card(r, i + 1, scale) for i, (_, r) in enumerate(leaders.iterrows())]
    results = html.Div(
        [
            ui.disclaimer(),
            html.Div([headline, *top], className="headline-row profile"),
            html.Div([likelihood_card, relative_card, table], className="charts-row"),
        ],
        className="results-grid",
    )
    return results, ""
