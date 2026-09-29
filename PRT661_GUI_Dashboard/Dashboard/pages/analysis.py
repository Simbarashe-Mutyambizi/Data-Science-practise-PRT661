from __future__ import annotations

import re

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from lthc_dashboard import components as ui
from lthc_dashboard.config import short_condition
from lthc_dashboard.census_analysis import (
    AGE_ORDER,
    ALL_CONDITIONS,
    ANY_LTHC,
    PROF_ORDER,
    SPECIFIC_CONDITIONS,
    YEARS_ORDER,
    age_standardised,
    fmt_int,
    fmt_pct,
    load_data,
    prevalence,
    risk_table,
    split_views,
    standard_weights,
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
    REFERENCE,
    SEQUENTIAL,
    SEX_COLOURS,
    TEAL,
    TEAL_RAMP_2,
    TEMPLATE,
)

dash.register_page(__name__, path="/analysis", name="Analysis", title="Analysis · LTHC dashboard", order=1)

DF = load_data()
MIG, LANG = split_views(DF)
STD_W = standard_weights(MIG)

REGIONS = sorted(r for r in MIG["region"].unique() if r != "Not specified")
COUNTRIES = sorted(c for c in MIG.loc[MIG["population"] > 0, "country"].unique())
NON_MAPPABLE = MIG["country"].str.contains(r", nfd|, nec|At sea", regex=True)

GREY = "#8a96a8"
SEX_COLORS = {"Male": SEX_COLOURS["Male"], "Female": SEX_COLOURS["Female"]}
YEARS_COLORS = dict(zip(YEARS_ORDER, TEAL_RAMP_2, strict=True))
PROF_COLORS = {"Very well or well": TEAL_RAMP_2[1], "Not well or Not at all": TEAL_RAMP_2[0]}
PROF_LABELS = {"Not well or Not at all": "Not well or not at all"}
RISK_COLORS = {"Elevated": CORAL, "Typical": "#b9c3cf", "Lower": TEAL}
SEQ = SEQUENTIAL
DIVERGING = [[0, TEAL], [0.5, "#f3f5f8"], [1, CORAL]]
px.defaults.template = TEMPLATE

GRAPH_CFG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
LABELS = {
    "rate": "Prevalence (%)",
    "asr": "Age-standardised prevalence (%)",
    "crude_rate": "Crude prevalence (%)",
    "cases": "People with condition",
    "population": "Population",
    "age": "Age group",
    "sex": "Sex",
    "years": "Years in Australia",
    "proficiency": "Spoken English",
    "region": "Region of birth",
    "subregion": "Sub-region",
    "country": "Country of birth",
    "condition": "Condition",
    "language": "Language used at home",
    "rate_ratio": "Rate ratio",
    "baseline_rate": "Baseline (%)",
}


def empty_fig(msg: str = "No data for the current selection"):
    fig = go.Figure()
    fig.update_layout(
        template=TEMPLATE,
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[dict(text=msg, showarrow=False, font=dict(size=14, color=MUTED))],
    )
    return fig


def filt(df: pd.DataFrame, sex=None, ages=None, regions=None, condition=None) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if sex:
        m &= df["sex"] == sex
    if ages:
        m &= df["age"].isin(ages)
    if regions:
        m &= df["region"].isin(regions)
    if condition:
        m &= df["condition"].isin([condition] if isinstance(condition, str) else condition)
    return df[m]


def _hbar_axes(fig: go.Figure, title: str, x_max: float | None = None) -> None:
    fig.update_xaxes(showgrid=True, gridcolor=GRID, showline=False, title_text=title, ticksuffix="")
    if x_max:
        fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(showgrid=False, showline=True, linecolor=BASELINE, title_text=None, tickfont=dict(color=INK_2))


def card(title, caption, *children, controls=None, aside=None, cls=""):
    head = html.Div(
        [
            html.Div([html.H3(title), html.P(caption, className="caption", title=caption)], className="card-heading"),
            html.Div(aside, className="card-aside") if aside is not None else None,
        ],
        className="card-head",
    )
    body = [head]
    if controls is not None:
        body.append(html.Div(controls, className="card-controls"))
    body.append(html.Div(list(children), className="card-body"))
    return html.Div(body, className=f"card {cls}".strip())


MAP_CFG = {**GRAPH_CFG, "topojsonURL": "/assets/topojson/"}


def graph(id_, config=None):
    return dcc.Graph(
        id=id_,
        config=config or GRAPH_CFG,
        responsive=True,
        style={"height": "100%", "width": "100%"},
        className="graph-fill",
    )


def kpi(id_, label, cls="teal"):
    return ui.kpi(label, "–", "", accent=cls, id=id_)


MIN_POP_STEPS = [0, 500, 1000, 5000, 10000, 20000]


def min_pop_slider(id_, value=1000):
    marks = {i: (f"{v // 1000}k" if v >= 1000 else str(v)) for i, v in enumerate(MIN_POP_STEPS)}
    return html.Div(
        [
            html.Label("Minimum population", className="control-label"),
            html.Div(
                dcc.Slider(
                    id=id_, min=0, max=len(MIN_POP_STEPS) - 1, step=None, marks=marks, value=MIN_POP_STEPS.index(value)
                ),
                className="slider",
            ),
        ],
        className="slider-field",
    )


def _min_pop(value) -> int:
    if value is None:
        return 0
    if 0 <= value < len(MIN_POP_STEPS) and float(value).is_integer():
        return MIN_POP_STEPS[int(value)]
    return int(value)


filter_bar = html.Div(
    html.Div(
        [
            html.Div(
                [
                    html.Label("Health condition", className="title"),
                    dcc.Dropdown(id="f-condition", options=ALL_CONDITIONS, value=ANY_LTHC, clearable=False),
                ],
                className="filter",
            ),
            html.Div(
                [
                    html.Label("Sex", className="title"),
                    dcc.RadioItems(
                        id="f-sex",
                        options=["Persons", "Male", "Female"],
                        value="Persons",
                        inline=True,
                        className="segmented",
                    ),
                ],
                className="filter",
            ),
            html.Div(
                [
                    html.Label("Age group", className="title"),
                    dcc.Checklist(id="f-age", options=AGE_ORDER, value=AGE_ORDER, inline=True, className="segmented"),
                ],
                className="filter",
            ),
            html.Div(
                [
                    html.Label("Region of birth", className="title"),
                    dcc.Dropdown(id="f-region", options=REGIONS, value=[], multi=True, placeholder="All regions"),
                ],
                className="filter",
            ),
        ],
        className="filter-bar",
        id="filter-bar",
    ),
    className="toolbar",
)

kpis = ui.kpi_row(
    [
        kpi("k-pop", "Population in scope", "navy"),
        kpi("k-cases", "People with condition"),
        kpi("k-crude", "Crude prevalence"),
        kpi("k-asr", "Age-standardised prevalence", "amber"),
        kpi("k-region", "Highest-risk region", "coral"),
        kpi("k-top", "Most common condition"),
    ],
)
kpis.id = "kpi-row"

CELL_STYLE = {
    "fontFamily": FONT_FAMILY,
    "fontSize": 12.5,
    "padding": "6px 10px",
    "textAlign": "left",
    "border": "none",
    "borderBottom": "1px solid #e3e8ef",
    "color": INK,
}
HEADER_STYLE = {
    "fontWeight": 650,
    "fontSize": 12,
    "backgroundColor": "#f4f6fa",
    "color": NAVY,
    "borderBottom": "1px solid #e3e8ef",
    "whiteSpace": "normal",
}
FILTER_STYLE = {"backgroundColor": "#fbfcfd", "borderBottom": "1px solid #e3e8ef"}
FILTER_OPTIONS = {"case": "insensitive", "placeholder_text": "Filter…"}
TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell=CELL_STYLE,
    style_header=HEADER_STYLE,
    style_filter=FILTER_STYLE,
    style_as_list_view=True,
    filter_options=FILTER_OPTIONS,
)


def field(label, component, cls=""):
    return html.Div([html.Label(label, className="control-label"), component], className=cls)


def tab(label, value, grid, *children):
    return dcc.Tab(
        label=label,
        value=value,
        className="tab",
        selected_className="tab--selected",
        children=html.Div(list(children), className=f"tab-body {grid}"),
    )


tab_overview = tab(
    "Overview",
    "overview",
    "tab-overview",
    card(
        "Prevalence by condition",
        "Share of the selected population reporting each condition; the selected condition is in coral.",
        graph("g-conditions"),
    ),
    card(
        "Age and sex profile",
        "Age-specific prevalence of the selected condition for females and males.",
        graph("g-age-sex"),
    ),
    card(
        "Where the burden sits",
        "Region → sub-region → country. Box size = people with the condition; colour = crude prevalence. "
        "Click a box to drill down.",
        graph("g-treemap"),
    ),
)

tab_geo = tab(
    "Country of birth",
    "geo",
    "tab-geo",
    card(
        "Global view",
        "Age-standardised prevalence by country of birth (UK constituent countries combined).",
        graph("g-map", MAP_CFG),
        aside=min_pop_slider("s-geo-minpop", 1000),
        cls="geo-map",
    ),
    card(
        "Highest-prevalence countries",
        "Top 12 countries of birth by age-standardised prevalence; hover a bar for its region and counts.",
        graph("g-top-countries"),
        cls="geo-top",
    ),
    card(
        "Region comparison",
        "Age-standardised vs crude prevalence by region of birth; the gap is what age structure alone explains.",
        graph("g-regions"),
        cls="geo-regions",
    ),
)

tab_risk = tab(
    "Elevated risk",
    "risk",
    "tab-risk",
    card(
        "Largest excess prevalence",
        "The 15 elevated groups with the highest rate ratio: the group's prevalence ÷ the rate for all "
        "overseas-born people of the same age group.",
        graph("g-risk"),
    ),
    card(
        "Population groups with elevated risk",
        "Country × age × length-of-stay groups ranked by binomial z-score against all overseas-born people of the "
        "same age group. Elevated = z ≥ 3 and rate ratio ≥ 1.2; groups of 1,000 people or more.",
        html.Div(
            dash_table.DataTable(
                id="t-risk",
                page_size=10,
                sort_action="native",
                filter_action="native",
                **TABLE_STYLE,
                style_data_conditional=[{"if": {"column_id": "Rate ratio"}, "color": "#c2461f", "fontWeight": 700}],
            ),
            className="scroll",
        ),
        cls="table-card",
    ),
)

tab_mig = tab(
    "Years in Australia",
    "migration",
    "tab-migration",
    card(
        "Length of stay by age group",
        "Recent arrivals are younger and healthier (the healthy migrant effect), so compare within age groups.",
        graph("g-years-age"),
    ),
    card(
        "Length of stay by region",
        "Age-standardised prevalence by region of birth: light dot 0–10 years, dark dot more than 10 years.",
        graph("g-years-region"),
    ),
    card(
        "Which conditions increase with length of stay?",
        "Rate ratio: more than 10 years ÷ 0–10 years, within each age group. Coral: higher with longer stay; "
        "teal: lower.",
        graph("g-years-heat"),
    ),
)

tab_lang = tab(
    "Language & English",
    "language",
    "tab-language",
    card(
        "English proficiency by age group",
        "Prevalence among people who speak English very well or well, and not well or not at all.",
        graph("g-prof-age"),
    ),
    card(
        "Proficiency gap by condition",
        "Rate ratio: not well or not at all ÷ very well or well, within each age group. Coral: higher with "
        "limited English; teal: lower.",
        graph("g-prof-heat"),
    ),
    card(
        "Language used at home",
        "Age-standardised prevalence for the 20 largest language groups by spoken English; dot size shows the "
        "population.",
        graph("g-lang-dumbbell"),
        aside=min_pop_slider("s-lang-minpop", 1000),
    ),
)

tab_data = tab(
    "Data explorer",
    "data",
    "tab-data",
    card(
        "Underlying data",
        "Filtered by the controls above; use the column filters (for example >1000) and export to CSV. Aggregated "
        "census counts only: ABS perturbs small cells, so Male + Female may not equal Persons.",
        html.Div(
            dash_table.DataTable(
                id="t-data",
                page_size=14,
                sort_action="native",
                filter_action="native",
                export_format="csv",
                export_headers="display",
                style_table={"overflowX": "auto"},
                style_cell={**CELL_STYLE, "maxWidth": 260, "overflow": "hidden", "textOverflow": "ellipsis"},
                style_header=HEADER_STYLE,
                style_filter=FILTER_STYLE,
                style_as_list_view=True,
                filter_options=FILTER_OPTIONS,
            ),
            className="scroll",
        ),
        controls=[
            html.Label("Table", className="control-label"),
            dcc.RadioItems(
                id="d-view",
                value="migration",
                className="segmented",
                options=[
                    {"label": "Years in Australia (Table 10)", "value": "migration"},
                    {"label": "Language & English proficiency (Tables 13/14)", "value": "language"},
                ],
            ),
        ],
        cls="table-card",
    ),
)

layout = ui.screen(
    "analysis",
    "Census analysis",
    f"Overseas-born Australians by country of birth, years in Australia, language and English proficiency: "
    f"{len(COUNTRIES)} countries, {LANG['language'].nunique()} languages, age-standardised comparisons.",
    kpis,
    dcc.Tabs(
        id="tabs",
        value="overview",
        className="tabs-container",
        children=[tab_overview, tab_geo, tab_risk, tab_mig, tab_lang, tab_data],
    ),
    action=("Profile a population group", "/profile"),
    toolbar=filter_bar,
    class_name="analysis-screen analysis-app",
)


GLOBAL = [Input("f-condition", "value"), Input("f-sex", "value"), Input("f-age", "value"), Input("f-region", "value")]


@callback(
    [
        Output(f"{k}-{p}", "children")
        for k in ["k-pop", "k-cases", "k-crude", "k-asr", "k-region", "k-top"]
        for p in ["value", "sub"]
    ],
    GLOBAL,
)
def update_kpis(condition, sex, ages, regions):
    d = filt(MIG, sex, ages, regions, condition)
    if d["population"].sum() == 0:
        return ["–", ""] * 6
    tot = prevalence(d).iloc[0]
    asr = age_standardised(d.assign(_all=1), "_all", STD_W)["asr"].iloc[0]
    reg = age_standardised(d[d["region"] != "Not specified"], "region", STD_W)
    reg = reg[(reg["population"] >= 1000) & (reg["asr"] > 0)].sort_values("asr", ascending=False)
    conds = prevalence(filt(MIG, sex, ages, regions, SPECIFIC_CONDITIONS), "condition").sort_values(
        "rate", ascending=False
    )
    top_reg = reg.iloc[0] if len(reg) else None
    return [
        fmt_int(tot["population"]),
        "overseas-born, in selection",
        fmt_int(tot["cases"]),
        short_condition(condition),
        fmt_pct(tot["rate"]),
        "cases ÷ population",
        fmt_pct(asr),
        "adjusted to the overseas-born age mix",
        top_reg["region"] if top_reg is not None else "–",
        f"{fmt_pct(top_reg['asr'])} age-standardised" if top_reg is not None else "",
        conds.iloc[0]["condition"] if len(conds) else "–",
        f"{fmt_pct(conds.iloc[0]['rate'])} of the population" if len(conds) else "",
    ]


@callback(Output("g-conditions", "figure"), Output("g-age-sex", "figure"), Output("g-treemap", "figure"), GLOBAL)
def update_overview(condition, sex, ages, regions):
    conds = SPECIFIC_CONDITIONS + ([condition] if condition not in SPECIFIC_CONDITIONS else [])
    c = prevalence(filt(MIG, sex, ages, regions, conds), "condition")
    if c["population"].sum() == 0:
        return empty_fig(), empty_fig(), empty_fig()
    c = c.sort_values("rate")
    c["highlight"] = np.where(c["condition"] == condition, "Selected", "Other")
    f1 = px.bar(
        c,
        x="rate",
        y="condition",
        orientation="h",
        color="highlight",
        color_discrete_map={"Selected": CORAL, "Other": TEAL},
        text=c["rate"].map("{:.1f}%".format),
        hover_data={"cases": ":,", "population": ":,", "highlight": False, "rate": ":.2f"},
        labels=LABELS,
    )
    f1.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=INK_2, size=12))
    f1.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, margin=dict(r=40))
    _hbar_axes(f1, "Prevalence (%)", float(c["rate"].max()) * 1.14)

    a = prevalence(filt(MIG, None, ages, regions, condition).query("sex != 'Persons'"), ["age", "sex"])
    f2 = px.bar(
        a,
        x="age",
        y="rate",
        color="sex",
        barmode="group",
        color_discrete_map=SEX_COLORS,
        category_orders={"age": AGE_ORDER, "sex": ["Female", "Male"]},
        text=a["rate"].map("{:.1f}%".format),
        hover_data={"cases": ":,", "population": ":,", "rate": ":.2f"},
        labels=LABELS,
    )
    f2.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=INK_2, size=12))
    f2.update_layout(
        yaxis_title="Prevalence (%)",
        xaxis_title=None,
        legend_title_text="Sex",
        yaxis_range=[0, float(a["rate"].max()) * 1.15 if len(a) else 1],
    )

    t = prevalence(filt(MIG, sex, ages, regions, condition), ["region", "subregion", "country"])
    t = t[t["cases"] > 0]
    if t.empty:
        f3 = empty_fig()
    else:
        f3 = px.treemap(
            t,
            path=[px.Constant("All overseas-born"), "region", "subregion", "country"],
            values="cases",
            color="rate",
            color_continuous_scale=SEQ,
            hover_data={"population": ":,", "rate": ":.1f"},
            labels=LABELS,
        )
        f3.update_traces(
            root_color="#eef3f8",
            marker=dict(line=dict(color="#ffffff", width=1)),
            tiling=dict(pad=2),
            textfont=dict(family=FONT_FAMILY),
            hovertemplate="<b>%{label}</b><br>People with condition: %{value:,}"
            "<br>Prevalence: %{color:.1f}%<extra></extra>",
        )
        f3.update_layout(margin=dict(t=6, l=0, r=0, b=0), coloraxis_colorbar=dict(title="%", thickness=12))
    return f1, f2, f3


@callback(
    Output("g-map", "figure"),
    Output("g-top-countries", "figure"),
    Output("g-regions", "figure"),
    GLOBAL + [Input("s-geo-minpop", "value")],
)
def update_geo(condition, sex, ages, regions, min_pop):
    d = filt(MIG, sex, ages, regions, condition)
    if d["population"].sum() == 0:
        return empty_fig(), empty_fig(), empty_fig()

    m = age_standardised(d[~NON_MAPPABLE.loc[d.index]], "map_country", STD_W)
    min_pop = _min_pop(min_pop)
    m = m[m["population"] >= min_pop]
    f1 = px.choropleth(
        m,
        locations="map_country",
        locationmode="country names",
        color="asr",
        hover_name="map_country",
        color_continuous_scale=SEQ,
        projection="natural earth",
        hover_data={"map_country": False, "asr": ":.1f", "crude_rate": ":.1f", "cases": ":,", "population": ":,"},
        labels=LABELS,
    )
    f1.update_geos(
        showcountries=True,
        countrycolor="#ffffff",
        showcoastlines=False,
        showframe=False,
        landcolor="#e6ebf1",
        bgcolor="rgba(0,0,0,0)",
        lataxis_range=[-48, 76],
    )
    f1.update_traces(marker_line_color="#ffffff", marker_line_width=0.5)
    f1.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="Age-standardised %", side="top"),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.02,
            yanchor="top",
            len=0.45,
            thickness=10,
            outlinewidth=0,
        ),
    )

    c = age_standardised(d, ["country", "region"], STD_W)
    c = c[c["population"] >= min_pop].nlargest(12, "asr").sort_values("asr")
    f2 = go.Figure()
    f2.add_bar(
        x=c["asr"],
        y=c["country"],
        orientation="h",
        marker=dict(color=TEAL),
        text=[f"<b>{v:.1f}%</b>" for v in c["asr"]],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=INK, size=12),
        customdata=np.column_stack([c["crude_rate"], c["cases"], c["population"], c["region"]]) if len(c) else None,
        hovertemplate="<b>%{y}</b> · %{customdata[3]}<br>Age-standardised: %{x:.1f}%<br>Crude: %{customdata[0]:.1f}%"
        "<br>People with condition: %{customdata[1]:,.0f}<br>Population: %{customdata[2]:,.0f}<extra></extra>",
    )
    f2.update_layout(template=TEMPLATE, showlegend=False, margin=dict(r=48), bargap=0.3)
    _hbar_axes(f2, "Age-standardised prevalence (%)", float(c["asr"].max()) * 1.08 if len(c) else None)
    if len(c):
        top_tick = int(np.ceil(float(c["asr"].max()) / 10) * 10)
        f2.update_xaxes(tickvals=list(range(0, top_tick + 1, 10)), ticksuffix="%")
    f2.update_yaxes(dtick=1, tickfont=dict(size=12))

    r = age_standardised(d[d["region"] != "Not specified"], "region", STD_W).sort_values("asr")
    r_long = r.melt(
        id_vars=["region", "cases", "population"],
        value_vars=["asr", "crude_rate"],
        var_name="measure",
        value_name="value",
    )
    r_long["measure"] = r_long["measure"].map({"asr": "Age-standardised", "crude_rate": "Crude"})
    f3 = px.bar(
        r_long,
        x="value",
        y="region",
        color="measure",
        barmode="group",
        orientation="h",
        color_discrete_map={"Age-standardised": TEAL, "Crude": REFERENCE},
        category_orders={"region": r["region"].tolist()[::-1], "measure": ["Age-standardised", "Crude"]},
        hover_data={"cases": ":,", "population": ":,", "value": ":.2f"},
        labels={**LABELS, "value": "Prevalence (%)"},
    )
    f3.update_layout(legend_title_text="", bargap=0.25, bargroupgap=0.08)
    _hbar_axes(f3, "Prevalence (%)")
    f3.update_yaxes(dtick=1)
    return f1, f2, f3


def ratio_heatmap(df, split_col, numerator, denominator, ages):
    p = prevalence(df, ["condition", "age", split_col])
    w = p.pivot_table(index=["condition", "age"], columns=split_col, values="rate").reset_index()
    if numerator not in w or denominator not in w:
        return empty_fig()
    w["ratio"] = w[numerator] / w[denominator].replace(0, np.nan)
    mat = w.pivot(index="condition", columns="age", values="ratio")
    mat = mat.reindex(
        index=[c for c in ALL_CONDITIONS if c in mat.index],
        columns=[a for a in AGE_ORDER if a in (ages or AGE_ORDER) and a in mat.columns],
    )
    if mat.empty:
        return empty_fig()
    mat = mat.astype(float)
    mat.index = [short_condition(c) for c in mat.index]
    logm = np.log2(mat).replace([np.inf, -np.inf], np.nan)
    finite = logm.values[np.isfinite(logm.values)]
    top = min(max(np.abs(finite).max() if finite.size else 1.0, 0.5), 3)
    fig = px.imshow(
        logm,
        text_auto=False,
        aspect="auto",
        color_continuous_scale=DIVERGING,
        zmin=-top,
        zmax=top,
        labels=dict(x="Age group", y="", color="log₂ ratio"),
    )
    fig.update_traces(
        text=mat.map(lambda v: f"{v:.2f}×" if pd.notna(v) else "").values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        xgap=3,
        ygap=3,
        customdata=mat.values,
        hovertemplate="%{y}<br>Age %{x}<br>Rate ratio: %{customdata:.2f}<extra></extra>",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(side="top", showline=False, title=None, tickangle=0, tickfont=dict(color=INK_2)),
        yaxis=dict(showgrid=False, dtick=1, tickfont=dict(color=INK_2)),
        margin=dict(t=8, r=4),
    )
    return fig


def years_dumbbell(r: pd.DataFrame) -> go.Figure:
    wide = r.pivot(index="region", columns="years", values="asr").dropna(how="all")
    order_col = YEARS_ORDER[-1] if YEARS_ORDER[-1] in wide else wide.columns[-1]
    wide = wide.sort_values(order_col)
    fig = go.Figure()
    for region, row in wide.iterrows():
        values = [row.get(y) for y in YEARS_ORDER if pd.notna(row.get(y))]
        if len(values) == 2:
            fig.add_scatter(
                x=values,
                y=[region, region],
                mode="lines",
                line=dict(color=BASELINE, width=3),
                hoverinfo="skip",
                showlegend=False,
            )
    for years in YEARS_ORDER:
        if years not in wide:
            continue
        sub = r[r["years"] == years].set_index("region").reindex(wide.index)
        fig.add_scatter(
            x=sub["asr"],
            y=wide.index,
            mode="markers",
            name=years,
            marker=dict(size=13, color=YEARS_COLORS[years], line=dict(color="#ffffff", width=2)),
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate="<b>%{y}</b><br>" + years + ": %{x:.1f}% (age-standardised)"
            "<br>People with condition: %{customdata[0]:,.0f}<br>Population: %{customdata[1]:,.0f}<extra></extra>",
        )
    fig.update_layout(template=TEMPLATE, legend_title_text="Years in Australia")
    _hbar_axes(fig, "Age-standardised prevalence (%)")
    fig.update_yaxes(showline=False)
    return fig


@callback(Output("g-years-age", "figure"), Output("g-years-region", "figure"), Output("g-years-heat", "figure"), GLOBAL)
def update_migration(condition, sex, ages, regions):
    d = filt(MIG, sex, ages, regions, condition)
    if d["population"].sum() == 0:
        return empty_fig(), empty_fig(), empty_fig()

    a = prevalence(d, ["age", "years"])
    f1 = px.bar(
        a,
        x="age",
        y="rate",
        color="years",
        barmode="group",
        color_discrete_map=YEARS_COLORS,
        category_orders={"age": AGE_ORDER, "years": YEARS_ORDER},
        text=a["rate"].map("{:.1f}%".format),
        hover_data={"cases": ":,", "population": ":,", "rate": ":.2f"},
        labels=LABELS,
    )
    f1.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=INK_2, size=12))
    f1.update_layout(
        yaxis_title="Prevalence (%)",
        xaxis_title=None,
        legend_title_text="Years in Australia",
        yaxis_range=[0, float(a["rate"].max()) * 1.15 if len(a) else 1],
    )

    r = age_standardised(d[d["region"] != "Not specified"], ["region", "years"], STD_W)
    f2 = years_dumbbell(r) if not r.empty else empty_fig()

    f3 = ratio_heatmap(filt(MIG, sex, ages, regions), "years", "More than 10 years", "0–10 years", ages)
    return f1, f2, f3


@callback(
    Output("g-prof-age", "figure"),
    Output("g-prof-heat", "figure"),
    Output("g-lang-dumbbell", "figure"),
    GLOBAL + [Input("s-lang-minpop", "value")],
)
def update_language(condition, sex, ages, regions, min_pop):
    d = filt(LANG, sex, ages, regions, condition)
    if d["population"].sum() == 0:
        return empty_fig(), empty_fig(), empty_fig()

    a = prevalence(d, ["age", "proficiency"])
    f1 = px.bar(
        a,
        x="age",
        y="rate",
        color="proficiency",
        barmode="group",
        color_discrete_map=PROF_COLORS,
        category_orders={"age": AGE_ORDER, "proficiency": PROF_ORDER},
        text=a["rate"].map("{:.1f}%".format),
        hover_data={"cases": ":,", "population": ":,", "rate": ":.2f"},
        labels=LABELS,
    )
    f1.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=INK_2, size=12))
    f1.update_layout(
        yaxis_title="Prevalence (%)",
        xaxis_title=None,
        legend_title_text="",
        yaxis_range=[0, float(a["rate"].max()) * 1.15 if len(a) else 1],
    )

    f1.for_each_trace(lambda t: t.update(name=PROF_LABELS.get(t.name, t.name)))
    f2 = ratio_heatmap(
        filt(LANG, sex, ages, regions), "proficiency", "Not well or Not at all", "Very well or well", ages
    )

    langs = age_standardised(d, ["language", "proficiency"], STD_W)
    langs = langs[langs["population"] >= _min_pop(min_pop)]
    size = langs.groupby("language")["population"].sum().nlargest(20)
    langs = langs[langs["language"].isin(size.index)]
    if langs.empty:
        f3 = empty_fig("No language groups meet the minimum population")
    else:
        order = (langs.groupby("language")["asr"].max().sort_values()).index.tolist()
        f3 = px.line(
            langs.sort_values(["language", "asr"]),
            x="asr",
            y="language",
            line_group="language",
            color_discrete_sequence=[BASELINE],
        )
        f3.update_traces(hoverinfo="skip", hovertemplate=None, line_width=3, showlegend=False)
        pts = px.scatter(
            langs,
            x="asr",
            y="language",
            color="proficiency",
            color_discrete_map=PROF_COLORS,
            size="population",
            size_max=18,
            category_orders={"proficiency": PROF_ORDER},
            hover_data={"crude_rate": ":.1f", "cases": ":,", "population": ":,", "asr": ":.2f"},
            labels=LABELS,
        )
        pts.update_traces(marker=dict(line=dict(color="#ffffff", width=1.5), opacity=1, sizemin=6))
        for tr in pts.data:
            f3.add_trace(tr)
        f3.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=order),
            legend_title_text="Spoken English",
        )
        _hbar_axes(f3, "Age-standardised prevalence (%)")
        f3.update_yaxes(showline=False)
        f3.for_each_trace(lambda t: t.update(name=PROF_LABELS.get(t.name, t.name)))
    return f1, f2, f3


def elevated_groups(condition, sex, ages, regions) -> pd.DataFrame:
    d = filt(MIG, sex, ages, None, condition)
    if d.empty:
        return pd.DataFrame()
    r = risk_table(d, ["country", "region", "age", "years"], ["age"], min_population=1000)
    if regions:
        r = r[r["region"].isin(regions)]
    return r[r["risk_level"] == "Elevated"]


YEARS_SHORT = {"0–10 years": "0–10 yrs", "More than 10 years": "10+ yrs"}


COUNTRY_SHORT = {"Korea, Republic of (South)": "South Korea"}


def short_country(name: str) -> str:
    if name in COUNTRY_SHORT:
        return COUNTRY_SHORT[name]
    return re.sub(r"\s*\(.*?\)", "", name).split(",")[0].strip() or name


@callback(Output("g-risk", "figure"), GLOBAL)
def update_risk_chart(condition, sex, ages, regions):
    r = elevated_groups(condition, sex, ages, regions)
    if r.empty:
        return empty_fig("No elevated groups for the current selection")
    top = r.nlargest(15, "rate_ratio").sort_values("rate_ratio")
    labels = [
        f"{short_country(country)} · {age} · {YEARS_SHORT.get(years, years)}"
        for country, age, years in zip(top["country"], top["age"], top["years"], strict=True)
    ]
    fig = go.Figure()
    fig.add_bar(
        x=top["rate_ratio"],
        y=labels,
        orientation="h",
        marker=dict(color=CORAL),
        text=[f"<b>{v:.2f}×</b>" for v in top["rate_ratio"]],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=INK, size=12),
        customdata=np.column_stack(
            [top["region"], top["rate"], top["baseline_rate"], top["population"], top["z_score"], top["country"]]
        ),
        hovertemplate="<b>%{customdata[5]}</b><br>%{y}<br>Prevalence %{customdata[1]:.1f}% vs "
        "%{customdata[2]:.1f}% for the same age group<br>Population %{customdata[3]:,.0f} · z = "
        "%{customdata[4]:.1f}<extra></extra>",
    )
    fig.add_vline(x=1, line=dict(color=NAVY, width=1.5, dash="dot"))
    fig.update_layout(template=TEMPLATE, showlegend=False, margin=dict(r=52), bargap=0.3)
    _hbar_axes(fig, "Rate ratio (1× = rate for the same age group)")
    fig.update_xaxes(range=[0, float(top["rate_ratio"].max()) * 1.1], ticksuffix="×")
    fig.update_yaxes(dtick=1, tickfont=dict(size=12))
    return fig


@callback(Output("t-risk", "data"), Output("t-risk", "columns"), GLOBAL)
def update_risk_table(condition, sex, ages, regions):
    r = elevated_groups(condition, sex, ages, regions)
    if r.empty:
        return [], []
    r = r.head(100)
    out = pd.DataFrame(
        {
            "Country of birth": r["country"],
            "Region": r["region"],
            "Age group": r["age"],
            "Years in Australia": r["years"],
            "Population": r["population"],
            "Cases": r["cases"],
            "Prevalence (%)": r["rate"].round(1),
            "Baseline (%)": r["baseline_rate"].round(1),
            "Rate ratio": r["rate_ratio"].round(2),
            "z-score": r["z_score"].round(1),
        }
    )
    cols = [
        {"name": c, "id": c, "type": "numeric", "format": {"specifier": ","}}
        if c in ("Population", "Cases")
        else {"name": c, "id": c, "type": "numeric"}
        if out[c].dtype.kind in "fi"
        else {"name": c, "id": c}
        for c in out.columns
    ]
    return out.to_dict("records"), cols


@callback(Output("t-data", "data"), Output("t-data", "columns"), GLOBAL + [Input("d-view", "value")])
def update_table(condition, sex, ages, regions, view):
    src = MIG if view == "migration" else LANG
    d = filt(src, sex, ages, regions, condition)
    d = d[d["population"] > 0]
    keep = ["country", "region", "subregion", "age", "sex", "condition", "cases", "population", "rate"]
    keep.insert(3, "years" if view == "migration" else "proficiency")
    if view == "language":
        keep.insert(3, "language")
    d = d[keep].assign(rate=lambda x: x["rate"].round(2)).sort_values(["country", "age"])
    number_formats = {"cases": {"specifier": ","}, "population": {"specifier": ","}, "rate": {"specifier": ".2f"}}
    cols = [
        {"name": LABELS.get(c, c), "id": c, "type": "numeric", "format": number_formats[c]}
        if c in number_formats
        else {"name": LABELS.get(c, c), "id": c}
        for c in keep
    ]
    return d.to_dict("records"), cols
