"""Plotly figure builders for every chart in the dashboard.

Conventions: thin bars with rounded tips, hairline grid, values in tooltips,
direct labels only where they carry the story, one hue per series in a fixed
order, sequential blue for magnitude.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .analysis import Comparison
from .config import (
    AGE,
    AGE_GROUPS,
    COMMON_CONDITIONS,
    LTHC,
    REGION,
    SEX,
    YEARS,
    YEARS_OPTIONS,
    age_label,
    sex_label,
    short_condition,
)
from .theme import (
    AGE_COLOURS,
    BASELINE,
    BLUE,
    DE_EMPHASIS,
    GRID,
    INK,
    INK_2,
    MUTED,
    SEQUENTIAL_SCALE,
    SEX_COLOURS,
    SURFACE,
    TEMPLATE,
    TRAIN_TEST_COLOURS,
    YEARS_COLOURS,
)

LABEL_FONT = dict(color=INK_2, size=12)
GAP_LINE = dict(color=SURFACE, width=1.5)  # surface-coloured gap between touching bars


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _figure(height: int, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, height=height, **layout)
    return fig


def _pct_axis(title: str = "Prevalence (%)", maximum: float | None = None, **extra) -> dict:
    axis = dict(title=dict(text=title), ticksuffix="%", rangemode="tozero", **extra)
    if maximum is not None and maximum > 0:
        axis["range"] = [0, maximum]
    return axis


def _headroom(values, factor: float = 1.16) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (max(values) if values else 1.0) * factor


def _horizontal_layout(fig: go.Figure, x_max: float, title: str = "Prevalence (%)") -> None:
    fig.update_layout(
        xaxis=dict(
            title=dict(text=title),
            ticksuffix="%",
            range=[0, x_max],
            showgrid=True,
            gridcolor=GRID,
            showline=False,
        ),
        yaxis=dict(showgrid=False, showline=True, linecolor=BASELINE, autorange="reversed"),
        bargap=0.32,
    )


def empty_figure(message: str, height: int = 260) -> go.Figure:
    fig = _figure(height)
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(color=MUTED, size=13)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def age_by_condition(table: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> go.Figure:
    labels = [wrap_label(short_condition(c), 12) for c in conditions]
    fig = _figure(370)
    for age in AGE_GROUPS:
        sub = table[table[AGE] == age].set_index(LTHC).reindex(conditions)
        is_oldest = age == AGE_GROUPS[-1]
        fig.add_bar(
            name=age_label(age),
            x=labels,
            y=sub["prevalence"],
            marker=dict(color=AGE_COLOURS[age], line=GAP_LINE),
            text=[f"{v:.1f}%" for v in sub["prevalence"]] if is_oldest else None,
            textposition="outside",
            textfont=LABEL_FONT,
            cliponaxis=False,
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate=(
                "<b>%{y:.1f}%</b> · age " + age_label(age) + "<br>%{x}"
                "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people<extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="group",
        yaxis=_pct_axis(maximum=_headroom(table["prevalence"])),
        xaxis=dict(tickangle=0),
        legend=dict(title=dict(text="Age group  ")),
    )
    return fig


def sex_by_condition(table: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> go.Figure:
    labels = [wrap_label(short_condition(c), 12) for c in conditions]
    fig = _figure(370)
    for sex in ["Female", "Male", "Persons"]:
        sub = table[table[SEX] == sex].set_index(LTHC).reindex(conditions)
        fig.add_bar(
            name=sex_label(sex),
            x=labels,
            y=sub["prevalence"],
            marker=dict(color=SEX_COLOURS[sex], line=GAP_LINE),
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate=(
                "<b>%{y:.1f}%</b> · " + sex_label(sex) + "<br>%{x}"
                "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people<extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="group", xaxis=dict(tickangle=0), yaxis=_pct_axis(maximum=_headroom(table["prevalence"], 1.1))
    )
    return fig


def region_heatmap(table: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> go.Figure:
    pivot = table.pivot(index=REGION, columns=LTHC, values="prevalence").reindex(columns=conditions)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    return _heatmap(pivot, [short_condition(c) for c in conditions], list(pivot.index))


def residency_by_age(table: pd.DataFrame) -> go.Figure:
    labels = [age_label(a) for a in AGE_GROUPS]
    fig = _figure(340)
    for years in YEARS_OPTIONS:
        sub = table[table[YEARS] == years].set_index(AGE).reindex(AGE_GROUPS)
        fig.add_bar(
            name=years,
            x=labels,
            y=sub["prevalence"],
            marker=dict(color=YEARS_COLOURS[years], line=GAP_LINE),
            text=[f"{v:.1f}%" for v in sub["prevalence"]],
            textposition="outside",
            textfont=LABEL_FONT,
            cliponaxis=False,
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate=(
                "<b>%{y:.1f}%</b> · " + years + " in Australia<br>Age %{x}"
                "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people<extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="group",
        bargap=0.45,
        yaxis=_pct_axis(maximum=_headroom(table["prevalence"], 1.18)),
        xaxis=dict(title=dict(text="Age group")),
        legend=dict(title=dict(text="Years in Australia  ")),
    )
    return fig


# ---------------------------------------------------------------------------
# Explore
# ---------------------------------------------------------------------------
def ranked_groups(comparison: Comparison, top_n: int = 20) -> go.Figure:
    table = comparison.table
    if table.empty:
        return empty_figure("No published data for this combination of filters.")
    shown = table if comparison.dimension.order else table.head(top_n)
    n = len(shown)
    height = max(230, 34 * n + 90)
    x_max = _headroom(list(shown["prevalence"]) + [comparison.overall or 0], 1.2)
    fig = _figure(height)
    fig.add_bar(
        orientation="h",
        x=shown["prevalence"],
        y=shown["group"],
        marker=dict(color=BLUE),
        text=[f"{v:.1f}%" for v in shown["prevalence"]],
        textposition="outside",
        textfont=LABEL_FONT,
        cliponaxis=False,
        customdata=np.column_stack([shown["cases"], shown["population"], shown["cells"]]),
        hovertemplate=(
            "<b>%{x:.1f}%</b> · %{y}<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people"
            "<br>%{customdata[2]} census cells pooled<extra></extra>"
        ),
        showlegend=False,
    )
    _horizontal_layout(fig, x_max)
    if comparison.overall is not None:
        fig.add_vline(x=comparison.overall, line=dict(color=INK_2, width=1))
        fig.add_annotation(
            x=comparison.overall,
            y=1,
            yref="paper",
            yanchor="bottom",
            text=f"All groups {comparison.overall:.1f}%",
            showarrow=False,
            font=dict(color=INK_2, size=11),
        )
        fig.update_layout(margin=dict(t=30))
    return fig


def profile_heatmap(
    profile: pd.DataFrame, column: str, groups: list[str], display: dict[str, str], conditions: list[str]
) -> go.Figure:
    if profile.empty:
        return empty_figure("No published data for this combination of filters.")
    pivot = profile.pivot(index=column, columns=LTHC, values="prevalence").reindex(index=groups, columns=conditions)
    return _heatmap(pivot, [short_condition(c) for c in conditions], [display.get(g, g) for g in groups])


def wrap_label(text: str, width: int = 11) -> str:
    """Break long axis labels onto several lines so heatmap columns do not collide."""
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _heatmap(pivot: pd.DataFrame, x_labels: list[str], y_labels: list[str]) -> go.Figure:
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    z_max = float(np.ceil(finite.max())) if finite.size else 1.0
    text = [["" if not np.isfinite(v) else f"{v:.1f}" for v in row] for row in values]
    wrapped = [wrap_label(label, 10) for label in x_labels]
    rows = [wrap_label(label, 22) for label in y_labels]
    header_lines = max(label.count("<br>") + 1 for label in wrapped)
    fig = _figure(max(240, 40 * len(y_labels) + 60 + 16 * header_lines))
    fig.add_heatmap(
        z=values,
        x=wrapped,
        y=rows,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=SEQUENTIAL_SCALE,
        zmin=0,
        zmax=z_max,
        xgap=2,
        ygap=2,
        hoverongaps=False,
        showscale=False,
        hovertemplate="<b>%{z:.1f}%</b><br>%{y}<br>%{x}<extra></extra>",
    )
    fig.update_layout(
        xaxis=dict(side="top", showline=False, showgrid=False, tickangle=0),
        yaxis=dict(showgrid=False, autorange="reversed"),
        plot_bgcolor=SURFACE,
        margin=dict(t=8),
    )
    return fig


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def age_trajectory(trajectory: pd.DataFrame, selected_sex: str, selected_age: str) -> go.Figure:
    labels = [age_label(a) for a in AGE_GROUPS]
    fig = _figure(340)
    # "All people" first so the Female/Male lines sit on top; legendrank keeps the reading order.
    for sex, rank in [("Persons", 3), ("Female", 1), ("Male", 2)]:
        sub = trajectory[trajectory["sex"] == sex].set_index("age").reindex(AGE_GROUPS)
        fig.add_scatter(
            x=labels,
            y=sub["predicted"],
            mode="lines+markers",
            name=sex_label(sex),
            legendrank=rank,
            line=dict(color=SEX_COLOURS[sex], width=2),
            marker=dict(size=9, color=SEX_COLOURS[sex], line=dict(color=SURFACE, width=2)),
            hovertemplate="<b>%{y:.1f}%</b> predicted · " + sex_label(sex) + "<extra></extra>",
        )
    observed = trajectory.dropna(subset=["observed"])
    if not observed.empty:
        fig.add_scatter(
            x=[age_label(a) for a in observed["age"]],
            y=observed["observed"],
            mode="markers",
            name="Published census value",
            legendrank=4,
            marker=dict(size=13, symbol="circle-open", color=INK, line=dict(width=1.5)),
            customdata=[sex_label(s) for s in observed["sex"]],
            hovertemplate="<b>%{y:.1f}%</b> published · %{customdata}<extra></extra>",
        )
    selected = trajectory[(trajectory["sex"] == selected_sex) & (trajectory["age"] == selected_age)]
    if not selected.empty:
        value = float(selected["predicted"].iloc[0])
        position = AGE_GROUPS.index(selected_age)
        fig.add_annotation(
            x=age_label(selected_age),
            y=value,
            text=f"<b>{value:.1f}%</b> {sex_label(selected_sex).lower()}",
            showarrow=True,
            arrowhead=0,
            arrowcolor=MUTED,
            ax={0: 44, 1: 0, 2: -56}[position],
            ay=-34,
            font=dict(color=INK, size=12),
            bgcolor=SURFACE,
        )
    all_values = list(trajectory["predicted"]) + list(observed["observed"])
    fig.update_layout(
        yaxis=_pct_axis(maximum=_headroom(all_values, 1.2)),
        xaxis=dict(title=dict(text="Age group")),
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def condition_profile(profile: pd.DataFrame, selected_condition: str) -> go.Figure:
    ordered = profile.sort_values("predicted", ascending=False)
    labels = [short_condition(c) for c in ordered["condition"]]
    colours = [BLUE if c == selected_condition else DE_EMPHASIS for c in ordered["condition"]]
    height = max(260, 32 * len(ordered) + 100)
    fig = _figure(height)
    fig.add_bar(
        orientation="h",
        x=ordered["predicted"],
        y=labels,
        marker=dict(color=colours),
        name="Model estimate",
        showlegend=False,
        hovertemplate="<b>%{x:.1f}%</b> predicted<br>%{y}<extra></extra>",
    )
    observed = ordered.dropna(subset=["observed"])
    if not observed.empty:
        fig.add_scatter(
            x=observed["observed"],
            y=[short_condition(c) for c in observed["condition"]],
            mode="markers",
            name="Published census value",
            marker=dict(size=12, symbol="circle-open", color=INK, line=dict(width=1.5)),
            hovertemplate="<b>%{x:.1f}%</b> published<br>%{y}<extra></extra>",
        )
    # Values sit in their own column at the right edge so they never collide with the markers.
    fig.add_annotation(
        x=1,
        xref="paper",
        xanchor="left",
        xshift=10,
        y=1,
        yref="paper",
        yanchor="bottom",
        text="Predicted",
        showarrow=False,
        font=dict(color=MUTED, size=11),
    )
    for condition, label, value in zip(ordered["condition"], labels, ordered["predicted"], strict=True):
        selected = condition == selected_condition
        fig.add_annotation(
            x=1,
            xref="paper",
            xanchor="left",
            xshift=10,
            y=label,
            yref="y",
            text=f"<b>{value:.1f}%</b>" if selected else f"{value:.1f}%",
            showarrow=False,
            font=dict(color=INK if selected else INK_2, size=12),
        )
    x_max = _headroom(list(ordered["predicted"]) + list(observed["observed"]), 1.08)
    _horizontal_layout(fig, x_max, "Predicted prevalence (%)")
    fig.update_layout(margin=dict(r=72, t=30))
    return fig


# ---------------------------------------------------------------------------
# Model page
# ---------------------------------------------------------------------------
def r2_dumbbell(overfitting: pd.DataFrame) -> go.Figure:
    data = overfitting.sort_values("Test R²", ascending=True)
    fig = _figure(300)
    for _, row in data.iterrows():
        fig.add_scatter(
            x=[row["Test R²"], row["Train R²"]],
            y=[row["Model"], row["Model"]],
            mode="lines",
            line=dict(color=BASELINE, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    for key, column, label in [("train", "Train R²", "Training groups"), ("test", "Test R²", "Held-out test groups")]:
        fig.add_scatter(
            x=data[column],
            y=data["Model"],
            mode="markers+text" if key == "test" else "markers",
            name=label,
            marker=dict(size=11, color=TRAIN_TEST_COLOURS[key], line=dict(color=SURFACE, width=2)),
            text=[f"{v:.3f}" for v in data[column]] if key == "test" else None,
            textposition="middle left",
            textfont=LABEL_FONT,
            hovertemplate="<b>R² %{x:.3f}</b> · " + label + "<br>%{y}<extra></extra>",
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(text="R² (higher is better)"), range=[0.6, 1.005], showgrid=True, gridcolor=GRID, showline=False
        ),
        yaxis=dict(showgrid=False, showline=True, linecolor=BASELINE),
        margin=dict(l=8, r=24),
    )
    return fig


FEATURE_PREFIXES = {
    "Long-term health condition (LTHC)_": "Condition",
    "Age group_": "Age group",
    "Country of birth of person_": "Country of birth",
    "Language used at home_": "Language at home",
    "Years spent in Australia_": "Years in Australia",
    "Proficiency in spoken English_": "English proficiency",
    "Region_class_": "Region",
    "Subregion_class_": "Subregion",
    "Sex_": "Sex",
}


def pretty_feature(name: str) -> str:
    name = name.removeprefix("categorical__")
    for prefix, label in FEATURE_PREFIXES.items():
        if name.startswith(prefix):
            value = name[len(prefix) :]
            value = age_label(value) if label == "Age group" else short_condition(value)
            return f"{label}: {value}"
    return name


def feature_importance(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = importance.sort_values("Importance", ascending=False).head(top_n)
    labels = [pretty_feature(f) for f in top["Feature"]]
    share = top["Importance"] * 100
    fig = _figure(max(300, 30 * len(top) + 80))
    fig.add_bar(
        orientation="h",
        x=share,
        y=labels,
        marker=dict(color=BLUE),
        text=[f"{v:.1f}%" for v in share],
        textposition="outside",
        textfont=LABEL_FONT,
        cliponaxis=False,
        showlegend=False,
        hovertemplate="<b>%{x:.2f}%</b> of total gain<br>%{y}<extra></extra>",
    )
    _horizontal_layout(fig, _headroom(share, 1.2), "Share of XGBoost feature importance (%)")
    return fig


def actual_vs_predicted(predictions: pd.DataFrame) -> go.Figure:
    actual = predictions["Actual Prevalence"]
    predicted = predictions["Predicted Prevalence"]
    upper = float(np.ceil(max(actual.max(), predicted.max()) / 10) * 10)
    fig = _figure(430)
    fig.add_scatter(
        x=[0, upper],
        y=[0, upper],
        mode="lines",
        line=dict(color=INK_2, width=1),
        name="Perfect prediction",
        hoverinfo="skip",
    )
    fig.add_scattergl(
        x=actual,
        y=predicted,
        mode="markers",
        name="Held-out census cell",
        marker=dict(size=6, color=BLUE, opacity=0.35, line=dict(width=0)),
        hovertemplate="Predicted <b>%{y:.1f}%</b><br>Published %{x:.1f}%<extra></extra>",
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Published census prevalence (%)"),
            range=[0, upper],
            ticksuffix="%",
            showgrid=True,
            gridcolor=GRID,
        ),
        yaxis=dict(
            title=dict(text="Model prediction (%)"),
            range=[min(0, float(predicted.min()) - 2), upper],
            ticksuffix="%",
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return fig
