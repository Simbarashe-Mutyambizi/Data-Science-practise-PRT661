from __future__ import annotations

import math
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    CORAL,
    GRID,
    INK,
    INK_2,
    MUTED,
    NAVY,
    SEQUENTIAL_SCALE,
    SEX_COLOURS,
    SURFACE,
    TEAL,
    TEMPLATE,
    TRAIN_TEST_COLOURS,
    YEARS_COLOURS,
)

LABEL_FONT = dict(color=INK_2, size=12)
GAP_LINE = dict(color=SURFACE, width=1)


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
        bargap=0.3,
    )


def empty_figure(message: str, height: int = 260) -> go.Figure:
    fig = _figure(height)
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(color=MUTED, size=13)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def wrap_label(text: str, width: int = 11) -> str:
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def age_by_condition(table: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> go.Figure:
    labels = [wrap_label(short_condition(c), 14) for c in conditions]
    fig = _figure(360)
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
            textfont=dict(color=INK, size=12),
            cliponaxis=False,
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate=(
                "<b>%{y:.1f}%</b> aged " + age_label(age) + "<br>%{x}"
                "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people<extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="group",
        yaxis=_pct_axis(maximum=_headroom(table["prevalence"], 1.14)),
        xaxis=dict(tickangle=0, tickfont=dict(color=INK_2, size=12.5)),
        legend=dict(title=dict(text="Age group")),
    )
    return fig


def sex_by_condition(table: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> go.Figure:
    labels = [wrap_label(short_condition(c), 14) for c in conditions]
    fig = _figure(360)
    for sex in ["Female", "Male"]:
        sub = table[table[SEX] == sex].set_index(LTHC).reindex(conditions)
        fig.add_bar(
            name=sex_label(sex),
            x=labels,
            y=sub["prevalence"],
            marker=dict(color=SEX_COLOURS[sex], line=GAP_LINE),
            text=[f"{v:.1f}%" for v in sub["prevalence"]],
            textposition="outside",
            textfont=dict(color=INK_2, size=11.5),
            cliponaxis=False,
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate=(
                "<b>%{y:.1f}%</b> of " + sex_label(sex).lower() + "s<br>%{x}"
                "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people<extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="group",
        bargap=0.34,
        xaxis=dict(tickangle=0, tickfont=dict(color=INK_2, size=12.5)),
        yaxis=_pct_axis(maximum=_headroom(table.loc[table[SEX] != "Persons", "prevalence"], 1.14)),
        legend=dict(title=dict(text="Sex")),
    )
    return fig


def region_heatmap(table: pd.DataFrame, conditions: list[str] = COMMON_CONDITIONS) -> go.Figure:
    pivot = table.pivot(index=REGION, columns=LTHC, values="prevalence").reindex(columns=conditions)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    return _heatmap(pivot, [short_condition(c) for c in conditions], list(pivot.index))


def residency_by_age(table: pd.DataFrame) -> go.Figure:
    labels = [age_label(a) for a in AGE_GROUPS]
    fig = _figure(360)
    for years in YEARS_OPTIONS:
        sub = table[table[YEARS] == years].set_index(AGE).reindex(AGE_GROUPS)
        fig.add_bar(
            name=years,
            x=labels,
            y=sub["prevalence"],
            marker=dict(color=YEARS_COLOURS[years], line=GAP_LINE),
            text=[f"{v:.1f}%" for v in sub["prevalence"]],
            textposition="outside",
            textfont=dict(color=INK, size=12),
            cliponaxis=False,
            customdata=np.column_stack([sub["cases"], sub["population"]]),
            hovertemplate=(
                "<b>%{y:.1f}%</b> · " + years + " in Australia<br>Aged %{x}"
                "<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} people<extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="group",
        bargap=0.42,
        yaxis=_pct_axis(maximum=_headroom(table["prevalence"], 1.14)),
        xaxis=dict(title=dict(text="Age group"), tickfont=dict(color=INK_2, size=12.5)),
        legend=dict(title=dict(text="Years in Australia")),
    )
    return fig


def _heatmap(pivot: pd.DataFrame, x_labels: list[str], y_labels: list[str]) -> go.Figure:
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    z_max = float(np.ceil(finite.max())) if finite.size else 1.0
    text = [["" if not np.isfinite(v) else f"{v:.1f}%" for v in row] for row in values]
    wrapped = [wrap_label(label, 10) for label in x_labels]
    header_lines = max(label.count("<br>") + 1 for label in wrapped)
    fig = _figure(max(260, 38 * len(y_labels) + 40 + 16 * header_lines))
    fig.add_heatmap(
        z=values,
        x=wrapped,
        y=y_labels,
        text=text,
        customdata=[list(x_labels) for _ in y_labels],
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=SEQUENTIAL_SCALE,
        zmin=0,
        zmax=z_max,
        xgap=3,
        ygap=3,
        hoverongaps=False,
        showscale=False,
        hovertemplate="<b>%{z:.1f}%</b><br>%{y}<br>%{customdata}<extra></extra>",
    )
    fig.update_layout(
        xaxis=dict(side="top", showline=False, showgrid=False, tickangle=0, tickfont=dict(color=INK_2, size=12)),
        yaxis=dict(showgrid=False, autorange="reversed", dtick=1, tickfont=dict(color=INK_2, size=12)),
        plot_bgcolor=SURFACE,
        margin=dict(t=8, r=4),
    )
    return fig


def age_trajectory(trajectory: pd.DataFrame, selected_sex: str, selected_age: str) -> go.Figure:
    labels = [age_label(a) for a in AGE_GROUPS]
    fig = _figure(350)
    for sex, rank in [("Persons", 3), ("Female", 1), ("Male", 2)]:
        sub = trajectory[trajectory["sex"] == sex].set_index("age").reindex(AGE_GROUPS)
        colour = SEX_COLOURS[sex]
        fig.add_scatter(
            x=labels,
            y=sub["predicted"],
            mode="lines+markers",
            name=sex_label(sex),
            legendrank=rank,
            line=dict(color=colour, width=3 if sex == selected_sex else 2, dash="dash" if sex == "Persons" else None),
            marker=dict(size=9, color=colour, line=dict(color=SURFACE, width=2)),
            hovertemplate="<b>%{y:.1f}%</b> predicted · " + sex_label(sex) + "<extra></extra>",
        )
        observed = sub.dropna(subset=["observed"])
        if not observed.empty:
            fig.add_scatter(
                x=[age_label(a) for a in observed.index],
                y=observed["observed"],
                mode="markers",
                name="Published census value",
                legendgroup="published",
                showlegend=False,
                marker=dict(size=14, symbol="circle-open", color=colour, line=dict(width=2)),
                hovertemplate="<b>%{y:.1f}%</b> published · " + sex_label(sex) + "<extra></extra>",
            )
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="Published census value",
        legendgroup="published",
        legendrank=4,
        marker=dict(size=13, symbol="circle-open", color=INK_2, line=dict(width=2)),
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
            ax={0: 48, 1: 0, 2: -60}[position],
            ay=-36,
            font=dict(color=INK, size=12.5),
            bgcolor=SURFACE,
            bordercolor=BASELINE,
            borderpad=4,
        )
    observed_all = trajectory["observed"].dropna()
    all_values = list(trajectory["predicted"]) + list(observed_all)
    fig.update_layout(
        yaxis=_pct_axis(maximum=_headroom(all_values, 1.22)),
        xaxis=dict(title=dict(text="Age group"), tickfont=dict(color=INK_2, size=12.5)),
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def condition_profile(profile: pd.DataFrame, selected_condition: str) -> go.Figure:
    ordered = profile.sort_values("predicted", ascending=False)
    labels = [short_condition(c) for c in ordered["condition"]]
    colours = [CORAL if c == selected_condition else TEAL for c in ordered["condition"]]
    height = max(280, 32 * len(ordered) + 100)
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
            marker=dict(size=12, symbol="circle-open", color=NAVY, line=dict(width=2)),
            hovertemplate="<b>%{x:.1f}%</b> published<br>%{y}<extra></extra>",
        )
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
        font=dict(color=MUTED, size=11.5),
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
            font=dict(color=CORAL if selected else INK_2, size=12.5),
        )
    x_max = _headroom(list(ordered["predicted"]) + list(observed["observed"]), 1.08)
    _horizontal_layout(fig, x_max, "Predicted prevalence (%)")
    fig.update_layout(margin=dict(r=76, t=30), yaxis=dict(tickfont=dict(color=INK_2, size=12.5)))
    return fig


def r2_dumbbell(overfitting: pd.DataFrame) -> go.Figure:
    data = overfitting.sort_values("Test R²", ascending=True)
    fig = _figure(320)
    for _, row in data.iterrows():
        fig.add_scatter(
            x=[row["Test R²"], row["Train R²"]],
            y=[row["Model"], row["Model"]],
            mode="lines",
            line=dict(color=BASELINE, width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    for key, column, label in [("train", "Train R²", "Training groups"), ("test", "Test R²", "Held-out test groups")]:
        fig.add_scatter(
            x=data[column],
            y=data["Model"],
            mode="markers+text" if key == "test" else "markers",
            name=label,
            marker=dict(size=13, color=TRAIN_TEST_COLOURS[key], line=dict(color=SURFACE, width=2)),
            text=[f"{v:.3f}" for v in data[column]] if key == "test" else None,
            textposition="middle left",
            textfont=dict(color=INK, size=12),
            hovertemplate="<b>R² %{x:.3f}</b> · " + label + "<br>%{y}<extra></extra>",
        )
    fig.update_layout(
        xaxis=dict(
            title=dict(text="R² (higher is better)"), range=[0.6, 1.005], showgrid=True, gridcolor=GRID, showline=False
        ),
        yaxis=dict(showgrid=False, showline=True, linecolor=BASELINE, tickfont=dict(color=INK_2, size=12.5)),
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
    fig = _figure(max(320, 29 * len(top) + 90))
    fig.add_bar(
        orientation="h",
        x=share,
        y=labels,
        marker=dict(color=[TEAL if i < 2 else "#5cbcaf" for i in range(len(top))]),
        text=[f"{v:.1f}%" for v in share],
        textposition="outside",
        textfont=dict(color=INK_2, size=12),
        cliponaxis=False,
        showlegend=False,
        hovertemplate="<b>%{x:.2f}%</b> of total gain<br>%{y}<extra></extra>",
    )
    _horizontal_layout(fig, _headroom(share, 1.2), "Share of importance (%)")
    fig.update_layout(yaxis=dict(dtick=1, tickfont=dict(color=INK_2, size=12)))
    return fig


def actual_vs_predicted(predictions: pd.DataFrame) -> go.Figure:
    actual = predictions["Actual Prevalence"]
    predicted = predictions["Predicted Prevalence"]
    upper = float(np.ceil(max(actual.max(), predicted.max()) / 10) * 10)
    fig = _figure(500)
    fig.add_scatter(
        x=[0, upper],
        y=[0, upper],
        mode="lines",
        line=dict(color=NAVY, width=1.5, dash="dash"),
        name="Perfect prediction",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=actual,
        y=predicted,
        mode="markers",
        name="Held-out census cell",
        marker=dict(size=6, color=TEAL, opacity=0.4, line=dict(width=0)),
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
        ),
    )
    return fig
