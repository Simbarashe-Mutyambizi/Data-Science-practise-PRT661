"""Reusable Dash layout pieces (cards, stat tiles, tables, notices)."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import pandas as pd
from dash import dcc, html

from .theme import GRAPH_CONFIG


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}%"


def fmt_int(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f}"


def page_header(title: str, lede: str | Sequence, eyebrow: str | None = None) -> html.Header:
    children = []
    if eyebrow:
        children.append(html.P(eyebrow, className="eyebrow"))
    children += [html.H1(title), html.P(lede, className="lede")]
    return html.Header(children, className="page-header")


def stat_tile(
    label: str,
    value: str,
    detail=None,
    *,
    status: str | None = None,
    icon: str | None = None,
    hero: bool = False,
    class_name: str = "",
) -> html.Div:
    value_children = []
    if icon:
        value_children.append(
            html.Span(icon, className=f"status-icon status-{status or 'neutral'}", **{"aria-hidden": "true"})
        )
    value_children.append(html.Span(value))
    return html.Div(
        [
            html.P(label, className="tile-label"),
            html.P(value_children, className="tile-value hero" if hero else "tile-value"),
            html.P(detail, className="tile-detail") if detail else None,
        ],
        className=f"tile {class_name}".strip(),
    )


def graph_style(figure) -> dict:
    """Explicit pixel height: a responsive Plotly graph fills its container, so the
    container must carry the figure's intended height."""
    height = getattr(figure.layout, "height", None) or 360
    return {"height": f"{int(height)}px", "width": "100%"}


def graph(figure, graph_id: str | None = None) -> dcc.Graph:
    kwargs = {"id": graph_id} if graph_id else {}
    return dcc.Graph(
        figure=figure,
        config=GRAPH_CONFIG,
        responsive=True,
        style=graph_style(figure),
        className="graph",
        **kwargs,
    )


def data_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, Callable | None]],
    *,
    max_rows: int | None = None,
    bold_rows: Iterable[int] = (),
) -> html.Div:
    """columns: (frame column, header, formatter)."""
    shown = frame if max_rows is None else frame.head(max_rows)
    bold = set(bold_rows)
    header = html.Thead(html.Tr([html.Th(title, scope="col") for _, title, _ in columns]))
    body_rows = []
    for position, (_, row) in enumerate(shown.iterrows()):
        cells = []
        for column, _, formatter in columns:
            value = row[column]
            text = formatter(value) if formatter else value
            cells.append(html.Td(text, className="num" if formatter else None))
        body_rows.append(html.Tr(cells, className="bold" if position in bold else None))
    table = html.Table([header, html.Tbody(body_rows)], className="data-table")
    note = None
    if max_rows is not None and len(frame) > max_rows:
        note = html.P(f"Showing {max_rows} of {len(frame)} rows.", className="table-note")
    return html.Div([table, note], className="table-wrap")


def table_view(table: html.Div, summary: str = "Show the data as a table") -> html.Details:
    return html.Details([html.Summary(summary), table], className="table-view")


def card(
    title: str, *children, subtitle: str | None = None, class_name: str = "", id: str | None = None
) -> html.Section:
    head = [html.H2(title)]
    if subtitle:
        head.append(html.P(subtitle, className="card-subtitle"))
    kwargs = {"id": id} if id else {}
    return html.Section(
        [html.Div(head, className="card-head"), *children], className=f"card {class_name}".strip(), **kwargs
    )


def insight(text, class_name: str = "") -> html.P:
    return html.P(text, className=f"insight {class_name}".strip())


def notice(children, kind: str = "info", icon: str = "i") -> html.Div:
    return html.Div(
        [
            html.Span(icon, className="notice-icon", **{"aria-hidden": "true"}),
            html.Div(children, className="notice-body"),
        ],
        className=f"notice notice-{kind}",
        role="note",
    )


def disclaimer() -> html.Div:
    return notice(
        [
            html.Strong("Population-level estimates only. "),
            "Figures describe groups of people in aggregated 2021 Census data. They are not an individual "
            "diagnosis or a personal health-risk assessment; health decisions should stay with qualified professionals.",
        ],
        kind="caution",
        icon="!",
    )


def field(
    label: str, control, help_text: str | None = None, html_for: str | None = None, class_name: str = ""
) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=html_for, className="field-label"),
            control,
            html.P(help_text, className="field-help") if help_text else None,
        ],
        className=f"field {class_name}".strip(),
    )
