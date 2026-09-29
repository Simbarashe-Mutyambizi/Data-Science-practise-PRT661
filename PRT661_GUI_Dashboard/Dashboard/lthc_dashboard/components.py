from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

import pandas as pd
from dash import MATCH, Input, Output, clientside_callback, dcc, html

from .theme import GRAPH_CONFIG

NAV_ITEMS = [
    ("overview", "Overview", "/"),
    ("analysis", "Analysis", "/analysis"),
    ("predict", "Predict", "/predict"),
    ("profile", "Group profile", "/profile"),
    ("model", "Model & methods", "/model"),
]


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}%"


def fmt_int(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f}"


NUMBER_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?(?:%|×| percentage points| points\b|\s?pp\b))")


def emphasise_numbers(text: str) -> list:
    parts: list = []
    for index, piece in enumerate(NUMBER_PATTERN.split(text)):
        if not piece:
            continue
        parts.append(html.Strong(piece) if index % 2 else piece)
    return parts


def one_in(pct: float | None) -> str:
    if pct is None or pd.isna(pct) or pct <= 0:
        return "–"
    share = pct / 100
    if share >= 0.95:
        return "almost everyone"
    n = round(1 / share)
    if n >= 1000:
        return "fewer than 1 in 1,000"
    if n >= 2 and abs(1 / n - share) / share < 0.08:
        return f"about 1 in {n:,}"
    for denominator in (3, 4, 5, 10):
        for numerator in range(2, denominator):
            if abs(numerator / denominator - share) / share < 0.06:
                return f"about {numerator} in {denominator}"
    return f"about {share * 100:.0f} in 100"


def appbar(active: str, title: str, lede: str | Sequence, *, action: tuple[str, str] | None = None) -> html.Header:
    nav = html.Nav(
        [
            dcc.Link(
                label,
                href=href,
                className="nav-link active" if key == active else "nav-link",
                title=f"{label} (current page)" if key == active else label,
            )
            for key, label, href in NAV_ITEMS
        ],
        className="nav",
        **{"aria-label": "Main"},
    )
    brand = dcc.Link(
        [
            html.Img(src="/assets/logo.svg", alt="", className="brand-logo"),
            html.Span(
                [
                    html.Span("LTHC Population Health Dashboard", className="brand-name"),
                    html.Span("Long-term conditions among CALD Australians, 2021 Census", className="brand-sub"),
                ],
                className="brand-text",
            ),
        ],
        href="/",
        className="brand",
    )
    heading = [
        html.Div(
            [html.H1(title), html.P(lede, className="lede", title=lede if isinstance(lede, str) else None)],
            className="appbar-heading",
        )
    ]
    if action:
        label, href = action
        heading.append(
            dcc.Link(
                [label, html.Span(className="icon-chevron", **{"aria-hidden": "true"})],
                href=href,
                className="appbar-action",
            )
        )
    return html.Header(
        html.Div(
            [html.Div([brand, nav], className="appbar-top"), html.Div(heading, className="appbar-title")],
            className="appbar-inner",
        ),
        className="appbar",
    )


def screen(
    active: str,
    title: str,
    lede: str | Sequence,
    *children,
    action: tuple[str, str] | None = None,
    toolbar=None,
    class_name: str = "",
) -> html.Div:
    return html.Div(
        [
            appbar(active, title, lede, action=action),
            toolbar,
            html.Div(list(children), id="content", className="screen-body"),
        ],
        className=f"screen {class_name}".strip(),
    )


def kpi(label: str, value, sub=None, *, accent: str = "teal", id: str | None = None) -> html.Div:
    value_kwargs = {"id": f"{id}-value"} if id else {}
    sub_kwargs = {"id": f"{id}-sub"} if id else {}
    return html.Div(
        [
            html.Div(label, className="label"),
            html.Div(value, className="value", **value_kwargs),
            html.Div(sub or "", className="sub", **sub_kwargs),
        ],
        className=f"kpi {accent}",
    )


def kpi_row(items: list, class_name: str = "") -> html.Div:
    return html.Div(items, className=f"kpi-row {class_name}".strip())


clientside_callback(
    "function(value) { return 'switch-box show-' + (value || 'chart'); }",
    Output({"type": "switch-box", "index": MATCH}, "className"),
    Input({"type": "switch", "index": MATCH}, "value"),
)


def switch(key: str, panes: list[tuple[str, str, object]], default: str | None = None):
    default = default or panes[0][1]
    control = dcc.RadioItems(
        id={"type": "switch", "index": key},
        options=[{"label": label, "value": value} for label, value, _ in panes],
        value=default,
        className="segmented mini",
    )
    box = html.Div(
        [html.Div(content, className=f"pane pane-{value}") for _, value, content in panes],
        id={"type": "switch-box", "index": key},
        className=f"switch-box show-{default}",
    )
    return control, box


def graph(figure, graph_id: str | None = None) -> dcc.Graph:
    kwargs = {"id": graph_id} if graph_id else {}
    figure.update_layout(height=None, autosize=True)
    return dcc.Graph(
        figure=figure,
        config=GRAPH_CONFIG,
        responsive=True,
        style={"height": "100%", "width": "100%"},
        className="graph-fill",
        **kwargs,
    )


def card(
    title: str,
    *children,
    subtitle: str | Sequence | None = None,
    class_name: str = "",
    id: str | None = None,
    aside=None,
) -> html.Section:
    head = [
        html.Div(
            [
                html.H2(title, className="card-title"),
                html.P(subtitle, className="caption", title=subtitle if isinstance(subtitle, str) else None)
                if subtitle
                else None,
            ],
            className="card-heading",
        )
    ]
    if aside is not None:
        head.append(html.Div(aside, className="card-aside"))
    kwargs = {"id": id} if id else {}
    return html.Section(
        [html.Div(head, className="card-head"), *children], className=f"card {class_name}".strip(), **kwargs
    )


def chart_card(
    title: str,
    figure,
    table: html.Div | None = None,
    *,
    key: str,
    subtitle: str | None = None,
    foot=None,
    class_name: str = "",
) -> html.Section:
    if table is None:
        return card(
            title,
            html.Div(graph(figure), className="card-body"),
            html.P(foot, className="card-foot") if foot else None,
            subtitle=subtitle,
            class_name=class_name,
        )
    control, box = switch(key, [("Chart", "chart", graph(figure)), ("Table", "table", table)])
    return card(
        title,
        box,
        html.P(foot, className="card-foot") if foot else None,
        subtitle=subtitle,
        aside=control,
        class_name=class_name,
    )


def insight(text, label: str = "Key finding", class_name: str = "") -> html.Div:
    content = emphasise_numbers(text) if isinstance(text, str) else text
    return html.Div(
        [html.Span(label, className="insight-label"), html.P(content)],
        className=f"insight {class_name}".strip(),
    )


def legend(items: list[tuple[str, str, str]]) -> html.Div:
    return html.Div(
        [
            html.Span([html.I(className=f"key-{shape}", style={"--key": colour}), label], className="legend-item")
            for label, colour, shape in items
        ],
        className="legend",
    )


def note(children, class_name: str = "") -> html.Div:
    return html.Div(children, className=f"note {class_name}".strip())


def badge(text: str, kind: str, icon: str | None = None) -> html.Span:
    children = [html.Span(icon, className="badge-icon", **{"aria-hidden": "true"})] if icon else []
    return html.Span(children + [text], className=f"badge badge-{kind}")


def data_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, Callable | None]],
    *,
    max_rows: int | None = None,
    bold_rows: Iterable[int] = (),
    tag_rows: dict[int, str] | None = None,
) -> html.Div:
    shown = frame if max_rows is None else frame.head(max_rows)
    bold = set(bold_rows)
    tags = tag_rows or {}
    header = html.Thead(
        html.Tr([html.Th(title, scope="col", className="num" if fmt else None) for _, title, fmt in columns])
    )
    body_rows = []
    for position, (_, row) in enumerate(shown.iterrows()):
        cells = []
        for index, (column, _, formatter) in enumerate(columns):
            value = row[column]
            text = formatter(value) if formatter else value
            if index == 0 and position in tags:
                text = [text, html.Span(tags[position], className="tag tag-teal")]
            cells.append(html.Td(text, className="num" if formatter else None))
        body_rows.append(html.Tr(cells, className="highlight" if position in bold else None))
    table = html.Table([header, html.Tbody(body_rows)], className="data-table")
    note_el = None
    if max_rows is not None and len(frame) > max_rows:
        note_el = html.P(f"Showing {max_rows} of {len(frame)} rows.", className="table-note")
    return html.Div([table, note_el], className="table-wrap")


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
            "Figures describe groups of people in aggregated 2021 Census data, not an individual diagnosis or a "
            "personal health-risk assessment.",
        ],
        kind="caution",
        icon="!",
    )


def field(
    label: str, control, help_text: str | None = None, html_for: str | None = None, class_name: str = ""
) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=html_for, className="control-label"),
            control,
            html.P(help_text, className="hint") if help_text else None,
        ],
        className=f"form-field {class_name}".strip(),
    )


def step(number: int, title: str, *children) -> html.Div:
    return html.Div(
        [
            html.Div([html.Span(str(number), className="step-num"), html.Span(title)], className="step-title"),
            *children,
        ],
        className="step",
    )
