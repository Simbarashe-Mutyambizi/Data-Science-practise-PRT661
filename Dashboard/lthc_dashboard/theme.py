"""Colour roles and the Plotly template.

Colours come from a validated data-viz palette (categorical slots checked for
colour-vision-deficiency separation; ordinal ramps checked for monotone
lightness). Roles, not raw hex, are used everywhere else in the app.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# Surfaces and ink
PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots (identity) - used in this fixed order, never cycled
BLUE = "#2a78d6"
ORANGE = "#eb6834"
DE_EMPHASIS = "#a9a8a0"  # "everything else" in emphasis charts (table view provides relief)
CONTEXT_GRAY = "#898781"  # context series such as "All people"

SEX_COLOURS = {"Female": BLUE, "Male": ORANGE, "Persons": CONTEXT_GRAY}

# Ordinal ramps (one hue, light -> dark), validated with --ordinal
AGE_COLOURS = {"00–44": "#86b6ef", "45–64": "#2a78d6", "65 and over": "#104281"}
YEARS_COLOURS = {"0–10 years": "#86b6ef", "More than 10 years": "#184f95"}
TRAIN_TEST_COLOURS = {"train": "#86b6ef", "test": "#184f95"}

# Sequential ramp (magnitude) for heatmaps: blue 100 -> 700
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SEQUENTIAL_SCALE = [[index / (len(SEQUENTIAL_BLUE) - 1), colour] for index, colour in enumerate(SEQUENTIAL_BLUE)]

# Status (reserved meaning; always shipped with an icon and a text label)
STATUS = {"good": "#0ca30c", "neutral": MUTED, "serious": "#ec835a", "critical": "#d03b3b"}

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtons": [["toImage"]],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _template() -> go.layout.Template:
    axis_common = dict(
        tickfont=dict(color=MUTED, size=12),
        title=dict(font=dict(color=INK_2, size=12)),
        linecolor=BASELINE,
        linewidth=1,
        ticks="",
        zeroline=False,
        automargin=True,
    )
    return go.layout.Template(
        # Value labels keep their size instead of shrinking to the bar width
        data=dict(bar=[go.Bar(constraintext="none")]),
        layout=dict(
            font=dict(family=FONT_FAMILY, size=13, color=INK_2),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            colorway=[BLUE, ORANGE],
            xaxis=dict(showgrid=False, showline=True, **axis_common),
            yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, **axis_common),
            legend=dict(
                orientation="h",
                x=0,
                xanchor="left",
                y=1.02,
                yanchor="bottom",
                font=dict(color=INK_2, size=12),
                bgcolor="rgba(0,0,0,0)",
                itemclick="toggleothers",
            ),
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor=GRID,
                font=dict(family=FONT_FAMILY, color=INK, size=12),
                align="left",
            ),
            margin=dict(l=8, r=16, t=16, b=8),
            barcornerradius=4,
            bargap=0.35,
            bargroupgap=0.12,
            hovermode="closest",
            separators=".,",
        ),
    )


pio.templates["lthc"] = _template()
TEMPLATE = "lthc"
