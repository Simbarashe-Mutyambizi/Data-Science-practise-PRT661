from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = '"Inter Variable", Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

NAVY = "#0b2545"
NAVY_2 = "#13315c"
TEAL = "#0f9d8f"
TEAL_INK = "#0b7a70"
CORAL = "#e4572e"
AMBER = "#f2a541"

PAGE = "#f4f6fa"
SURFACE = "#ffffff"
INK = "#1c2733"
INK_2 = "#3d4a5c"
MUTED = "#5f6b7a"
GRID = "#e8edf3"
BASELINE = "#cfd7e2"
REFERENCE = "#c3ccd8"

BLUE = TEAL
DE_EMPHASIS = REFERENCE
CONTEXT_GRAY = "#8a96a8"

MALE = "#2c5a93"
FEMALE = "#e39b2f"
SEX_COLOURS = {"Female": FEMALE, "Male": MALE, "Persons": CONTEXT_GRAY}

TEAL_RAMP_3 = ["#6cc2b6", "#1f9e8f", "#0b5c57"]
TEAL_RAMP_2 = ["#6cc2b6", "#0b5c57"]
AGE_COLOURS = dict(zip(["00–44", "45–64", "65 and over"], TEAL_RAMP_3, strict=True))
YEARS_COLOURS = {"0–10 years": TEAL_RAMP_2[0], "More than 10 years": TEAL_RAMP_2[1]}
TRAIN_TEST_COLOURS = {"train": REFERENCE, "test": TEAL}

SEQUENTIAL = ["#e8f5f3", "#c3e7e1", "#95d3ca", "#5cbcaf", "#249f91", "#0f8078", "#12606a", "#143f5c", "#0b2545"]
SEQUENTIAL_SCALE = [[index / (len(SEQUENTIAL) - 1), colour] for index, colour in enumerate(SEQUENTIAL)]

LEVEL_COLOURS = {"lower": TEAL, "about": "#b9c3cf", "higher": CORAL, "much-higher": "#b8321a"}
STATUS = {"good": TEAL, "neutral": CONTEXT_GRAY, "serious": CORAL, "critical": "#b8321a"}

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtons": [["toImage"]],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _template() -> go.layout.Template:
    axis_common = dict(
        tickfont=dict(color=MUTED, size=12),
        title=dict(font=dict(color=INK_2, size=12.5), standoff=10),
        linecolor=BASELINE,
        linewidth=1,
        ticks="",
        zeroline=False,
        automargin=True,
        gridcolor=GRID,
        gridwidth=1,
    )
    return go.layout.Template(
        data=dict(bar=[go.Bar(constraintext="none", textfont=dict(color=INK_2, size=12))]),
        layout=dict(
            font=dict(family=FONT_FAMILY, size=13, color=INK),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            colorway=[TEAL, MALE, CORAL, FEMALE, NAVY, "#7b5ea7", CONTEXT_GRAY],
            xaxis=dict(showgrid=False, showline=True, **axis_common),
            yaxis=dict(showgrid=True, showline=False, **axis_common),
            legend=dict(
                orientation="h",
                x=0,
                xanchor="left",
                y=1.02,
                yanchor="bottom",
                font=dict(color=INK_2, size=12.5),
                title=dict(font=dict(color=MUTED, size=12.5)),
                bgcolor="rgba(0,0,0,0)",
                itemclick="toggleothers",
            ),
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor=BASELINE,
                font=dict(family=FONT_FAMILY, color=INK, size=12.5),
                align="left",
            ),
            coloraxis=dict(colorbar=dict(thickness=12, outlinewidth=0, tickfont=dict(color=MUTED, size=11.5))),
            margin=dict(l=8, r=16, t=16, b=8),
            barcornerradius=4,
            bargap=0.3,
            bargroupgap=0.1,
            hovermode="closest",
            separators=".,",
        ),
    )


pio.templates["lthc"] = _template()
TEMPLATE = "lthc"
