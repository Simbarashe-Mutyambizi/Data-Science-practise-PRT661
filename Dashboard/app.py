"""LTHC Population Health Dashboard - Plotly Dash entry point.

Local:   python app.py                       -> http://127.0.0.1:8050
Render:  gunicorn app:server --bind 0.0.0.0:$PORT

Set LTHC_API_URL to the FastAPI service (see api/main.py) to serve predictions
through the API; without it the bundled copy of the same model is used.
"""

from __future__ import annotations

import os

import dash
from dash import Dash, Input, Output, dcc, html

from lthc_dashboard import theme  # noqa: F401  (registers the Plotly template)
from lthc_dashboard.data import get_data
from lthc_dashboard.inference import warm_up_api

app = Dash(
    __name__,
    use_pages=True,
    title="LTHC Population Health Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {
            "name": "description",
            "content": "Long-term health conditions among culturally and linguistically diverse "
            "Australians (AIHW, 2021 Census): findings explorer and prevalence predictor.",
        },
    ],
)
server = app.server  # for gunicorn


@server.route("/healthz")
def healthz():
    return {"status": "ok"}


get_data()  # load and index the dataset once at start-up
warm_up_api()  # wake the inference API in the background, if one is configured

NAV_ITEMS = [
    ("overview", "Overview", "/"),
    ("explore", "Explore", "/explore"),
    ("predict", "Predict", "/predict"),
    ("model", "Model & methods", "/model"),
]


def topbar() -> html.Header:
    return html.Header(
        html.Div(
            [
                dcc.Link(
                    [
                        html.Span("LTHC", className="brand-mark"),
                        html.Span(
                            [
                                html.Span("Population Health Dashboard", className="brand-name"),
                                html.Span("CALD Australians · 2021 Census · AIHW", className="brand-sub"),
                            ],
                            className="brand-text",
                        ),
                    ],
                    href="/",
                    className="brand",
                ),
                html.Nav(
                    [
                        dcc.Link(label, href=href, id=f"nav-{key}", className="nav-link")
                        for key, label, href in NAV_ITEMS
                    ],
                    className="nav",
                    **{"aria-label": "Main"},
                ),
            ],
            className="topbar-inner",
        ),
        className="topbar",
    )


def footer() -> html.Footer:
    return html.Footer(
        html.Div(
            [
                html.P(
                    [
                        html.Strong("Population-level estimates only. "),
                        "Built on aggregated census cells, not individual records; not a diagnosis or a personal "
                        "health-risk assessment.",
                    ]
                ),
                html.P(
                    [
                        "Data: Australian Institute of Health and Welfare (2021), ",
                        html.A(
                            "Chronic conditions among culturally and linguistically diverse Australians, 2021",
                            href="https://www.aihw.gov.au/reports/cald-australians/chronic-conditions-cald-2021/data",
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                        ". PRT661 Data Science Practice group project, Charles Darwin University.",
                    ]
                ),
            ],
            className="footer-inner",
        ),
        className="footer",
    )


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.A("Skip to content", href="#main", className="skip-link"),
        topbar(),
        html.Main(dash.page_container, id="main", className="page"),
        footer(),
    ],
    className="app",
)


@app.callback(
    [Output(f"nav-{key}", "className") for key, _, _ in NAV_ITEMS],
    Input("url", "pathname"),
)
def highlight_nav(pathname: str | None):
    pathname = (pathname or "/").rstrip("/") or "/"
    return ["nav-link active" if pathname == href else "nav-link" for _, _, href in NAV_ITEMS]


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DASH_DEBUG") == "1",
    )
