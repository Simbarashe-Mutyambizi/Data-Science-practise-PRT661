from __future__ import annotations

import os

import dash
from dash import Dash, html

from lthc_dashboard import theme  # noqa: F401
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
        {"name": "theme-color", "content": "#0b2545"},
        {
            "name": "description",
            "content": "Long-term health conditions among culturally and linguistically diverse "
            "Australians (AIHW, 2021 Census): findings, census analysis and a prevalence predictor.",
        },
    ],
)
server = app.server

app.index_string = """<!DOCTYPE html>
<html lang="en">
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/logo.svg">
        <link rel="preload" href="/assets/fonts/inter-latin-opsz-normal.woff2" as="font" type="font/woff2" crossorigin>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


@server.route("/healthz")
def healthz():
    return {"status": "ok"}


get_data()
warm_up_api()


def status_bar() -> html.Footer:
    return html.Footer(
        [
            html.P(
                [
                    html.Strong("Population-level estimates only: "),
                    "aggregated census counts, not a diagnosis or a personal health-risk assessment.",
                ]
            ),
            html.P(
                [
                    "Source: ",
                    html.A(
                        "AIHW (2021), Chronic conditions among CALD Australians",
                        href="https://www.aihw.gov.au/reports/cald-australians/chronic-conditions-cald-2021/data",
                        target="_blank",
                        rel="noopener noreferrer",
                        title="AIHW (2021), Chronic conditions among culturally and linguistically diverse Australians",
                    ),
                    " · PRT661 Data Science Practice, Charles Darwin University",
                ],
                className="statusbar-source",
            ),
        ],
        className="statusbar",
    )


app.layout = html.Div(
    [
        html.A("Skip to content", href="#content", className="skip-link"),
        html.Main(dash.page_container, id="main"),
        status_bar(),
    ],
    className="app",
)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DASH_DEBUG") == "1",
    )
