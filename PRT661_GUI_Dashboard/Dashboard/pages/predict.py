from __future__ import annotations

import json
import math
from urllib.parse import urlparse

import dash
from dash import Input, Output, State, callback, dcc, html

from lthc_dashboard import components as ui, figures
from lthc_dashboard.config import (
    AGE_GROUPS,
    COMPOSITE_CONDITIONS,
    COUNTRY,
    ENGLISH,
    ENGLISH_OPTIONS,
    LANGUAGE,
    NOT_SPECIFIED,
    REGION,
    SOURCE_LABELS,
    SPECIFIC_CONDITIONS,
    SUBREGION,
    YEARS,
    YEARS_OPTIONS,
    age_label,
    sex_label,
    short_condition,
)
from lthc_dashboard.data import get_data
from lthc_dashboard.inference import get_client
from lthc_dashboard.model import to_api_record
from lthc_dashboard.prediction import (
    MODE_COUNTRY,
    MODE_LANGUAGE,
    SEX_NOUNS,
    PredictionOutcome,
    PredictionRequest,
    benchmark_scope,
    describe_group,
    run_prediction,
    validate_request,
)
from lthc_dashboard.theme import TEAL

dash.register_page(__name__, path="/predict", name="Predict", title="Predict · LTHC dashboard", order=2)

NOT_RECORDED = "__not_recorded__"
DEFAULTS = {
    "country": "China (excludes SARs and Taiwan)",
    "years": "More than 10 years",
    "language": "Vietnamese",
    "english": "Very well or well",
}
DEFAULT_AGE, DEFAULT_SEX, DEFAULT_CONDITION = "65 and over", "Female", "Diabetes"

RISK_BADGES = {
    "below": ("lower", "▼"),
    "average": ("about", "●"),
    "above": ("higher", "▲"),
    "well-above": ("much-higher", "▲"),
    "unknown": ("unknown", "–"),
}
RISK_ACCENTS = {"below": "teal", "average": "slate", "above": "coral", "well-above": "coral", "unknown": "slate"}


def _options(values: list[str], not_recorded_label: str) -> list[dict]:
    return [{"label": v, "value": v} for v in values] + [
        {"label": not_recorded_label, "value": NOT_RECORDED, "disabled": True}
    ]


def _segmented(id_: str, options: list[dict], value: str) -> dcc.RadioItems:
    return dcc.RadioItems(id=id_, options=options, value=value, className="segmented")


def _form() -> html.Div:
    lookups = get_data().lookups
    return html.Div(
        [
            html.H2("Describe the population group"),
            html.P(
                "AIHW publishes country of birth and home language in separate census tables, so pick one; the "
                "model learned them separately.",
                className="caption",
            ),
            ui.step(
                1,
                "Census table",
                ui.field(
                    "Describe the group by",
                    _segmented(
                        "pred-mode",
                        [
                            {"label": "Country of birth", "value": MODE_COUNTRY},
                            {"label": "Home language", "value": MODE_LANGUAGE},
                        ],
                        MODE_COUNTRY,
                    ),
                ),
            ),
            ui.step(
                2,
                "Background",
                ui.field(
                    "Country of birth",
                    dcc.Dropdown(
                        id="pred-country",
                        options=_options(lookups.countries, "Not recorded in the language table"),
                        value=DEFAULTS["country"],
                        clearable=False,
                        placeholder="Search countries",
                    ),
                    html_for="pred-country",
                ),
                ui.field(
                    "Years in Australia",
                    dcc.Dropdown(
                        id="pred-years",
                        options=_options(YEARS_OPTIONS, "Not recorded in the language table"),
                        value=DEFAULTS["years"],
                        clearable=False,
                        searchable=False,
                    ),
                    html_for="pred-years",
                ),
                ui.field(
                    "Language used at home",
                    dcc.Dropdown(
                        id="pred-language",
                        options=_options(lookups.languages, "Not recorded in the country table"),
                        value=NOT_RECORDED,
                        clearable=False,
                        disabled=True,
                        placeholder="Search languages",
                    ),
                    html_for="pred-language",
                ),
                ui.field(
                    "Proficiency in spoken English",
                    dcc.Dropdown(
                        id="pred-english",
                        options=_options(ENGLISH_OPTIONS, "Not recorded in the country table"),
                        value=NOT_RECORDED,
                        clearable=False,
                        searchable=False,
                        disabled=True,
                    ),
                    html_for="pred-english",
                ),
            ),
            ui.step(
                3,
                "Age, sex and condition",
                ui.field(
                    "Age group",
                    _segmented("pred-age", [{"label": age_label(a), "value": a} for a in AGE_GROUPS], DEFAULT_AGE),
                ),
                ui.field(
                    "Sex",
                    _segmented(
                        "pred-sex",
                        [{"label": sex_label(s), "value": s} for s in ["Female", "Male", "Persons"]],
                        DEFAULT_SEX,
                    ),
                ),
                ui.field(
                    "Long-term health condition",
                    dcc.Dropdown(
                        id="pred-condition",
                        options=[{"label": c, "value": c} for c in SPECIFIC_CONDITIONS]
                        + [{"label": f"{c} (combined)", "value": c} for c in COMPOSITE_CONDITIONS],
                        value=DEFAULT_CONDITION,
                        clearable=False,
                    ),
                    html_for="pred-condition",
                ),
            ),
            html.Div(
                html.Button("Predict prevalence", id="pred-submit", n_clicks=0, className="btn-primary"),
                className="form-actions",
            ),
            html.Div(id="pred-derived", className="derived"),
            dcc.Store(id="pred-memory", data=dict(DEFAULTS)),
        ],
        className="card form-card",
    )


def layout(**_kwargs):
    return ui.screen(
        "predict",
        "Estimate prevalence for a population group",
        "Inputs are validated and formatted for the XGBoost model, sent to the FastAPI inference service and "
        "returned with the published census value, a relative risk level, charts and insights.",
        html.Div(
            [
                _form(),
                dcc.Loading(
                    html.Div(id="pred-results", className="results"),
                    color=TEAL,
                    delay_show=300,
                    overlay_style={"visibility": "visible", "opacity": 0.55},
                    parent_className="results-loading",
                ),
            ],
            className="form-layout",
        ),
        action=("How the model works", "/model"),
    )


@callback(
    Output("pred-country", "value"),
    Output("pred-country", "disabled"),
    Output("pred-years", "value"),
    Output("pred-years", "disabled"),
    Output("pred-language", "value"),
    Output("pred-language", "disabled"),
    Output("pred-english", "value"),
    Output("pred-english", "disabled"),
    Output("pred-memory", "data"),
    Output("pred-derived", "children"),
    Input("pred-mode", "value"),
    Input("pred-country", "value"),
    Input("pred-years", "value"),
    Input("pred-language", "value"),
    Input("pred-english", "value"),
    State("pred-memory", "data"),
)
def sync_fields(mode, country, years, language, english, memory):
    lookups = get_data().lookups
    memory = dict(memory or DEFAULTS)
    if country in lookups.country_language:
        memory["country"] = country
    if years in YEARS_OPTIONS:
        memory["years"] = years
    if language in lookups.language_country:
        memory["language"] = language
    if english in ENGLISH_OPTIONS:
        memory["english"] = english

    if mode == MODE_LANGUAGE:
        values = (NOT_RECORDED, True, NOT_RECORDED, True, memory["language"], False, memory["english"], False)
        paired_country = lookups.language_country[memory["language"]]
        items = [
            html.Li(
                [
                    "Country of birth ",
                    html.Strong(paired_country),
                    f", paired with {memory['language']} during data cleaning; it sets the region (",
                    lookups.country_region[paired_country],
                    ", ",
                    lookups.country_subregion[paired_country],
                    ").",
                ]
            ),
            html.Li(["Years in Australia ", html.Strong(NOT_SPECIFIED), " (not recorded in the language table)."]),
        ]
    else:
        values = (memory["country"], False, memory["years"], False, NOT_RECORDED, True, NOT_RECORDED, True)
        country_value = memory["country"]
        items = [
            html.Li(
                [
                    "Region ",
                    html.Strong(lookups.country_region[country_value]),
                    " and subregion ",
                    html.Strong(lookups.country_subregion[country_value]),
                    " (from country of birth).",
                ]
            ),
            html.Li(
                [
                    "Home language ",
                    html.Strong(lookups.country_language[country_value]),
                    ", the language paired with this country during data cleaning.",
                ]
            ),
            html.Li(["English proficiency ", html.Strong(NOT_SPECIFIED), " (not recorded in this table)."]),
        ]
    derived = [html.P("What the model also receives", className="derived-title"), html.Ul(items)]
    return (*values, memory, derived)


@callback(
    Output("pred-results", "children"),
    Input("pred-submit", "n_clicks"),
    State("pred-mode", "value"),
    State("pred-country", "value"),
    State("pred-years", "value"),
    State("pred-language", "value"),
    State("pred-english", "value"),
    State("pred-age", "value"),
    State("pred-sex", "value"),
    State("pred-condition", "value"),
    State("pred-memory", "data"),
)
def predict(_n_clicks, mode, country, years, language, english, age, sex, condition, memory):
    memory = memory or DEFAULTS
    mode = mode or MODE_COUNTRY
    if mode == MODE_COUNTRY:
        request = PredictionRequest(
            mode=mode,
            age=age,
            sex=sex,
            condition=condition,
            country=country if country != NOT_RECORDED else memory.get("country"),
            years=years if years != NOT_RECORDED else memory.get("years"),
        )
    else:
        request = PredictionRequest(
            mode=mode,
            age=age,
            sex=sex,
            condition=condition,
            language=language if language != NOT_RECORDED else memory.get("language"),
            english=english if english != NOT_RECORDED else memory.get("english"),
        )
    data = get_data()
    problems = validate_request(request, data)
    if problems:
        return ui.notice(
            [html.Strong("Please check the form. "), html.Ul([html.Li(p) for p in problems])], kind="error", icon="!"
        )
    outcome = run_prediction(request, data, get_client())
    return render_outcome(outcome)


def _served_by(outcome: PredictionOutcome) -> html.P:
    if outcome.served_by == "api":
        host = urlparse(outcome.api_url or "").netloc or (outcome.api_url or "")
        text = ["Served by the FastAPI service ", html.Code(host)]
        kind = "ok"
    elif outcome.note:
        text = [outcome.note]
        kind = "warn"
    else:
        text = [
            "Served by the bundled copy of the model (no inference API configured; set ",
            html.Code("LTHC_API_URL"),
            ").",
        ]
        kind = "plain"
    return html.P(text, className=f"served-by served-{kind}")


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def _summary(outcome: PredictionOutcome) -> html.Dl:
    request, model_input = outcome.request, outcome.model_input
    derived = "derived"
    rows: list[tuple[str, object, str | None]] = [
        ("Census table", SOURCE_LABELS[outcome.source], None),
    ]
    if request.mode == MODE_COUNTRY:
        rows += [
            ("Country of birth", model_input[COUNTRY], None),
            ("Years in Australia", model_input[YEARS], None),
            ("Language used at home", f"{model_input[LANGUAGE]} (paired during cleaning)", derived),
            ("English proficiency", "Not recorded in this table", derived),
        ]
    else:
        rows += [
            ("Language used at home", model_input[LANGUAGE], None),
            ("English proficiency", model_input[ENGLISH], None),
            ("Country of birth", f"{model_input[COUNTRY]} (paired during cleaning)", derived),
            ("Years in Australia", "Not recorded in this table", derived),
        ]
    rows += [
        ("Region and subregion", f"{model_input[REGION]}, {model_input[SUBREGION]}", derived),
        ("Age group and sex", f"{age_label(request.age)}, {sex_label(request.sex).lower()}", None),
        ("Condition", request.condition, None),
        (
            "Group population (2021 Census)",
            f"{outcome.group_population:,.0f} people" if outcome.group_population else "Not published",
            None,
        ),
        (
            "Model training status",
            {
                "test": "Held-out test group, not seen during training",
                "train": "Training group, seen during training",
            }.get(outcome.split or "", "Not in the published data"),
            None,
        ),
    ]
    items = []
    for label, value, tag in rows:
        items.append(html.Dt(label))
        items.append(html.Dd([value, html.Span("derived", className="tag") if tag else None]))
    return html.Dl(items, className="summary-list")


def _compare(outcome: PredictionOutcome) -> html.Div:
    values = [v for v in (outcome.predicted, outcome.observed, outcome.benchmark) if v is not None]
    raw = max(values) * 1.2 if values and max(values) > 0 else 1.0
    step = 1 if raw <= 5 else 2 if raw <= 10 else 5 if raw <= 40 else 10
    scale = min(100.0, math.ceil(raw / step) * step)

    def left(value: float) -> str:
        return f"{max(0.0, min(value / scale * 100, 100)):.2f}%"

    markers, legend = [], []
    if outcome.benchmark is not None:
        markers.append(html.Span(className="compare-marker benchmark", style={"left": left(outcome.benchmark)}))
        legend.append(
            html.Span(
                [
                    html.I(className="benchmark"),
                    f"Average for {SEX_NOUNS[outcome.request.sex]} aged {age_label(outcome.request.age)} ",
                    html.B(ui.fmt_pct(outcome.benchmark)),
                ]
            )
        )
    if outcome.observed is not None:
        markers.append(html.Span(className="compare-marker observed", style={"left": left(outcome.observed)}))
        legend.append(
            html.Span([html.I(className="observed"), "Published census value ", html.B(ui.fmt_pct(outcome.observed))])
        )
    markers.append(html.Span(className="compare-marker predicted", style={"left": left(outcome.predicted)}))
    legend.insert(0, html.Span([html.I(className="predicted"), "Predicted ", html.B(ui.fmt_pct(outcome.predicted))]))
    return html.Div(
        [
            html.Div("How the estimate compares", className="eyebrow-label"),
            html.Div(
                markers,
                className="compare-track",
                role="img",
                **{"aria-label": "Comparison of predicted, published and average prevalence"},
            ),
            html.Div(
                [html.Span("0%"), html.Span(f"{scale / 2:g}%"), html.Span(f"{scale:g}%")], className="compare-scale"
            ),
            html.Div(legend, className="compare-legend"),
        ],
        className="compare",
    )


def render_outcome(outcome: PredictionOutcome) -> html.Div:
    request = outcome.request
    risk = outcome.risk
    condition_name = short_condition(request.condition).lower()
    badge_kind, badge_icon = RISK_BADGES.get(risk.key, ("unknown", "–"))

    status = []
    if outcome.split == "test":
        status.append(ui.badge("Held-out test group", "outline"))
    elif outcome.split == "train":
        status.append(ui.badge("Training group", "outline"))
    status.append(_served_by(outcome))

    headline = html.Div(
        [
            html.Div("Population group", className="eyebrow-label"),
            html.Div(_sentence_case(describe_group(outcome)), className="profile-text"),
            html.Div(status, className="badge-row"),
            html.Div(
                [
                    html.Div(f"Predicted prevalence of {condition_name}", className="eyebrow-label"),
                    html.Div(ui.fmt_pct(outcome.predicted), className="big"),
                    html.Div(f"{ui.one_in(outcome.predicted).capitalize()} (XGBoost estimate)", className="sub"),
                ],
                className="headline-value",
            ),
        ],
        className="card headline-card",
    )
    compare = html.Div(_compare(outcome), className="card compare-card")

    if outcome.observed is not None:
        observed_kpi = ui.kpi(
            "Published (census)",
            ui.fmt_pct(outcome.observed),
            f"AIHW 2021 value for {outcome.group_population:,.0f} people; the estimate is "
            f"{abs(outcome.predicted - outcome.observed):.1f} points "
            f"{'above' if outcome.predicted > outcome.observed else 'below'}",
            accent="navy",
        )
    else:
        observed_kpi = ui.kpi(
            "Published (census)",
            "Not published",
            "AIHW suppressed this cell (n.p.) or the group is not in the tables; the model fills the gap",
            accent="navy",
        )
    if outcome.benchmark is not None:
        benchmark_kpi = ui.kpi(
            "Benchmark",
            ui.fmt_pct(outcome.benchmark),
            f"{SEX_NOUNS[request.sex].capitalize()} aged {age_label(request.age)}, "
            f"{benchmark_scope(outcome.source).replace('across ', '')}",
            accent="teal",
        )
    else:
        benchmark_kpi = ui.kpi(
            "Benchmark", "—", "No published benchmark for this age, sex and condition", accent="teal"
        )
    risk_kpi = ui.kpi(
        "Relative risk level",
        ui.badge(risk.label, badge_kind, badge_icon),
        f"{risk.ratio:.2f}× the benchmark; bands at 0.8×, 1.25× and 2×"
        if risk.ratio is not None
        else "No benchmark to compare against",
        accent=RISK_ACCENTS.get(risk.key, "slate"),
    )

    trajectory_table = ui.data_table(
        outcome.trajectory.assign(
            age_l=outcome.trajectory["age"].map(age_label), sex_l=outcome.trajectory["sex"].map(sex_label)
        ),
        [
            ("age_l", "Age group", None),
            ("sex_l", "Sex", None),
            ("predicted", "Predicted", ui.fmt_pct),
            ("observed", "Published", ui.fmt_pct),
        ],
    )
    profile_table = ui.data_table(
        outcome.profile.sort_values("predicted", ascending=False).assign(name=lambda t: t["condition"]),
        [("name", "Condition", None), ("predicted", "Predicted", ui.fmt_pct), ("observed", "Published", ui.fmt_pct)],
    )
    profile_rows = outcome.profile[
        outcome.profile["condition"].isin(SPECIFIC_CONDITIONS) | (outcome.profile["condition"] == request.condition)
    ]

    insight_items = []
    for text in outcome.insights:
        caution = text.startswith("Caution")
        insight_items.append(html.Li(ui.emphasise_numbers(text), className="caution" if caution else None))

    control, panes = ui.switch(
        "pred-details",
        [
            ("Insights", "insights", html.Ul(insight_items, className="insight-list")),
            ("Summary", "summary", _summary(outcome)),
            (
                "Model input",
                "input",
                html.Pre(
                    json.dumps(to_api_record(outcome.model_input), indent=2, ensure_ascii=False),
                    className="code-block",
                ),
            ),
        ],
    )
    details = ui.card(
        "Health insights",
        panes,
        subtitle="Generated from the prediction and the published census data",
        aside=control,
    )

    return html.Div(
        [
            ui.disclaimer(),
            html.Div([headline, observed_kpi, benchmark_kpi, risk_kpi, compare], className="headline-row"),
            html.Div(
                [
                    ui.chart_card(
                        f"{short_condition(request.condition)} across age groups",
                        figures.age_trajectory(outcome.trajectory, request.sex, request.age),
                        trajectory_table,
                        key="pred-trajectory",
                        subtitle="Same background, every age group and sex",
                        foot="Lines: model estimates (dashed: all people). Rings: published census values.",
                    ),
                    ui.chart_card(
                        "Condition profile for this group",
                        figures.condition_profile(profile_rows, request.condition),
                        profile_table,
                        key="pred-profile",
                        subtitle=f"Each specific condition, {SEX_NOUNS[request.sex]} aged {age_label(request.age)}",
                        foot="Coral: the condition you chose. Rings: published census values.",
                    ),
                    details,
                ],
                className="charts-row",
            ),
        ],
        className="results-grid",
    )
