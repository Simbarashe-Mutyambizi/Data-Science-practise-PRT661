"""Predict: the deployment phase of the workflow - select a population group, get a prediction."""

from __future__ import annotations

import json

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
    PredictionOutcome,
    PredictionRequest,
    benchmark_scope,
    describe_group,
    run_prediction,
    validate_request,
)
from lthc_dashboard.theme import BLUE

dash.register_page(__name__, path="/predict", name="Predict", title="Predict · LTHC dashboard", order=2)

NOT_RECORDED = "__not_recorded__"
DEFAULTS = {
    "country": "China (excludes SARs and Taiwan)",
    "years": "More than 10 years",
    "language": "Vietnamese",
    "english": "Very well or well",
}
DEFAULT_AGE, DEFAULT_SEX, DEFAULT_CONDITION = "65 and over", "Female", "Diabetes"


def _options(values: list[str], not_recorded_label: str) -> list[dict]:
    return [{"label": v, "value": v} for v in values] + [
        {"label": not_recorded_label, "value": NOT_RECORDED, "disabled": True}
    ]


def _form() -> html.Div:
    lookups = get_data().lookups
    return html.Div(
        [
            html.H2("1 · Describe the population group", className="form-title"),
            ui.field(
                "Describe the group by",
                dcc.RadioItems(
                    id="pred-mode",
                    options=[
                        {"label": "Country of birth", "value": MODE_COUNTRY},
                        {"label": "Home language", "value": MODE_LANGUAGE},
                    ],
                    value=MODE_COUNTRY,
                    className="segmented",
                    inputClassName="segmented-input",
                    labelClassName="segmented-option",
                ),
                "AIHW publishes these in separate census tables, so the model learned them separately. "
                "The fields the chosen table does not record are locked.",
            ),
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
            ui.field(
                "Age group",
                dcc.RadioItems(
                    id="pred-age",
                    options=[{"label": age_label(a), "value": a} for a in AGE_GROUPS],
                    value=DEFAULT_AGE,
                    className="segmented",
                    inputClassName="segmented-input",
                    labelClassName="segmented-option",
                ),
            ),
            ui.field(
                "Sex",
                dcc.RadioItems(
                    id="pred-sex",
                    options=[{"label": sex_label(s), "value": s} for s in ["Female", "Male", "Persons"]],
                    value=DEFAULT_SEX,
                    className="segmented",
                    inputClassName="segmented-input",
                    labelClassName="segmented-option",
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
            html.Div(id="pred-derived", className="derived"),
            html.Button("Predict prevalence", id="pred-submit", n_clicks=0, className="button-primary"),
            dcc.Store(id="pred-memory", data=dict(DEFAULTS)),
        ],
        className="card form-card",
    )


def layout(**_kwargs):
    return html.Div(
        [
            ui.page_header(
                "Estimate prevalence for a population group",
                [
                    "Choose a group, and the inputs are checked and formatted for the model, sent to the inference "
                    "API, and returned as a population-level prevalence estimate. Results show the ",
                    html.Strong("predicted"),
                    " and ",
                    html.Strong("published"),
                    " prevalence, a ",
                    html.Strong("relative risk level"),
                    ", a ",
                    html.Strong("demographic summary"),
                    ", charts and plain-language insights.",
                ],
                eyebrow="Machine-learning inference",
            ),
            html.Div(
                [
                    _form(),
                    dcc.Loading(
                        html.Div(id="pred-results", className="results"),
                        color=BLUE,
                        delay_show=300,
                        overlay_style={"visibility": "visible", "opacity": 0.55},
                        parent_className="results-loading",
                    ),
                ],
                className="predict-layout",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Keep the locked fields consistent with the chosen census table
# ---------------------------------------------------------------------------
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
        derived = [
            html.P("What the model also receives", className="derived-title"),
            html.Ul(
                [
                    html.Li(
                        [
                            "Country of birth: ",
                            html.Strong(paired_country),
                            f" — paired with {memory['language']} during data cleaning; it sets the region (",
                            lookups.country_region[paired_country],
                            ", ",
                            lookups.country_subregion[paired_country],
                            ").",
                        ]
                    ),
                    html.Li(
                        ["Years in Australia: ", html.Strong(NOT_SPECIFIED), " (not recorded in the language table)."]
                    ),
                ]
            ),
        ]
    else:
        values = (memory["country"], False, memory["years"], False, NOT_RECORDED, True, NOT_RECORDED, True)
        country_value = memory["country"]
        derived = [
            html.P("What the model also receives", className="derived-title"),
            html.Ul(
                [
                    html.Li(
                        [
                            "Region: ",
                            html.Strong(lookups.country_region[country_value]),
                            " · Subregion: ",
                            html.Strong(lookups.country_subregion[country_value]),
                            " (from country of birth).",
                        ]
                    ),
                    html.Li(
                        [
                            "Language used at home: ",
                            html.Strong(lookups.country_language[country_value]),
                            " — the language paired with this country during data cleaning (not recorded in this table).",
                        ]
                    ),
                    html.Li(["English proficiency: ", html.Strong(NOT_SPECIFIED), " (not recorded in this table)."]),
                ]
            ),
        ]
    return (*values, memory, derived)


# ---------------------------------------------------------------------------
# Run the prediction and render the results
# ---------------------------------------------------------------------------
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


def _served_by(outcome: PredictionOutcome) -> html.Div:
    if outcome.served_by == "api":
        text = ["Served by the FastAPI inference service at ", html.Code(outcome.api_url or "")]
        kind = "ok"
    elif outcome.note:
        text = [outcome.note]
        kind = "warn"
    else:
        text = [
            "Served by the bundled copy of the model (no inference API configured — set ",
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
        ("Population group", _sentence_case(describe_group(outcome)), None),
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
        ("Region · subregion", f"{model_input[REGION]} · {model_input[SUBREGION]}", derived),
        ("Age group · sex", f"{age_label(request.age)} · {sex_label(request.sex)}", None),
        ("Condition", request.condition, None),
        (
            "Group population (2021 Census)",
            f"{outcome.group_population:,.0f} people" if outcome.group_population else "Not published",
            None,
        ),
        (
            "Model training status",
            {
                "test": "Held-out test group — not seen during training",
                "train": "Training group — seen during training",
            }.get(outcome.split or "", "Not in the published data"),
            None,
        ),
    ]
    items = []
    for label, value, tag in rows:
        items.append(html.Dt(label))
        items.append(html.Dd([value, html.Span("derived", className="tag") if tag else None]))
    return html.Dl(items, className="summary-list")


def render_outcome(outcome: PredictionOutcome) -> html.Div:
    request = outcome.request
    risk = outcome.risk
    if outcome.observed is not None:
        observed_tile = ui.stat_tile(
            "Published prevalence (census)",
            ui.fmt_pct(outcome.observed),
            f"AIHW 2021 value for this group of {outcome.group_population:,.0f} people",
        )
    else:
        observed_tile = ui.stat_tile(
            "Published prevalence (census)",
            "Not published",
            "AIHW suppressed this cell (n.p.) or the group is not in the tables; the model fills the gap",
        )
    if risk.ratio is not None and outcome.benchmark is not None:
        risk_detail = (
            f"{risk.ratio:.1f}× the benchmark of {outcome.benchmark:.1f}%: {sex_label(request.sex).lower()} aged "
            f"{age_label(request.age)} with the same condition, {benchmark_scope(outcome.source)}"
        )
    else:
        risk_detail = "No published benchmark for this age, sex and condition."
    tiles = html.Div(
        [
            ui.stat_tile(
                "Predicted prevalence",
                ui.fmt_pct(outcome.predicted),
                "XGBoost estimate for this group",
                hero=True,
                class_name="tile-accent",
            ),
            observed_tile,
            ui.stat_tile("Relative risk level", risk.label, risk_detail, status=risk.status, icon=risk.icon),
        ],
        className="tiles tiles-3",
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

    return html.Div(
        [
            ui.disclaimer(),
            _served_by(outcome),
            html.H2("2 · Results", className="section-title"),
            tiles,
            html.Div(
                [
                    ui.card("Demographic summary", _summary(outcome), subtitle="What the model was asked about"),
                    ui.card(
                        "Health insights",
                        html.Ul([html.Li(text) for text in outcome.insights], className="insight-list"),
                        subtitle="Generated from the prediction and the published census data",
                    ),
                ],
                className="grid-2",
            ),
            html.Div(
                [
                    ui.card(
                        f"{short_condition(request.condition)} across age groups",
                        ui.graph(figures.age_trajectory(outcome.trajectory, request.sex, request.age)),
                        html.P(
                            "Lines are model estimates; open circles are published census values where AIHW released them.",
                            className="card-foot",
                        ),
                        ui.table_view(trajectory_table),
                        subtitle="Same group, every age group and sex",
                    ),
                    ui.card(
                        "Condition profile for this group",
                        ui.graph(figures.condition_profile(profile_rows, request.condition)),
                        html.P(
                            "Blue is the condition you selected; open circles are published census values.",
                            className="card-foot",
                        ),
                        ui.table_view(profile_table),
                        subtitle=f"Predicted prevalence of each specific condition · {sex_label(request.sex).lower()}, "
                        f"aged {age_label(request.age)}",
                    ),
                ],
                className="grid-2",
            ),
            html.Details(
                [
                    html.Summary("Show the exact input sent to the model"),
                    html.Pre(
                        json.dumps(to_api_record(outcome.model_input), indent=2, ensure_ascii=False),
                        className="code-block",
                    ),
                ],
                className="table-view",
            ),
        ]
    )
