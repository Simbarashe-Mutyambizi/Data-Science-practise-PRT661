from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    AGE,
    AGE_GROUPS,
    ALL_CONDITIONS,
    ANY_CONDITION,
    COMPOSITE_CONDITIONS,
    COUNTRY,
    ENGLISH,
    ENGLISH_OPTIONS,
    LANGUAGE,
    LTHC,
    NOT_SPECIFIED,
    POPULATION,
    REGION,
    SEX,
    SEXES,
    SOURCE_COUNTRY,
    SOURCE_LANGUAGE,
    SPECIFIC_CONDITIONS,
    SUBREGION,
    TARGET,
    YEARS,
    YEARS_OPTIONS,
    age_label,
)
from .data import DashboardData, select_rows, weighted_prevalence
from .inference import InferenceClient
from .model import to_api_record

MODE_COUNTRY = SOURCE_COUNTRY
MODE_LANGUAGE = SOURCE_LANGUAGE

FIELD_LABELS = {
    "country_of_birth": "Country of birth",
    "years_in_australia": "Years in Australia",
    "age_group": "Age group",
    "sex": "Sex",
    "lthc": "Condition",
    "language": "Language used at home",
    "english_proficiency": "English proficiency",
    "region": "Region",
    "subregion": "Subregion",
}


@dataclass
class PredictionRequest:
    mode: str
    age: str
    sex: str
    condition: str
    country: str | None = None
    years: str | None = None
    language: str | None = None
    english: str | None = None


def validate_request(request: PredictionRequest, data: DashboardData) -> list[str]:
    lookups = data.lookups
    problems: list[str] = []
    if request.mode not in (MODE_COUNTRY, MODE_LANGUAGE):
        problems.append("Choose whether to describe the group by country of birth or by home language.")
    if request.age not in AGE_GROUPS:
        problems.append("Choose an age group.")
    if request.sex not in SEXES:
        problems.append("Choose a sex.")
    if request.condition not in ALL_CONDITIONS:
        problems.append("Choose a long-term health condition.")
    if request.mode == MODE_COUNTRY:
        if request.country not in lookups.country_language:
            problems.append("Choose a country of birth from the list.")
        if request.years not in YEARS_OPTIONS:
            problems.append("Choose how long the group has lived in Australia.")
    elif request.mode == MODE_LANGUAGE:
        if request.language not in lookups.language_country:
            problems.append("Choose a language used at home from the list.")
        if request.english not in ENGLISH_OPTIONS:
            problems.append("Choose a level of spoken English.")
    return problems


def build_model_input(request: PredictionRequest, data: DashboardData) -> dict[str, str]:
    lookups = data.lookups
    if request.mode == MODE_COUNTRY:
        country, years = request.country, request.years
        language, english = lookups.country_language[country], NOT_SPECIFIED
    else:
        language, english = request.language, request.english
        country, years = lookups.language_country[language], NOT_SPECIFIED
    return {
        COUNTRY: country,
        YEARS: years,
        AGE: request.age,
        SEX: request.sex,
        LANGUAGE: language,
        ENGLISH: english,
        REGION: lookups.country_region[country],
        SUBREGION: lookups.country_subregion[country],
        LTHC: request.condition,
    }


@dataclass(frozen=True)
class RiskLevel:
    key: str
    label: str
    icon: str
    status: str
    ratio: float | None


RISK_BANDS = [
    (0.8, RiskLevel("below", "Below average", "▼", "good", None)),
    (1.25, RiskLevel("average", "Around average", "●", "neutral", None)),
    (2.0, RiskLevel("above", "Above average", "▲", "serious", None)),
    (float("inf"), RiskLevel("well-above", "Well above average", "▲", "critical", None)),
]


def risk_level(predicted: float, benchmark: float | None) -> RiskLevel:
    if benchmark is None or benchmark <= 0:
        return RiskLevel("unknown", "No benchmark", "–", "neutral", None)
    ratio = predicted / benchmark
    for upper, level in RISK_BANDS:
        if ratio < upper:
            return RiskLevel(level.key, level.label, level.icon, level.status, ratio)
    raise AssertionError("unreachable")


def benchmark_prevalence(data: DashboardData, condition: str, age: str, sex: str, source: str) -> float | None:
    rows = select_rows(data.df, conditions=[condition], ages=[age], sexes=[sex], sources=[source])
    table = weighted_prevalence(rows)
    return float(table["prevalence"].iloc[0]) if len(table) else None


@dataclass
class PredictionOutcome:
    request: PredictionRequest
    model_input: dict[str, str]
    source: str
    predicted: float
    observed: float | None
    observed_cases: float | None
    group_population: float | None
    split: str | None
    benchmark: float | None
    risk: RiskLevel
    profile: pd.DataFrame
    trajectory: pd.DataFrame
    unseen_inputs: list[str]
    served_by: str
    api_url: str | None
    note: str | None
    insights: list[str] = field(default_factory=list)


def _scenario_inputs(model_input: dict[str, str]) -> list[dict[str, str]]:
    scenarios = [dict(model_input)]
    for condition in ALL_CONDITIONS:
        scenarios.append({**model_input, LTHC: condition})
    for age in AGE_GROUPS:
        for sex in SEXES:
            scenarios.append({**model_input, AGE: age, SEX: sex})
    unique, seen = [], set()
    for scenario in scenarios:
        key = tuple(sorted(scenario.items()))
        if key not in seen:
            seen.add(key)
            unique.append(scenario)
    return unique


def _observed_value(data: DashboardData, record: dict[str, str]) -> float | None:
    row = data.find_cell(record)
    return None if row is None else float(row[TARGET])


def run_prediction(request: PredictionRequest, data: DashboardData, client: InferenceClient) -> PredictionOutcome:
    model_input = build_model_input(request, data)
    scenarios = _scenario_inputs(model_input)
    batch = client.predict([to_api_record(s) for s in scenarios])
    predicted_by_key = {tuple(sorted(s.items())): v for s, v in zip(scenarios, batch.values, strict=True)}

    def predicted_for(record: dict[str, str]) -> float:
        return predicted_by_key[tuple(sorted(record.items()))]

    predicted = predicted_for(model_input)
    unseen = batch.unseen_inputs[0] if batch.unseen_inputs else []

    cell = data.find_cell(model_input)
    group = data.group_rows(model_input)
    group_population = float(group[POPULATION].iloc[0]) if len(group) else None
    split = str(group["split"].iloc[0]) if len(group) else None

    profile = pd.DataFrame(
        [
            {
                "condition": condition,
                "predicted": predicted_for({**model_input, LTHC: condition}),
                "observed": _observed_value(data, {**model_input, LTHC: condition}),
            }
            for condition in ALL_CONDITIONS
        ]
    )
    trajectory = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "predicted": predicted_for({**model_input, AGE: age, SEX: sex}),
                "observed": _observed_value(data, {**model_input, AGE: age, SEX: sex}),
            }
            for age in AGE_GROUPS
            for sex in SEXES
        ]
    )

    benchmark = benchmark_prevalence(data, request.condition, request.age, request.sex, request.mode)
    outcome = PredictionOutcome(
        request=request,
        model_input=model_input,
        source=request.mode,
        predicted=predicted,
        observed=None if cell is None else float(cell[TARGET]),
        observed_cases=None if cell is None else float(cell["Number of people reporting LTHC(s)"]),
        group_population=group_population,
        split=split,
        benchmark=benchmark,
        risk=risk_level(predicted, benchmark),
        profile=profile,
        trajectory=trajectory,
        unseen_inputs=unseen,
        served_by=batch.served_by,
        api_url=batch.api_url,
        note=batch.note,
    )
    outcome.insights = build_insights(outcome)
    return outcome


SEX_NOUNS = {"Female": "females", "Male": "males", "Persons": "people"}


def benchmark_scope(source: str) -> str:
    if source == MODE_COUNTRY:
        return "across all published country-of-birth groups"
    return "across all published home-language groups"


def describe_group(outcome: PredictionOutcome) -> str:
    request, model_input = outcome.request, outcome.model_input
    people = f"{SEX_NOUNS[request.sex]} aged {age_label(request.age)}"
    if request.mode == MODE_COUNTRY:
        years = "0–10 years" if model_input[YEARS] == "0–10 years" else "more than 10 years"
        return f"{people} born in {model_input[COUNTRY]} who have lived in Australia for {years}"
    english = model_input[ENGLISH].lower()
    return f"{people} who speak {model_input[LANGUAGE]} at home and speak English {english}"


CONDITION_PHRASES = {
    "Lung condition": "a lung condition",
    "Mental health condition": "a mental health condition",
    "Any other long-term health condition(s)": "another long-term health condition (outside the ten listed)",
    ANY_CONDITION: "one or more long-term health conditions",
}


def condition_phrase(condition: str) -> str:
    return CONDITION_PHRASES.get(condition, condition.lower())


def build_insights(outcome: PredictionOutcome) -> list[str]:
    request = outcome.request
    insights: list[str] = []
    group = describe_group(outcome)
    condition = condition_phrase(request.condition)

    headline = f"The model estimates that {outcome.predicted:.1f}% of {group} report {condition}."
    if outcome.benchmark is not None and outcome.risk.ratio is not None:
        headline += (
            f" That is {outcome.risk.ratio:.1f}× the average for {SEX_NOUNS[request.sex]} aged "
            f"{age_label(request.age)} {benchmark_scope(outcome.source)} ({outcome.benchmark:.1f}%)."
        )
    insights.append(headline)

    if outcome.observed is not None:
        difference = outcome.predicted - outcome.observed
        direction = "above" if difference > 0 else "below"
        sentence = (
            f"AIHW published {outcome.observed:.1f}% for this exact group"
            + (f" ({outcome.group_population:,.0f} people)" if outcome.group_population else "")
            + f"; the estimate is {abs(difference):.1f} percentage points {direction} it."
        )
        if outcome.split == "test":
            sentence += " This group was held out from training, so this is a genuine out-of-sample check."
        elif outcome.split == "train":
            sentence += " This group was part of the model's training data."
        insights.append(sentence)
    elif outcome.group_population:
        insights.append(
            f"AIHW did not publish a value for this condition in this group of {outcome.group_population:,.0f} "
            "people (small counts are suppressed as ‘n.p.’), so the model fills the gap."
        )
    else:
        insights.append(
            "This exact group does not appear in the published census tables (it was suppressed or has no "
            "residents), so there is no census figure to compare against."
        )

    ordered = outcome.trajectory[outcome.trajectory["sex"] == request.sex].set_index("age").reindex(AGE_GROUPS)
    young, old = ordered["predicted"].iloc[0], ordered["predicted"].iloc[-1]
    if old > young:
        insights.append(
            f"For this group, predicted prevalence rises from {young:.1f}% at 0–44 to {old:.1f}% at 65 and over."
        )
    else:
        insights.append(
            f"For this group, predicted prevalence does not rise with age ({young:.1f}% at 0–44, {old:.1f}% at 65 and over)."
        )

    same_age = outcome.trajectory[outcome.trajectory["age"] == request.age].set_index("sex")["predicted"]
    female, male = same_age.get("Female"), same_age.get("Male")
    if female is not None and male is not None and abs(female - male) >= 0.5:
        higher = "females" if female > male else "males"
        insights.append(
            f"At this age the estimate is higher for {higher} ({female:.1f}% of females vs {male:.1f}% of males)."
        )

    specific = outcome.profile[outcome.profile["condition"].isin(SPECIFIC_CONDITIONS)].sort_values(
        "predicted", ascending=False
    )
    top = [f"{row.condition.lower()} ({row.predicted:.1f}%)" for row in specific.head(3).itertuples()]
    listing = ", ".join(top[:-1]) + f" and {top[-1]}" if len(top) > 1 else top[0]
    insights.append(f"The most common specific conditions predicted for this group are {listing}.")
    any_row = outcome.profile[outcome.profile["condition"] == ANY_CONDITION]
    if not any_row.empty and request.condition != ANY_CONDITION:
        insights.append(
            f"Overall, {float(any_row['predicted'].iloc[0]):.1f}% of this group are predicted to report at least one long-term condition."
        )

    if outcome.unseen_inputs:
        insights.append(unseen_warning(outcome))
    return insights


def unseen_warning(outcome: PredictionOutcome) -> str:
    api_record = to_api_record(outcome.model_input)
    parts = [f"{FIELD_LABELS.get(f, f).lower()} ‘{api_record[f]}’" for f in outcome.unseen_inputs]
    listing = " and ".join(parts)
    return (
        f"Caution: the model never saw {listing} during training (it only appears in held-out groups), "
        "so the estimate leans on the other characteristics such as region, subregion, age, sex and condition."
    )


def is_composite(condition: str) -> bool:
    return condition in COMPOSITE_CONDITIONS
