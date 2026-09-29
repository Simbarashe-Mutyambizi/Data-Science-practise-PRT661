from __future__ import annotations

from functools import lru_cache

import dash
import pandas as pd
from dash import html

from lthc_dashboard import components as ui, figures
from lthc_dashboard.config import (
    AGE,
    AGE_GROUPS,
    ANY_CONDITION,
    FEATURE_IMPORTANCE_FILE,
    FINAL_PREDICTIONS_FILE,
    LTHC,
    MODEL_EVALUATION_FILE,
    OVERFITTING_FILE,
    TARGET,
)
from lthc_dashboard.data import dataset_summary, get_data
from lthc_dashboard.model import get_model

dash.register_page(__name__, path="/model", name="Model & methods", title="Model & methods · LTHC dashboard", order=3)


def _metrics() -> pd.DataFrame:
    evaluation = pd.read_csv(MODEL_EVALUATION_FILE)
    overfitting = pd.read_csv(OVERFITTING_FILE)
    merged = evaluation.merge(overfitting[["Model", "Train R²", "R² Difference"]], on="Model")
    return merged.sort_values("RMSE").reset_index(drop=True)


def error_insight() -> str:
    df = get_data().df
    test = df[df["split"] == "test"].copy()
    test["abs_error"] = (get_model().predict_frame(test) - test[TARGET]).abs()
    low = test.loc[test[TARGET] <= 10, "abs_error"].mean()
    high_rows = test[test[TARGET] > 40]
    high = high_rows["abs_error"].mean()
    any_share = (high_rows[LTHC] == ANY_CONDITION).mean() * 100
    older_share = high_rows[AGE].isin(AGE_GROUPS[1:]).mean() * 100
    return (
        f"Average miss: {low:.1f} points where prevalence is 10% or less, {high:.1f} points above 40% "
        f"({len(high_rows) / len(test) * 100:.1f}% of test cells; {any_share:.0f}% of them ‘one or more conditions’, "
        f"{older_share:.0f}% aged 45 and over)."
    )


PIPELINE = [
    (
        "Acquire",
        "AIHW 2021 CALD Tables S10 and S13",
        "S10: country of birth and years in Australia. S13: language used at home and spoken English.",
    ),
    (
        "Clean",
        "120,978 cells → 16,184 published",
        "Tables combined, language–country gaps filled and suppressed ‘n.p.’ cells removed.",
    ),
    ("Explore", "Weighted prevalence by group", "Univariate and bivariate analysis: pooled cases ÷ pooled population."),
    (
        "Prepare",
        "Nine categories → 402 one-hot features",
        "Count columns dropped so the target cannot leak into the features.",
    ),
    ("Split", "2,173 training · 544 test groups", "Group-aware split: no population group is in both sets."),
    ("Train", "Four regressors, baseline to XGBoost", "Linear regression, decision tree, random forest and XGBoost."),
    ("Evaluate", "MAE, RMSE, R² and train–test gap", "Scored on the held-out test groups; XGBoost selected."),
    (
        "Deploy",
        "FastAPI service + this dashboard",
        "FastAPI serves the XGBoost model and this dashboard calls it; both run on Render.",
    ),
]

LIMITATIONS = [
    "Groups, not people. Each row is an aggregated census cell, so predictions describe the share of a group "
    "reporting a condition — never an individual's risk.",
    "Self-reported conditions. The census counts people who say a doctor or nurse told them they have the "
    "condition, so under-diagnosis and reporting differences between communities carry through.",
    "Suppressed cells. AIHW hides small counts (‘n.p.’). Only 13.4% of cells were published, so small communities "
    "and Male/Female splits are under-represented; ‘All people’ totals are available far more often.",
    "Separate tables. Country of birth and home language are never observed together. The model learned each table "
    "separately, so the Predict page only allows the combinations the census publishes.",
    "Approximate language–country pairings. During cleaning, every country was paired with one language and every "
    "language with one country (for example China with Uygur, India with Kannada, Arabic with Saudi Arabia). "
    "The model was trained on these pairings, so the dashboard uses them unchanged, but region-level results for "
    "language-table rows inherit their approximations.",
    "Crude all-age rates. Comparisons that pool age groups are driven by age structure (for example, many "
    "European-born Australians are older). Use an age filter for like-with-like comparisons.",
    "Raw regression output can fall slightly below 0% (123 of 3,124 test predictions). The API clips "
    "predictions to the 0–100% range.",
]

ETHICS = [
    (
        "Population level only",
        "Every result carries a notice that it is a population estimate, not a diagnosis. "
        "The Assessment 2 risk register flags misreading it as personal advice as the main risk.",
    ),
    (
        "Human oversight",
        "Funding, service and care decisions should stay with qualified health professionals, using "
        "this as one input among many.",
    ),
    (
        "Transparency",
        "Feature importance shows what drives predictions, and every prediction shows the exact input "
        "sent to the model and, where available, the published census value to compare against.",
    ),
    (
        "Privacy",
        "Only de-identified, aggregated AIHW counts are used. The dashboard never goes below the level AIHW "
        "publishes, and respects its suppression of small cells.",
    ),
    (
        "Responsible AI use",
        "The language–country mapping was generated with AI assistance and reviewed by the team; "
        "it is kept in the repository for inspection and correction.",
    ),
]


@lru_cache(maxsize=1)
def _content() -> html.Div:
    metrics = _metrics()
    importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    predictions = pd.read_csv(FINAL_PREDICTIONS_FILE)
    overfitting = pd.read_csv(OVERFITTING_FILE)
    summary = dataset_summary(get_data().df)
    xgb = metrics.set_index("Model").loc["XGBoost"]
    best_index = int(metrics.index[metrics["Model"] == "XGBoost"][0])

    kpis = ui.kpi_row(
        [
            ui.kpi("R² on held-out groups", f"{xgb['R²']:.3f}", "Share of the variation explained", accent="teal"),
            ui.kpi("Mean absolute error", f"{xgb['MAE']:.2f} pp", "Average miss, percentage points", accent="navy"),
            ui.kpi("Root mean squared error", f"{xgb['RMSE']:.2f} pp", "Penalises large misses more", accent="navy"),
            ui.kpi("Train–test R² gap", f"{xgb['R² Difference']:.3f}", "Small gap: little overfitting", accent="amber"),
            ui.kpi("Test groups", "544", f"{summary['test_rows']:,} cells never seen in training", accent="navy"),
            ui.kpi("Features", "402", "Nine categories, one-hot encoded", accent="navy"),
        ]
    )

    comparison_table = ui.data_table(
        metrics,
        [
            ("Model", "Model", None),
            ("MAE", "MAE (pp)", lambda v: f"{v:.3f}"),
            ("RMSE", "RMSE (pp)", lambda v: f"{v:.3f}"),
            ("R²", "Test R²", lambda v: f"{v:.3f}"),
            ("Train R²", "Train R²", lambda v: f"{v:.3f}"),
            ("R² Difference", "Gap", lambda v: f"{v:.3f}"),
        ],
        bold_rows=[best_index],
        tag_rows={best_index: "Deployed"},
    )
    importance_table = ui.data_table(
        importance.sort_values("Importance", ascending=False)
        .head(30)
        .assign(feature=lambda t: t["Feature"].map(figures.pretty_feature), share=lambda t: t["Importance"] * 100),
        [("feature", "Feature", None), ("share", "Share of importance", lambda v: f"{v:.2f}%")],
    )

    def lead_in(text: str) -> list:
        head, _, rest = text.partition(". ")
        return [html.Strong(head + ". "), rest]

    limits_control, limits_panes = ui.switch(
        "model-limits",
        [
            (
                "Limitations",
                "limitations",
                html.Ul([html.Li(lead_in(item)) for item in LIMITATIONS], className="plain-list"),
            ),
            (
                "Ethics",
                "ethics",
                html.Dl(sum([[html.Dt(title), html.Dd(text)] for title, text in ETHICS], []), className="ethics-list"),
            ),
        ],
    )

    return ui.screen(
        "model",
        "How the model works and how well it performs",
        "An XGBoost regressor predicts the share of a population group reporting a long-term condition; it was "
        "tested on demographic groups it never saw in training and is served by a FastAPI inference service.",
        kpis,
        html.Div(
            [
                ui.card(
                    "Four models compared",
                    html.Div(
                        [
                            comparison_table,
                            ui.insight(
                                "XGBoost: lowest RMSE, highest test R² and a small train–test gap, so it was "
                                "deployed. Random forest is a close second (lowest MAE).",
                                label="Model choice",
                            ),
                        ],
                        className="card-body scroll",
                    ),
                    subtitle="Errors in percentage points (pp): lower is better. R²: higher is better.",
                ),
                ui.chart_card(
                    "Little overfitting",
                    figures.r2_dumbbell(overfitting),
                    subtitle="R² on training groups compared with held-out test groups",
                    key="model-overfit",
                    foot=[
                        html.Strong("Key finding: "),
                        "every model scores almost as well on unseen groups as on training groups.",
                    ],
                ),
                ui.chart_card(
                    "Predicted vs published prevalence",
                    figures.actual_vs_predicted(predictions),
                    subtitle=f"{len(predictions):,} census cells from held-out test groups",
                    key="model-scatter",
                    foot=ui.emphasise_numbers(error_insight()),
                ),
                ui.chart_card(
                    "What drives the predictions",
                    figures.feature_importance(importance, top_n=10),
                    importance_table,
                    key="model-importance",
                    subtitle="Top 10 of 402 features by share of XGBoost importance. The condition asked about and "
                    "age 65+ dominate; region, language and residence refine the estimate.",
                ),
                ui.card(
                    "From census tables to predictions",
                    html.Div(
                        html.Ol(
                            [
                                html.Li(
                                    [
                                        html.Span(str(i), className="step-num"),
                                        html.Span(step, className="pipeline-step"),
                                        html.Span(summary_text, className="pipeline-text"),
                                    ],
                                    title=detail,
                                    className="deploy" if step == "Deploy" else None,
                                )
                                for i, (step, summary_text, detail) in enumerate(PIPELINE, start=1)
                            ],
                            className="pipeline",
                        ),
                        className="card-body scroll",
                    ),
                    subtitle="From raw AIHW tables to the deployed service; hover a step for detail",
                ),
                ui.card(
                    "Limitations and ethics",
                    limits_panes,
                    subtitle="Read these before using any number from this dashboard",
                    aside=limits_control,
                ),
            ],
            className="grid-3x2",
        ),
        action=("Try a prediction", "/predict"),
    )


def layout(**_kwargs):
    return _content()
