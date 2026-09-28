"""Model & methods: performance on held-out groups, what drives predictions, limitations, ethics."""

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
    return merged


def error_insight() -> str:
    """Describe where the deployed model's errors are largest, from the held-out test groups."""
    df = get_data().df
    test = df[df["split"] == "test"].copy()
    test["abs_error"] = (get_model().predict_frame(test) - test[TARGET]).abs()
    low = test.loc[test[TARGET] <= 10, "abs_error"].mean()
    high_rows = test[test[TARGET] > 40]
    high = high_rows["abs_error"].mean()
    any_share = (high_rows[LTHC] == ANY_CONDITION).mean() * 100
    older_share = high_rows[AGE].isin(AGE_GROUPS[1:]).mean() * 100
    return (
        f"Points close to the line are accurate. The average miss is {low:.1f} percentage points for groups at or "
        f"below 10% prevalence and {high:.1f} points above 40%. Those high-prevalence cells are only "
        f"{len(high_rows) / len(test) * 100:.1f}% of the test set; {any_share:.0f}% of them are ‘one or more "
        f"conditions’ and {older_share:.0f}% are people aged 45 and over."
    )


PIPELINE = [
    (
        "Acquire",
        "AIHW ‘Chronic conditions among culturally and linguistically diverse Australians, 2021’: Table S10 "
        "(country of birth, years in Australia) and Table S13 (home language, English proficiency).",
    ),
    (
        "Clean",
        "Tables concatenated (120,978 cells); missing language and country filled with an AI-assisted "
        "language–country mapping; cells AIHW suppressed as ‘n.p.’ removed, leaving 16,184 published cells.",
    ),
    ("Explore", "Univariate and bivariate analysis with weighted prevalence (pooled cases ÷ pooled population)."),
    (
        "Prepare",
        "Nine categorical features, one-hot encoded to 402 columns. ‘Number of people reporting’ and "
        "‘Population’ dropped: the target is their ratio, so keeping them would leak the answer.",
    ),
    (
        "Split",
        "Group-aware split on the demographic group: 2,173 training and 544 test groups, no overlap, so the "
        "test score measures performance on groups the model has never seen.",
    ),
    ("Train", "Linear regression (baseline), decision tree, random forest and XGBoost regressors."),
    (
        "Evaluate",
        "MAE, RMSE and R² on the test groups, plus the train–test gap to check for overfitting. "
        "XGBoost selected: lowest RMSE and smallest R² gap.",
    ),
    (
        "Deploy",
        "FastAPI serves the XGBoost model; this Dash dashboard calls it. Both are set up to be hosted on Render.",
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

    tiles = html.Div(
        [
            ui.stat_tile(
                "R² on held-out groups",
                f"{xgb['R²']:.3f}",
                "Share of variation in prevalence explained",
                hero=True,
                class_name="tile-accent",
            ),
            ui.stat_tile("Mean absolute error", f"{xgb['MAE']:.2f} pp", "Average miss, in percentage points"),
            ui.stat_tile("Root mean squared error", f"{xgb['RMSE']:.2f} pp", "Penalises large misses more"),
            ui.stat_tile("Train–test R² gap", f"{xgb['R² Difference']:.3f}", "Small gap: little overfitting"),
        ],
        className="tiles",
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
    )
    importance_table = ui.data_table(
        importance.sort_values("Importance", ascending=False)
        .head(30)
        .assign(feature=lambda t: t["Feature"].map(figures.pretty_feature), share=lambda t: t["Importance"] * 100),
        [("feature", "Feature", None), ("share", "Share of importance", lambda v: f"{v:.2f}%")],
    )

    return html.Div(
        [
            ui.page_header(
                "How the model works and how well it performs",
                "An XGBoost regressor predicts the age-specific percentage of a population group reporting a "
                "long-term health condition. It was tested on demographic groups it never saw during training.",
                eyebrow="Model & methods",
            ),
            tiles,
            html.Div(
                [
                    ui.card(
                        "Four models compared",
                        comparison_table,
                        html.P(
                            f"Test set: {summary['test_rows']:,} census cells from 544 held-out groups · training set: "
                            f"{summary['train_rows']:,} cells. Bold row is the deployed model.",
                            className="card-foot",
                        ),
                        subtitle="Errors are in percentage points (pp); lower is better. R²: higher is better",
                    ),
                    ui.card(
                        "Little overfitting",
                        ui.graph(figures.r2_dumbbell(overfitting)),
                        ui.insight(
                            "Each model scores almost as well on unseen groups as on training groups. Linear regression "
                            "is the weakest because it cannot capture how age and condition combine."
                        ),
                        subtitle="R² on training groups vs held-out test groups",
                    ),
                ],
                className="grid-2",
            ),
            html.Div(
                [
                    ui.card(
                        "What drives the predictions",
                        ui.graph(figures.feature_importance(importance)),
                        ui.insight(
                            "Which condition is asked about and whether the group is aged 65 and over dominate. Region, "
                            "language and time in Australia refine the estimate."
                        ),
                        ui.table_view(importance_table, "Show the top 30 features as a table"),
                        subtitle="Top 15 of 402 one-hot features, share of XGBoost importance",
                    ),
                    ui.card(
                        "Predicted vs published prevalence",
                        ui.graph(figures.actual_vs_predicted(predictions)),
                        ui.insight(error_insight()),
                        subtitle=f"{len(predictions):,} census cells from held-out test groups",
                    ),
                ],
                className="grid-2",
            ),
            ui.card(
                "From census tables to predictions",
                html.Ol(
                    [html.Li([html.Strong(step), " — ", text]) for step, text in PIPELINE],
                    className="pipeline",
                ),
                subtitle="The training phase of the project workflow",
            ),
            html.Div(
                [
                    ui.card(
                        "Limitations",
                        html.Ul([html.Li(item) for item in LIMITATIONS], className="plain-list"),
                        subtitle="Read these before using any number from this dashboard",
                    ),
                    ui.card(
                        "Ethics and responsible use",
                        html.Dl(
                            sum([[html.Dt(title), html.Dd(text)] for title, text in ETHICS], []),
                            className="summary-list stacked",
                        ),
                        subtitle="Carried over from the project's ethics, privacy and security plan",
                    ),
                ],
                className="grid-2",
            ),
        ]
    )


def layout(**_kwargs):
    return _content()
