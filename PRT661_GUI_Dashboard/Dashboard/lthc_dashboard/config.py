from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LTHC_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.environ.get("LTHC_MODELS_DIR", BASE_DIR / "models"))

DATA_FILE = DATA_DIR / "LTHC_phase2_cleaned.csv"
MODEL_EVALUATION_FILE = DATA_DIR / "model_evaluation.csv"
OVERFITTING_FILE = DATA_DIR / "overfitting_analysis.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "xgboost_feature_importance.csv"
FINAL_PREDICTIONS_FILE = DATA_DIR / "final_predictions.csv"

PREPROCESSOR_FILE = MODELS_DIR / "lthc_preprocessor.pkl"
MODEL_FILE = MODELS_DIR / "xgboost_regression_model.pkl"

API_URL = os.environ.get("LTHC_API_URL", "").strip().rstrip("/")
API_TIMEOUT_SECONDS = float(os.environ.get("LTHC_API_TIMEOUT", "10"))

COUNTRY = "Country of birth of person"
YEARS = "Years spent in Australia"
AGE = "Age group"
SEX = "Sex"
LTHC = "Long-term health condition (LTHC)"
CASES = "Number of people reporting LTHC(s)"
POPULATION = "Population"
TARGET = "Age-specific percentage of population reporting LTHC(s)"
LANGUAGE = "Language used at home"
ENGLISH = "Proficiency in spoken English"
REGION = "Region_class"
SUBREGION = "Subregion_class"

NOT_SPECIFIED = "Not specified"

MODEL_FEATURES = [COUNTRY, YEARS, AGE, SEX, LANGUAGE, ENGLISH, REGION, SUBREGION, LTHC]
GROUP_COLUMNS = [COUNTRY, YEARS, AGE, SEX, LANGUAGE, ENGLISH, REGION, SUBREGION]
SPLIT_TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

API_FIELDS = {
    "country_of_birth": COUNTRY,
    "years_in_australia": YEARS,
    "age_group": AGE,
    "sex": SEX,
    "lthc": LTHC,
    "language": LANGUAGE,
    "english_proficiency": ENGLISH,
    "region": REGION,
    "subregion": SUBREGION,
}

AGE_GROUPS = ["00–44", "45–64", "65 and over"]
AGE_LABELS = {"00–44": "0–44", "45–64": "45–64", "65 and over": "65 and over"}

SEXES = ["Female", "Male", "Persons"]
SEX_LABELS = {"Female": "Female", "Male": "Male", "Persons": "All people"}

YEARS_OPTIONS = ["0–10 years", "More than 10 years"]
ENGLISH_OPTIONS = ["Very well or well", "Not well or Not at all"]

SPECIFIC_CONDITIONS = [
    "Arthritis",
    "Asthma",
    "Cancer",
    "Dementia",
    "Diabetes",
    "Heart disease",
    "Kidney disease",
    "Lung condition",
    "Mental health condition",
    "Stroke",
]
COMPOSITE_CONDITIONS = [
    "Heart disease or stroke",
    "Any other long-term health condition(s)",
    "One or more long-term health condition(s)",
]
ALL_CONDITIONS = SPECIFIC_CONDITIONS + COMPOSITE_CONDITIONS
COMMON_CONDITIONS = [
    "Arthritis",
    "Asthma",
    "Diabetes",
    "Mental health condition",
    "Heart disease or stroke",
]
ANY_CONDITION = "One or more long-term health condition(s)"

CONDITION_SHORT = {
    "Any other long-term health condition(s)": "Any other condition",
    "One or more long-term health condition(s)": "One or more conditions",
    "Mental health condition": "Mental health",
    "Heart disease or stroke": "Heart disease or stroke",
}

SOURCE_COUNTRY = "country"
SOURCE_LANGUAGE = "language"
SOURCE_LABELS = {
    SOURCE_COUNTRY: "Country of birth table (AIHW Table S10)",
    SOURCE_LANGUAGE: "Language used at home table (AIHW Table S13)",
}

WARNING_TEXT = (
    "This is a population-level estimate and is not an individual diagnosis or personal disease-risk prediction."
)


def short_condition(name: str) -> str:
    return CONDITION_SHORT.get(name, name)


def age_label(value: str) -> str:
    return AGE_LABELS.get(value, value)


def sex_label(value: str) -> str:
    return SEX_LABELS.get(value, value)
