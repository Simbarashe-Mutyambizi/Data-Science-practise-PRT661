from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lthc_dashboard.config import WARNING_TEXT  # noqa: E402
from lthc_dashboard.model import get_model  # noqa: E402

app = FastAPI(
    title="LTHC Population Prevalence Prediction API",
    description=(
        "Predicts the age-specific percentage of a population group reporting a "
        "long-term health condition (AIHW 2021 Census, CALD Australians). "
        "Population-level estimates only."
    ),
    version="1.1",
)

model = get_model()


class LTHCInput(BaseModel):
    country_of_birth: str
    years_in_australia: str
    age_group: str
    sex: str
    lthc: str
    language: str
    english_proficiency: str
    region: str
    subregion: str


class LTHCBatchInput(BaseModel):
    records: List[LTHCInput] = Field(..., min_length=1, max_length=500)


INTERPRETATION = "Estimated population-level prevalence"


@app.get("/")
def root():
    return {"message": "LTHC Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "running", "model": "LTHC population prevalence model"}


@app.post("/predict")
def predict(data: LTHCInput):
    record = data.model_dump()
    prediction = model.predict_records([record])[0]
    return {
        "predicted_prevalence": prediction,
        "unit": "percent",
        "interpretation": INTERPRETATION,
        "warning": WARNING_TEXT,
        "unseen_inputs": model.unseen_inputs(record),
    }


@app.post("/predict/batch")
def predict_batch(batch: LTHCBatchInput):
    records = [item.model_dump() for item in batch.records]
    predictions = model.predict_records(records)
    return {
        "predictions": [
            {"predicted_prevalence": value, "unseen_inputs": model.unseen_inputs(record)}
            for value, record in zip(predictions, records, strict=True)
        ],
        "unit": "percent",
        "interpretation": INTERPRETATION,
        "warning": WARNING_TEXT,
    }
