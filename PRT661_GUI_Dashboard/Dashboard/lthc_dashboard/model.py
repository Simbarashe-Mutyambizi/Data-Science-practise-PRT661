from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import API_FIELDS, MODEL_FEATURES, MODEL_FILE, PREPROCESSOR_FILE

COLUMN_TO_FIELD = {column: field for field, column in API_FIELDS.items()}


class LTHCModel:
    def __init__(
        self,
        preprocessor_path: Path | str = PREPROCESSOR_FILE,
        model_path: Path | str = MODEL_FILE,
    ) -> None:
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)
        encoder = self.preprocessor.named_transformers_["categorical"]
        encoded_columns = self.preprocessor.transformers_[0][2]
        self.known_categories: dict[str, set[str]] = {
            column: set(categories) for column, categories in zip(encoded_columns, encoder.categories_, strict=True)
        }

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        processed = self.preprocessor.transform(frame[MODEL_FEATURES])
        predictions = self.model.predict(processed)
        return np.clip(predictions.astype(float), 0.0, 100.0)

    def predict_records(self, records: list[dict[str, str]]) -> list[float]:
        frame = pd.DataFrame([to_columns(record) for record in records])
        return [round(float(value), 2) for value in self.predict_frame(frame)]

    def unseen_inputs(self, record: dict[str, str]) -> list[str]:
        columns = to_columns(record)
        return [
            COLUMN_TO_FIELD[column]
            for column in MODEL_FEATURES
            if columns[column] not in self.known_categories.get(column, set())
        ]


def to_columns(record: dict[str, str]) -> dict[str, str]:
    if all(column in record for column in MODEL_FEATURES):
        return {column: record[column] for column in MODEL_FEATURES}
    missing = [field for field in API_FIELDS if field not in record]
    if missing:
        raise KeyError(f"Missing input field(s): {', '.join(missing)}")
    return {column: record[field] for field, column in API_FIELDS.items()}


def to_api_record(columns: dict[str, str]) -> dict[str, str]:
    return {field: columns[column] for field, column in API_FIELDS.items()}


@lru_cache(maxsize=1)
def get_model() -> LTHCModel:
    return LTHCModel()
