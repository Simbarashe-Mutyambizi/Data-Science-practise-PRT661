"""The bundled model files reproduce the evaluation reported on the Machine_Learning branch."""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lthc_dashboard.config import MODEL_FEATURES, TARGET
from lthc_dashboard.model import to_api_record


def test_reproduces_reported_test_metrics(data, model):
    test = data.df[data.df["split"] == "test"]
    raw = model.model.predict(model.preprocessor.transform(test[MODEL_FEATURES]))  # unclipped, as in the notebook
    assert mean_absolute_error(test[TARGET], raw) == pytest.approx(1.834, abs=0.0005)
    assert np.sqrt(mean_squared_error(test[TARGET], raw)) == pytest.approx(2.840, abs=0.0005)
    assert r2_score(test[TARGET], raw) == pytest.approx(0.965, abs=0.0005)


def test_predictions_are_clipped_to_percent_range(data, model):
    predictions = model.predict_frame(data.df)
    assert predictions.min() >= 0
    assert predictions.max() <= 100


def test_unseen_inputs_are_reported(data, model):
    row = data.df[data.df["Language used at home"] == "Catalan"].iloc[0]
    record = to_api_record({column: row[column] for column in MODEL_FEATURES})
    assert set(model.unseen_inputs(record)) == {"country_of_birth", "language"}


def test_known_inputs_have_no_warning(data, model):
    row = data.df[data.df["split"] == "train"].iloc[0]
    record = to_api_record({column: row[column] for column in MODEL_FEATURES})
    assert model.unseen_inputs(record) == []
