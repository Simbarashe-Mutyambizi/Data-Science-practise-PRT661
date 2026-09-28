"""The Dash app starts, registers its pages, and every callback runs for every option."""

import dash
import pytest

import app as dashboard  # noqa: F401  (instantiates the Dash app and registers pages)
from lthc_dashboard.analysis import DIMENSIONS


def test_pages_registered():
    paths = {page["path"] for page in dash.page_registry.values()}
    assert paths == {"/", "/explore", "/predict", "/model"}


def test_static_pages_build():
    from pages import model, overview

    assert overview.layout() is not None
    assert model.layout() is not None


@pytest.mark.parametrize("dimension", list(DIMENSIONS))
@pytest.mark.parametrize("age", ["all", "65 and over"])
def test_explore_callback_every_dimension(dimension, age):
    from pages.explore import update_explore

    outputs = update_explore(dimension, "Arthritis", age, "Persons", "both", 1000)
    assert len(outputs) == 13
    title, _subtitle, figure, style = outputs[0], outputs[1], outputs[2], outputs[3]
    assert "Arthritis" in title
    assert figure.data
    assert style["height"].endswith("px")


@pytest.mark.parametrize("mode", ["country", "language"])
def test_predict_callbacks(mode):
    from pages.predict import DEFAULTS, NOT_RECORDED, predict, sync_fields

    synced = sync_fields(mode, DEFAULTS["country"], DEFAULTS["years"], NOT_RECORDED, NOT_RECORDED, dict(DEFAULTS))
    country, _, years, _, language, _, english, _, memory, _derived = synced
    result = predict(1, mode, country, years, language, english, "45–64", "Male", "Asthma", memory)
    assert result is not None
    assert "Please check the form" not in str(result)


def test_predict_callback_reports_invalid_input():
    from pages.predict import DEFAULTS, predict

    result = predict(
        1, "country", "Atlantis", "More than 10 years", None, None, "45–64", "Male", "Asthma", dict(DEFAULTS)
    )
    assert "Please check the form" in str(result)
