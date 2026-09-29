import dash
import pytest

import app as dashboard  # noqa: F401


def test_pages_registered():
    paths = {page["path"] for page in dash.page_registry.values()}
    assert paths == {"/", "/analysis", "/predict", "/profile", "/model"}


def test_static_pages_build():
    from pages import model, overview

    assert overview.layout() is not None
    assert model.layout() is not None


ALL_AGES = ["0–44", "45–64", "65 and over"]


@pytest.mark.parametrize("condition", ["One or more long-term health condition(s)", "Diabetes", "Dementia"])
@pytest.mark.parametrize("sex", ["Persons", "Female"])
def test_analysis_callbacks(condition, sex):
    from pages import analysis

    kpis = analysis.update_kpis(condition, sex, ALL_AGES, [])
    assert len(kpis) == 12
    if condition == "Dementia" and sex != "Persons":
        assert kpis[0] == "–"
        return
    assert kpis[0] != "–"
    for callback, extra in [
        (analysis.update_overview, []),
        (analysis.update_geo, [1000]),
        (analysis.update_migration, []),
        (analysis.update_language, [1000]),
    ]:
        figures = callback(condition, sex, ALL_AGES, [], *extra)
        assert len(figures) == 3
        assert all(fig.data for fig in figures)
    rows, columns = analysis.update_risk_table(condition, sex, ALL_AGES, [])
    assert columns
    assert analysis.update_risk_chart(condition, sex, ALL_AGES, []).data
    data, columns = analysis.update_table(condition, sex, ALL_AGES, [], "language")
    assert data and columns


def test_analysis_region_filter_and_single_age():
    from pages import analysis

    figures = analysis.update_overview("Arthritis", "Male", ["65 and over"], ["North-West Europe"])
    assert all(fig.data for fig in figures)


def test_profile_requires_age_and_sex():
    from pages import profile

    _, error = profile.show_results(1, "Any", "Any", "Any", None, None, "Any", "Any")
    assert "age group" in error and "sex" in error


def test_profile_results_for_a_group():
    from pages import profile

    results, error = profile.show_results(
        1, "Any", "Any", "Any", "45–64", "Female", "More than 10 years", "Very well or well"
    )
    assert error == ""
    text = str(results)
    assert "Women aged 45–64" in text
    assert "38%" in text


def test_profile_country_and_language_lists_narrow():
    from pages import profile

    countries, value = profile.update_country_options("South-East Asia", "Any")
    names = {option["value"] for option in countries}
    assert "Vietnam" in names and "Italy" not in names
    languages, _ = profile.update_language_options("South-East Asia", "Vietnam", "Any")
    assert "Vietnamese" in {option["value"] for option in languages}


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
