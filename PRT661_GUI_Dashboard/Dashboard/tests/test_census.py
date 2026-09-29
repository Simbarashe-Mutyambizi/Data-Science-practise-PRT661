import pytest

from lthc_dashboard.census_analysis import (
    ANY_LTHC,
    age_standardised,
    estimate_profile,
    load_data,
    prevalence,
    risk_table,
    split_views,
    standard_weights,
)


@pytest.fixture(scope="module")
def views():
    return split_views(load_data())


def test_both_census_views_load(views):
    migration, language = views
    assert len(migration) == 63_180
    assert len(language) == 57_798
    assert (migration["years"] != "Not specified").all()
    assert (language["proficiency"] != "Not specified").all()


def test_overall_prevalence_matches_the_dashboard_kpis(views):
    migration, _ = views
    total = prevalence(migration[(migration["sex"] == "Persons") & (migration["condition"] == ANY_LTHC)]).iloc[0]
    assert total["population"] == 6_611_949
    assert total["rate"] == pytest.approx(30.6, abs=0.05)


def test_age_standardisation_weights_sum_to_one(views):
    migration, _ = views
    weights = standard_weights(migration)
    assert sum(weights.values()) == pytest.approx(1.0)
    subset = migration[(migration["sex"] == "Persons") & (migration["condition"] == ANY_LTHC)]
    regions = age_standardised(subset, "region", weights)
    assert regions["asr"].between(0, 100).all()


def test_risk_table_flags(views):
    migration, _ = views
    subset = migration[(migration["sex"] == "Persons") & (migration["condition"] == "Diabetes")]
    table = risk_table(subset, ["country", "age"], ["age"], min_population=1000)
    assert set(table["risk_level"]) <= {"Elevated", "Typical", "Lower"}
    elevated = table[table["risk_level"] == "Elevated"]
    assert (elevated["rate_ratio"] >= 1.2).all() and (elevated["z_score"] >= 3).all()


def test_profile_estimate_reproduces_the_original_app(views):
    migration, language = views
    result, info = estimate_profile(
        migration, language, age="45–64", sex="Female", years="More than 10 years", proficiency="Very well or well"
    )
    assert result.iloc[0]["likelihood"] == pytest.approx(38.0, abs=0.5)
    assert info["confidence"] == "High"
    assert result["likelihood"].between(0, 99).all()
