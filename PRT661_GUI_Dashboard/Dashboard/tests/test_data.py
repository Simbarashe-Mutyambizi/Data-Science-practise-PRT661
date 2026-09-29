import pytest

from lthc_dashboard import analysis
from lthc_dashboard.config import AGE, COUNTRY, LANGUAGE, LTHC, REGION, SEX
from lthc_dashboard.data import dataset_summary, select_rows, weighted_prevalence


def test_dataset_shape_and_groups(data):
    summary = dataset_summary(data.df)
    assert summary["rows"] == 16_184
    assert summary["groups"] == 2_717
    assert summary["conditions"] == 13


def test_split_matches_saved_training_split(data):
    df = data.df
    assert (df["split"] == "train").sum() == 13_060
    assert (df["split"] == "test").sum() == 3_124
    train_groups = set(df.loc[df["split"] == "train", "group_key"])
    test_groups = set(df.loc[df["split"] == "test", "group_key"])
    assert len(train_groups) == 2_173
    assert len(test_groups) == 544
    assert not train_groups & test_groups


def test_every_row_comes_from_exactly_one_census_table(data):
    df = data.df
    assert set(df["source"]) == {"country", "language"}
    country_rows = df[df["source"] == "country"]
    language_rows = df[df["source"] == "language"]
    assert (country_rows["Proficiency in spoken English"] == "Not specified").all()
    assert (language_rows["Years spent in Australia"] == "Not specified").all()
    assert country_rows.groupby(COUNTRY)[LANGUAGE].nunique().max() == 1
    assert language_rows.groupby(LANGUAGE)[COUNTRY].nunique().max() == 1


def test_weighted_prevalence_matches_report_sex_figures(data):
    table = analysis.prevalence_by_sex(data.df).set_index([LTHC, SEX])["prevalence"].round(2)
    assert table[("Arthritis", "Male")] == pytest.approx(5.67)
    assert table[("Mental health condition", "Female")] == pytest.approx(5.99)
    assert table[("Mental health condition", "Male")] == pytest.approx(3.90)
    assert table[("Mental health condition", "Persons")] == pytest.approx(5.13)
    assert table[("Diabetes", "Male")] == pytest.approx(6.67)
    assert table[("Diabetes", "Female")] == pytest.approx(5.33)
    assert table[("Heart disease or stroke", "Male")] == pytest.approx(5.33)
    assert table[("Heart disease or stroke", "Female")] == pytest.approx(3.18)


def test_weighted_prevalence_matches_report_female_arthritis_by_age(data):
    rows = select_rows(data.df, conditions=["Arthritis"], sexes=["Female"])
    table = weighted_prevalence(rows, [AGE]).set_index(AGE)["prevalence"].round(2)
    assert table["00–44"] == pytest.approx(0.93)
    assert table["45–64"] == pytest.approx(10.70)
    assert table["65 and over"] == pytest.approx(33.03)


def test_region_heatmap_persons_rule(data):
    arthritis = select_rows(data.df, conditions=["Arthritis"])
    pooled = weighted_prevalence(arthritis, [REGION]).set_index(REGION)["prevalence"]
    assert pooled["Southern and Eastern Europe"] == pytest.approx(16.26, abs=0.005)
    persons = analysis.prevalence_by_region(data.df).query(f"`{LTHC}` == 'Arthritis'").set_index(REGION)["prevalence"]
    assert persons["Southern and Eastern Europe"] == pytest.approx(15.54, abs=0.005)
