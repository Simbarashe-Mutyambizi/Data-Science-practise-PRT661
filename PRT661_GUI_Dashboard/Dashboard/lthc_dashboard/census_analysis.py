from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

RAW_FILE = DATA_DIR / "Cleaned_data_final.xlsx"
CACHE_FILE = DATA_DIR / "census_tables_clean.csv.gz"

COLUMN_MAP = {
    "Country of birth of person": "country",
    "Years spent in Australia": "years",
    "Age group": "age",
    "Sex": "sex",
    "Long-term health condition (LTHC)": "condition",
    "Number of people reporting LTHC(s)": "cases",
    "Population": "population",
    "Age-specific percentage of population reporting LTHC(s)": "rate",
    "Language used at home": "language",
    "Proficiency in spoken English": "proficiency",
    "Region_class": "region",
    "Subregion_class": "subregion",
}

NS = "Not specified"
ANY_LTHC = "One or more long-term health condition(s)"
AGE_ORDER = ["0–44", "45–64", "65 and over"]
SEX_ORDER = ["Persons", "Male", "Female"]
YEARS_ORDER = ["0–10 years", "More than 10 years"]
PROF_ORDER = ["Very well or well", "Not well or Not at all"]

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
SUMMARY_CONDITIONS = [
    ANY_LTHC,
    "Heart disease or stroke",
    "Any other long-term health condition(s)",
]
ALL_CONDITIONS = SUMMARY_CONDITIONS[:1] + SPECIFIC_CONDITIONS + SUMMARY_CONDITIONS[1:]

MAP_NAME_FIX = {
    "China (excludes SARs and Taiwan)": "China",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "United Kingdom, Channel Islands and Isle of Ma": "United Kingdom",
    "Hong Kong (SAR of China)": "Hong Kong",
    "Macau (SAR of China)": "Macao",
    "Korea, Republic of (South)": "South Korea",
    "Korea, Democratic People's Republic of (North)": "North Korea",
    "Congo, Democratic Republic of": "Democratic Republic of the Congo",
    "Congo, Republic of": "Republic of the Congo",
    "Russian Federation": "Russia",
    "Gaza Strip and West Bank": "Palestine",
    "Micronesia, Federated States of": "Micronesia",
    "Samoa, American": "American Samoa",
    "Eswatini": "Swaziland",
    "Cabo Verde": "Cape Verde",
    "Brunei Darussalam": "Brunei",
    "Czechia": "Czech Republic",
    "Timor-Leste": "East Timor",
    "North Macedonia": "Macedonia",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Virgin Islands, British": "British Virgin Islands",
    "Virgin Islands, United States": "United States Virgin Islands",
    "Cote d'Ivoire": "Ivory Coast",
}


def _fix_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("â€“", "–", regex=False)
        .str.replace("â€”", "—", regex=False)
        .str.replace("â€™", "’", regex=False)
        .str.strip()
    )


def load_data(raw_path: Path | str = RAW_FILE, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE, keep_default_na=False, na_values=[""])

    raw_path = Path(raw_path)
    if raw_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(raw_path)
    else:
        df = pd.read_csv(raw_path, encoding="utf-8")

    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df = df.rename(columns=COLUMN_MAP)

    for col in ["country", "years", "age", "sex", "condition", "language", "proficiency", "region", "subregion"]:
        df[col] = _fix_text(df[col])
    df["age"] = df["age"].replace({"00–44": "0–44"})
    df["language"] = df["language"].replace({".ATSI lang": "Aboriginal & Torres Strait Islander languages"})

    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0).astype(int)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    df["view"] = np.where(df["years"] != NS, "migration", "language")

    df["available"] = df["population"] > 0
    key_lang = np.where(df["view"] == "language", df["language"], "")
    df["group_population"] = (
        df.assign(_k=key_lang)
        .groupby(["view", "country", "years", "_k", "proficiency", "age", "sex"])["population"]
        .transform("max")
    )
    df["rate"] = np.where(df["population"] > 0, df["cases"] / df["population"] * 100, 0.0)

    df["map_country"] = df["country"].replace(MAP_NAME_FIX)

    if use_cache:
        try:
            df.to_csv(CACHE_FILE, index=False, compression="gzip")
        except OSError:
            pass
    return df


def split_views(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mig = df[df["view"] == "migration"].copy()
    lang = df[df["view"] == "language"].copy()
    return mig, lang


def prevalence(df: pd.DataFrame, by: list[str] | str | None = None) -> pd.DataFrame:
    if by is None or by == []:
        c, p = df["cases"].sum(), df["population"].sum()
        return pd.DataFrame({"cases": [c], "population": [p], "rate": [c / p * 100 if p else np.nan]})
    by = [by] if isinstance(by, str) else list(by)
    out = df.groupby(by, observed=True, as_index=False)[["cases", "population"]].sum()
    out["rate"] = np.where(out["population"] > 0, out["cases"] / out["population"] * 100, np.nan)
    return out


def standard_weights(mig: pd.DataFrame) -> dict[str, float]:
    base = mig[(mig["sex"] == "Persons") & (mig["condition"] == ANY_LTHC)]
    w = base.groupby("age")["population"].sum()
    return (w / w.sum()).to_dict()


def age_standardised(df: pd.DataFrame, by: list[str] | str, weights: dict[str, float]) -> pd.DataFrame:
    by = [by] if isinstance(by, str) else list(by)
    g = prevalence(df, by + ["age"])
    g = g[g["population"] > 0].copy()
    g["w"] = g["age"].map(weights).fillna(0)
    g["wr"] = g["rate"] * g["w"]
    out = g.groupby(by, as_index=False).agg(
        wr=("wr", "sum"), w=("w", "sum"), cases=("cases", "sum"), population=("population", "sum")
    )
    out["asr"] = np.where(out["w"] > 0, out["wr"] / out["w"], np.nan)
    out["crude_rate"] = out["cases"] / out["population"] * 100
    return out.drop(columns=["wr", "w"])


def risk_table(
    df: pd.DataFrame, group_cols: list[str], baseline_cols: list[str], min_population: int = 500
) -> pd.DataFrame:
    grp = prevalence(df, group_cols)
    base = prevalence(df, baseline_cols).rename(columns={"rate": "baseline_rate", "cases": "_bc", "population": "_bp"})
    out = grp.merge(base[baseline_cols + ["baseline_rate"]], on=baseline_cols, how="left")
    out = out[out["population"] >= min_population].copy()

    p0 = out["baseline_rate"] / 100
    se = np.sqrt(p0 * (1 - p0) / out["population"])
    out["z_score"] = np.where(se > 0, (out["rate"] / 100 - p0) / se, 0.0)
    out["rate_ratio"] = np.where(out["baseline_rate"] > 0, out["rate"] / out["baseline_rate"], np.nan)
    out["risk_level"] = np.select(
        [(out["z_score"] >= 3) & (out["rate_ratio"] >= 1.2), (out["z_score"] <= -3) & (out["rate_ratio"] <= 0.8)],
        ["Elevated", "Lower"],
        default="Typical",
    )
    return out.sort_values("z_score", ascending=False)


ANY = "Any"
SMOOTHING_K = 200


def _rates(df: pd.DataFrame, conds: list[str]) -> pd.DataFrame:
    g = df.groupby("condition")[["cases", "population"]].sum().reindex(conds, fill_value=0).astype(float)
    g["rate"] = np.where(g["population"] > 0, g["cases"] / g["population"], np.nan)
    return g


def _smooth(group: pd.DataFrame, prior_rate: pd.Series, k: float = SMOOTHING_K) -> pd.Series:
    prior = prior_rate.fillna(0)
    return (group["cases"] + k * prior) / (group["population"] + k)


def estimate_profile(
    mig: pd.DataFrame,
    lang: pd.DataFrame,
    *,
    age: str,
    sex: str,
    region: str = ANY,
    country: str = ANY,
    years: str = ANY,
    language: str = ANY,
    proficiency: str = ANY,
    conditions: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    conds = conditions or ([ANY_LTHC] + SPECIFIC_CONDITIONS)
    info: dict = {"notes": []}

    m_age = mig[mig["age"] == age]
    l_age = lang[lang["age"] == age]
    if years != ANY:
        m_age = m_age[m_age["years"] == years]

    def _sex_factor(df):
        if sex == "Persons":
            return pd.Series(1.0, index=conds)
        r_s = _rates(df[df["sex"] == sex], conds)["rate"]
        r_p = _rates(df[df["sex"] == "Persons"], conds)["rate"]
        return (r_s / r_p).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    f_mig = _sex_factor(m_age)
    f_lang = _sex_factor(l_age)

    def rates_sex(df, factor):
        r = _rates(df[df["sex"] == sex], conds)
        r["fallback"] = False
        if sex != "Persons":
            p = _rates(df[df["sex"] == "Persons"], conds)
            miss = (r["population"] == 0) & (p["population"] > 0)
            r.loc[miss, "cases"] = p.loc[miss, "cases"] * factor[miss]
            r.loc[miss, "population"] = p.loc[miss, "population"]
            r.loc[miss, "fallback"] = True
            r["rate"] = np.where(r["population"] > 0, r["cases"] / r["population"], np.nan)
        return r

    avg_all = rates_sex(mig[mig["age"] == age], _sex_factor(mig[mig["age"] == age]))
    start = rates_sex(m_age, f_mig)
    est = start["rate"].fillna(avg_all["rate"]).fillna(0)

    levels = []
    if country != ANY:
        row = mig.loc[mig["country"] == country, ["region", "subregion"]].iloc[0]
        levels = [("region", row["region"]), ("subregion", row["subregion"]), ("country", country)]
        info["place"] = country
    elif region != ANY:
        levels = [("region", region)]
        info["place"] = region
    else:
        info["place"] = "any country"

    own = pd.Series(True, index=conds)
    for col, val in levels:
        r = rates_sex(m_age[m_age[col] == val], f_mig)
        est = (r["cases"] + SMOOTHING_K * est) / (r["population"] + SMOOTHING_K)
        own = r["population"] > 0

    grp_rows = m_age[(m_age["sex"] == sex) & (m_age["condition"] == ANY_LTHC)]
    if levels:
        col, val = levels[-1]
        grp_rows = grp_rows[grp_rows[col] == val]
    info["population"] = int(grp_rows["group_population"].sum())

    ratio = pd.Series(1.0, index=conds)
    if language != ANY or proficiency != ANY:
        place_l = l_age
        if levels:
            col, val = levels[-1]
            place_l = place_l[place_l[col] == val]
        sel = place_l
        if language != ANY:
            sel = sel[sel["language"] == language]
        if proficiency != ANY:
            sel = sel[sel["proficiency"] == proficiency]
        ref = rates_sex(place_l, f_lang)
        s = rates_sex(sel, f_lang)
        info["language_population"] = int(
            sel.loc[(sel["sex"] == sex) & (sel["condition"] == ANY_LTHC), "group_population"].sum()
        )
        if s["population"].max() > 0:
            ref_rate = ref["rate"]
            ratio = _smooth(s, ref_rate) / ref_rate.replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.25, 4)
        else:
            info["notes"].append(
                "No census records for this language / English combination – language effect not applied."
            )

    likelihood = (est * ratio).clip(upper=0.99) * 100
    average = avg_all["rate"] * 100
    out = pd.DataFrame(
        {
            "condition": conds,
            "likelihood": likelihood.values,
            "average": average.values,
            "language_effect": ratio.values,
            "own_data": own.values if levels else True,
        }
    )
    out["relative"] = np.where(out["average"] > 0, out["likelihood"] / out["average"], np.nan)
    out["level"] = np.select(
        [out["average"].isna() | (out["average"] <= 0), out["relative"] >= 1.2, out["relative"] <= 0.8],
        ["Not reported", "Higher than average", "Lower than average"],
        "About average",
    )
    out["one_in"] = np.where(out["likelihood"] > 0, np.round(100 / out["likelihood"]), np.nan)

    n = info["population"]
    info["confidence"] = "High" if n >= 5000 else "Medium" if n >= 500 else "Low"
    if n < 500:
        info["notes"].append("Small census group – the estimate leans on the wider region.")
    if levels and not bool(own.all()):
        info["notes"].append(
            "Some conditions were not published for this exact group; "
            "those estimates come from the wider sub-region / region."
        )
    return out, info


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}" if pd.notna(n) else "–"


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%" if pd.notna(x) else "–"


if __name__ == "__main__":
    data = load_data(use_cache=False)
    mig, lang = split_views(data)
    print(f"Rows: {len(data):,}  (migration view {len(mig):,}, language view {len(lang):,})")
    tot = prevalence(mig[(mig.sex == "Persons") & (mig.condition == ANY_LTHC)])
    print(f"Overseas-born population covered: {tot.population[0]:,}")
    print(f"Reporting 1+ LTHC: {tot.cases[0]:,} ({tot.rate[0]:.1f}% crude)")
