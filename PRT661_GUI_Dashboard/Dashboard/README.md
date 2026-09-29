# LTHC Population Health Dashboard

The dashboard stage of the PRT661 project, built to the **Final Project Architecture**: a **Plotly Dash** dashboard
that calls a **FastAPI** machine-learning inference API, both hosted on **Render**. It turns the cleaned AIHW data and
the XGBoost model from the `Machine_Learning` branch into the two things the project objectives promised: key findings
from the exploratory analysis, and population-level prevalence predictions.

It combines two pieces of team work: the prediction dashboard (Overview, Predict, Model & methods, FastAPI service) and
the census analysis dashboard and Risk Identifier built as stand-alone apps, now the **Analysis** and **Group profile**
pages of one app.

![Predict page](docs/screenshots/predict.png)

More screenshots (1920 × 1080 browser window): [Overview](docs/screenshots/overview.png) · Analysis tabs:
[overview](docs/screenshots/analysis_overview.png), [country of birth](docs/screenshots/analysis_country.png),
[elevated risk](docs/screenshots/analysis_risk.png), [years in Australia](docs/screenshots/analysis_years.png),
[language & English](docs/screenshots/analysis_language.png), [data explorer](docs/screenshots/analysis_data_explorer.png)
· [Group profile](docs/screenshots/group_profile.png) · [Model & methods](docs/screenshots/model.png) ·
[Phone](docs/screenshots/mobile_predict.png)

## Design

All five pages share one visual system, so the merged pages read as one product:

- **One page = one screen.** Every page fills the browser window edge to edge, like a Power BI report: a navy header
  (navigation, page title, one-line summary, one action), a row of headline KPI cards, then a grid of chart cards that
  stretch to the window, and a slim status bar with the population-level caveat and the data source. Nothing scrolls
  on a laptop or desktop; long tables and notes scroll inside their own card, with a soft shadow when there is more.
- **Same proportions on any screen.** Pages are laid out for a 1600 × 900 to 1920 × 1080 window. In a smaller window
  (a 1366 × 768 laptop, or a browser that is not full screen) the whole page is scaled down to fit, and on a larger
  monitor it is scaled up, so it always looks like the design (`assets/fit.js`; Plotly tooltips still line up).
  Tablets and phones get a scrolling single-column layout instead.
- **Components.** KPI cards with a coloured top edge, chart cards with a Chart / Table switch, numbered form steps on
  the Predict and Group profile pages (`lthc_dashboard/components.py`, `assets/style.css`).
- **Type.** Inter, bundled in `assets/fonts` (SIL Open Font License), so the dashboard looks the same on every computer.
- **Chart colours** (`lthc_dashboard/theme.py`), checked for colour-vision deficiency: one series in teal and the
  selected item in coral; light-to-dark teal for "more" of an ordered thing (age, years in Australia, English
  proficiency); male blue and female amber; teal-to-navy for magnitude in maps and heatmaps; lower / about / higher
  than average in teal / light slate / coral, always with an icon and a label.

## Pages

| Page | What it shows |
|---|---|
| **Overview** (`/`) | Headline findings with charts and data tables: age, sex, region of birth, years in Australia, and the leading condition by region and sex. |
| **Analysis** (`/analysis`) | Census analysis dashboard with global filters, live KPIs and six tabs: overview (conditions, age × sex, treemap), country of birth (world map, top countries, crude vs age-standardised regions), elevated risk (groups with the largest excess prevalence and the table of statistically elevated groups), years in Australia, language & English, and a data explorer with CSV export. |
| **Predict** (`/predict`) | The deployment phase of the workflow: pick a population group and get the XGBoost prediction, the published census value, a relative risk level, a demographic summary, charts and insights. |
| **Group profile** (`/profile`) | Census-based profile of every condition for a population group (empirical-Bayes smoothing), compared with all overseas-born people of the same age and sex. |
| **Model & methods** (`/model`) | Test-set performance of the four models, overfitting check, feature importance, predicted vs published values, pipeline, limitations and ethics. |

### How the Predict page follows the workflow diagram (deployment phase)

| Workflow step | In the dashboard |
|---|---|
| Selects parameters for population group (country of birth, age group, sex, years in Australia, language, English proficiency, LTHC) | The form shows all seven parameters. |
| Validate and preprocess selected parameters, format them for the AI model | `prediction.validate_request` and `prediction.build_model_input` check the inputs and build the nine model features the way the training rows were built (see *Census tables* below). The exact input sent is shown in the *Model input* tab. |
| Load the formatted data into the model | `inference.InferenceClient` posts to the FastAPI service (`/predict/batch`). |
| Predict population-level LTHC prevalence (%) | XGBoost model, clipped to 0–100%. |
| Display results: predicted prevalence, observed prevalence, risk level, demographic summary | The prediction card, tiles for the published census value, the benchmark and the relative risk level, a comparison bar, and the demographic summary in the *Summary* tab. |
| Generate data visualisations and produce health insights | Age-by-sex trajectory, condition profile and generated plain-language insights. |

## Run it locally (Python 3.11)

```bash
cd Dashboard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Terminal 1 - inference API (http://127.0.0.1:8000/docs for the Swagger UI)
uvicorn api.main:app --port 8000

# Terminal 2 - dashboard (http://127.0.0.1:8050)
export LTHC_API_URL=http://127.0.0.1:8000     # PowerShell: $env:LTHC_API_URL="http://127.0.0.1:8000"
python app.py
```

Without `LTHC_API_URL` the dashboard uses its bundled copy of the same model, and the Predict page says which one
served each prediction. If the API is set but unreachable (a sleeping free Render instance, for example), the dashboard
falls back to the bundled model and says so.

Run the tests with `python -m pytest` from the `Dashboard` folder.

| Environment variable | Default | Purpose |
|---|---|---|
| `LTHC_API_URL` | empty | Base URL of the FastAPI service |
| `LTHC_API_TIMEOUT` | `10` | Seconds to wait for the API before using the bundled model |
| `LTHC_DATA_DIR`, `LTHC_MODELS_DIR` | `data/`, `models/` | Override file locations |

## Deploy on Render

`render.yaml` in the repository root is a Render Blueprint that creates both services from this folder on the `GUI`
branch.

1. Push the `GUI` branch to GitHub.
2. In Render: **New + → Blueprint**, connect the repository, select the `GUI` branch and apply.
3. When asked for `LTHC_API_URL`, enter the API's public URL (normally `https://lthc-api.onrender.com`; check the URL
   Render gives the `lthc-api` service and update the variable under the dashboard's *Environment* settings if it
   differs). It can be left empty at first.
4. Open the `lthc-dashboard` URL. The Predict page reports "Served by the FastAPI inference service" once connected.

Free Render services sleep when idle and can take about a minute to wake; the dashboard pings the API as it starts so both wake together, and uses its bundled model for any request made before the API is up. The dashboard uses about 300 MB of memory and the API about 200 MB,
within the free tier's 512 MB. The Python version comes from `.python-version` (3.11).

## Folder structure

```
Dashboard/
├── app.py                  Dash entry point (gunicorn app:server)
├── api/main.py             FastAPI service: /, /health, /predict, /predict/batch
├── pages/                  overview.py, analysis.py, predict.py, profile.py, model.py
├── lthc_dashboard/
│   ├── config.py           paths, environment settings, column names, category lists
│   ├── data.py             loading, census-table tagging, train/test split, weighted prevalence
│   ├── analysis.py         descriptive tables for Overview (report method)
│   ├── census_analysis.py  age-standardisation, risk tables and group profiles (Analysis, Group profile)
│   ├── model.py            loads the preprocessor + XGBoost model (shared by API and dashboard)
│   ├── inference.py        API client with fallback to the bundled model
│   ├── prediction.py       Predict-page logic: validation, inputs, benchmark, risk level, insights
│   ├── figures.py          Plotly charts
│   ├── components.py       layout pieces (cards, tiles, tables, notices)
│   └── theme.py            colours and Plotly template
├── assets/                 style.css, fit.js (fit each page to the window), logo.svg, fonts/ (Inter),
│                           topojson/world_110m.json (map outlines served locally)
├── data/                   LTHC_phase2_cleaned.csv + evaluation outputs (from Machine_Learning/Attempt_2),
│                           census_tables_clean.csv.gz (all 120,978 cells, used by Analysis and Group profile)
├── models/                 lthc_preprocessor.pkl, xgboost_regression_model.pkl (from Fast_API/models)
├── tests/                  47 tests
└── docs/screenshots/
```

`data/` and `models/` are unchanged copies of files on the `Machine_Learning` branch. The preprocessor was pickled with
scikit-learn 1.8.0, so that version is pinned. `api/main.py` keeps the endpoints, request fields and response keys of
`Fast_API/api/main.py`, and adds `/predict/batch`, an `unseen_inputs` field and file paths that work from any
directory.

## Method notes

- **Two complementary methods.** Overview reproduces the Assessment 2 report (weighted prevalence, both census tables
  pooled, all ages). Analysis and Group profile work within one census table at a time and age-standardise
  comparisons to the overseas-born age structure, so regions with older migrants (for example Europe) are not
  flagged just for being older. Both are stated on the pages.
- **Weighted prevalence.** Every descriptive figure is pooled cases ÷ pooled population for the group, as in the
  Assessment 2 report. The tests check that it reproduces the report's figures (for example female arthritis by age:
  0.93%, 10.70%, 33.03%).
- **No double counting.** "Persons" is the Male + Female total, so when sex is not the comparison only Persons rows are
  used, following the report's univariate rule. The report's region heatmap pooled Persons with Male and Female rows,
  so a few region figures differ slightly (Southern and Eastern Europe arthritis: 15.5% here, 16.3% in the report).
- **Census tables.** AIHW publishes country of birth + years in Australia (Table S10) separately from home language +
  English proficiency (Table S13). Every cleaned row comes from exactly one table, so:
  - comparisons by country or years in Australia use Table S10 only, and comparisons by language or English
    proficiency use Table S13 only;
  - the Predict form locks the fields the chosen table does not record, and fills them the way the training rows were
    filled (the paired language or country from cleaning, and "Not specified").
- **Crude rates.** "All ages" comparisons are crude. The age filter gives like-with-like comparisons.
- **Relative risk level.** Predicted prevalence ÷ benchmark, where the benchmark is the weighted prevalence for the same
  condition, age group and sex across all published groups in the same census table. Below 0.8× is below average,
  0.8–1.25× around average, 1.25–2× above average and 2× or more well above average.
- **Group profile estimate.** Starts from all overseas-born people of the same age, sex and years in Australia and
  moves down region → subregion → country, shrinking small groups towards the wider area:
  `(cases + 200 × previous estimate) / (population + 200)`. A chosen language or English level multiplies the result
  by that group's relative rate. Confidence is High / Medium / Low for groups of ≥ 5,000 / ≥ 500 / < 500 people.
- **Training status.** The group-aware split (`GroupShuffleSplit`, `test_size=0.2`, `random_state=42`) is recreated
  exactly, so each result says whether the group was held out from training.

## Data-quality note for the team

The language–country pairings made during cleaning are often not the most common language: China → Uygur,
India → Kannada, Philippines → Bisaya, England → "Northern European Languages, nfd", New Zealand → Māori, and in the
other direction Arabic → Saudi Arabia and Spanish → Spain. The model was trained on these pairings, so the dashboard
uses them unchanged and shows them as "derived". Fixing the mapping would mean re-running cleaning and retraining.
The language-table rows' regions also come from these pairings.

## Maintenance notes

- `dash_table.DataTable` (used on the Analysis and Group profile pages) still works in Dash 4.4.1 but is deprecated;
  Dash recommends `dash-ag-grid` for a future upgrade.
- To rebuild `census_tables_clean.csv.gz` from the team's `Cleaned_data_final.xlsx`, put the workbook in `data/`, delete
  the `.csv.gz`, install `openpyxl` and restart the app.
