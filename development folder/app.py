import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html
from dash import dash_table
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
# =========================================================
# 1. LOAD DATA & UTILITIES
# =========================================================
file_path = "Cleaned_data_final.csv"
FAST_API_URL = "https://data-science-practise-prt661-api.onrender.com"

# FAST API INFERENCE SECTION
PREDICT_ENDPOINT = f"{FAST_API_URL}/predict"   
API_TIMEOUT = 90                               


API_FIELD_MAP = {
    "country": "country_of_birth",
    "age": "age_group",
    "sex": "sex",
    "years": "years_in_australia",
    "language": "language",
    "english": "english_proficiency",
    "region": "region",
    "sub_region": "subregion",
    "lthc": "lthc",
}

try:
    raw_df = pd.read_csv(file_path)
    df = raw_df.loc[raw_df["Number of people reporting LTHC(s)"] > 0]
except Exception as e:
    raise RuntimeError(f"Could not load data from {file_path}: {e}")


def weight_prevalence(df, col1, col2, num1, num2):
    weighted_df = df.groupby([col1, col2])[[num1, num2]].sum()
    weighted_df["Prevalence (%)"] = (
        weighted_df[num1] / weighted_df[num2]
    ) * 100
    return weighted_df


def top_ten_filter(df):
    top_conditions = [
        "Arthritis",
        "Asthma",
        "Diabetes",
        "Mental health condition",
        "Heart disease or stroke",
    ]
    return df.loc[
        df["Long-term health condition (LTHC)"].isin(top_conditions)
    ]


# Descriptions for each graph in each drop down
DESCRIPTIONS = {
    "g1": """This chart illustrates distinct gender disparities in long-term health conditions (LTHCs) across Australia, highlighting where targeted health interventions and resource allocations yield the highest impact.

Gender Prevalence Breakdown:
• Female-Skewed Conditions: Arthritis (9.0% Females vs. 5.7% Males), Mental Health (5.9% Females vs. 3.9% Males).
• Male-Skewed Conditions: Diabetes (6.7% Males vs. 5.3% Females), Heart Disease / Stroke (5.3% Males vs. 3.2% Females).

Strategic Applications: Targeted Awareness enables organizations to route cardiovascular campaigns through male-centric channels, while focusing mental health outreach toward female demographics.""",
    "g2": """We grouped findings based on Age group and Sex to determine how these LTHC prevalence percentages vary across conditions. The data shows how rapidly the percentage prevalence grows per age and sex group. 

Referring to our previous discovery of Arthritis being more prevalent in females, we see prevalence changes: 0–44: 0.93%, 45–64: 10.70%, and 65 and over: 33%. 

This is important as it shows that without early support, rapid growth in occurrence can strain resources. Having such data helps forecast and proactively prepare healthcare systems.""",
    "g3": """We performed an analysis between Years spent in Australia, Age group, and LTHC. We noticed that the more years a person stays in Australia, the higher the prevalence of incurring a condition. However, Age is a confounding variable with a high effect.

This graph serves as a suggestion for health bodies to investigate why the prevalence of LTHC rises alongside years spent in Australia. A possible reason may be under-reporting in the early years of migration due to systemic barriers or fear.""",
    "g4": """One of the most prominent observations is the increase in percentage prevalence of each health condition as the age group increases, supporting the notion that an aging population experiences higher LTHC rates. 

However, this graph also shows LTHCs that are pre-dominant per age group. Amongst the 0–44 group, Asthma and Mental health conditions have higher occurrences compared to other LTCHs. 

Because asthma and mental health are predominant in the 0–44 demographic, public health funding—such as school-based asthma plans or youth mental health services—should be directed toward young-adult community settings for early support.""",
    "g6": """This treemap breaks down overseas-born Australians by region, sub-region and country of birth. Tile size shows the number of people reporting a long-term health condition; colour shows prevalence (%).

Large, dark tiles flag groups with both a big affected population and a high rate, which is where targeted support is likely to matter most. Click a tile to zoom in, and click the top bar to zoom back out.""",
    "g5": """This heatmap serves two key analytical purposes: it highlights the leading condition within any given region, and it allows for cross-regional comparisons of specific diseases. 

For example, while Arthritis is the primary health condition affecting the Southern and Eastern European cohort (with a 16.26% prevalence), the heatmap reveals that the Northern European demographic actually experiences the highest overall rate of Arthritis across all nine regions.""",
}


# INFERENCE PAGE DROPDOWN OPTIONS
# =========================================================
COUNTRY_OF_BIRTH = ['Oceania and Antarctica, nfd', 'New Zealand', 'Melanesia, nfd',
       'New Caledonia', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu',
       'Micronesia, nfd', 'Guam', 'Kiribati', 'Marshall Islands',
       'Micronesia, Federated States of', 'Nauru',
       'Northern Mariana Islands', 'Palau', 'Cook Islands', 'Fiji',
       'French Polynesia', 'Niue', 'Samoa', 'Samoa, American', 'Tokelau',
       'Tonga', 'Tuvalu', 'Wallis and Futuna', 'Pitcairn Islands',
       'Polynesia (excludes Hawaii), nec', 'Antarctica, nfd',
       'Chilean Antarctic Territory',
       'United Kingdom, Channel Islands and Isle of Ma', 'England',
       'Isle of Man', 'Northern Ireland', 'Scotland', 'Wales', 'Guernsey',
       'Jersey', 'Ireland', 'Western Europe, nfd', 'Austria', 'Belgium',
       'France', 'Germany', 'Liechtenstein', 'Luxembourg', 'Monaco',
       'Netherlands', 'Switzerland', 'Northern Europe, nfd', 'Denmark',
       'Faroe Islands', 'Finland', 'Greenland', 'Iceland', 'Norway',
       'Sweden', 'Aland Islands', 'Southern and Eastern Europe, nfd',
       'Andorra', 'Gibraltar', 'Holy See', 'Italy', 'Malta', 'Portugal',
       'San Marino', 'Spain', 'South Eastern Europe, nfd', 'Albania',
       'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus',
       'North Macedonia', 'Greece', 'Moldova', 'Romania', 'Slovenia',
       'Montenegro', 'Serbia', 'Kosovo', 'Eastern Europe, nfd', 'Belarus',
       'Czechia', 'Estonia', 'Hungary', 'Latvia', 'Lithuania', 'Poland',
       'Russian Federation', 'Slovakia', 'Ukraine',
       'North Africa and the Middle East, nfd', 'North Africa, nfd',
       'Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia',
       'Western Sahara', 'South Sudan', 'Middle East, nfd', 'Bahrain',
       'Gaza Strip and West Bank', 'Iran', 'Iraq', 'Israel', 'Jordan',
       'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria',
       'Turkey', 'United Arab Emirates', 'Yemen', 'South-East Asia, nfd',
       'Mainland South-East Asia, nfd', 'Myanmar', 'Cambodia', 'Laos',
       'Thailand', 'Vietnam', 'Maritime South-East Asia, nfd',
       'Brunei Darussalam', 'Indonesia', 'Malaysia', 'Philippines',
       'Singapore', 'Timor-Leste',
       'Chinese Asia (includes Mongolia), nfd',
       'China (excludes SARs and Taiwan)', 'Hong Kong (SAR of China)',
       'Macau (SAR of China)', 'Mongolia', 'Taiwan', 'Japan',
       "Korea, Democratic People's Republic of (North)",
       'Korea, Republic of (South)', 'Southern and Central Asia, nfd',
       'Southern Asia, nfd', 'Bangladesh', 'Bhutan', 'India', 'Maldives',
       'Nepal', 'Pakistan', 'Sri Lanka', 'Central Asia, nfd',
       'Afghanistan', 'Armenia', 'Azerbaijan', 'Georgia', 'Kazakhstan',
       'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan',
       'Americas, nfd', 'Northern America, nfd', 'Bermuda', 'Canada',
       'United States of America', 'South America, nfd', 'Argentina',
       'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',
       'Falkland Islands', 'French Guiana', 'Guyana', 'Paraguay', 'Peru',
       'Suriname', 'Uruguay', 'Venezuela', 'Central America, nfd',
       'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras',
       'Mexico', 'Nicaragua', 'Panama', 'Caribbean, nfd', 'Anguilla',
       'Antigua and Barbuda', 'Aruba', 'Bahamas', 'Barbados',
       'Cayman Islands', 'Cuba', 'Dominica', 'Dominican Republic',
       'Grenada', 'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique',
       'Montserrat', 'Puerto Rico', 'St Kitts and Nevis', 'St Lucia',
       'St Vincent and the Grenadines', 'Trinidad and Tobago',
       'Turks and Caicos Islands', 'Virgin Islands, British',
       'Virgin Islands, United States', 'St Martin (French part)',
       'Bonaire, Sint Eustatius and Saba', 'Curacao',
       'Sint Maarten (Dutch part)', 'Sub-Saharan Africa, nfd',
       'Central and West Africa, nfd', 'Benin', 'Burkina Faso',
       'Cameroon', 'Cabo Verde', 'Central African Republic', 'Chad',
       'Congo, Republic of', 'Congo, Democratic Republic of',
       "Cote d'Ivoire", 'Equatorial Guinea', 'Gabon', 'Gambia', 'Ghana',
       'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania',
       'Niger', 'Nigeria', 'Sao Tome and Principe', 'Senegal',
       'Sierra Leone', 'Togo', 'Southern and East Africa, nfd', 'Angola',
       'Botswana', 'Burundi', 'Comoros', 'Djibouti', 'Eritrea',
       'Ethiopia', 'Kenya', 'Lesotho', 'Madagascar', 'Malawi',
       'Mauritius', 'Mayotte', 'Mozambique', 'Namibia', 'Reunion',
       'Rwanda', 'St Helena', 'Seychelles', 'Somalia', 'South Africa',
       'Eswatini', 'Tanzania', 'Uganda', 'Zambia', 'Zimbabwe', 'At sea']

AGE_GROUPS = ['00–44', '45–64', '65 and over']

SEX_OPTIONS = ['Male', 'Female','Persons']

YEARS_IN_AUSTRALIA = ['0–10 years', 'More than 10 years', 'Not specified']

LANGUAGES = ['.ATSI lang', 'Maori (New Zealand)',
       'Pacific Austronesian Languages, nfd', 'French',
       'Papua New Guinea Languages, nfd', 'Solomon Islands Pijin',
       'Bislama', 'Tagalog', 'Gilbertese', 'Yapese', 'Nauruan',
       'Maori (Cook Island)', 'Fijian Hindustani', 'Niue', 'Samoan',
       'Tokelauan', 'Tongan', 'Tuvaluan', "Norf'k-Pitcairn",
       'Other Languages, nfd', 'Spanish',
       'Northern European Languages, nfd', 'Celtic, nfd', 'Irish',
       'Gaelic (Scotland)', 'Welsh', 'German', 'Dutch', 'Letzeburgish',
       'Swiss, so described', 'Danish', 'Finnish', 'Icelandic',
       'Norwegian', 'Swedish', 'Other Southern European Languages, nec',
       'Catalan', 'Latin', 'Italian', 'Maltese', 'Portuguese',
       'Iberian Romance, nfd', 'South Slavic, nfd', 'Albanian', 'Bosnian',
       'Bulgarian', 'Croatian', 'Cypriot, so described', 'Macedonian',
       'Greek', 'Romanian', 'Slovene', 'Serbian', 'Yiddish',
       'Belorussian', 'Czech', 'Estonian', 'Hungarian', 'Latvian',
       'Lithuanian', 'Polish', 'Russian', 'Slovak', 'Ukrainian', 'Arabic',
       'Moro (Nuba Moro)', 'Dinka', 'Iranic, nfd',
       'Persian (excluding Dari)', 'Kurdish', 'Hebrew', 'Turkish',
       'Southeast Asian Languages, nfd', 'Mon-Khmer, nec',
       'Burmese and Related Languages, nfd', 'Khmer', 'Hmong', 'Thai',
       'Vietnamese', 'Southeast Asian Austronesian Languages, nfd',
       'Malay', 'Indonesian', 'Bisaya', 'Mandarin', 'Tetum',
       'Eastern Asian Languages, nfd', 'Uygur', 'Cantonese', 'Mongolian',
       'Min Nan', 'Japanese', 'Korean', 'Southern Asian Languages, nfd',
       'Bengali', 'Tibetan', 'Kannada', 'Dhivehi', 'Nepali', 'Balochi',
       'Sinhalese', 'Turkic, nfd', 'Pashto', 'Armenian', 'Azeri',
       'Georgian', 'Turkmen', 'Uzbek', 'Creole, nfd',
       'American Languages', 'French Creole, nfd', 'Spanish Creole, nfd',
       'African Languages, nfd', 'Portuguese Creole, nfd', 'Lingala',
       'Mandinka', 'Akan', 'Bassa', 'Yoruba', 'Krio', 'Tswana',
       'Kirundi (Rundi)', 'Tigre', 'Oromo', 'Kikuyu', 'Nyanja (Chichewa)',
       'Mauritian Creole', 'Afrikaans', 'Kinyarwanda (Rwanda)',
       'Seychelles Creole', 'Somali', 'Zulu', 'Swahili', 'Acholi',
       'Bemba', 'Shona', 'Not specified', 'Celtic, nec', 'Frisian',
       'Scandinavian, nfd', 'Scandinavian, nec',
       'Finnish and Related Languages, nec', 'Iberian Romance, nec',
       'Basque', 'Eastern European Languages, nfd', 'Baltic, nfd',
       'East Slavic, nfd', 'Serbo-Croatian/Yugoslavian, so described',
       'Czechoslovakian, so described', 'Aromunian (Macedo-Romanian)',
       'Romany', 'Other Eastern European Languages, nec', 'Dari',
       'Hazaraghi', 'Iranic, nec',
       'Middle Eastern Semitic Languages, nfd', 'Assyrian Neo-Aramaic',
       'Chaldean Neo-Aramaic', 'Mandaean (Mandaic)',
       'Middle Eastern Semitic Languages, nec', 'Tatar', 'Turkic, nec',
       'Other Southwest and Central Asian Languages, n', 'Malayalam',
       'Tamil', 'Telugu', 'Tulu', 'Dravidian, nec', 'Indo-Aryan, nfd',
       'Gujarati', 'Hindi', 'Konkani', 'Marathi', 'Punjabi', 'Sindhi',
       'Urdu', 'Assamese', 'Kashmiri', 'Oriya', 'Indo-Aryan, nec',
       'Other Southern Asian Languages', 'Burmese', 'Chin Haka', 'Karen',
       'Rohingya', 'Zomi', 'Burmese and Related Languages, nec', 'Mon',
       'Lao', 'Tai, nec', 'Cebuano', 'IIokano', 'Timorese', 'Filipino',
       'Acehnese', 'Balinese', 'Bikol', 'Iban', 'Ilonggo (Hiligaynon)',
       'Javanese', 'Pampangan',
       'Southeast Asian Austronesian Languages, nec',
       'Other Southeast Asian Languages', 'Chinese, nfd', 'Hakka', 'Wu',
       'Chinese, nec', 'Other Eastern Asian Languages, nec',
       'Non-verbal, so described', 'Pidgin, nfd', 'Amharic', 'Ewe', 'Ga',
       'Harari', 'Hausa', 'Igbo', 'Luganda', 'Luo', 'Ndebele', 'Nuer',
       'Shilluk', 'Tigrinya', 'Xhosa', 'Anuak', 'Bari', 'Dan (Gio-Dan)',
       'Fulfulde', 'Kpelle', 'Krahn', 'Liberian (Liberian English)',
       'Loma (Lorma)', 'Madi', 'Mann', 'Themne', 'African Languages, nec',
       'Fijian', 'Rotuman', 'Pacific Austronesian Languages, nec',
       'Oceanian Pidgins and Creoles, nfd',
       'Oceanian Pidgins and Creoles, nec', 'Kiwai', 'Motu (HiriMotu)',
       'Tok Pisin (Neomelanesian)', 'Papua New Guinea Languages, nec',
       'Invented Languages', 'Sign Languages, nfd', 'Auslan',
       'Key Word Sign Australia', 'Sign Languages, nec']

ENGLISH_PROFICIENCY = ['Not specified', 'Very well or well', 'Not well or Not at all']

LTHC_OPTIONS = ['Arthritis', 'Asthma', 'Cancer', 'Dementia', 'Diabetes',
       'Heart disease', 'Kidney disease', 'Lung condition',
       'Mental health condition', 'Stroke', 'Heart disease or stroke',
       'Any other long-term health condition(s)',
       'One or more long-term health condition(s)']

#excluding these conditions
EXCLUDED_FROM_ALL = {
    'One or more long-term health condition(s)',
    'Any other long-term health condition(s)',
    'Heart disease or stroke',
}
ALL_CONDITIONS = [c for c in LTHC_OPTIONS if c not in EXCLUDED_FROM_ALL]
HEADLINE_KEYS = ("prediction", "predicted_prevalence", "prevalence", "result")

Region=['Oceania and Antarctica', 'North-West Europe',
       'Southern and Eastern Europe', 'North Africa and the Middle East',
       'South-East Asia', 'North-East Asia', 'Southern and Central Asia',
       'Americas', 'Sub-Saharan Africa', 'Not specified']
Sub_region= ['Oceania (nfd)', 'New Zealand', 'Melanesia', 'Micronesia',
       'Polynesia', 'Antarctica', 'United Kingdom', 'Ireland',
       'Western Europe', 'Northern Europe', 'Southern Europe',
       'South Eastern Europe', 'Eastern Europe',
       'North Africa and Middle East (nfd)', 'North Africa',
       'Middle East', 'South-East Asia (nfd)', 'Mainland South-East Asia',
       'Maritime South-East Asia', 'Chinese Asia', 'Japan and the Koreas',
       'Southern and Central Asia (nfd)', 'Southern Asia', 'Central Asia',
       'Americas (nfd)', 'Northern America', 'South America',
       'Central America', 'Caribbean', 'Sub-Saharan Africa (nfd)',
       'Central and West Africa', 'Southern and East Africa',
       'Not specified']

# (label shown in UI, component id, options list)
INFERENCE_FIELDS = [
    ("Country of Birth", "inf-country", COUNTRY_OF_BIRTH),
    ("Age Group", "inf-age", AGE_GROUPS),
    ("Sex", "inf-sex", SEX_OPTIONS),
    ("Years in Australia", "inf-years", YEARS_IN_AUSTRALIA),
    ("Language Used at Home", "inf-language", LANGUAGES),
    ("English Proficiency", "inf-english", ENGLISH_PROFICIENCY),
    ("Region","inf-region",Region),
    ("Sub-region","inf-sub_region",Sub_region)
]



# 2. PLOTLY GRAPHS
# =========================================================

def lthc_to_gender(data):
    sex_to_LTHC = data.groupby(["Long-term health condition (LTHC)", "Sex"])[
        ["Number of people reporting LTHC(s)", "Population"]
    ].sum()
    sex_to_LTHC["Prevalence (%)"] = (
        sex_to_LTHC["Number of people reporting LTHC(s)"]
        / sex_to_LTHC["Population"]
    ) * 100

    fig1 = px.bar(
        sex_to_LTHC.reset_index(),
        x="Long-term health condition (LTHC)",
        y="Prevalence (%)",
        color="Sex",
        barmode="group",
        title="Prevalence (%) of LTHC by Gender",
        template="plotly_white",
        color_discrete_sequence=["#2b5c8f", "#d95f02"],
    )
    fig1.update_layout(
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig1


def lthc_to_gender_age(data):
    lthc_sex_age_group = data.groupby(
        ["Long-term health condition (LTHC)", "Sex", "Age group"]
    )[["Number of people reporting LTHC(s)", "Population"]].sum()
    lthc_sex_age_group["Prevalence (%)"] = (
        lthc_sex_age_group["Number of people reporting LTHC(s)"]
        / lthc_sex_age_group["Population"]
    ) * 100

    table_df = lthc_sex_age_group["Prevalence (%)"].unstack().reset_index()
    table_df = table_df.round(2)
    return table_df


def ltch_to_age_years_in_aussie(data):
    most_common_without_unspecified = data.loc[
        data["Years spent in Australia"] != "Not specified"
    ]
    age_to_lthc_years_spent = most_common_without_unspecified.groupby(
        ["Years spent in Australia", "Long-term health condition (LTHC)", "Age group"]
    )[["Number of people reporting LTHC(s)", "Population"]].sum()
    age_to_lthc_years_spent["Prevalence (%)"] = (
        age_to_lthc_years_spent["Number of people reporting LTHC(s)"]
        / age_to_lthc_years_spent["Population"]
    ) * 100

    table_two = age_to_lthc_years_spent["Prevalence (%)"].unstack().reset_index()
    table_two = table_two.round(2)
    return table_two


def age_to_lthc(data):
    age_to_lthc_df = data.groupby(
        ["Long-term health condition (LTHC)", "Age group"]
    )[["Number of people reporting LTHC(s)", "Population"]].sum()
    age_to_lthc_df["Prevalence (%)"] = (
        age_to_lthc_df["Number of people reporting LTHC(s)"]
        / age_to_lthc_df["Population"]
    ) * 100

    fig4 = px.bar(
        age_to_lthc_df.reset_index(),
        x="Age group",
        y="Prevalence (%)",
        color="Long-term health condition (LTHC)",
        barmode="group",
        title="Age-Specific LTHC Prevalence (%)",
        template="plotly_white",
    )
    fig4.update_layout(
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    return fig4


def heatmap(data):
    grouped_region_to_lthc = (
        weight_prevalence(
            data,
            "Region_class",
            "Long-term health condition (LTHC)",
            "Number of people reporting LTHC(s)",
            "Population",
        )["Prevalence (%)"]
        .round(2)
        .reset_index()
    )

    val_pivot = grouped_region_to_lthc.pivot(
        index="Region_class",
        columns="Long-term health condition (LTHC)",
        values="Prevalence (%)",
    ).fillna(0)

    fig5 = px.imshow(
        val_pivot,
        labels=dict(
            x="Long-term health condition (LTHC)",
            y="Region Class",
            color="Prevalence (%)",
        ),
        color_continuous_scale="YlOrRd",
        text_auto=".2f",
        aspect="auto",
        title="Top LTHC & Prevalence by Region",
        template="plotly_white",
    )
    fig5.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig5



REGION_COL = "Region_class"
SUBREGION_COL = "Subregion_class"
COUNTRY_COL = "Country of birth of person"


def treemap(data):
    """Treemap region -> sub-region -> country: size = cases, colour = prevalence."""
    path_cols = [REGION_COL, SUBREGION_COL, COUNTRY_COL]
    if not all(c in data.columns for c in path_cols):
        return empty_treemap_fig(f"Missing column(s): {', '.join(c for c in path_cols if c not in data.columns)}")

    t = (
        data.loc[~data[path_cols].isin(["Not specified"]).any(axis=1)]
        .groupby(path_cols)[["Number of people reporting LTHC(s)", "Population"]]
        .sum()
        .reset_index()
    )
    t = t[t["Number of people reporting LTHC(s)"] > 0]
    if t.empty:
        return empty_treemap_fig("No data for this selection")

    t["Prevalence (%)"] = t["Number of people reporting LTHC(s)"] / t["Population"] * 100

    fig6 = px.treemap(
        t,
        path=[px.Constant("All overseas-born"), REGION_COL, SUBREGION_COL, COUNTRY_COL],
        values="Number of people reporting LTHC(s)",
        color="Prevalence (%)",
        color_continuous_scale="YlOrRd",
        template="plotly_white",
    )
    fig6.update_traces(
        root_color="#eef3f8",
        hovertemplate="<b>%{label}</b><br>People with condition: %{value:,}"
                      "<br>Prevalence: %{color:.1f}%<extra></extra>",
    )
    fig6.update_layout(margin=dict(t=10, l=0, r=0, b=0), coloraxis_colorbar_title="%")
    return fig6


def empty_treemap_fig(message):
    fig = px.scatter(template="plotly_white")
    fig.update_layout(
        xaxis_visible=False, yaxis_visible=False,
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
    )
    return fig



# 3. Graph cards
# =========================================================
def create_graph_card(card_title, component_id, description_text, width_lg=6, is_table=False):
    if is_table:
        visual_component = html.Div(id=component_id, style={"height": "420px", "overflow": "auto"})
    else:
        visual_component = dcc.Graph(
            id=component_id,
            config={"displayModeBar": True, "responsive": True},
            style={"height": "420px"},
        )

    return dbc.Col(
        xs=12, md=12, lg=width_lg, className="mb-4",
        children=[
            dbc.Card(
                className="shadow-sm border-0 rounded-3 h-100",
                children=[
                    dbc.CardHeader(
                        html.H5(
                            card_title,
                            className="mb-0 text-dark fw-bold",
                            style={"fontSize": "1.1rem"},
                        ),
                        className="bg-white border-bottom-0 pt-3 pb-2",
                    ),
                    dbc.CardBody(
                        children=[
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        dcc.Markdown(description_text),
                                        title="ℹ️ Click to read description & insights",
                                    )
                                ],
                                start_collapsed=True,
                                className="mb-3 border-0",
                            ),
                            visual_component,
                        ]
                    ),
                ],
            )
        ],
    )


def create_inference_dropdown(label, component_id, options):
    """A labelled, searchable dropdown row for the Inferences form."""
    return dbc.Row(
        className="mb-3 align-items-center",
        children=[
            dbc.Col(
                html.Label(label, htmlFor=component_id, className="fw-semibold mb-0"),
                xs=12, md=4,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id=component_id,
                    options=[{"label": o, "value": o} for o in options],
                    placeholder=f"Select {label.lower()}...",
                    searchable=True,
                    clearable=True,
                    maxHeight=300,
                ),
                xs=12, md=8,
            ),
        ],
    )


# 4. DASH APPLICATION, LAYOUTS & ROUTING
# =========================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="LTHC Analytics Dashboard",
)
server = app.server
# --- Navigation Bar ---
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Inferences & Insights", href="/inferences", active="exact")),
    ],
    brand="Health Analytics Hub",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-4 shadow-sm"
)

# --- Home Page Layout ---
home_layout = dbc.Container(
    fluid=True,
    className="px-4 pb-4",
    children=[
        dbc.Card(
            className="shadow-sm border-0 mb-4 bg-primary text-white rounded-3",
            children=[
                dbc.CardBody(
                    children=[
                        dbc.Row(
                            className="align-items-center g-3",
                            children=[
                                dbc.Col(
                                    xs=12, md=8,
                                    children=[
                                        html.H2(
                                            "Long-Term Health Conditions (LTHC) Analytics",
                                            className="fw-bold mb-1 text-white",
                                        ),
                                        html.P(
                                            "Interactive demographical, regional, and age prevalence insights.",
                                            className="text-white-50 mb-0",
                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    xs=12, md=4,
                                    children=[
                                        html.Label("Filter Health Conditions:", className="fw-semibold text-white mb-2"),
                                        dcc.Dropdown(
                                            id="condition-filter-dropdown",
                                            options=[
                                                {"label": "🌐 All Long-Term Conditions", "value": "ALL"},
                                                {"label": "⭐ Top LTHC Conditions", "value": "TOP_TEN"},
                                            ],
                                            value="ALL",
                                            clearable=False,
                                            className="text-dark",
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ]
                )
            ],
        ),
        dbc.Row(
            children=[
                create_graph_card("Gender Prevalence", "graph-1", DESCRIPTIONS["g1"], width_lg=6),
                create_graph_card("Condition, Sex & Age Stack Table", "graph-2", DESCRIPTIONS["g2"], width_lg=6, is_table=True),
                create_graph_card("Age & Residency Correlation", "graph-3", DESCRIPTIONS["g3"], width_lg=6, is_table=True),
                create_graph_card("Age-Specific Prevalence", "graph-4", DESCRIPTIONS["g4"], width_lg=6),
                create_graph_card("Regional LTHC Heatmap Matrix", "graph-5", DESCRIPTIONS["g5"], width_lg=12),
                create_graph_card("Cases & Prevalence by Country of Birth", "graph-6", DESCRIPTIONS["g6"], width_lg=12),
            ]
        ),
    ],
)

# --- Inferences Page Layout ---
inferences_layout = dbc.Container(
    className="px-4 pb-4",
    style={"maxWidth": "900px"},
    children=[
        # Header banner
        dbc.Card(
            className="shadow-sm border-0 mb-4 bg-primary text-white rounded-3",
            children=[
                dbc.CardBody(
                    children=[
                        html.H3(
                            "Population Long-Term Health Condition Analytics",
                            className="fw-bold mb-1 text-white text-center",
                        ),
                        html.P(
                            "Population-level analysis and prediction of LTHC prevalence",
                            className="text-white-50 mb-2 text-center",
                        ),
                        html.P(
                            "⚠️ NOT a medical diagnostic tool",
                            className="fw-semibold mb-0 text-center text-warning",
                        ),
                    ]
                )
            ],
        ),
        # Selection form
        dbc.Card(
            className="shadow-sm border-0 rounded-3 mb-4",
            children=[
                dbc.CardHeader(
                    html.H5("1. Select Population Group", className="mb-0 fw-bold text-dark"),
                    className="bg-white border-bottom-0 pt-3 pb-2",
                ),
                dbc.CardBody(
                    children=[
                        *[
                            create_inference_dropdown(label, comp_id, opts)
                            for label, comp_id, opts in INFERENCE_FIELDS
                        ],
                        html.Hr(),
                        dbc.Row(
                            className="mb-3 align-items-center",
                            children=[
                                dbc.Col(
                                    html.Label("Prediction Mode", className="fw-semibold mb-0"),
                                    xs=12, md=4,
                                ),
                                dbc.Col(
                                    dbc.RadioItems(
                                        id="inf-mode",
                                        options=[
                                            {"label": "Single condition", "value": "single"},
                                            {"label": "All conditions", "value": "all"},
                                        ],
                                        value="single",
                                        inline=True,
                                    ),
                                    xs=12, md=8,
                                ),
                            ],
                        ),
                        html.Div(
                            id="inf-lthc-container",
                            children=create_inference_dropdown(
                                "Long-Term Health Condition (LTHC)", "inf-lthc", LTHC_OPTIONS
                            ),
                        ),
                        html.Div(
                            dbc.Button(
                                "Analyse Population",
                                id="analyse-button",
                                color="primary",
                                size="lg",
                                n_clicks=0,
                            ),
                            className="text-center mt-4",
                        ),
                    ]
                ),
            ],
        ),
        # Results placeholder
        dcc.Loading(html.Div(id="inference-results"), type="circle"),
    ],
)

#Main App Layout
app.layout = html.Div(
    className="bg-light min-vh-100",
    children=[
        dcc.Location(id='url', refresh=False),
        navbar,
        html.Div(id='page-content')
    ]
)


#5. CALLBACKS
# =========================================================

# Callback to handle page routing
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/inferences':
        return inferences_layout
    else:
        # Default to home layout for '/' or any undefined route
        return home_layout


# Callback for updating Dashboard Graphs
@app.callback(
    [
        Output("graph-1", "figure"),
        Output("graph-2", "children"),
        Output("graph-3", "children"),
        Output("graph-4", "figure"),
        Output("graph-5", "figure"),
        Output("graph-6", "figure"),
    ],
    Input("condition-filter-dropdown", "value"),
)
def update_dashboard_graphs(selected_filter):
    filtered_df = top_ten_filter(df) if selected_filter == "TOP_TEN" else df

    fig1 = lthc_to_gender(filtered_df)
    table_df = lthc_to_gender_age(filtered_df)
    table_two = ltch_to_age_years_in_aussie(filtered_df)
    fig4 = age_to_lthc(filtered_df)
    fig5 = heatmap(filtered_df)
    fig6 = treemap(filtered_df)

    data_table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": str(i), "id": str(i)} for i in table_df.columns],
        style_table={'overflowX': 'auto', 'overflowY': 'auto', 'height': '400px'},
        style_cell={'textAlign': 'left', 'padding': '8px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
    )

    data_table_two = dash_table.DataTable(
        data=table_two.to_dict("records"),
        columns=[{"name": str(i), "id": str(i)} for i in table_two.columns],
        style_table={'overflowX': 'auto', 'overflowY': 'auto', 'height': '400px'},
        style_cell={'textAlign': 'left', 'padding': '8px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
    )

    return fig1, data_table, data_table_two, fig4, fig5, fig6


def render_api_result(selections, result):
    """Build the results card from the FastAPI JSON response."""
    if not isinstance(result, dict):
        result = {"prediction": result}

    # Pick out the headline value if the API uses a common key name
    headline_key = next(
        (k for k in ("prediction", "predicted_prevalence", "prevalence", "result") if k in result),
        None,
    )
    headline = result.get(headline_key) if headline_key else None
    if isinstance(headline, float):
        # 4 significant figures, so small values (e.g. 0.0032) don't collapse to 0.0
        headline = f"{headline:.4g}"

    children = []
    if headline is not None:
        children.append(
            html.Div(
                [
                    html.P("Model result", className="text-muted mb-1 text-center"),
                    html.H2(str(headline), className="fw-bold text-primary text-center mb-3"),
                ]
            )
        )

    other_items = {k: v for k, v in result.items() if k != headline_key}
    if other_items:
        children.append(
            dbc.ListGroup(
                [dbc.ListGroupItem(f"{k}: {v}") for k, v in other_items.items()],
                flush=True,
                className="mb-3",
            )
        )

    children.append(
        html.Details(
            [
                html.Summary("Raw API response (for debugging)"),
                html.Pre(json.dumps(result, indent=2, default=str), className="small bg-light p-2 mt-2"),
            ],
            className="mb-3",
        )
    )

    children.append(html.H6("Inputs sent to the model", className="fw-bold mt-3"))
    children.append(
        dbc.ListGroup(
            [dbc.ListGroupItem(f"{k}: {v}") for k, v in selections.items()],
            flush=True,
        )
    )

    return dbc.Card(
        className="shadow-sm border-0 rounded-3",
        children=[
            dbc.CardHeader(
                html.H5("2. Inference Result", className="mb-0 fw-bold"),
                className="bg-white border-bottom-0 pt-3 pb-2",
            ),
            dbc.CardBody(children),
        ],
    )


def fmt_value(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def fetch_prediction(payload):
    response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


def predict_all_conditions(base_payload):
    """Call the API once per condition (in parallel). Returns (ranked, failed)."""
    lthc_key = API_FIELD_MAP["lthc"]

    def one(cond):
        try:
            result = fetch_prediction({**base_payload, lthc_key: cond})
        except Exception as e:
            return cond, None, str(e)
        if not isinstance(result, dict):
            result = {"prediction": result}
        key = next((k for k in HEADLINE_KEYS if k in result), None)
        try:
            return cond, (float(result[key]), key, result), None
        except (TypeError, ValueError, KeyError):
            return cond, None, "no numeric prediction in response"

    with ThreadPoolExecutor(max_workers=6) as ex:
        outcomes = list(ex.map(one, ALL_CONDITIONS))

    ranked = sorted(
        [(c, v, k, r) for c, out, err in outcomes if out for v, k, r in [out]],
        key=lambda x: x[1],
        reverse=True,
    )
    failed = {c: err for c, out, err in outcomes if err}
    return ranked, failed


def render_all_conditions_result(selections, ranked, failed):
    """Results card for 'All conditions' mode."""
    top_cond, top_val, top_key, top_result = ranked[0]

    children = [
        html.P("Most prevalent condition", className="text-muted mb-1 text-center"),
        html.H3(top_cond, className="fw-bold text-center mb-1"),
        html.H2(fmt_value(top_val), className="fw-bold text-primary text-center mb-3"),
    ]

    # Extra info (units, percentage, etc.) is shown for the top condition only
    top_extra = {k: v for k, v in top_result.items() if k != top_key}
    if top_extra:
        children.append(html.H6(f"Details for {top_cond}", className="fw-bold"))
        children.append(
            dbc.ListGroup(
                [dbc.ListGroupItem(f"{k}: {v}") for k, v in top_extra.items()],
                flush=True,
                className="mb-3",
            )
        )

    others = ranked[1:]
    if others:
        children.append(html.H6("Other conditions (prevalence)", className="fw-bold"))
        children.append(
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"{i}. {c}: {fmt_value(v)}")
                    for i, (c, v, _, _) in enumerate(others, start=2)
                ],
                flush=True,
                className="mb-3",
            )
        )

    if failed:
        children.append(
            dbc.Alert(
                "No prediction returned for: " + ", ".join(failed),
                color="warning",
                className="mb-3",
            )
        )

    children.append(
        html.Details(
            [
                html.Summary("Raw API responses (for debugging)"),
                html.Pre(
                    json.dumps({c: r for c, _, _, r in ranked}, indent=2, default=str),
                    className="small bg-light p-2 mt-2",
                ),
            ],
            className="mb-3",
        )
    )
    children.append(html.H6("Inputs sent to the model", className="fw-bold mt-3"))
    children.append(
        dbc.ListGroup(
            [dbc.ListGroupItem(f"{k}: {v}") for k, v in selections.items()],
            flush=True,
        )
    )

    return dbc.Card(
        className="shadow-sm border-0 rounded-3",
        children=[
            dbc.CardHeader(
                html.H5("2. Inference Result", className="mb-0 fw-bold"),
                className="bg-white border-bottom-0 pt-3 pb-2",
            ),
            dbc.CardBody(children),
        ],
    )


# Show/hide the single-condition dropdown depending on the selected mode
@app.callback(
    Output("inf-lthc-container", "style"),
    Input("inf-mode", "value"),
)
def toggle_lthc_dropdown(mode):
    return {"display": "none"} if mode == "all" else {}


# Callback for the "Analyse Population" button: sends the selections to FastAPI
@app.callback(
    Output("inference-results", "children"),
    Input("analyse-button", "n_clicks"),
    [
        State("inf-country", "value"),
        State("inf-age", "value"),
        State("inf-sex", "value"),
        State("inf-years", "value"),
        State("inf-language", "value"),
        State("inf-english", "value"),
        State("inf-region", "value"),
        State("inf-sub_region", "value"),
        State("inf-lthc", "value"),
        State("inf-mode", "value"),
    ],
    prevent_initial_call=True,
)
def analyse_population(
    n_clicks, country, age, sex, years, language, english, region, sub_region, lthc, mode
):
    selections = {
        "Country of Birth": country,
        "Age Group": age,
        "Sex": sex,
        "Years in Australia": years,
        "Language Used at Home": language,
        "English Proficiency": english,
        "Region": region,
        "Sub-region": sub_region,
        "LTHC": "All conditions" if mode == "all" else lthc,
    }
    missing = [k for k, v in selections.items() if not v]
    if missing:
        return dbc.Alert(
            f"Please select a value for: {', '.join(missing)}",
            color="warning",
        )

    form_values = {
        "country": country, "age": age, "sex": sex, "years": years,
        "language": language, "english": english, "region": region,
        "sub_region": sub_region, "lthc": lthc,
    }
   
    def clean(v):
        return v.replace("\u2013", "-").replace("\u2014", "-") if isinstance(v, str) else v

    payload = {API_FIELD_MAP[k]: clean(v) for k, v in form_values.items()}

    if mode == "all":
        ranked, failed = predict_all_conditions(payload)
        if not ranked:
            return dbc.Alert(
                [
                    html.Strong("Could not get predictions for any condition. "),
                    html.Code("; ".join(f"{c}: {e}" for c, e in list(failed.items())[:3])[:1000]),
                ],
                color="danger",
            )
        return render_all_conditions_result(selections, ranked, failed)

    try:
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.Timeout:
        return dbc.Alert(
            "The inference API took too long to respond. It may be waking up - please try again in a moment.",
            color="warning",
        )
    except requests.exceptions.HTTPError:
        return dbc.Alert(
            [
                html.Strong(f"API returned an error ({response.status_code}). "),
                html.Code(response.text[:3000]),
            ],
            color="danger",
        )
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Could not reach the inference API: {e}", color="danger")
    except ValueError:
        return dbc.Alert("The API response was not valid JSON.", color="danger")

    return render_api_result(selections, result)



# 6. RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
