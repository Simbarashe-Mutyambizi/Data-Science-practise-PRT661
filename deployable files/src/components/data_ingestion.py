# -*- coding: utf-8 -*-
"""
data_ingestion.py - PRT661 Group Dan3

Loads the AIHW 2021 CALD workbook, cleans it and hands the cleaned file to
DataTransformation, ModelTrainer, the FastAPI server and the dashboard.

Cleaning follows Data_Cleaning.py (Assessment 3 feedback):
  * NO country <-> language imputation. Table S10 (country of birth, years in
    Australia) and Table S13 (language used at home, English proficiency) are
    kept as separate sources. Columns a table never collected are set to
    "Not collected" instead of being filled by an AI-generated mapping.
  * Cells AIHW suppressed ("n.p") are dropped, not converted to 0. The
    published data contains no 0% cells.
  * "Persons" rows are kept (Persons = Male and Female combined). Downstream,
    the train/test grouping must leave Sex out of the group key so a cell's
    Persons, Male and Female rows always land on the same side of a split.
  * Region and subregion are looked up from country of birth, so they only
    exist for S10 rows; S13 rows get "Not collected".
"""
import os
import sys
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.components.model_trainer import ModelTrainer
import pandas as pd
from dataclasses import dataclass
import uvicorn
from src.components.api_server import app as api_app
from src.components.app import app as dashboard_app
import threading
import time
import requests

NOT_COLLECTED = "Not collected"

COUNTRY = "Country of birth of person"
YEARS = "Years spent in Australia"
LANGUAGE = "Language used at home"
PROFICIENCY = "Proficiency in spoken English"
AGE = "Age group"
SEX = "Sex"
LTHC = "Long-term health condition (LTHC)"
NUMBER = "Number of people reporting LTHC(s)"
POPULATION = "Population"
TARGET = "Age-specific percentage of population reporting LTHC(s)"

CATEGORICAL_COLUMNS = [COUNTRY, YEARS, LANGUAGE, PROFICIENCY,
                       "Region_class", "Subregion_class", AGE, SEX, LTHC]
NUMERIC_COLUMNS = [NUMBER, POPULATION, TARGET]
# Same column order as the previous cleaned file, so DataTransformation keeps working
OUTPUT_COLUMNS = [COUNTRY, YEARS, AGE, SEX, LTHC, NUMBER, POPULATION, TARGET,
                  LANGUAGE, PROFICIENCY, "Region_class", "Subregion_class"]


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(
        'src', 'components',
        'aihw-phe-320-age-specific-percentages-chronic-conditions-cald-2021.xlsx')
    cleaned_data_path: str = os.path.join('artifacts', 'cleaned', 'Cleaned_data_final.csv')
    # Removing Persons would cut 16,184 rows to 6,278 and drop every country /
    # language that AIHW only publishes for Persons (168 -> 67 countries,
    # 147 -> 61 languages), so it stays False.
    remove_persons: bool = False


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Loading raw data")
        try:
            # 1. Load the two source tables separately
            raw = self.ingestion_config.raw_data_path
            s10 = pd.read_excel(raw, sheet_name='Table S10', skiprows=3)
            s13 = pd.read_excel(raw, sheet_name='Table S13', skiprows=3)
            logging.info(f"Dataframes created: S10 {len(s10):,} rows, S13 {len(s13):,} rows")

            # 2. Region and subregion come from country of birth (S10 only).
            #    This is a geographic lookup, not a language guess.
            s10["Region_class"] = s10[COUNTRY].map(COUNTRY_TO_REGION)
            s10["Subregion_class"] = s10[COUNTRY].map(COUNTRY_TO_SUBREGION)

            # 3. Stack the tables. Columns a table never collected become NaN here.
            s10_rows = len(s10)
            concat_df = pd.concat([s10, s13], ignore_index=True, sort=False)
            is_s10 = pd.Series([True] * s10_rows + [False] * len(s13))
            logging.info("Dataframes merged into one dataset (no imputation)")

            # 4. Drop suppressed cells ("n.p"). Count, population and percentage
            #    are always suppressed together, so these rows hold no data.
            for col in NUMERIC_COLUMNS:
                concat_df[col] = pd.to_numeric(concat_df[col], errors="coerce")
            published = concat_df[TARGET].notna()
            logging.info(f"Suppressed (n.p) rows dropped: {(~published).sum():,}")
            concat_df, is_s10 = concat_df.loc[published], is_s10.loc[published]

            # 5. Persons rows are kept unless the config says otherwise
            if self.ingestion_config.remove_persons:
                keep = concat_df[SEX] != "Persons"
                logging.info(f"Persons rows removed: {(~keep).sum():,}")
                concat_df, is_s10 = concat_df.loc[keep], is_s10.loc[keep]

            # 6. Every categorical gap is a variable the source table did not collect
            unmapped = sorted(set(concat_df.loc[is_s10, COUNTRY]) - set(COUNTRY_TO_REGION))
            if unmapped:
                raise ValueError(f"Countries with no region mapping: {unmapped}")
            concat_df[CATEGORICAL_COLUMNS] = concat_df[CATEGORICAL_COLUMNS].fillna(NOT_COLLECTED)
            logging.info('Null values set to "Not collected"')

            # 7. Checks
            if concat_df[OUTPUT_COLUMNS].isna().sum().sum() != 0:
                raise ValueError("Null values remain after cleaning")
            if not ((concat_df[TARGET] > 0) & (concat_df[TARGET] <= 100)).all():
                raise ValueError("Prevalence outside (0, 100]")
            if not (concat_df.loc[is_s10, [LANGUAGE, PROFICIENCY]] == NOT_COLLECTED).all().all():
                raise ValueError("S10 rows should not have language or proficiency values")
            s13_only = [COUNTRY, YEARS, "Region_class", "Subregion_class"]
            if not (concat_df.loc[~is_s10, s13_only] == NOT_COLLECTED).all().all():
                raise ValueError("S13 rows should not have country, years or region values")

            print(concat_df.isnull().sum())
            print("Duplicate rows:", concat_df.duplicated().sum())
            logging.info(
                f"Cleaned data: {len(concat_df):,} rows "
                f"(S10 {int(is_s10.sum()):,}, S13 {int((~is_s10).sum()):,}), "
                f"sex counts {concat_df[SEX].value_counts().to_dict()}"
            )

            # 8. Save
            logging.info("Load cleaned data to artifacts")
            os.makedirs(os.path.dirname(self.ingestion_config.cleaned_data_path), exist_ok=True)
            concat_df[OUTPUT_COLUMNS].to_csv(self.ingestion_config.cleaned_data_path,
                                             index=False, header=True)
            logging.info("Cleaning completed, moving to transformation")
            print(len(concat_df))
            return self.ingestion_config.cleaned_data_path

        except Exception as e:
            raise CustomException(e, sys)


# --------------------------------------------------------------------------
# Country of birth -> subregion / region (ABS SACC groups). Applied to S10
# rows only; S13 rows record language, not country.
# --------------------------------------------------------------------------
COUNTRY_TO_SUBREGION = {
    # Oceania and Antarctica
    'Oceania and Antarctica, nfd': 'Oceania (nfd)',
    'New Zealand': 'New Zealand',
    'Melanesia, nfd': 'Melanesia',
    'New Caledonia': 'Melanesia',
    'Papua New Guinea': 'Melanesia',
    'Solomon Islands': 'Melanesia',
    'Vanuatu': 'Melanesia',
    'Micronesia, nfd': 'Micronesia',
    'Guam': 'Micronesia',
    'Kiribati': 'Micronesia',
    'Marshall Islands': 'Micronesia',
    'Micronesia, Federated States of': 'Micronesia',
    'Nauru': 'Micronesia',
    'Northern Mariana Islands': 'Micronesia',
    'Palau': 'Micronesia',
    'Cook Islands': 'Polynesia',
    'Fiji': 'Polynesia',
    'French Polynesia': 'Polynesia',
    'Niue': 'Polynesia',
    'Samoa': 'Polynesia',
    'Samoa, American': 'Polynesia',
    'Tokelau': 'Polynesia',
    'Tonga': 'Polynesia',
    'Tuvalu': 'Polynesia',
    'Wallis and Futuna': 'Polynesia',
    'Pitcairn Islands': 'Polynesia',
    'Polynesia (excludes Hawaii), nec': 'Polynesia',
    'Antarctica, nfd': 'Antarctica',
    'Chilean Antarctic Territory': 'Antarctica',

    # North-West Europe
    'United Kingdom, Channel Islands and Isle of Ma': 'United Kingdom',
    'England': 'United Kingdom',
    'Isle of Man': 'United Kingdom',
    'Northern Ireland': 'United Kingdom',
    'Scotland': 'United Kingdom',
    'Wales': 'United Kingdom',
    'Guernsey': 'United Kingdom',
    'Jersey': 'United Kingdom',
    'Ireland': 'Ireland',
    'Western Europe, nfd': 'Western Europe',
    'Austria': 'Western Europe',
    'Belgium': 'Western Europe',
    'France': 'Western Europe',
    'Germany': 'Western Europe',
    'Liechtenstein': 'Western Europe',
    'Luxembourg': 'Western Europe',
    'Monaco': 'Western Europe',
    'Netherlands': 'Western Europe',
    'Switzerland': 'Western Europe',
    'Northern Europe, nfd': 'Northern Europe',
    'Denmark': 'Northern Europe',
    'Faroe Islands': 'Northern Europe',
    'Finland': 'Northern Europe',
    'Greenland': 'Northern Europe',
    'Iceland': 'Northern Europe',
    'Norway': 'Northern Europe',
    'Sweden': 'Northern Europe',
    'Aland Islands': 'Northern Europe',

    # Southern and Eastern Europe
    'Southern and Eastern Europe, nfd': 'Southern Europe',
    'Andorra': 'Southern Europe',
    'Gibraltar': 'Southern Europe',
    'Holy See': 'Southern Europe',
    'Italy': 'Southern Europe',
    'Malta': 'Southern Europe',
    'Portugal': 'Southern Europe',
    'San Marino': 'Southern Europe',
    'Spain': 'Southern Europe',
    'South Eastern Europe, nfd': 'South Eastern Europe',
    'Albania': 'South Eastern Europe',
    'Bosnia and Herzegovina': 'South Eastern Europe',
    'Bulgaria': 'South Eastern Europe',
    'Croatia': 'South Eastern Europe',
    'Cyprus': 'South Eastern Europe',
    'North Macedonia': 'South Eastern Europe',
    'Greece': 'South Eastern Europe',
    'Moldova': 'South Eastern Europe',
    'Romania': 'South Eastern Europe',
    'Slovenia': 'South Eastern Europe',
    'Montenegro': 'South Eastern Europe',
    'Serbia': 'South Eastern Europe',
    'Kosovo': 'South Eastern Europe',
    'Eastern Europe, nfd': 'Eastern Europe',
    'Belarus': 'Eastern Europe',
    'Czechia': 'Eastern Europe',
    'Estonia': 'Eastern Europe',
    'Hungary': 'Eastern Europe',
    'Latvia': 'Eastern Europe',
    'Lithuania': 'Eastern Europe',
    'Poland': 'Eastern Europe',
    'Russian Federation': 'Eastern Europe',
    'Slovakia': 'Eastern Europe',
    'Ukraine': 'Eastern Europe',

    # North Africa and the Middle East
    'North Africa and the Middle East, nfd': 'North Africa and Middle East (nfd)',
    'North Africa, nfd': 'North Africa',
    'Algeria': 'North Africa',
    'Egypt': 'North Africa',
    'Libya': 'North Africa',
    'Morocco': 'North Africa',
    'Sudan': 'North Africa',
    'Tunisia': 'North Africa',
    'Western Sahara': 'North Africa',
    'South Sudan': 'North Africa',
    'Middle East, nfd': 'Middle East',
    'Bahrain': 'Middle East',
    'Gaza Strip and West Bank': 'Middle East',
    'Iran': 'Middle East',
    'Iraq': 'Middle East',
    'Israel': 'Middle East',
    'Jordan': 'Middle East',
    'Kuwait': 'Middle East',
    'Lebanon': 'Middle East',
    'Oman': 'Middle East',
    'Qatar': 'Middle East',
    'Saudi Arabia': 'Middle East',
    'Syria': 'Middle East',
    'Turkey': 'Middle East',
    'United Arab Emirates': 'Middle East',
    'Yemen': 'Middle East',

    # South-East Asia
    'South-East Asia, nfd': 'South-East Asia (nfd)',
    'Mainland South-East Asia, nfd': 'Mainland South-East Asia',
    'Myanmar': 'Mainland South-East Asia',
    'Cambodia': 'Mainland South-East Asia',
    'Laos': 'Mainland South-East Asia',
    'Thailand': 'Mainland South-East Asia',
    'Vietnam': 'Mainland South-East Asia',
    'Maritime South-East Asia, nfd': 'Maritime South-East Asia',
    'Brunei Darussalam': 'Maritime South-East Asia',
    'Indonesia': 'Maritime South-East Asia',
    'Malaysia': 'Maritime South-East Asia',
    'Philippines': 'Maritime South-East Asia',
    'Singapore': 'Maritime South-East Asia',
    'Timor-Leste': 'Maritime South-East Asia',

    # North-East Asia
    'Chinese Asia (includes Mongolia), nfd': 'Chinese Asia',
    'China (excludes SARs and Taiwan)': 'Chinese Asia',
    'Hong Kong (SAR of China)': 'Chinese Asia',
    'Macau (SAR of China)': 'Chinese Asia',
    'Mongolia': 'Chinese Asia',
    'Taiwan': 'Chinese Asia',
    'Japan': 'Japan and the Koreas',
    "Korea, Democratic People's Republic of (North)": 'Japan and the Koreas',
    'Korea, Republic of (South)': 'Japan and the Koreas',

    # Southern and Central Asia
    'Southern and Central Asia, nfd': 'Southern and Central Asia (nfd)',
    'Southern Asia, nfd': 'Southern Asia',
    'Bangladesh': 'Southern Asia',
    'Bhutan': 'Southern Asia',
    'India': 'Southern Asia',
    'Maldives': 'Southern Asia',
    'Nepal': 'Southern Asia',
    'Pakistan': 'Southern Asia',
    'Sri Lanka': 'Southern Asia',
    'Central Asia, nfd': 'Central Asia',
    'Afghanistan': 'Central Asia',
    'Armenia': 'Central Asia',
    'Azerbaijan': 'Central Asia',
    'Georgia': 'Central Asia',
    'Kazakhstan': 'Central Asia',
    'Kyrgyzstan': 'Central Asia',
    'Tajikistan': 'Central Asia',
    'Turkmenistan': 'Central Asia',
    'Uzbekistan': 'Central Asia',

    # Americas
    'Americas, nfd': 'Americas (nfd)',
    'Northern America, nfd': 'Northern America',
    'Bermuda': 'Northern America',
    'Canada': 'Northern America',
    'United States of America': 'Northern America',
    'South America, nfd': 'South America',
    'Argentina': 'South America',
    'Bolivia': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Ecuador': 'South America',
    'Falkland Islands': 'South America',
    'French Guiana': 'South America',
    'Guyana': 'South America',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Suriname': 'South America',
    'Uruguay': 'South America',
    'Venezuela': 'South America',
    'Central America, nfd': 'Central America',
    'Belize': 'Central America',
    'Costa Rica': 'Central America',
    'El Salvador': 'Central America',
    'Guatemala': 'Central America',
    'Honduras': 'Central America',
    'Mexico': 'Central America',
    'Nicaragua': 'Central America',
    'Panama': 'Central America',
    'Caribbean, nfd': 'Caribbean',
    'Anguilla': 'Caribbean',
    'Antigua and Barbuda': 'Caribbean',
    'Aruba': 'Caribbean',
    'Bahamas': 'Caribbean',
    'Barbados': 'Caribbean',
    'Cayman Islands': 'Caribbean',
    'Cuba': 'Caribbean',
    'Dominica': 'Caribbean',
    'Dominican Republic': 'Caribbean',
    'Grenada': 'Caribbean',
    'Guadeloupe': 'Caribbean',
    'Haiti': 'Caribbean',
    'Jamaica': 'Caribbean',
    'Martinique': 'Caribbean',
    'Montserrat': 'Caribbean',
    'Puerto Rico': 'Caribbean',
    'St Kitts and Nevis': 'Caribbean',
    'St Lucia': 'Caribbean',
    'St Vincent and the Grenadines': 'Caribbean',
    'Trinidad and Tobago': 'Caribbean',
    'Turks and Caicos Islands': 'Caribbean',
    'Virgin Islands, British': 'Caribbean',
    'Virgin Islands, United States': 'Caribbean',
    'St Martin (French part)': 'Caribbean',
    'Bonaire, Sint Eustatius and Saba': 'Caribbean',
    'Curacao': 'Caribbean',
    'Sint Maarten (Dutch part)': 'Caribbean',

    # Sub-Saharan Africa
    'Sub-Saharan Africa, nfd': 'Sub-Saharan Africa (nfd)',
    'Central and West Africa, nfd': 'Central and West Africa',
    'Benin': 'Central and West Africa',
    'Burkina Faso': 'Central and West Africa',
    'Cameroon': 'Central and West Africa',
    'Cabo Verde': 'Central and West Africa',
    'Central African Republic': 'Central and West Africa',
    'Chad': 'Central and West Africa',
    'Congo, Republic of': 'Central and West Africa',
    'Congo, Democratic Republic of': 'Central and West Africa',
    "Cote d'Ivoire": 'Central and West Africa',
    'Equatorial Guinea': 'Central and West Africa',
    'Gabon': 'Central and West Africa',
    'Gambia': 'Central and West Africa',
    'Ghana': 'Central and West Africa',
    'Guinea': 'Central and West Africa',
    'Guinea-Bissau': 'Central and West Africa',
    'Liberia': 'Central and West Africa',
    'Mali': 'Central and West Africa',
    'Mauritania': 'Central and West Africa',
    'Niger': 'Central and West Africa',
    'Nigeria': 'Central and West Africa',
    'Sao Tome and Principe': 'Central and West Africa',
    'Senegal': 'Central and West Africa',
    'Sierra Leone': 'Central and West Africa',
    'Togo': 'Central and West Africa',
    'Southern and East Africa, nfd': 'Southern and East Africa',
    'Angola': 'Southern and East Africa',
    'Botswana': 'Southern and East Africa',
    'Burundi': 'Southern and East Africa',
    'Comoros': 'Southern and East Africa',
    'Djibouti': 'Southern and East Africa',
    'Eritrea': 'Southern and East Africa',
    'Ethiopia': 'Southern and East Africa',
    'Kenya': 'Southern and East Africa',
    'Lesotho': 'Southern and East Africa',
    'Madagascar': 'Southern and East Africa',
    'Malawi': 'Southern and East Africa',
    'Mauritius': 'Southern and East Africa',
    'Mayotte': 'Southern and East Africa',
    'Mozambique': 'Southern and East Africa',
    'Namibia': 'Southern and East Africa',
    'Reunion': 'Southern and East Africa',
    'Rwanda': 'Southern and East Africa',
    'St Helena': 'Southern and East Africa',
    'Seychelles': 'Southern and East Africa',
    'Somalia': 'Southern and East Africa',
    'South Africa': 'Southern and East Africa',
    'Eswatini': 'Southern and East Africa',
    'Tanzania': 'Southern and East Africa',
    'Uganda': 'Southern and East Africa',
    'Zambia': 'Southern and East Africa',
    'Zimbabwe': 'Southern and East Africa'
}


COUNTRY_TO_REGION = {
    # 1. Oceania and Antarctica
    'Oceania and Antarctica, nfd': 'Oceania and Antarctica',
    'New Zealand': 'Oceania and Antarctica',
    'Melanesia, nfd': 'Oceania and Antarctica',
    'New Caledonia': 'Oceania and Antarctica',
    'Papua New Guinea': 'Oceania and Antarctica',
    'Solomon Islands': 'Oceania and Antarctica',
    'Vanuatu': 'Oceania and Antarctica',
    'Micronesia, nfd': 'Oceania and Antarctica',
    'Guam': 'Oceania and Antarctica',
    'Kiribati': 'Oceania and Antarctica',
    'Marshall Islands': 'Oceania and Antarctica',
    'Micronesia, Federated States of': 'Oceania and Antarctica',
    'Nauru': 'Oceania and Antarctica',
    'Northern Mariana Islands': 'Oceania and Antarctica',
    'Palau': 'Oceania and Antarctica',
    'Cook Islands': 'Oceania and Antarctica',
    'Fiji': 'Oceania and Antarctica',
    'French Polynesia': 'Oceania and Antarctica',
    'Niue': 'Oceania and Antarctica',
    'Samoa': 'Oceania and Antarctica',
    'Samoa, American': 'Oceania and Antarctica',
    'Tokelau': 'Oceania and Antarctica',
    'Tonga': 'Oceania and Antarctica',
    'Tuvalu': 'Oceania and Antarctica',
    'Wallis and Futuna': 'Oceania and Antarctica',
    'Pitcairn Islands': 'Oceania and Antarctica',
    'Polynesia (excludes Hawaii), nec': 'Oceania and Antarctica',
    'Antarctica, nfd': 'Oceania and Antarctica',
    'Chilean Antarctic Territory': 'Oceania and Antarctica',

    # 2. North-West Europe
    'United Kingdom, Channel Islands and Isle of Ma': 'North-West Europe',
    'England': 'North-West Europe',
    'Isle of Man': 'North-West Europe',
    'Northern Ireland': 'North-West Europe',
    'Scotland': 'North-West Europe',
    'Wales': 'North-West Europe',
    'Guernsey': 'North-West Europe',
    'Jersey': 'North-West Europe',
    'Ireland': 'North-West Europe',
    'Western Europe, nfd': 'North-West Europe',
    'Austria': 'North-West Europe',
    'Belgium': 'North-West Europe',
    'France': 'North-West Europe',
    'Germany': 'North-West Europe',
    'Liechtenstein': 'North-West Europe',
    'Luxembourg': 'North-West Europe',
    'Monaco': 'North-West Europe',
    'Netherlands': 'North-West Europe',
    'Switzerland': 'North-West Europe',
    'Northern Europe, nfd': 'North-West Europe',
    'Denmark': 'North-West Europe',
    'Faroe Islands': 'North-West Europe',
    'Finland': 'North-West Europe',
    'Greenland': 'North-West Europe',
    'Iceland': 'North-West Europe',
    'Norway': 'North-West Europe',
    'Sweden': 'North-West Europe',
    'Aland Islands': 'North-West Europe',

    # 3. Southern and Eastern Europe
    'Southern and Eastern Europe, nfd': 'Southern and Eastern Europe',
    'Andorra': 'Southern and Eastern Europe',
    'Gibraltar': 'Southern and Eastern Europe',
    'Holy See': 'Southern and Eastern Europe',
    'Italy': 'Southern and Eastern Europe',
    'Malta': 'Southern and Eastern Europe',
    'Portugal': 'Southern and Eastern Europe',
    'San Marino': 'Southern and Eastern Europe',
    'Spain': 'Southern and Eastern Europe',
    'South Eastern Europe, nfd': 'Southern and Eastern Europe',
    'Albania': 'Southern and Eastern Europe',
    'Bosnia and Herzegovina': 'Southern and Eastern Europe',
    'Bulgaria': 'Southern and Eastern Europe',
    'Croatia': 'Southern and Eastern Europe',
    'Cyprus': 'Southern and Eastern Europe',
    'North Macedonia': 'Southern and Eastern Europe',
    'Greece': 'Southern and Eastern Europe',
    'Moldova': 'Southern and Eastern Europe',
    'Romania': 'Southern and Eastern Europe',
    'Slovenia': 'Southern and Eastern Europe',
    'Montenegro': 'Southern and Eastern Europe',
    'Serbia': 'Southern and Eastern Europe',
    'Kosovo': 'Southern and Eastern Europe',
    'Eastern Europe, nfd': 'Southern and Eastern Europe',
    'Belarus': 'Southern and Eastern Europe',
    'Czechia': 'Southern and Eastern Europe',
    'Estonia': 'Southern and Eastern Europe',
    'Hungary': 'Southern and Eastern Europe',
    'Latvia': 'Southern and Eastern Europe',
    'Lithuania': 'Southern and Eastern Europe',
    'Poland': 'Southern and Eastern Europe',
    'Russian Federation': 'Southern and Eastern Europe',
    'Slovakia': 'Southern and Eastern Europe',
    'Ukraine': 'Southern and Eastern Europe',

    # 4. North Africa and the Middle East
    'North Africa and the Middle East, nfd': 'North Africa and the Middle East',
    'North Africa, nfd': 'North Africa and the Middle East',
    'Algeria': 'North Africa and the Middle East',
    'Egypt': 'North Africa and the Middle East',
    'Libya': 'North Africa and the Middle East',
    'Morocco': 'North Africa and the Middle East',
    'Sudan': 'North Africa and the Middle East',
    'Tunisia': 'North Africa and the Middle East',
    'Western Sahara': 'North Africa and the Middle East',
    'South Sudan': 'North Africa and the Middle East',
    'Middle East, nfd': 'North Africa and the Middle East',
    'Bahrain': 'North Africa and the Middle East',
    'Gaza Strip and West Bank': 'North Africa and the Middle East',
    'Iran': 'North Africa and the Middle East',
    'Iraq': 'North Africa and the Middle East',
    'Israel': 'North Africa and the Middle East',
    'Jordan': 'North Africa and the Middle East',
    'Kuwait': 'North Africa and the Middle East',
    'Lebanon': 'North Africa and the Middle East',
    'Oman': 'North Africa and the Middle East',
    'Qatar': 'North Africa and the Middle East',
    'Saudi Arabia': 'North Africa and the Middle East',
    'Syria': 'North Africa and the Middle East',
    'Turkey': 'North Africa and the Middle East',
    'United Arab Emirates': 'North Africa and the Middle East',
    'Yemen': 'North Africa and the Middle East',

    # 5. South-East Asia
    'South-East Asia, nfd': 'South-East Asia',
    'Mainland South-East Asia, nfd': 'South-East Asia',
    'Myanmar': 'South-East Asia',
    'Cambodia': 'South-East Asia',
    'Laos': 'South-East Asia',
    'Thailand': 'South-East Asia',
    'Vietnam': 'South-East Asia',
    'Maritime South-East Asia, nfd': 'South-East Asia',
    'Brunei Darussalam': 'South-East Asia',
    'Indonesia': 'South-East Asia',
    'Malaysia': 'South-East Asia',
    'Philippines': 'South-East Asia',
    'Singapore': 'South-East Asia',
    'Timor-Leste': 'South-East Asia',

    # 6. North-East Asia
    'Chinese Asia (includes Mongolia), nfd': 'North-East Asia',
    'China (excludes SARs and Taiwan)': 'North-East Asia',
    'Hong Kong (SAR of China)': 'North-East Asia',
    'Macau (SAR of China)': 'North-East Asia',
    'Mongolia': 'North-East Asia',
    'Taiwan': 'North-East Asia',
    'Japan': 'North-East Asia',
    "Korea, Democratic People's Republic of (North)": 'North-East Asia',
    'Korea, Republic of (South)': 'North-East Asia',

    # 7. Southern and Central Asia
    'Southern and Central Asia, nfd': 'Southern and Central Asia',
    'Southern Asia, nfd': 'Southern and Central Asia',
    'Bangladesh': 'Southern and Central Asia',
    'Bhutan': 'Southern and Central Asia',
    'India': 'Southern and Central Asia',
    'Maldives': 'Southern and Central Asia',
    'Nepal': 'Southern and Central Asia',
    'Pakistan': 'Southern and Central Asia',
    'Sri Lanka': 'Southern and Central Asia',
    'Central Asia, nfd': 'Southern and Central Asia',
    'Afghanistan': 'Southern and Central Asia',
    'Armenia': 'Southern and Central Asia',
    'Azerbaijan': 'Southern and Central Asia',
    'Georgia': 'Southern and Central Asia',
    'Kazakhstan': 'Southern and Central Asia',
    'Kyrgyzstan': 'Southern and Central Asia',
    'Tajikistan': 'Southern and Central Asia',
    'Turkmenistan': 'Southern and Central Asia',
    'Uzbekistan': 'Southern and Central Asia',

    # 8. Americas
    'Americas, nfd': 'Americas',
    'Northern America, nfd': 'Americas',
    'Bermuda': 'Americas',
    'Canada': 'Americas',
    'United States of America': 'Americas',
    'South America, nfd': 'Americas',
    'Argentina': 'Americas',
    'Bolivia': 'Americas',
    'Brazil': 'Americas',
    'Chile': 'Americas',
    'Colombia': 'Americas',
    'Ecuador': 'Americas',
    'Falkland Islands': 'Americas',
    'French Guiana': 'Americas',
    'Guyana': 'Americas',
    'Paraguay': 'Americas',
    'Peru': 'Americas',
    'Suriname': 'Americas',
    'Uruguay': 'Americas',
    'Venezuela': 'Americas',
    'Central America, nfd': 'Americas',
    'Belize': 'Americas',
    'Costa Rica': 'Americas',
    'El Salvador': 'Americas',
    'Guatemala': 'Americas',
    'Honduras': 'Americas',
    'Mexico': 'Americas',
    'Nicaragua': 'Americas',
    'Panama': 'Americas',
    'Caribbean, nfd': 'Americas',
    'Anguilla': 'Americas',
    'Antigua and Barbuda': 'Americas',
    'Aruba': 'Americas',
    'Bahamas': 'Americas',
    'Barbados': 'Americas',
    'Cayman Islands': 'Americas',
    'Cuba': 'Americas',
    'Dominica': 'Americas',
    'Dominican Republic': 'Americas',
    'Grenada': 'Americas',
    'Guadeloupe': 'Americas',
    'Haiti': 'Americas',
    'Jamaica': 'Americas',
    'Martinique': 'Americas',
    'Montserrat': 'Americas',
    'Puerto Rico': 'Americas',
    'St Kitts and Nevis': 'Americas',
    'St Lucia': 'Americas',
    'St Vincent and the Grenadines': 'Americas',
    'Trinidad and Tobago': 'Americas',
    'Turks and Caicos Islands': 'Americas',
    'Virgin Islands, British': 'Americas',
    'Virgin Islands, United States': 'Americas',
    'St Martin (French part)': 'Americas',
    'Bonaire, Sint Eustatius and Saba': 'Americas',
    'Curacao': 'Americas',
    'Sint Maarten (Dutch part)': 'Americas',

    # 9. Sub-Saharan Africa
    'Sub-Saharan Africa, nfd': 'Sub-Saharan Africa',
    'Central and West Africa, nfd': 'Sub-Saharan Africa',
    'Benin': 'Sub-Saharan Africa',
    'Burkina Faso': 'Sub-Saharan Africa',
    'Cameroon': 'Sub-Saharan Africa',
    'Cabo Verde': 'Sub-Saharan Africa',
    'Central African Republic': 'Sub-Saharan Africa',
    'Chad': 'Sub-Saharan Africa',
    'Congo, Republic of': 'Sub-Saharan Africa',
    'Congo, Democratic Republic of': 'Sub-Saharan Africa',
    "Cote d'Ivoire": 'Sub-Saharan Africa',
    'Equatorial Guinea': 'Sub-Saharan Africa',
    'Gabon': 'Sub-Saharan Africa',
    'Gambia': 'Sub-Saharan Africa',
    'Ghana': 'Sub-Saharan Africa',
    'Guinea': 'Sub-Saharan Africa',
    'Guinea-Bissau': 'Sub-Saharan Africa',
    'Liberia': 'Sub-Saharan Africa',
    'Mali': 'Sub-Saharan Africa',
    'Mauritania': 'Sub-Saharan Africa',
    'Niger': 'Sub-Saharan Africa',
    'Nigeria': 'Sub-Saharan Africa',
    'Sao Tome and Principe': 'Sub-Saharan Africa',
    'Senegal': 'Sub-Saharan Africa',
    'Sierra Leone': 'Sub-Saharan Africa',
    'Togo': 'Sub-Saharan Africa',
    'Southern and East Africa, nfd': 'Sub-Saharan Africa',
    'Angola': 'Sub-Saharan Africa',
    'Botswana': 'Sub-Saharan Africa',
    'Burundi': 'Sub-Saharan Africa',
    'Comoros': 'Sub-Saharan Africa',
    'Djibouti': 'Sub-Saharan Africa',
    'Eritrea': 'Sub-Saharan Africa',
    'Ethiopia': 'Sub-Saharan Africa',
    'Kenya': 'Sub-Saharan Africa',
    'Lesotho': 'Sub-Saharan Africa',
    'Madagascar': 'Sub-Saharan Africa',
    'Malawi': 'Sub-Saharan Africa',
    'Mauritius': 'Sub-Saharan Africa',
    'Mayotte': 'Sub-Saharan Africa',
    'Mozambique': 'Sub-Saharan Africa',
    'Namibia': 'Sub-Saharan Africa',
    'Reunion': 'Sub-Saharan Africa',
    'Rwanda': 'Sub-Saharan Africa',
    'St Helena': 'Sub-Saharan Africa',
    'Seychelles': 'Sub-Saharan Africa',
    'Somalia': 'Sub-Saharan Africa',
    'South Africa': 'Sub-Saharan Africa',
    'Eswatini': 'Sub-Saharan Africa',
    'Tanzania': 'Sub-Saharan Africa',
    'Uganda': 'Sub-Saharan Africa',
    'Zambia': 'Sub-Saharan Africa',
    'Zimbabwe': 'Sub-Saharan Africa'
}


if __name__ == "__main__":
    obj = DataIngestion()
    cleaned_df_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    x_train_path, y_train_path, x_test_path, y_test_path = data_transformation.initiate_data_transformation(cleaned_df_path)

    x_train_arr, x_test_arr, y_train_arr, y_test_arr = DataTransformation.load_arrays(
        x_train_path, y_train_path, x_test_path, y_test_path
    )
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(x_train_arr, x_test_arr, y_train_arr, y_test_arr))
    
    # Run the FastAPI server in the background, then the dashboard
    api_server = uvicorn.Server(uvicorn.Config(api_app, host="127.0.0.1", port=8000))
    threading.Thread(target=api_server.run, daemon=True).start()

    # Wait until the API answers before launching the dashboard
    for _ in range(60):
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=1).status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    else:
        raise RuntimeError("API did not start on http://127.0.0.1:8000")

    print("Dashboard: http://127.0.0.1:8050")
    try:
        dashboard_app.run(debug=False, host="127.0.0.1", port=8050)
    finally:
        api_server.should_exit = True
    