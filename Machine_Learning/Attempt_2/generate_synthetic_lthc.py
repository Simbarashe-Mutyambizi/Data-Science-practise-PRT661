
import pandas as pd
import numpy as np
import requests
import time

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_PATH = "Machine_Learning\Attempt_2\LTHC_phase2_cleaned.csv"
API_URL = "http://127.0.0.1:8000/predict"

NUMBER_OF_TESTS = 10000
OUTPUT_FILE = "synthetic_api_test_results.csv"
REQUEST_TIMEOUT = 10

rng = np.random.default_rng(42)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Only use features accepted by the API.
# Exclude population, LTHC count and target prevalence
# to prevent target leakage.

feature_columns = [
    "Country of birth of person",
    "Years spent in Australia",
    "Age group",
    "Sex",
    "Long-term health condition (LTHC)",
    "Language used at home",
    "Proficiency in spoken English",
    "Region_class",
    "Subregion_class"
]

# Check that required columns exist
missing_columns = [
    col for col in feature_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

# Get observed values for each feature
feature_values = {}

for col in feature_columns:
    values = df[col].dropna().unique().tolist()

    if not values:
        raise ValueError(f"No usable values found for {col}")

    feature_values[col] = values

# --------------------------------------------------
# 3. Generate 10,000 synthetic input records
# --------------------------------------------------

synthetic_data = {}

for col in feature_columns:
    synthetic_data[col] = rng.choice(
        feature_values[col],
        size=NUMBER_OF_TESTS
    )

synthetic_df = pd.DataFrame(synthetic_data)

print(f"Generated {len(synthetic_df)} synthetic records.")
print("Starting API tests...\n")

# --------------------------------------------------
# 4. Test the API for each record
# --------------------------------------------------

results = []

start_time = time.time()

for index, row in synthetic_df.iterrows():

    # Map dataset column names to API field names
    payload = {
        "country_of_birth": row[
            "Country of birth of person"
        ],
        "years_in_australia": row[
            "Years spent in Australia"
        ],
        "age_group": row["Age group"],
        "sex": row["Sex"],
        "lthc": row[
            "Long-term health condition (LTHC)"
        ],
        "language": row["Language used at home"],
        "english_proficiency": row[
            "Proficiency in spoken English"
        ],
        "region": row["Region_class"],
        "subregion": row["Subregion_class"]
    }

    api_ran = "No"
    predicted_prevalence = np.nan
    http_status = np.nan
    error_message = ""

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        http_status = response.status_code

        if response.ok:
            try:
                response_data = response.json()
                prediction = float(
                    response_data["predicted_prevalence"]
                )

                if (
                    np.isfinite(prediction)
                    and 0 <= prediction <= 100
                ):
                    predicted_prevalence = prediction
                    api_ran = "Yes"
                else:
                    error_message = (
                        "Prediction was not a valid value "
                        "between 0 and 100"
                    )

            except (ValueError, TypeError, KeyError) as exc:
                error_message = (
                    f"Invalid API response: {exc}"
                )

        else:
            error_message = (
                f"HTTP request failed: {response.status_code}"
            )

    except requests.RequestException as exc:
        error_message = str(exc)

    # Save the outcome for this test
    result = row.to_dict()
    result["Predicted_Prevalence"] = predicted_prevalence
    result["HTTP_Status"] = http_status
    result["API_Ran"] = api_ran
    result["Error_Message"] = error_message

    results.append(result)

    # Print progress every 100 tests
    completed = index + 1

    if completed % 100 == 0:
        elapsed = time.time() - start_time
        print(
            f"Completed {completed}/{NUMBER_OF_TESTS} "
            f"tests | Elapsed: {elapsed:.1f}s"
        )

    # Periodically save progress in case the script stops
    if completed % 500 == 0:
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

# --------------------------------------------------
# 5. Save final results
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 6. Display summary
# --------------------------------------------------

successful = (results_df["API_Ran"] == "Yes").sum()
failed = NUMBER_OF_TESTS - successful
success_rate = (successful / NUMBER_OF_TESTS) * 100

elapsed = time.time() - start_time

print("\n========== API TEST SUMMARY ==========")
print(f"Total tests:       {NUMBER_OF_TESTS}")
print(f"Successful tests:  {successful}")
print(f"Failed tests:      {failed}")
print(f"Success rate:      {success_rate:.2f}%")
print(f"Total time:        {elapsed:.2f} seconds")
print(f"Average per test:  {elapsed / NUMBER_OF_TESTS:.3f} seconds")
print(f"Results saved to:  {OUTPUT_FILE}")
print("======================================")
