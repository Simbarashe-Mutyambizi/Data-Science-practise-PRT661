from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="LTHC Population Prevalence Prediction API",
    description="API for predicting population-level LTHC prevalence",
    version="1.0"
)


# --------------------------------------------------
# 2. Load trained ML components
# --------------------------------------------------

preprocessor = joblib.load("../models/lthc_preprocessor.pkl")
model = joblib.load("../models/lthc_final_model.pkl")


# --------------------------------------------------
# 3. Define expected input
# --------------------------------------------------

class LTHCInput(BaseModel):

    country_of_birth: str
    years_in_australia: str
    age_group: str
    sex: str
    lthc: str
    language: str
    english_proficiency: str
    region: str
    subregion: str


# --------------------------------------------------
# 4. Health check endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "LTHC Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "running",
        "model": "LTHC population prevalence model"
    }


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: LTHCInput):

    # Convert API input into dataframe
    input_data = pd.DataFrame([{

        "Country of birth of person":
            data.country_of_birth,

        "Years spent in Australia":
            data.years_in_australia,

        "Age group":
            data.age_group,

        "Sex":
            data.sex,

        "Long-term health condition (LTHC)":
            data.lthc,

        "Language used at home":
            data.language,

        "Proficiency in spoken English":
            data.english_proficiency,

        "Region_class":
            data.region,

        "Subregion_class":
            data.subregion

    }])


    # --------------------------------------------------
    # Preprocess input using SAME preprocessor used
    # during model training
    # --------------------------------------------------

    processed_data = preprocessor.transform(input_data)


    # --------------------------------------------------
    # Generate prediction
    # --------------------------------------------------

    prediction = model.predict(processed_data)[0]


    # --------------------------------------------------
    # Keep prediction within valid percentage range
    # --------------------------------------------------

    prediction = max(0, min(100, prediction))


    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {

        "predicted_prevalence": round(float(prediction), 2),

        "unit": "percent",

        "interpretation":
            "Estimated population-level prevalence",

        "warning":
            "This is a population-level estimate and "
            "is not an individual diagnosis or personal "
            "disease-risk prediction."
    }