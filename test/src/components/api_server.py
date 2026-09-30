from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from src.exception import CustomException
from src.logger import logging
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

from fastapi.middleware.cors import CORSMiddleware



# Allow your local dashboard to communicate with the Render API


try:
    app = FastAPI(
        title="LTHC Population Prevalence Prediction API",
        description="API for predicting population-level LTHC prevalence",
        version="1.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, replace "*" with your specific local/live URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])
    # --------------------------------------------------
    # 2. Load trained ML components
    # --------------------------------------------------
    
    preprocessor = joblib.load("artifacts/lthc_preprocessor.pkl")
    with open("artifacts/xgboost_regression_model.pkl","rb") as f:
        model=joblib.load(f)
    
    
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
        features = [
            "Country of birth of person",
            "Years spent in Australia",
            "Age group",
            "Sex",
            "Language used at home",
            "Proficiency in spoken English",
            "Region_class",
            "Subregion_class",
            "Long-term health condition (LTHC)" 
        ]
        
      
        
        preprocessors = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    # Set sparse_output to False to output dense arrays for CSV saving
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    features
                )
            ],
            remainder="passthrough"
        )
        #final_predict_data=preprocessor.fit_transform(input_data)
        final_predict_data=preprocessor.transform(input_data)

        
    
    
        # --------------------------------------------------
        # Generate prediction
        # --------------------------------------------------
    
        prediction = model.predict(final_predict_data)[0]
    
        
        # --------------------------------------------------
        # Keep prediction within valid percentage range
        # --------------------------------------------------
    
        prediction = max(0, min(100, prediction))
    
        print(prediction)
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
        }# -*- coding: utf-8 -*-
            
    
except Exception as e:
    raise CustomException(e,sys)            

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
    