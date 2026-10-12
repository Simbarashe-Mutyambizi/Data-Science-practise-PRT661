import sys
import pandas as pd
from src.exception import CustomException
from src.utilid import load_object
import joblib
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder

with open("artifacts/model.pkl","rb") as f:
    model=joblib.load(f)

class Predictions:
    def __init__(self,data):
        self.data=data
        
        
    def column_transformer(self):
        try:
            logging.info("Transforming inserted data from user")
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
            
            cleaned_data_df=self.data[features]
            
            preprocessor = ColumnTransformer(
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
            final_predict_data=preprocessor.fit_transform(cleaned_data_df)
            predictions= model.predict(final_predict_data)
            prediction_df=pd.Series(predictions)
            prediction_df.columns="Predictions"
            df_with_predictions=self.data.join(prediction_df)
            return df_with_predictions
        except Exception as e:
            raise CustomException(e,sys)