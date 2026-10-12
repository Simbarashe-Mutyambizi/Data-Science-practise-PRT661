import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from dataclasses import dataclass 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy import sparse
import joblib
from src.exception import CustomException
from src.logger import logging
from xgboost import XGBRegressor
from src.utilid import save_object
from scipy import sparse

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,x_train_arr,x_test_arr,y_train_arr,y_test_arr):
        try:
            print(type(x_train_arr), x_train_arr.shape, x_test_arr.shape)
            print("sparse:", sparse.issparse(x_train_arr))
            print(x_train_arr[:3].sum(axis=1))
            print(y_train_arr[:5], y_train_arr.mean(), y_test_arr.mean())
            logging.info("Loading train and test data for best model")
            X_train,X_test,y_train,y_test=x_train_arr,x_test_arr,y_train_arr,y_test_arr
            model = XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
            )
        
            model.fit(
            X_train,
            y_train
            )

            logging.info("XGBoost training complete.")
            
            logging.info("Instantiating testing")
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            overfitting_results = []
            train_r2 = r2_score(
            y_train,
            train_predictions
            )
    
            test_r2 = r2_score(
            y_test,
            test_predictions
            )
    
            train_mae = mean_absolute_error(
            y_train,
            train_predictions
            )
    
            test_mae = mean_absolute_error(
            y_test,
            test_predictions
            )
    
            overfitting_results.append({
            "Model":"XGBoost",
            "Train MAE": train_mae,
            "Test MAE": test_mae,
            "Train R²": train_r2,
            "Test R²": test_r2,
            "R² Difference": train_r2 - test_r2
            })
    
            overfitting_df = pd.DataFrame(
                overfitting_results
                )

            overfitting_df.round(4)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            
            return overfitting_df
            



            
        except Exception as e:
            raise CustomException(e,sys)