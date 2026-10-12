import os
import sys
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from src.utilid import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    # Only paths and constants belong in the config. No logic here.
    x_train_path: str = os.path.join("artifacts", "train", "X_train.csv")
    y_train_path: str = os.path.join("artifacts", "train", "y_train.csv")
    x_test_path: str = os.path.join("artifacts", "test", "X_test.csv")
    y_test_path: str = os.path.join("artifacts", "test", "y_test.csv")
    
    target: str = "Age-specific percentage of population reporting LTHC(s)"
    
    features: tuple = (
        "Country of birth of person",
        "Years spent in Australia",
        "Age group",
        "Sex",
        "Language used at home",
        "Proficiency in spoken English",
        "Region_class",
        "Subregion_class",
        "Long-term health condition (LTHC)",
    )

    # Rows sharing these values form one demographic group. A group is kept
    # entirely in train OR test, so no group leaks across the split.
    group_columns: tuple = (
        "Country of birth of person",
        "Years spent in Australia",
        "Age group",
        "Sex",
        "Language used at home",
        "Proficiency in spoken English",
        "Region_class",
        "Subregion_class",
    )


class PreprocessorConfig:
    preprocessor_path=os.path.join("artifacts","lthc_preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.preprocessor_path_config=PreprocessorConfig()

    def initiate_data_transformation(self, cleaned_data_path):
        """Reads the cleaned CSV, splits it by demographic group, one-hot
        encodes the features, saves the four CSVs and returns their paths."""
        logging.info("Data transformation started")
        try:
            cfg = self.config

            # Create output folders (this must run inside a method, not the class body)
            os.makedirs(os.path.dirname(cfg.x_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(cfg.x_test_path), exist_ok=True)

            # Load the cleaned dataset produced by data_ingestion.py
            df = pd.read_csv(cleaned_data_path)
            logging.info(f"Loaded cleaned data from {cleaned_data_path}, shape {df.shape}")

            features = list(cfg.features)
            group_columns = list(cfg.group_columns)

            missing = [c for c in features + group_columns + [cfg.target] if c not in df.columns]
            if missing:
                raise ValueError(f"Columns missing from cleaned data: {missing}")

            # Build the demographic group column
            groups = df[group_columns].astype(str).agg("|".join, axis=1)

            X = df[features]
            y = df[cfg.target]

            logging.info("Train test split initiated")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))

            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            # Preprocessing: fit on train only, apply to test
            categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_features,
                    )
                ],
                remainder="passthrough",
            )

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            logging.info("Preprocessing complete")

            feature_names = preprocessor.get_feature_names_out()
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

            # Save the four CSV files
            X_train_df.to_csv(cfg.x_train_path, index=False, header=True)
            y_train.to_csv(cfg.y_train_path, index=False, header=True)
            X_test_df.to_csv(cfg.x_test_path, index=False, header=True)
            y_test.to_csv(cfg.y_test_path, index=False, header=True)
            save_object(
                file_path= self.preprocessor_path_config.preprocessor_path,
                obj=preprocessor
            )
            logging.info("Data transformation completed successfully")

            return (
                cfg.x_train_path,
                cfg.y_train_path,
                cfg.x_test_path,
                cfg.y_test_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def load_arrays(x_train_path, y_train_path, x_test_path, y_test_path):
        """Loads the saved CSVs as NumPy arrays for model training."""
        try:
            X_train = pd.read_csv(x_train_path).to_numpy()
            X_test = pd.read_csv(x_test_path).to_numpy()
            # ravel() turns the (n, 1) target into a flat (n,) array that sklearn expects
            y_train = pd.read_csv(y_train_path).to_numpy().ravel()
            y_test = pd.read_csv(y_test_path).to_numpy().ravel()
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

'''
if __name__ == "__main__":
    # Quick standalone run: python -m src.components.data_transformation
    cleaned_path = os.path.join("artifacts", "cleaned", "Cleaned_data_final.csv")
    paths = DataTransformation().initiate_data_transformation(cleaned_path)
    print(paths)
'''