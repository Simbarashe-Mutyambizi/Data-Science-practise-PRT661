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
from scipy import sparse

NOT_COLLECTED = "Not collected"


@dataclass
class DataTransformationConfig:
    # Only paths and constants belong in the config. No logic here.
    x_train_path: str = os.path.join("artifacts", "train", "X_train.npz")
    y_train_path: str = os.path.join("artifacts", "train", "y_train.csv")
    x_test_path: str = os.path.join("artifacts", "test", "X_test.npz")
    y_test_path: str = os.path.join("artifacts", "test", "y_test.csv")
    # Group label of every training row, for grouped cross-validation in model_trainer.py
    train_groups_path: str = os.path.join("artifacts", "train", "groups_train.csv")

    target: str = "Age-specific percentage of population reporting LTHC(s)"

    # Same features, in the same order, as Model_training_evaluation.py.
    # The order matters: it fixes the one-hot column order, which XGBoost's
    # column sampling depends on, so results reproduce exactly.
    features: tuple = (
        "Country of birth of person",
        "Years spent in Australia",
        "Language used at home",
        "Proficiency in spoken English",
        "Region_class",
        "Subregion_class",
        "Age group",
        "Sex",
        "Long-term health condition (LTHC)",
    )

    # Rows sharing these values form one demographic group, which is kept
    # entirely in train OR test. Sex is deliberately NOT included: Persons is
    # Male and Female combined, so a cell's Persons, Male and Female rows must
    # stay on the same side of the split, otherwise the test set leaks.
    # (Region and subregion are left out because they follow from country.)
    # Same key, in the same order, as demographic_group in Data_Cleaning.py.
    group_columns: tuple = (
        "source_table",
        "Country of birth of person",
        "Years spent in Australia",
        "Language used at home",
        "Proficiency in spoken English",
        "Age group",
    )


class PreprocessorConfig:
    preprocessor_path = os.path.join("artifacts", "lthc_preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.preprocessor_path_config = PreprocessorConfig()

    def initiate_data_transformation(self, cleaned_data_path):
        """Reads the cleaned CSV, splits it by demographic group, one-hot
        encodes the features, saves the outputs and returns their paths."""
        logging.info("Data transformation started")
        try:
            cfg = self.config

            os.makedirs(os.path.dirname(cfg.x_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(cfg.x_test_path), exist_ok=True)

            # Load the cleaned dataset produced by data_ingestion.py
            df = pd.read_csv(cleaned_data_path)
            logging.info(f"Loaded cleaned data from {cleaned_data_path}, shape {df.shape}")

            # Source table: S13 rows never record country of birth
            df["source_table"] = np.where(
                df["Country of birth of person"] == NOT_COLLECTED, "S13", "S10"
            )

            features = list(cfg.features)
            group_columns = list(cfg.group_columns)

            missing = [c for c in features + group_columns + [cfg.target] if c not in df.columns]
            if missing:
                raise ValueError(f"Columns missing from cleaned data: {missing}")

            # Build the demographic group column (no Sex, see config)
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
            groups_train = groups.iloc[train_idx]

            if set(groups_train).intersection(groups.iloc[test_idx]):
                raise ValueError("A demographic group appears in both train and test")
            logging.info(
                f"Train {len(train_idx):,} rows / {groups_train.nunique():,} groups, "
                f"test {len(test_idx):,} rows / {groups.iloc[test_idx].nunique():,} groups"
            )

            # Preprocessing: fit on train only, apply to test
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        features,
                    )
                ]
            )

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            logging.info(
                f"Preprocessing complete: {X_train_processed.shape[1]} one-hot features"
            )

            pd.Series(preprocessor.get_feature_names_out()).to_csv(
                os.path.join("artifacts", "feature_names.csv"), index=False, header=False
            )

            # Save outputs
            sparse.save_npz(cfg.x_train_path, sparse.csr_matrix(X_train_processed))
            y_train.to_csv(cfg.y_train_path, index=False, header=True)
            sparse.save_npz(cfg.x_test_path, sparse.csr_matrix(X_test_processed))
            y_test.to_csv(cfg.y_test_path, index=False, header=True)
            groups_train.to_frame("demographic_group").to_csv(
                cfg.train_groups_path, index=False, header=True
            )
            save_object(
                file_path=self.preprocessor_path_config.preprocessor_path,
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
        """Loads the saved arrays for model training."""
        try:
            X_train = sparse.load_npz(x_train_path)
            X_test = sparse.load_npz(x_test_path)
            y_train = pd.read_csv(y_train_path).to_numpy().ravel()
            y_test = pd.read_csv(y_test_path).to_numpy().ravel()
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def load_train_groups(train_groups_path=None):
        """Loads the demographic-group label of every training row."""
        try:
            path = train_groups_path or DataTransformationConfig().train_groups_path
            return pd.read_csv(path)["demographic_group"].to_numpy()
        except Exception as e:
            raise CustomException(e, sys)


'''
if __name__ == "__main__":
    # Quick standalone run: python -m src.components.data_transformation
    cleaned_path = os.path.join("artifacts", "cleaned", "Cleaned_data_final.csv")
    paths = DataTransformation().initiate_data_transformation(cleaned_path)
    print(paths)
'''