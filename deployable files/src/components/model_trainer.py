"""
model_trainer.py - PRT661 Group Dan3

Trains the final model for the CI/CD pipeline. Matches Model_training_evaluation.py:

  * XGBoost with objective="reg:gamma" (log link), so every prediction is
    above 0%. This replaces the squared-error model from Assessment 3, which
    produced negative predictions that had to be clipped to 0%.
  * Same settings as the model selected by grouped 5-fold cross-validation:
    n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
    colsample_bytree=0.9, random_state=42.
  * Optional grouped 5-fold cross-validation on the training set (pass
    train_groups), reported as mean +/- SD across folds.
  * Train and test MAE, RMSE and R2 (in percentage points), the train-test
    gap and the number of negative predictions, saved to artifacts/.

Model selection itself (comparing the four models and the naive baseline) is
done in Model_training_evaluation.py; this file trains the selected model.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utilid import save_object

RANDOM_STATE = 42
N_FOLDS = 5


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "model_metrics.csv")
    cv_metrics_file_path: str = os.path.join("artifacts", "model_cv_metrics.csv")


def build_model():
    """The model selected in Model_training_evaluation.py."""
    return XGBRegressor(
        objective="reg:gamma",      # log link: predictions are always > 0
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def score(y_true, y_pred):
    return {
        "MAE (pp)": mean_absolute_error(y_true, y_pred),
        "RMSE (pp)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Negative predictions": int((y_pred < 0).sum()),
    }


def take_rows(X, idx):
    """Row selection that works for numpy arrays, scipy sparse matrices and DataFrames."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, x_train_arr, x_test_arr, y_train_arr, y_test_arr,
                               train_groups=None):
        """
        x_*_arr : encoded features (numpy or scipy sparse)
        y_*_arr : prevalence (%), any 1-D shape
        train_groups : optional demographic-group label for every training row.
                       Build it WITHOUT Sex so a cell's Persons, Male and Female
                       rows share a fold. If given, grouped 5-fold CV is run.
        """
        try:
            X_train, X_test = x_train_arr, x_test_arr
            y_train = np.asarray(y_train_arr, dtype=float).ravel()
            y_test = np.asarray(y_test_arr, dtype=float).ravel()

            logging.info(
                f"Train {X_train.shape}, test {X_test.shape}, sparse={sparse.issparse(X_train)}"
            )
            # reg:gamma needs a strictly positive target. Suppressed "n.p" cells
            # must be dropped in data_ingestion.py, not set to 0.
            if (y_train <= 0).any() or (y_test <= 0).any():
                raise ValueError(
                    "Target contains values <= 0. Drop suppressed (n.p) rows in "
                    "data_ingestion.py instead of converting them to 0."
                )

            # Optional: grouped 5-fold cross-validation on the training set
            if train_groups is not None:
                groups = np.asarray(train_groups).ravel()
                if len(groups) != len(y_train):
                    raise ValueError("train_groups must have one label per training row")
                logging.info(f"Grouped {N_FOLDS}-fold cross-validation")
                fold_rows = []
                cv = GroupKFold(n_splits=N_FOLDS)
                for fold, (tr, va) in enumerate(cv.split(X_train, y_train, groups=groups), 1):
                    fold_model = clone(build_model()).fit(take_rows(X_train, tr), y_train[tr])
                    pred = fold_model.predict(take_rows(X_train, va))
                    fold_rows.append({"Fold": fold, **score(y_train[va], pred)})
                cv_df = pd.DataFrame(fold_rows)
                summary = cv_df.drop(columns="Fold").agg(["mean", "std"]).T.round(4)
                os.makedirs(os.path.dirname(self.model_trainer_config.cv_metrics_file_path),
                            exist_ok=True)
                cv_df.round(4).to_csv(self.model_trainer_config.cv_metrics_file_path, index=False)
                logging.info(f"CV mean +/- SD:\n{summary}")
                print(f"\nGrouped {N_FOLDS}-fold CV (training set):\n{summary}")

            # Train the final model on the full training set
            logging.info("Training XGBoost (gamma) on the full training set")
            model = build_model()
            model.fit(X_train, y_train)
            logging.info("XGBoost training complete.")

            # Evaluate once on the hold-out test set
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_scores = score(y_train, train_predictions)
            test_scores = score(y_test, test_predictions)

            results_df = pd.DataFrame([{
                "Model": "XGBoost (gamma)",
                "Train MAE (pp)": train_scores["MAE (pp)"],
                "Test MAE (pp)": test_scores["MAE (pp)"],
                "Train RMSE (pp)": train_scores["RMSE (pp)"],
                "Test RMSE (pp)": test_scores["RMSE (pp)"],
                "Train R2": train_scores["R2"],
                "Test R2": test_scores["R2"],
                "R2 Difference": train_scores["R2"] - test_scores["R2"],
                "Negative test predictions": test_scores["Negative predictions"],
                "Min test prediction (%)": float(test_predictions.min()),
                "Max test prediction (%)": float(test_predictions.max()),
            }]).round(4)

            os.makedirs(os.path.dirname(self.model_trainer_config.metrics_file_path),
                        exist_ok=True)
            results_df.to_csv(self.model_trainer_config.metrics_file_path, index=False)
            logging.info(f"Test metrics:\n{results_df.T}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            return results_df

        except Exception as e:
            raise CustomException(e, sys)