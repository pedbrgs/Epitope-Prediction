import copy
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from data.normalization import normalize_data


class BaselineFeatureSelector(ABC):
    """An abstract class for a baseline feature selection algorithm.

    Attributes
    ----------
    estimator : ClassifierMixin
        An unfitted classifier with `predict` and `predict_proba` methods.
    random_state : int, optional
        Random seed used to initialize the estimator and/or selector, by default 1234.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        random_state: int = 1234
    ):
        """Init baseline class."""
        self.base_estimator = estimator
        self.random_state = random_state

    @abstractmethod
    def select(self, X_train, y_train, n_features, **kwargs) -> List[str]:
        """Abstract method to select features. Must be implemented in subclass.

        Parameters
        ----------
        X_train : pd.DataFrame
            DataFrame containing all feature columns.
        y_train : pd.DataFrame
            Series or DataFrame with the target variable.
        n_features : int
        **kwargs : dict
            Additional arguments required by the specific selector.
        """
        pass

    def _reset_estimator(self) -> ClassifierMixin:
        """Return a fresh, unfitted copy of the base estimator.

        This method creates a deep copy of the original base estimator to reset any fitted
        parameters, ensuring the returned estimator is in its initial state.

        Returns
        -------
        ClassifierMixin
            An unfitted copy of the base estimator.
        """
        return copy.deepcopy(self.base_estimator)

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            estimator: ClassifierMixin,
            selected_features: List[str]
        ) -> ClassifierMixin:
        """Fit an estimator using the features selected by the baseline algorithm.

        Parameters
        ----------
        X_train : pd.DataFrame
            DataFrame containing all feature columns.
        y_train : pd.DataFrame
            Series or DataFrame with the target variable.
        estimator : ClassifierMixin
            Unfitted classifier with `predict` and `predict_proba` methods.
        selected_features : List[str]
            Name of the best features selected.

        Returns
        -------
        ClassifierMixin
            Classifier fitted with the selected features.
        """
        X_train_selected = X_train[selected_features].copy()
        return estimator.fit(X_train_selected, y_train)

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        eval_function: Callable,
        folds: pd.Series,
        step_size: float = 0.05,
        normalization_method: str = "minmax"
    ) -> dict:
        """Get best number of selected features using K-Fold CV with a general estimator.

        Parameters
        ----------
        X_train : pd.DataFrame
            DataFrame containing all feature columns.
        y_train : pd.DataFrame
            Series or DataFrame with the target variable.
        eval_function : Callable
            A function with signature `eval_function(y_true, y_pred)` that returns a scalar score.
        folds : pd.Series
            A Series indicating the fold assignment for each sample (e.g., fold IDs).
        step_size : float
            The step size of number of selected features.
        normalization_method : str
            The normalization method to use. Options are:
            - 'standard' : StandardScaler (mean=0, std=1)
            - 'minmax' : MinMaxScaler (scale to [0, 1])
            - 'robust' : RobustScaler (scale using median and IQR)

    Returns
    -------
        dict
        A dictionary containing tuning logs with the following keys
            - best_k (int): The number of features (k) that achieved the best average
            cross-validation score.
            - k_values (List[int]): The list of k values (number of features) evaluated during
            tuning.
            - cv_scores (List[float]): The list of average cross-validation scores corresponding
            to each k value.
            - cv_stds (List[float]): The list of standard deviations of the cross-validation
            scores for each k value.
        """
        print(f"Normalization method: {normalization_method}.")
        k_values = []
        cv_scores = []
        cv_stds = []

        n_features = X_train.shape[1]
        print(f"Number of features: {n_features}")

        # Calculate k values from s% to (100-s)% of features
        n_steps = int(1.0 / step_size)
        k_percentages = np.linspace(step_size, 1.0, n_steps)[:-1]
        k_features = [int(p * n_features) for p in k_percentages]
        print(f"Search space: {k_percentages}")

        for k in k_features:
            print(f"\nEvaluating {self.__name__} with k={k} features ({k/n_features*100:.1f}%)")

            fold_scores = []

            for fold in folds.unique():
                print(f"Fold: {fold+1}/{folds.nunique()}")
                estimator = self._reset_estimator()

                # Split into train and validation
                fold_train = folds != fold
                fold_val = folds == fold

                # Normalize the training and validation input data
                X_scaled_train_fold, X_scaler_val_fold = normalize_data(
                    X_train=X_train[fold_train],
                    X_test=X_train[fold_val],
                    method=normalization_method
                )

                # Run feature selection
                selected_features = self.select(
                    X_train=X_scaled_train_fold,
                    y_train=y_train[fold_train],
                    n_features=k,
                    estimator=estimator,
                    variance_threshold=0.95
                )

                # Train an estimator
                estimator = self.fit(
                    X_train=X_scaled_train_fold,
                    y_train=y_train[fold_train],
                    estimator=estimator,
                    selected_features=selected_features
                )

                # Predict and evaluate
                y_pred = estimator.predict(X_scaler_val_fold[selected_features])
                score = eval_function(y_true=y_train[fold_val], y_pred=y_pred)
                fold_scores.append(score)

                del estimator

            cv_score, cv_std = round(np.mean(fold_scores), 4), round(np.std(fold_scores), 4)
            print(f"{eval_function.__name__}: {cv_score} +- {cv_std}")

            k_values.append(k)
            cv_scores.append(cv_score)
            cv_stds.append(cv_std)

        # Get the number of selected features that maximize the evaluation function
        best_k_index = np.argmax(cv_scores)
        best_k = k_values[best_k_index]

        logs = {
            "best_k": best_k,
            "k_values": k_values,
            "cv_scores": cv_scores,
            "cv_stds": cv_stds
        }

        return logs
