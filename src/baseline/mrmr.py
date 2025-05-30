from typing import Tuple, List, Callable

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.base import ClassifierMixin


class MRMR():
    """Minimum Redundancy and Maximum Relevance (MRMR) feature selection algorithm.

    Attributes
    ----------
    estimator : ClassifierMixin
        A fitted classifier with `predict` and `predict_proba` methods.
    eval_function : Callable
        A function with signature `eval_function(y_true, y_pred)` that returns a scalar score.
    random_state : int, optional
        Random seed used to initialize the estimator, by default 42.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        eval_function: Callable,
        random_state: int = 1234
    ):
        """Init MRMR Baseline class."""
        self.estimator = estimator
        self.eval_function = eval_function
        self.random_state = random_state

    def tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        folds: pd.Series,
        feature_cols: List[str],
        step_size: float = 0.05,
    ) -> int:
        """Get best number of selected features using K-Fold CV with a general estimator.

        Parameters
        ----------
        X_train : pd.DataFrame
            DataFrame containing all feature columns.
        y_train : pd.DataFrame
            Series or DataFrame with the target variable.
        folds : pd.Series
            A Series indicating the fold assignment for each sample (e.g., fold IDs).
        feature_cols : List[str]
            The list of feature names.
        step_size : float
            The step size of number of selected features.

        Returns
        -------
        best_k : int
            Best number of features to select.
        """

        k_values = []
        cv_scores = []
        cv_stds = []

        n_features = len(feature_cols)

        # Calculate k values from s% to (100-s)% of features
        k_percentages = np.arange(step_size, 1.0, step_size)
        k_features = [int(p * n_features) for p in k_percentages]
        print(f"Search space: {k_percentages}")

        for k in k_features:
            print(f"\nEvaluating MRMR with k={k} features ({k/n_features*100:.1f}%)")

            fold_scores = []

            for fold in folds.unique():
                print(f"Fold: {fold+1}/{folds.nunique()}")

                # Split into train and validation
                fold_train = folds != fold
                fold_val = folds == fold

                # Run MRMR feature selection
                selected_features = mrmr_classif(
                    X=X_train[fold_train],
                    y=y_train[fold_train],
                    K=k
                )

                # Train an estimator
                self.estimator.fit(X_train[selected_features][fold_train], y_train[fold_train])

                # Predict and evaluate
                y_pred = self.estimator.predict(X_train[selected_features][fold_val])
                score = self.eval_function(y_true=y_train[fold_val], y_pred=y_pred)
                fold_scores.append(score)

            cv_score, cv_std = np.mean(fold_scores), np.std(fold_scores)
            print(f"{self.eval_function.__name__}: {cv_score} +- {cv_std}")
        
            k_values.append(k)
            cv_scores.append(cv_score)
            cv_stds.append(cv_std)

        # Get the number of selected features that maximize the evaluation function
        best_k_index = np.argmax(cv_scores)
        best_k = k_values[best_k_index]
        return best_k
