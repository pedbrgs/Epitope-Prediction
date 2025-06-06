from typing import List

import pandas as pd
from sklearn.feature_selection import RFE

from .base import BaselineFeatureSelector


class RecursiveFeatureElimination(BaselineFeatureSelector):
    """Recursive Feature Elimination (RFE) feature selection algorithm."""

    __name__ = "RFE"

    def select(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_features: int,
            **kwargs
        ) -> List[str]:
        """Select the top `n_features` most relevant features using the RFE.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Target labels corresponding to the training data.
        n_features : int
            Number of top features to select.
        **kwargs : dict
            Additional arguments, must include:
                - estimator : sklearn.base.ClassifierMixin
                    A scikit-learn compatible estimator to use in RFE.

        Returns
        -------
        List[str]
            A list of selected feature names.
        """
        selector = RFE(
            kwargs["estimator"],
            n_features_to_select=n_features
        )
        selector = selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_].tolist()
        return selected_features
