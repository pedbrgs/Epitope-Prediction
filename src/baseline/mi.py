from typing import List

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from .base import BaselineFeatureSelector


class MutualInformationFeatureSelection(BaselineFeatureSelector):
    """Mutual Information-based feature selection."""

    __name__ = "MI"

    def select(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_features: int,
            **kwargs
        ) -> List[str]:
        """Select the top `n_features` most relevant features using the mutual information.

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
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        return selected_features
