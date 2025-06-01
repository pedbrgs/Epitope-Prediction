from typing import List

import pandas as pd
from mrmr import mrmr_classif

from .base import BaselineFeatureSelector


class MinimumRedundancyMaximumRelevance(BaselineFeatureSelector):
    """Minimum Redundancy and Maximum Relevance (MRMR)."""

    __name__ = "MRMR"

    def _select_features(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_features: int,
            **kwargs
        ) -> List[str]:
        """Select the top `n_features` most relevant features using the MRMR criterion.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Target labels corresponding to the training data.
        n_features : int
            Number of top features to select.

        Returns
        -------
        List[str]
            A list of selected feature names.
        """
        selected_features = mrmr_classif(
            X=X_train,
            y=y_train,
            K=n_features
        )
        return selected_features
