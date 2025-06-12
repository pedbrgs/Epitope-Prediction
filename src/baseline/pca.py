from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .base import BaselineFeatureSelector


class PrincipalComponentAnalysis(BaselineFeatureSelector):
    """Principal Component Analysis as a feature selection algorithm."""

    __name__ = "PCA"

    def select(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_features: int,
            **kwargs
        ) -> List[str]:
        """Select the top `n_features` most relevant features using PCA-based feature selection.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (already scaled).
        y_train : pd.Series
            Training target labels (not used in PCA, but kept for API consistency or future use).
        n_features : int
            Number of top features to select.
        **kwargs : dict
            Additional arguments.
                - variance_threshold : float, default=0.95
                    Cumulative variance threshold to determine number of PCA components (k).

        Returns
        -------
        List[str]
            A list of selected feature names.
        """
        # Full PCA to analyze all components
        pca = PCA(n_components=min(X_train.shape))
        pca.fit(X_train)
        # PCA identifies directions (components) with the most variance. Features with high
        # absolute loadings in top components are considered important because they contribute
        # more to the major variance in the data.
        loadings = np.abs(pca.components_)
        # Determine k (number of components that explain >=95% of variance)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(cumulative_variance >= kwargs["variance_threshold"]) + 1
        feature_importance = loadings[:k, :].mean(axis=0)
        selected_feature_indices = np.argsort(feature_importance)[::-1][:n_features]
        selected_features = X_train.columns[selected_feature_indices].tolist()
        return selected_features
