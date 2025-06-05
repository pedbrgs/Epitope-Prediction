from typing import List

import numpy as np
import pandas as pd
from pyccea.projection.vip import VIP
from sklearn.cross_decomposition import PLSRegression

from .base import BaselineFeatureSelector


class PLSRegressorVIP(BaselineFeatureSelector):
    """Partial Least Squares (PLS) with Variable Importance in Projection (VIP) feature selection.

    Attributes
    ----------
    estimator : ClassifierMixin
        A fitted classifier with `predict` and `predict_proba` methods.
    eval_function : Callable
        A function with signature `eval_function(y_true, y_pred)` that returns a scalar score.
    random_state : int, optional
        Random seed used to initialize the estimator, by default 42.
    """

    __name__ = "PLS+VIP"

    def select(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_features: int,
            **kwargs
        ) -> List[str]:
        """Select the top `n_features` most relevant features using the PLS+VIP.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Target labels corresponding to the training data.
        n_features : int
            Number of top features to select.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        List[str]
            A list of selected feature names.
        """
        X_train_normalized = X_train - X_train.mean(axis=0)
        y_train_encoded = pd.get_dummies(y_train).astype(int)
        pls = PLSRegression(n_components=X_train.shape[1], scale=True, copy=True)
        pls.fit(X_train_normalized, y_train_encoded)
        vip = VIP(model=pls)
        vip.compute()
        selected_feature_indices = np.argsort(vip.importances)[::-1][:n_features]
        selected_features = X_train.columns[selected_feature_indices].tolist()
        return selected_features
