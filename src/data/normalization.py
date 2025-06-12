from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def normalize_data(
        X_train: pd.DataFrame,
        X_test: pd.Series,
        method: str = "standard"
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Normalize training and test data using the specified scaling method.

    The scaler is fit on the training data and then applied to both training and test sets to
    avoid data leakage.

    Parameters
    ----------
    X_train : array-like of shape (n_samples_train, n_features)
        Training data to fit the scaler.
    X_test : array-like of shape (n_samples_test, n_features)
        Test data to be transformed using the scaler fit on X_train.
    method : str, default='standard'
        The normalization method to use. Options are:
        - 'standard' : StandardScaler (mean=0, std=1)
        - 'minmax' : MinMaxScaler (scale to [0, 1])
        - 'robust' : RobustScaler (scale using median and IQR)

    Returns
    -------
    X_train_scaled : ndarray of shape (n_samples_train, n_features)
        Normalized training data.
    X_test_scaled : ndarray of shape (n_samples_test, n_features)
        Normalized test data.
    """
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    if method not in SCALERS:
        raise ValueError(
            f"Invalid method: '{method}'. "
            f"Supported methods are: {list(SCALERS.keys())}"
        )

    scaler = SCALERS[method]()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled
