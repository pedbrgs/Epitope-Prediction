from typing import List
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.base import ClassifierMixin


def holdout_eval(
    model: ClassifierMixin,
    test_data: pd.DataFrame,
    selected_features: List[str],
    total_features: int,
    method: str,
    dataset_name: str,
    model_name: str,
    label_col: str = "label"
) -> pd.DataFrame:
    """Evaluate a fitted classifier on a hold-out test set using selected features.

    Parameters
    ----------
    model : ClassifierMixin
        A fitted classifier with `predict` and `predict_proba` methods.
    test_data : pd.DataFrame
        DataFrame containing the test data, including features and label.
    selected_features : list of str
        List of feature names selected for evaluation.
    total_features : int
        Total number of features available before selection.
    method : str
        Name of the feature selection method used (e.g., "MRMR").
    dataset_name : str
        Name of the dataset used for evaluation.
    model_name : str
        Name of the machine learning model used (e.g., "random_forest").
    label_col : str, optional
        Name of the column containing class labels. Default is "label".

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame with performance metrics and feature proportions:
        - balanced_accuracy : float
        - roc_auc : float
        - mcc : float
        - p_features : float
        - p_eng_features : float
        - p_deep_features : float
    """
    X_test_selected = test_data[selected_features].copy()
    y_test = test_data[label_col].copy()

    y_test_pred = model.predict(X_test_selected)
    y_test_proba_pred = model.predict_proba(X_test_selected)[:, 1]

    balanced_accuracy = round(balanced_accuracy_score(y_test, y_test_pred), 4)
    roc_auc = round(roc_auc_score(y_test, y_test_proba_pred), 4)
    mcc = round(matthews_corrcoef(y_test, y_test_pred), 4)

    selected_k = len(selected_features)
    count_feat_only = sum(
        f.startswith("feat_") and not f.startswith("feat_esm")
        for f in selected_features
    )
    count_feat_esm = sum(f.startswith("feat_esm") for f in selected_features)
    perc_feat_only = round(count_feat_only / selected_k, 4)
    perc_feat_esm = round(count_feat_esm / selected_k, 4)

    return pd.DataFrame({
        "dataset": [dataset_name],
        "total_features": [total_features],
        "method": [method],
        "model": [model_name],
        "balanced_accuracy": [balanced_accuracy],
        "roc_auc": [roc_auc],
        "mcc": [mcc],
        "features_%": [round(selected_k / total_features, 4)],
        "eng_features_%": [perc_feat_only],
        "deep_features_%": [perc_feat_esm],
    })
