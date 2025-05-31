import pandas as pd
from typing import Tuple, List


def train_test_split(data: pd.DataFrame, subset_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets based on the subset column.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing the data to be split.
    subset_col : str
        The column name where the subset is indicated.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training and testing DataFrames.
    """
    train_data = data[data[subset_col] == "train"].copy()
    test_data = data[data[subset_col] == "test"].copy()
    return train_data, test_data


def input_output_split(
        data: pd.DataFrame,
        feature_cols: List[str],
        class_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits the data into input and output data based on the target column.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing the data to be split.
    class_col : str
        The column name where the target is.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the input and output data.
    """
    X = data[feature_cols].copy()  # Features only
    y = data[class_col].copy()  # Target variable
    return X, y