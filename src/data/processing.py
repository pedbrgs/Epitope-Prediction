from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import GroupKFold


def _read_csv_data(file_path: str) -> pd.DataFrame:
    """Reads the CSV file and returns a DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file, with duplicates removed.

    Raises
    ------
    ValueError
        If the file does not have a .csv extension.
    """
    if not file_path.endswith('.csv'):
        raise ValueError("The file must be a CSV file with a .csv extension.")
    return pd.read_csv(file_path, sep=";").drop_duplicates()


def _get_feature_names(data: pd.DataFrame) -> List[str]:
    """Get the names of the data columns that are predictive features.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.

    Returns
    -------
    List[str]
        The feature names.
    """
    return [feature for feature in data.columns if feature.startswith("feat_")]


def _split_group_kfolds(
        data: pd.DataFrame,
        n_splits: int,
        group_col: str,
        split_col: str,
        random_state: int = 1234
    ) -> pd.DataFrame:
    """Splits the data into groups using GroupKFold.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing the data to be split.
    n_splits : int
        The number of splits to create.
    group_col : str
        The column name containing group identifiers.
    split_col : str
        The column name where the split indices will be stored.
    random_state : int, optional
        Random seed for reproducibility (default is 1234).

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column indicating the split index for each row.
    """
    data[f"new_{split_col}"] = -1
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (_, test_idx) in enumerate(splitter.split(data, groups=data[group_col])):
        data.iloc[test_idx, data.columns.get_loc(f"new_{split_col}")] = fold_idx
    return data


def _check_group_leakage(data: pd.DataFrame, group_col: str, split_col: str) -> None:
    """Checks for group leakage in the folds.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing the data to be checked.
    group_col : str
        The column name containing group identifiers.
    split_col : str
        The column name where the split indices are stored.

    Raises
    ------
    ValueError
        If any group appears in multiple splits, indicating group leakage.
    """
    group_to_splits = defaultdict(set)

    for split, group in zip(data[split_col], data[group_col]):
        group_to_splits[group].add(split)

    # Check if any group appears in multiple splits
    violating_groups = {
        group: splits
        for group, splits in group_to_splits.items()
        if len(splits) > 1
    }

    if violating_groups:
        for group, splits in violating_groups.items():
            print(f"Group '{group}' appears in splits {splits}")
        raise ValueError("Group leakage detected in the splits.")
    else:
        print("All splits contain unique groups only.")


def _train_test_split(
        data: pd.DataFrame,
        split_col: str,
        n_splits: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets based on the split column.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing the data to be split.
    split_col : str
        The column name where the split indices are stored.
    n_splits : int
        The number of splits used to determine the test set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training and testing DataFrames.
    """
    train_data = data[data[f"new_{split_col}"] != n_splits - 1].reset_index(drop=True)
    test_data = data[data[f"new_{split_col}"] == n_splits - 1].reset_index(drop=True)
    return train_data, test_data


def _count_class_rows_by_fold(
        data: pd.DataFrame,
        class_col: str,
        split_col: str,
        class_id: int
    ) -> Dict[int, int]:
    """Returns the count of observations for each fold within the specified class.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    class_col : str
        The column name containing class labels.
    split_col : str]
        The column name containing fold identifiers.
    class_id : int
        The specific class ID to count groups for.
        
    Returns
    -------
    Dict[int, int]
        A dictionary where keys are group identifiers and values are the counts of rows for the
        specified class.
    """
    return (
        data.loc[data[class_col] == class_id]
        .groupby(f"new_{split_col}", as_index=True)
        .size()
        .to_dict()
    )


def _get_min_rows_across_folds(fold_counts: Dict[int, int]) -> int:
    """Returns the minimum number of rows across all folds.

    Parameters
    ----------
    fold_counts : Dict[int, int]
        A dictionary where keys are class IDs and values are the counts of rows for each fold.

    Returns
    -------
    int
        The minimum number of rows across all folds.
    """
    return min([n_rows for n_rows in fold_counts.values()])


def _assign_rows_per_fold(
        fold_counts: Dict[int, int],
        n_rows_per_fold: int
    ) -> Dict[int, int]:
    """Assigns the number of rows per fold based on a specified count.

    Parameters
    ----------
    fold_counts : Dict[int, int]
        A dictionary where keys are class IDs and values are the counts of rows for each fold.
    n_rows_per_fold : int
        The number of rows to assign for all folds.

    Returns
    -------
    Dict[int, int]
        A dictionary where keys are class IDs and values are the assigned number of rows per fold.
    """
    return {key: n_rows_per_fold for key in fold_counts.keys()}


def _stratified_sample_with_balanced_groups(
        data: pd.DataFrame,
        split_col: str,
        group_col: str,
        target_sizes: dict,
        class_col: str = None,
        class_id: str = None,
        random_state: int = 1234):
    """Perform stratified sampling while preserving samples from each group within each split.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe.
    split_col : str
        The column name used for stratification (e.g., 'Info_split').
    group_col : str
        The column name containing group information (e.g., 'Info_group')
    target_sizes : dict
        Dictionary with split values as keys and target sample sizes as values.
    class_col : str, optional
        If provided, filter data by this column before sampling.
    class_id : any, optional
        If provided with class_col, only sample from rows where class_col equals this value
    random_state : int, optional
        Seed for random number generation.

    Returns
    -------
    pandas.DataFrame
        The sampled dataframe with balanced group representation
    """
    # Filter by class if specified
    if class_col is not None and class_id is not None:
        data = data.query(f"{class_col} == @class_id")
    
    sampled_rows = []
    
    for split_value, target_size in target_sizes.items():
        split_df = data[data[split_col] == split_value]
        
        # Get groups in this split and their sizes
        group_sizes = split_df[group_col].value_counts()
        n_groups = len(group_sizes)
        
        # Calculate target samples per group (at least 1)
        base_samples_per_group = target_size // n_groups
        extra_samples = target_size % n_groups
        
        # First round: sample from each group proportionally
        guaranteed_samples = []
        remaining_target = target_size
        
        for group, group_size in group_sizes.items():
            group_df = split_df[split_df[group_col] == group]
            
            # Calculate samples for this group
            n_samples = min(
                base_samples_per_group + (1 if extra_samples > 0 else 0),
                group_size
            )
            extra_samples = max(0, extra_samples - 1)
            remaining_target -= n_samples
            
            if n_samples > 0:
                guaranteed_samples.append(
                    group_df.sample(n=n_samples, random_state=random_state)
                )
        
        # Combine the guaranteed samples
        guaranteed_df = pd.concat(guaranteed_samples) if guaranteed_samples else pd.DataFrame()
        
        # If we still need more samples, distribute them across groups
        if remaining_target > 0:
            # Remove already selected samples
            remaining_df = split_df[~split_df.index.isin(guaranteed_df.index)]
            
            if len(remaining_df) > 0:
                # Sample remaining rows randomly from any group
                additional_samples = remaining_df.sample(
                    n=min(remaining_target, len(remaining_df)),
                    random_state=random_state
                )
                sampled_rows.append(pd.concat([guaranteed_df, additional_samples]))
            else:
                sampled_rows.append(guaranteed_df)
        else:
            sampled_rows.append(guaranteed_df)
    
    result = pd.concat(sampled_rows).reset_index(drop=True)
    
    return result


def _rename_data_columns(data: pd.DataFrame, split_col: str, class_col: str) -> pd.DataFrame:
    """Rename split and class columns according to the PyCCEA defaults.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe.
    split_col : str
        The column name used for stratification (e.g., 'Info_split').
    group_col : str
        The column name containing group information (e.g., 'Info_group')

    Returns
    -------
    pd.DataFrame
        The dataframe with its columns renamed.
    """
    return data.rename(columns={
        f"new_{split_col}": "fold",
        class_col: "label"
    })


