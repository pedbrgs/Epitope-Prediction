import pandas as pd


def read_parquet_data(file_path: str) -> pd.DataFrame:
    """Reads the parquet file and returns a DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the parquet file, with duplicates removed.

    Raises
    ------
    ValueError
        If the file does not have a .parquet extension.
    """
    if not file_path.endswith('.parquet'):
        raise ValueError("The file must be a parquet file with a .parquet extension.")
    return pd.read_parquet(file_path).drop_duplicates()