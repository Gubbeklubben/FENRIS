from pathlib import Path

import pandas as pd
from pandas import DataFrame


def load_csv(file_path: str | Path) -> DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(str(file_path))
