from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    import pandas as pd

    return pd.read_csv(str(file_path))
