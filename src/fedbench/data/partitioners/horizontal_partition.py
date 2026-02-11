from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Tuple

def deterministic_shard(
    df: DataFrame,
    n_partitions: int,
    seed: int | None = None,
) -> List[DataFrame]:
    """
    Split *df* into *n_partitions* portions using a deterministic shuffle.

    The original column names and dtypes are preserved.  After shuffling,
    the function returns the exact number of shards requested – empty shards
    are omitted.

    Parameters
    ----------
    df
        DataFrame to split.
    n_partitions
        Desired number of shards.  Must be 1 ≤ *n_partitions* ≤ len(df).
    seed
        Integer seed for reproducibility; if ``None`` a new RNG will be
        used, resulting in a non‑deterministic shuffle.

    Returns
    -------
    List[DataFrame]
        List containing the shuffled shards in order 0 … *n_partitions*-1.
    """
    if df.empty:
        return []

    if n_partitions < 1 or n_partitions > len(df):
        raise ValueError(
            f"n_partitions={n_partitions} must be between 1 and len(df)={len(df)}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(df.index.to_list())
    shuffled = df.loc[perm].reset_index(drop=True)

    # Split indices rather than the DataFrame itself
    shards_idx = np.array_split(shuffled.index, n_partitions)
    shards = [shuffled.loc[idx] for idx in shards_idx if len(idx) > 0]
    return [shard.reset_index(drop=True) for shard in shards]