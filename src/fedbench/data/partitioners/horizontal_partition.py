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

    Parameters
    ----------
    df
        DataFrame to split.
    n_partitions
        Desired number of shards. The remaining rows will be evenly
        distributed.
    seed
        Integer seed for reproducibility; if `None` a new RNG will be used
        (but the user can  choose a constant elsewhere).

    Returns
    -------
    List[DataFrame]
        List containing the shuffled shards in order 0 … n‑1.
    """
    rng = np.random.default_rng(seed)

    # 1️⃣ Create a random permutation of row indices
    perm = rng.permutation(df.index)

    # 2️⃣ Reindex frame by the shuffled permutation
    shuffled = df.loc[perm].reset_index(drop=True)

    # 3️⃣ Split with np.array_split (stable split logic)
    shards = np.array_split(shuffled, n_partitions)

    # Convert each numpy slice back to a DataFrame
    return [shard.reset_index(drop=True) for shard in shards]
