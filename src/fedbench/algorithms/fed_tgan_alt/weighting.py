"""Fed-TGAN Table-Similarity-Aware Client Weighting.

Implements Section 4.2 of the Fed-TGAN paper:
  - JSD for categorical columns
  - Wasserstein distance for continuous columns
  - Divergence matrix → normalized weights → softmax
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def compute_jsd(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
    """Jensen--Shannon divergence between two probability distributions.

    ``scipy.spatial.distance.jensenshannon`` returns the *square root* of
    JSD, so the result is squared here.
    """
    return float(jensenshannon(p, q) ** 2)


def compute_wd(
    samples_a: NDArray[np.floating], samples_b: NDArray[np.floating]
) -> float:
    """First Wasserstein Distance between two 1-D sample arrays.

    Parameters
    ----------
    samples_a
        1-D array of samples from distribution A.
    samples_b
        1-D array of samples from distribution B.

    Returns
    -------
    float
        Wasserstein distance (earth mover's distance).
    """
    return float(wasserstein_distance(samples_a, samples_b))


def _generate_vgm_samples(
    vgm_params: dict,
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Generate synthetic 1-D samples from VGM parameters.

    Parameters
    ----------
    vgm_params
        Dict with keys ``"means"``, ``"covariances"``, ``"weights"``.
    n_samples
        Number of samples to draw.
    rng
        NumPy random generator instance.

    Returns
    -------
    ndarray
        1-D array of generated samples.
    """
    means = np.array(vgm_params["means"])
    covs = np.array(vgm_params["covariances"])
    weights = np.array(vgm_params["weights"])

    if len(means) == 0:
        return rng.standard_normal(n_samples)

    weights = weights / weights.sum()
    component_counts = rng.multinomial(n_samples, weights)

    samples: list[NDArray[np.floating]] = []
    for mean, cov, count in zip(means, covs, component_counts, strict=True):
        if count > 0:
            s = rng.normal(loc=mean, scale=np.sqrt(max(cov, 1e-8)), size=count)
            samples.append(s)

    return np.concatenate(samples) if samples else rng.standard_normal(n_samples)


def compute_client_weights(
    cat_freqs: list[dict[str, dict[str, int]]],
    cont_vgms: list[dict[str, dict]],
    client_sample_counts: list[int],
    cat_columns: list[str],
    cont_columns: list[str],
    seed: int = 42,
) -> list[float]:
    """Compute table-similarity-aware client weights (Fed-TGAN paper §4.2).

    Parameters
    ----------
    cat_freqs
        Per-client dicts mapping ``column_name → {category: count}``.
    cont_vgms
        Per-client dicts mapping ``column_name → VGM params``.
    client_sample_counts
        Number of data rows per client.
    cat_columns
        List of categorical column names.
    cont_columns
        List of continuous column names.
    seed
        Random seed for VGM sample generation.

    Returns
    -------
    list of float
        Similarity-weighted client weights (one per client), summing to ~1.0.
        Weights are higher for clients whose data is more similar to the
        global table distribution, and for clients with larger datasets.
    """
    n_clients = len(client_sample_counts)
    all_columns = cat_columns + cont_columns
    n_columns = len(all_columns)

    if n_clients <= 1 or n_columns == 0:
        return [1.0 / max(n_clients, 1)] * max(n_clients, 1)

    rng = np.random.default_rng(seed)
    total_samples = sum(client_sample_counts)

    # Step 0: Build P×Q divergence matrix S
    S = np.zeros((n_clients, n_columns))

    # Categorical columns: JSD
    for j, col in enumerate(cat_columns):
        # Build global frequency distribution
        global_freq: dict[str, int] = {}
        for client_freq in cat_freqs:
            for cat, count in client_freq.get(col, {}).items():
                global_freq[cat] = global_freq.get(cat, 0) + count

        all_cats = sorted(global_freq.keys())
        if not all_cats:
            continue

        global_total = sum(global_freq.values())
        if global_total == 0:
            continue
        global_dist = np.array([global_freq.get(c, 0) / global_total for c in all_cats])

        for i, client_freq in enumerate(cat_freqs):
            local_freq = client_freq.get(col, {})
            local_total = sum(local_freq.values())
            if local_total == 0:
                S[i, j] = 1.0  # Max divergence
                continue
            local_dist = np.array(
                [local_freq.get(c, 0) / local_total for c in all_cats]
            )
            S[i, j] = compute_jsd(local_dist, global_dist)

    # Continuous columns: Wasserstein Distance
    n_vgm_samples = 1000
    for j_offset, col in enumerate(cont_columns):
        j = len(cat_columns) + j_offset

        # Generate global distribution samples
        global_vgm_params: list[dict] = []
        for client_vgm in cont_vgms:
            if col in client_vgm:
                global_vgm_params.append(client_vgm[col])

        if not global_vgm_params:
            continue

        # Merge VGM samples for global reference
        global_samples: list[NDArray[np.floating]] = []
        for vgm_p, n_s in zip(global_vgm_params, client_sample_counts, strict=True):
            global_samples.append(_generate_vgm_samples(vgm_p, n_s, rng))
        global_combined = np.concatenate(global_samples)

        for i, client_vgm in enumerate(cont_vgms):
            if col not in client_vgm:
                S[i, j] = 1.0
                continue
            local_samples = _generate_vgm_samples(client_vgm[col], n_vgm_samples, rng)
            S[i, j] = compute_wd(local_samples, global_combined)

    # Step 1: Normalize each column (divide by column sum)
    col_sums = S.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    S_norm = S / col_sums[np.newaxis, :]

    # Step 2: Sum across columns per client → SS[i]
    SS = S_norm.sum(axis=1)

    # Step 3: Normalize SS to [0,1], complement, fuse with quantity ratio
    ss_min, ss_max = SS.min(), SS.max()
    if ss_max - ss_min > 1e-12:
        SS_normalized = (SS - ss_min) / (ss_max - ss_min)
    else:
        SS_normalized = np.zeros(n_clients)

    similarity = 1.0 - SS_normalized
    quantity_ratio = np.array(client_sample_counts) / total_samples
    SD = similarity * quantity_ratio

    # Step 4: Softmax → final weights
    exp_sd = np.exp(SD - SD.max())  # Stable softmax
    weights = exp_sd / exp_sd.sum()

    return weights.tolist()
