"""Fed-TGAN Data Sampler.

Implements CTGAN's "training-by-sampling" for conditional generation.
Ensures categorical levels are sampled proportionally during GAN training
to prevent mode collapse on minority categories.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fedbench.algorithms.fed_tgan.data_transformer import SpanInfo


class DataSampler:
    """Samples conditional vectors and matching real data for GAN training.

    Implements CTGAN's training-by-sampling strategy: samples discrete
    columns proportionally to prevent mode collapse on rare categories.
    Maintains per-category row indices and probability tables for efficient
    conditional sampling during training.
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        output_info: list[list[SpanInfo]],
        log_frequency: bool = True,
    ) -> None:
        self._data_length = len(data)

        def _is_discrete(col_info: list[SpanInfo]) -> bool:
            return len(col_info) == 1 and col_info[0].activation_fn == "softmax"

        n_discrete = sum(1 for ci in output_info if _is_discrete(ci))
        self._n_discrete_columns = n_discrete

        # For each discrete column, store row indices per category
        self._rid_by_cat_cols: list[list[NDArray[np.intp]]] = []
        self._discrete_column_matrix_st = np.zeros(n_discrete, dtype=np.int32)

        st = 0
        disc_idx = 0
        for col_info in output_info:
            if _is_discrete(col_info):
                span = col_info[0]
                ed = st + span.dim
                self._discrete_column_matrix_st[disc_idx] = st
                rid_by_cat: list[NDArray[np.intp]] = []
                for j in range(span.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                disc_idx += 1
                st = ed
            else:
                st += sum(s.dim for s in col_info)

        # Probability tables for conditional sampling
        max_category = max(
            (ci[0].dim for ci in output_info if _is_discrete(ci)),
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete, dtype=np.int32)
        self._discrete_column_n_category = np.zeros(n_discrete, dtype=np.int32)
        self._discrete_column_category_prob = np.zeros((n_discrete, max_category))

        self._n_categories = sum(ci[0].dim for ci in output_info if _is_discrete(ci))

        st = 0
        current_id = 0
        current_cond_st = 0
        for col_info in output_info:
            if _is_discrete(col_info):
                span = col_info[0]
                ed = st + span.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, : span.dim] = (
                    category_prob
                )
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span.dim
                current_cond_st += span.dim
                current_id += 1
                st = ed
            else:
                st += sum(s.dim for s in col_info)

    def dim_cond_vec(self) -> int:
        """Total number of categories across all discrete columns."""
        return self._n_categories

    def sample_condvec(
        self,
        batch: int,
    ) -> (
        tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.intp],
            NDArray[np.intp],
        ]
        | None
    ):
        """Sample a conditional vector for training.

        Parameters
        ----------
        batch
            Number of conditional vectors to produce.

        Returns
        -------
        tuple or None
            ``(cond, mask, discrete_column_id, category_id_in_col)``
            or ``None`` if there are no discrete columns.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(self._n_discrete_columns, batch)

        cond = np.zeros((batch, self._n_categories), dtype=np.float32)
        mask = np.zeros((batch, self._n_discrete_columns), dtype=np.float32)
        mask[np.arange(batch), discrete_column_id] = 1

        # Sample category weighted by probability
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(batch), axis=1)
        category_id_in_col = (probs.cumsum(axis=1) > r).argmax(axis=1)

        category_id = (
            self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        )
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch: int) -> NDArray[np.floating] | None:
        """Sample conditional vector using original frequency distribution.

        Unlike ``sample_condvec``, this samples across *all* categories
        proportionally, suitable for unconditional generation.
        """
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)
        col_idxs = np.random.choice(len(category_freq), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype=np.float32)
        cond[np.arange(batch), col_idxs] = 1

        return cond

    def sample_data(
        self,
        data: NDArray[np.floating],
        n: int,
        col: NDArray[np.intp] | None,
        opt: NDArray[np.intp] | None,
    ) -> NDArray[np.floating]:
        """Sample real data rows, optionally matching the conditional vector.

        When *col* and *opt* are provided, exactly ``len(col)`` rows are
        returned (one per conditional entry); *n* is ignored in that case.

        Parameters
        ----------
        data
            Encoded training data matrix.
        n
            Number of rows to sample (used only when ``col is None``).
        col
            Discrete column indices from ``sample_condvec``.
        opt
            Category indices within each selected column.

        Returns
        -------
        ndarray
            Sampled rows from *data*.
        """
        if col is None:
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        for c, o in zip(col, opt, strict=True):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return data[np.array(idx)]
