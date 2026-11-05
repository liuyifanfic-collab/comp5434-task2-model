"""Minimal CSR sparse matrix with exactly the two mat-vec kernels we need.

scipy is avoided on purpose: the kernels below are the precise operations a
Spark implementation would parallelise (map over non-zeros -> reduce by key),
so the local prototype (B) and the distributed pipeline (C) share the same
mathematical core. This makes the final "efficiency metrics on the merged
pipeline" step a drop-in replacement instead of a rewrite.
"""
from __future__ import annotations

import numpy as np


class CSRMatrix:
    """Compressed sparse row matrix X of shape (n_rows, n_features)."""

    def __init__(self, data, indices, indptr, n_features: int):
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        # int32 for indices / row ids: at full scale (~460M non-zeros) this
        # halves the index memory vs int64; values are always < 2^31 here.
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.n_features = int(n_features)

        n_rows = len(self.indptr) - 1
        self.shape = (n_rows, self.n_features)

        # Row id of every non-zero, pre-computed once so that BOTH mat-vec
        # products collapse to a single np.bincount reduction (fast, and a
        # direct analogue of Spark's reduceByKey).
        self._row_ids = np.repeat(
            np.arange(n_rows, dtype=np.int32), np.diff(self.indptr)
        )

    def dot(self, v: np.ndarray) -> np.ndarray:
        """X @ v  ->  shape (n_rows,)."""
        v = np.asarray(v, dtype=np.float64)
        contrib = self.data * v[self.indices]
        return np.bincount(self._row_ids, weights=contrib, minlength=self.shape[0])

    def T_dot(self, r: np.ndarray) -> np.ndarray:
        """X.T @ r  ->  shape (n_features,)."""
        r = np.asarray(r, dtype=np.float64)
        contrib = self.data * r[self._row_ids]
        return np.bincount(self.indices, weights=contrib, minlength=self.n_features)
