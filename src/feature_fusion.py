"""Fusion of C's dense statistical features with our sparse TF-IDF matrix.

C's v2 delivery (2026-11-03, "leak-free"):
- parquet directories, 21 columns, LOO statistics + Bayesian shrinkage
- no rating column; labels must be merged from the original train.csv
- is_val marks the validation fold (verified identical to split/val_ids.csv)

Design: AugmentedCSR exposes the same two kernels (dot / T_dot) that
RidgeRegressionGD already uses, so the model code is untouched -- it simply
sees one wider virtual matrix [ TF-IDF | standardised stats ]. This keeps
the "local prototype == distributed core" story intact for the report.

Leakage notes:
- C's features are leave-one-out statistics over the train fold only.
- The dense scaler (mean/std, log1p on counts) is fitted on the rows used
  for training (see run_ablation.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sparse import CSRMatrix

# Ablation groups over C's v2 schema (任务二特征规范.md §3)
USER_COLS = ["user_hist_cnt", "user_hist_mean", "user_hist_std",
             "user_hist_smooth", "user_hist_textlen", "user_has_hist"]
PROD_COLS = ["prod_hist_cnt", "prod_hist_mean", "prod_hist_std",
             "prod_hist_smooth", "prod_has_hist",
             "parent_hist_cnt", "parent_hist_mean", "parent_hist_smooth",
             "parent_has_hist"]
OTHER_COLS = ["time_hour", "time_dow", "votes", "purchased"]
ALL_COLS = USER_COLS + PROD_COLS + OTHER_COLS          # 19 dense features

# Count-like columns get a log1p transform before standardisation.
LOG1P_COLS = {"user_hist_cnt", "prod_hist_cnt", "parent_hist_cnt",
              "user_rating_cnt", "prod_rating_cnt"}    # legacy names kept for tests


class AugmentedCSR:
    """Virtual matrix [sparse CSR block | dense block] for gradient descent."""

    def __init__(self, sparse_part: CSRMatrix, dense_part=None):
        self.sparse = sparse_part
        self.dense = (None if dense_part is None
                      else np.ascontiguousarray(dense_part, dtype=np.float64))
        k = 0 if self.dense is None else self.dense.shape[1]
        self.shape = (sparse_part.shape[0], sparse_part.shape[1] + k)

    def dot(self, v: np.ndarray) -> np.ndarray:
        d = self.sparse.shape[1]
        out = self.sparse.dot(v[:d])
        if self.dense is not None:
            out = out + self.dense @ v[d:]
        return out

    def T_dot(self, r: np.ndarray) -> np.ndarray:
        head = self.sparse.T_dot(r)
        if self.dense is None:
            return head
        return np.concatenate([head, self.dense.T @ r])


def load_features(path) -> pd.DataFrame:
    """Read one of C's parquet feature directories and validate the schema."""
    df = pd.read_parquet(path)
    missing = [c for c in (["id"] + ALL_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing feature columns {missing}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicated ids")
    if df[ALL_COLS].isna().any().any():
        raise ValueError(f"{path}: NaN in feature columns")
    return df


def _extract(feat_df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = feat_df[cols].to_numpy(dtype=np.float64).copy()
    for j, c in enumerate(cols):
        if c in LOG1P_COLS:                       # counts -> log scale
            X[:, j] = np.log1p(np.clip(X[:, j], 0.0, None))
    return X


def fit_scaler(feat_df: pd.DataFrame, cols: list[str]) -> dict:
    """Fitted on the training rows only."""
    X = _extract(feat_df, cols)
    return {"mean": X.mean(axis=0).tolist(),
            "std": (X.std(axis=0) + 1e-8).tolist(),
            "cols": cols}


def apply_scaler(feat_df: pd.DataFrame, cols: list[str], scaler: dict):
    X = _extract(feat_df, cols)
    mean = np.asarray(scaler["mean"])
    std = np.asarray(scaler["std"])
    return (X - mean) / std
