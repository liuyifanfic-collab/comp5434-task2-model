"""Evaluation utilities: RMSE and a stratified train/validation splitter."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error -- the official metric for Dataset 2."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def stratified_split(y, val_ratio: float, seed: int):
    """Split indices so that every rating class keeps the same val fraction.

    Returns (train_mask, val_mask). Fixed seed -> identical split for every
    member, which is the anti-leakage contract in eval_protocol.md: all
    label-dependent statistics (C's user/product features) must be computed
    on the TRAIN side only.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        k = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(rng.choice(idx, size=k, replace=False).tolist())
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True
    return ~val_mask, val_mask
