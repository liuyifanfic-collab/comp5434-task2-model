"""Fusion module tests: kernel correctness + synthetic end-to-end.

1. AugmentedCSR.dot / T_dot must match a naive dense reference exactly.
2. Fusing user/product bias features must beat the text-only model.

Run from the task2_model folder:
    python -m tests.fusion_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import rmse, stratified_split           # noqa: E402
from src.feature_fusion import (AugmentedCSR, apply_scaler,  # noqa: E402
                                fit_scaler)
from src.model import RidgeRegressionGD                   # noqa: E402
from src.sparse import CSRMatrix                          # noqa: E402
from src.text_repr import TfidfVectorizer                 # noqa: E402
from tests.smoke_test import make_row                     # noqa: E402


def test_kernels() -> None:
    rng = np.random.default_rng(7)
    n, d, k = 6, 4, 3
    counts = [3, 0, 1, 4, 2, 2]
    data, indices, indptr = [], [], [0]
    for row, c in enumerate(counts):
        idx = np.sort(rng.choice(d, size=c, replace=False))
        indices.extend(idx.tolist())
        data.extend(rng.uniform(0.2, 1.0, size=c).tolist())
        indptr.append(len(indices))
    sparse = CSRMatrix(data, indices, indptr, d)

    dense = rng.normal(size=(n, k))
    aug = AugmentedCSR(sparse, dense)

    # Naive dense reference for the virtual [sparse | dense] matrix.
    ref = np.zeros((n, d))
    for row in range(n):
        for p in range(indptr[row], indptr[row + 1]):
            ref[row, indices[p]] = data[p]
    M = np.hstack([ref, dense])

    v = rng.normal(size=d + k)
    r = rng.normal(size=n)
    assert np.allclose(aug.dot(v), M @ v), "dot kernel mismatch"
    assert np.allclose(aug.T_dot(r), M.T @ r), "T_dot kernel mismatch"
    assert aug.shape == (n, d + k)
    print("[ok] AugmentedCSR kernels match dense reference")


def test_end_to_end() -> None:
    rng = np.random.default_rng(0)
    n_rows = 1500
    users = [f"u{i:03d}" for i in range(80)]
    prods = [f"p{i:03d}" for i in range(50)]
    user_bias = {u: rng.normal(0.0, 0.6) for u in users}
    prod_bias = {p: rng.normal(0.0, 0.4) for p in prods}

    rows = []
    for i in range(n_rows):
        r = make_row(rng)                     # sentiment-driven synthetic review
        u, p = rng.choice(users), rng.choice(prods)
        r["user_id"], r["prod_id"] = u, p
        r["parent_prod_id"] = "pp" + p[1:]
        # rating = sentiment + stable user/product tastes + noise
        score = (r["rating"]                                  # ~sentiment part
                 + user_bias[u] + prod_bias[p]
                 + rng.normal(0.0, 0.15))
        r["rating"] = int(np.clip(round(score), 1, 5))
        rows.append(r)
    df = pd.DataFrame(rows)

    tr_mask, va_mask = stratified_split(df["rating"].to_numpy(), 0.2, 5434)
    ytr = df.loc[tr_mask, "rating"].to_numpy(dtype=float)
    yva = df.loc[va_mask, "rating"].to_numpy(dtype=float)

    # -------- text-only baseline
    vec = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    Xtr_s = vec.fit_transform(df.loc[tr_mask, "comment"])
    Xva_s = vec.transform(df.loc[va_mask, "comment"])
    m_text = RidgeRegressionGD(lr=0.3, epochs=200, l2=1e-4,
                               patience=30).fit(Xtr_s, ytr, Xva_s, yva)

    # -------- C-like features: label stats from the TRAIN fold only
    fold = df.loc[tr_mask]
    u_mean = fold.groupby("user_id")["rating"].mean()
    p_mean = fold.groupby("prod_id")["rating"].mean()
    global_mean = float(fold["rating"].mean())

    def feat_row(row: pd.Series) -> dict:
        um = float(u_mean.get(row["user_id"], global_mean))
        pm = float(p_mean.get(row["prod_id"], global_mean))
        return {"user_rating_mean": um,
                "user_rating_std": 0.0,
                "user_rating_cnt": int(row["user_id"] in u_mean.index),
                "prod_rating_mean": pm,
                "prod_rating_cnt": int(row["prod_id"] in p_mean.index),
                "user_known_flag": int(row["user_id"] in u_mean.index),
                "prod_known_flag": int(row["prod_id"] in p_mean.index)}

    feats = pd.DataFrame([feat_row(r) for _, r in df.iterrows()])
    cols = ["user_rating_mean", "user_rating_std", "user_rating_cnt",
            "prod_rating_mean", "prod_rating_cnt",
            "user_known_flag", "prod_known_flag"]
    scaler = fit_scaler(feats.loc[tr_mask], cols)          # train fold only
    Xtr = AugmentedCSR(Xtr_s, apply_scaler(feats.loc[tr_mask], cols, scaler))
    Xva = AugmentedCSR(Xva_s, apply_scaler(feats.loc[va_mask], cols, scaler))

    m_fuse = RidgeRegressionGD(lr=0.3, epochs=200, l2=1e-4,
                               patience=30).fit(Xtr, ytr, Xva, yva)

    print(f"[i] text-only val RMSE {m_text.best_val_rmse_:.4f} | "
          f"fused val RMSE {m_fuse.best_val_rmse_:.4f}")
    assert m_fuse.best_val_rmse_ < m_text.best_val_rmse_ - 0.05, (
        "fusion should beat text-only on synthetic bias data")
    print("[ok] fusion beats text-only")


if __name__ == "__main__":
    test_kernels()
    test_end_to_end()
    print("FUSION TEST PASSED")
