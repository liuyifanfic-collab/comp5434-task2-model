"""Diagnostic: does (user, product) pair matching explain the 0.4x scores?

Checks on the VALIDATION fold (labels available, leakage-free):
    1. How many val rows have their (user, prod) or (user, parent_prod)
       pair present in the TRAIN fold?
    2. If we predict the matched pair's historical mean rating, what is
       the RMSE on those rows?
    3. Hybrid estimate: lookup on matched rows + current model (1.4033)
       on the rest.

Also reports match rates on the TEST rows (no labels, coverage only),
matched against the FULL training set.

Usage: python -m src.diag_pair_match
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> None:
    tr = pd.read_csv(config.TRAIN_CSV,
                     usecols=["id", "user_id", "prod_id", "parent_prod_id",
                              "rating"])
    val_ids = set(pd.read_csv(config.SPLIT_FILE)["id"])
    is_val = tr["id"].isin(val_ids).to_numpy()
    trainf = tr[~is_val]
    valf = tr[is_val]
    print(f"train fold {len(trainf)} | val fold {len(valf)}")

    # Integer codes for cheap composite keys (factorised over FULL train).
    u_codes, u_uniq = pd.factorize(tr["user_id"])
    p_codes, _ = pd.factorize(tr["prod_id"])
    pp_codes, pp_uniq = pd.factorize(tr["parent_prod_id"])
    tr = tr.assign(u=u_codes, p=p_codes, pp=pp_codes)
    tr["up"] = tr["u"].astype(np.int64) * len(pp_uniq) + tr["p"]
    tr["upp"] = tr["u"].astype(np.int64) * len(pp_uniq) + tr["pp"]
    trainf = tr[~is_val]
    valf = tr[is_val]

    up_map = trainf.groupby("up")["rating"].mean()
    upp_map = trainf.groupby("upp")["rating"].mean()

    # ----------------------------------------------- validation fold stats
    m_up = valf["up"].isin(up_map.index).to_numpy()
    m_upp_only = (valf["upp"].isin(upp_map.index).to_numpy() & ~m_up)
    print(f"\n[val] exact (user,prod) match rate:    {m_up.mean():.4f}")
    print(f"[val] parent-only match rate:          {m_upp_only.mean():.4f}")

    if m_up.any():
        pred = valf.loc[m_up, "up"].map(up_map).to_numpy()
        truth = valf.loc[m_up, "rating"].to_numpy()
        print(f"[val] lookup RMSE on exact matches:    {rmse(truth, pred):.4f}")
    if m_upp_only.any():
        pred = valf.loc[m_upp_only, "upp"].map(upp_map).to_numpy()
        truth = valf.loc[m_upp_only, "rating"].to_numpy()
        print(f"[val] lookup RMSE on parent matches:   {rmse(truth, pred):.4f}")

    # Hybrid estimate vs our current model on the val fold.
    p_exact = float(m_up.mean())
    p_parent = float(m_upp_only.mean())
    r_exact = rmse(valf.loc[m_up, "up"].map(up_map).to_numpy(),
                   valf.loc[m_up, "rating"].to_numpy()) if m_up.any() else 0.0
    r_parent = rmse(valf.loc[m_upp_only, "upp"].map(upp_map).to_numpy(),
                    valf.loc[m_upp_only, "rating"].to_numpy()) if m_upp_only.any() else 0.0
    model_rmse = 1.4033          # current fusion, pilot
    hybrid = np.sqrt(p_exact * r_exact ** 2
                     + p_parent * r_parent ** 2
                     + (1 - p_exact - p_parent) * model_rmse ** 2)
    print(f"[val] hybrid estimate (lookup + model): {hybrid:.4f}")

    # ------------------------------------------------- test-side coverage
    te = pd.read_csv(config.TEST_CSV,
                     usecols=["id", "user_id", "prod_id", "parent_prod_id"])
    te_u = pd.Index(u_uniq).get_indexer(te["user_id"])
    te_p = pd.Index(pd.factorize(tr["prod_id"])[1]).get_indexer(te["prod_id"])
    te_pp = pd.Index(pp_uniq).get_indexer(te["parent_prod_id"])
    # NOTE: pair maps above are train-fold only; for the final submission the
    # lookup would use the FULL train (incl. val fold). Report both.
    full_up = set(tr["up"])
    full_upp = set(tr["upp"])
    te_up = te_u.astype(np.int64) * len(pp_uniq) + te_p
    te_upp = te_u.astype(np.int64) * len(pp_uniq) + te_pp
    known = te_u >= 0
    print(f"\n[test] user known in train:            {known.mean():.4f}")
    print(f"[test] (user,prod) in FULL train:      "
          f"{(np.isin(te_up, list(full_up))).mean():.4f}")
    print(f"[test] (user,parent) in FULL train:    "
          f"{(np.isin(te_upp, list(full_upp))).mean():.4f}")


if __name__ == "__main__":
    main()
