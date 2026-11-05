"""Diagnostic 2: does exact TEXT matching explain the 0.4x scores?

If the dataset contains (near-)templated reviews, the same comment text
maps to the same rating regardless of user -- a text-hash lookup would
then be a near-perfect predictor.

Checks (leakage-free, labels only from the train fold):
    1. On the val fold: how often does a comment (or title+comment) hash
       appear in the train fold, and what is the lookup RMSE?
    2. On test rows: match rate against the FULL train set (coverage only).

Usage: python -m src.diag_text_match
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> None:
    tr = pd.read_csv(config.TRAIN_CSV,
                     usecols=["id", "title", "comment", "rating"])
    val_ids = set(pd.read_csv(config.SPLIT_FILE)["id"])
    is_val = tr["id"].isin(val_ids).to_numpy()

    text = (tr["title"].fillna("").astype(str) + " ¦ "
            + tr["comment"].fillna("").astype(str))
    h = text.map(hash)                       # 64-bit exact-text hash
    tr = tr.assign(h=h)

    trainf = tr[~is_val]
    valf = tr[is_val]

    # Rating consistency per text hash inside the TRAIN fold (top groups only
    # for reporting); map hash -> mean rating for hashes seen in train fold.
    h_map = trainf.groupby("h")["rating"].agg(["mean", "count"])

    m = valf["h"].isin(h_map.index).to_numpy()
    print(f"[val] text-hash match rate vs train fold: {m.mean():.4f}")
    if m.any():
        pred = valf.loc[m, "h"].map(h_map["mean"]).to_numpy()
        truth = valf.loc[m, "rating"].to_numpy()
        print(f"[val] lookup RMSE on text matches:      {rmse(truth, pred):.4f}")
        # spread of ratings within a duplicated text (train fold)
        multi = h_map[h_map["count"] >= 2]
        if len(multi):
            print(f"[val] duplicated texts in train fold:  {len(multi)} "
                  f"(max count {int(multi['count'].max())})")

    # ------------------------------------------------ test-side coverage
    te = pd.read_csv(config.TEST_CSV, usecols=["id", "title", "comment"])
    te_text = (te["title"].fillna("").astype(str) + " ¦ "
               + te["comment"].fillna("").astype(str))
    te_h = te_text.map(hash)
    full_h = set(tr["h"])
    hit = te_h.isin(full_h).to_numpy()
    print(f"\n[test] text-hash match rate vs FULL train: {hit.mean():.4f}")


if __name__ == "__main__":
    main()
