"""One-time canonical train/val split for Task 2 (B -> C handoff artifact).

Writes:
    split/val_ids.csv  -- validation-fold ids (+rating) over the FULL train.csv
    split/stats.json   -- fold sizes + train-fold global mean rating
                          (the fallback value for unknown users/products)

Why a file instead of re-computing on each side: C works on Spark, where
row order and RNG differ from pandas; a shared id list is the only way to
guarantee both sides use the IDENTICAL split, which the anti-leakage rule
in eval_protocol.md requires (label-dependent statistics may only use the
train fold).

Usage (run ONCE, then treat as frozen):
    python -m src.make_split
"""
from __future__ import annotations

import json
import sys

import pandas as pd

import config
from src.evaluate import stratified_split


def main() -> None:
    if config.SPLIT_FILE.exists():
        print(f"[!] {config.SPLIT_FILE} already exists -- the split is frozen.")
        print("    Delete it manually if you truly need to regenerate.")
        sys.exit(1)
    if not config.TRAIN_CSV.exists():
        print(f"[!] {config.TRAIN_CSV} not found.")
        sys.exit(1)

    # Only id + rating are needed -> cheap on the 1.9 GB csv.
    df = pd.read_csv(config.TRAIN_CSV, usecols=["id", "rating"])
    train_mask, val_mask = stratified_split(
        df["rating"].to_numpy(), config.VAL_RATIO, config.SEED)

    config.SPLIT_DIR.mkdir(exist_ok=True)
    val = df.loc[val_mask, ["id", "rating"]]
    val.to_csv(config.SPLIT_FILE, index=False)

    stats = {
        "seed": config.SEED,
        "val_ratio": config.VAL_RATIO,
        "n_train_fold": int(train_mask.sum()),
        "n_val_fold": int(val_mask.sum()),
        "train_fold_mean_rating": float(df.loc[train_mask, "rating"].mean()),
    }
    (config.SPLIT_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[i] wrote {config.SPLIT_FILE} ({len(val)} ids) + stats.json")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
