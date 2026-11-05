"""Train the text-only baseline for Task 2 (Member B).

Eval-protocol compliant pipeline:
    1. load data/train.csv
    2. stratified 90/10 split by rating (fixed seed, see config.py)
    3. fit TF-IDF on the TRAIN FOLD ONLY  (no validation leakage)
    4. train self-implemented ridge regression with early stopping
    5. report validation RMSE against the global-mean baseline
    6. persist vectoriser + weights to artifacts/

Also prints the offline-time efficiency metric (preprocessing + training)
that the report requires; C later re-measures the same hooks on the merged
Spark pipeline.

Usage (run from the task2_model folder):
    python -m src.train
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import config
from src.data_loader import TEXT_COL, load_reviews
from src.evaluate import rmse, stratified_split
from src.model import RidgeRegressionGD
from src.text_repr import TfidfVectorizer


def main() -> None:
    t_offline_start = time.perf_counter()

    # ------------------------------------------------------------- 1. load --
    if not config.TRAIN_CSV.exists():
        print(f"[!] {config.TRAIN_CSV} not found.")
        print("    Download train.csv from the Task 2 Kaggle page and put it "
              "in data/. See README.md.")
        sys.exit(1)
    df = load_reviews(config.TRAIN_CSV, is_train=True)
    print(f"[i] train rows: {len(df)}")

    # ------------------------------------------- 2. canonical split (B->C) --
    if config.SPLIT_FILE.exists():
        # Use the frozen handoff split so C's label-dependent features are
        # guaranteed to be computed on exactly this train fold.
        val_ids = set(pd.read_csv(config.SPLIT_FILE)["id"].tolist())
        val_mask = df["id"].isin(val_ids).to_numpy()
        train_mask = ~val_mask
        print(f"[i] canonical split loaded: {train_mask.sum()} train / "
              f"{val_mask.sum()} val (from {config.SPLIT_FILE.name})")
    else:
        print("[!] split/val_ids.csv missing -- in-process fallback split. "
              "Run `python -m src.make_split` and share split/ with C.")
        train_mask, val_mask = stratified_split(
            df["rating"].to_numpy(), config.VAL_RATIO, config.SEED)
        print(f"[i] split: {train_mask.sum()} train / {val_mask.sum()} val "
              f"(seed={config.SEED})")

    if config.SAMPLE_FRAC:
        # Pilot mode: subsample the TRAIN fold only; keep the validation
        # fold intact so pilot RMSE stays comparable across runs.
        rng = np.random.default_rng(config.SEED)
        tr_pos = np.where(train_mask)[0]
        keep = rng.choice(tr_pos, size=int(len(tr_pos) * config.SAMPLE_FRAC),
                          replace=False)
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[keep] = True
        print(f"[i] pilot: train fold sampled to {len(keep)} rows "
              f"(val fold kept full: {val_mask.sum()})")

    # ------------------------------------------ 3. TF-IDF (train fold only) --
    t0 = time.perf_counter()
    vec = TfidfVectorizer(min_df=config.MIN_DF,
                          max_features=config.MAX_FEATURES,
                          ngram_range=config.NGRAM_RANGE)
    Xtr = vec.fit_transform(df.loc[train_mask, TEXT_COL])
    Xva = vec.transform(df.loc[val_mask, TEXT_COL])
    t_tfidf = time.perf_counter() - t0
    print(f"[i] tf-idf: {Xtr.shape[1]} features "
          f"(fit on train fold only, {t_tfidf:.1f}s)")

    # ------------------------------------------------------------ 4. train --
    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=float)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=float)

    baseline = rmse(yva, np.full(len(yva), ytr.mean()))
    print(f"[i] global-mean baseline val RMSE: {baseline:.4f}")

    model = RidgeRegressionGD(lr=config.LEARNING_RATE, epochs=config.EPOCHS,
                              l2=config.L2, lr_decay=config.LR_DECAY,
                              patience=config.PATIENCE).fit(Xtr, ytr, Xva, yva)

    print(f"[i] model val RMSE: {model.best_val_rmse_:.4f} "
          f"(baseline {baseline:.4f})")

    # ---------------------------------------------------------- 5. persist --
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    vec.save(config.VECTORIZER_FILE)
    model.save(config.MODEL_FILE)
    print(f"[i] artifacts saved to {config.ARTIFACT_DIR}/")

    # ----------------------------------------------- 6. efficiency metric --
    offline = time.perf_counter() - t_offline_start
    avg_epoch = sum(model.epoch_times_) / max(1, len(model.epoch_times_))
    print(f"[perf] offline time (load+preprocess+train): {offline:.1f}s | "
          f"avg {avg_epoch:.3f}s/epoch over {len(model.epoch_times_)} epochs")


if __name__ == "__main__":
    main()
