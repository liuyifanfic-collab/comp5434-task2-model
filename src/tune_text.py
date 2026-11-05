"""Text-side hyper-parameter sweep (pilot runs on sampled train fold).

Runs three configs sequentially, sharing one data load and the canonical
split; saves the best config's artifacts so predict.py can use them
directly. Results go to the console -- record them in experiments.md.

    A-ep300   : 30k features, 300 epochs      -> isolates the epochs effect
    B-feat60k : 60k features, 300 epochs      -> adds the vocabulary effect
    C-data30  : 60k features, 300 ep, 30% data-> adds the data-amount effect

Usage: python -m src.tune_text
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from src.data_loader import TEXT_COL, load_reviews
from src.model import RidgeRegressionGD
from src.text_repr import TfidfVectorizer

CONFIGS = [
    {"name": "A-ep300",   "feats": 30000, "epochs": 300, "frac": 0.1, "l2": 1e-4},
    {"name": "B-feat60k", "feats": 60000, "epochs": 300, "frac": 0.1, "l2": 1e-4},
    {"name": "C-data30",  "feats": 60000, "epochs": 300, "frac": 0.3, "l2": 1e-4},
]


def main() -> None:
    t_all = time.perf_counter()
    df = load_reviews(config.TRAIN_CSV, is_train=True)
    print(f"[i] loaded {len(df)} rows")

    val_ids = set(pd.read_csv(config.SPLIT_FILE)["id"].tolist())
    val_mask = df["id"].isin(val_ids).to_numpy()
    base_train_mask = ~val_mask
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=float)
    print(f"[i] canonical split: {base_train_mask.sum()} train / "
          f"{val_mask.sum()} val")

    best = None   # (rmse, cfg, vec, model)
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        print(f"\n=== {cfg['name']}: feats={cfg['feats']} "
              f"epochs={cfg['epochs']} frac={cfg['frac']} l2={cfg['l2']} ===")

        # Sample the train fold for this config (same seed -> deterministic).
        rng = np.random.default_rng(config.SEED)
        tr_pos = np.where(base_train_mask)[0]
        keep = rng.choice(tr_pos, size=int(len(tr_pos) * cfg["frac"]),
                          replace=False)
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[keep] = True

        vec = TfidfVectorizer(min_df=config.MIN_DF,
                              max_features=cfg["feats"],
                              ngram_range=config.NGRAM_RANGE)
        Xtr = vec.fit_transform(df.loc[train_mask, TEXT_COL])
        Xva = vec.transform(df.loc[val_mask, TEXT_COL])
        ytr = df.loc[train_mask, "rating"].to_numpy(dtype=float)
        t_prep = time.perf_counter() - t0

        model = RidgeRegressionGD(lr=0.5, epochs=cfg["epochs"], l2=cfg["l2"],
                                  lr_decay=0.995, patience=15,
                                  verbose=False).fit(Xtr, ytr, Xva, yva)
        dt = time.perf_counter() - t0
        n_ep = len(model.epoch_times_)
        print(f"[result] {cfg['name']} val RMSE {model.best_val_rmse_:.4f} | "
              f"ran {n_ep} epochs | prep {t_prep:.0f}s | total {dt:.0f}s")

        if best is None or model.best_val_rmse_ < best[0]:
            best = (model.best_val_rmse_, cfg, vec, model)

        del Xtr, Xva   # free memory before the next config

    print("\n=== sweep summary ===")
    print(f"best: {best[1]['name']} val RMSE {best[0]:.4f}")
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    best[2].save(config.VECTORIZER_FILE)
    best[3].save(config.MODEL_FILE)
    print(f"[i] best-config artifacts saved to {config.ARTIFACT_DIR}/ -- "
          f"`python -m src.predict` now produces the v2 submission")
    print(f"[perf] sweep offline time: {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
