"""Focused tuning of the winning 'all' fusion group.

Same protocol as run_ablation (pilot sample, is_val split, scaler fitted on
training rows). TF-IDF and the dense block are built ONCE and reused across
configs -- only the optimisation schedule changes.

    F1: lr 0.5, 500 epochs   (text sweep showed lr 0.5 > 0.3)
    F2: lr 0.4, 500 epochs   (safety point, dense block raises lambda_max)

Divergence guard: a config whose validation RMSE goes non-finite is skipped
(early stopping already keeps the best finite weights).

Usage: python -m src.tune_fusion
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import config
from src.data_loader import TEXT_COL, load_reviews
from src.feature_fusion import (ALL_COLS, AugmentedCSR, apply_scaler,
                                fit_scaler, load_features)
from src.model import RidgeRegressionGD
from src.text_repr import TfidfVectorizer

CONFIGS = [
    {"name": "F1-lr.5-ep500", "lr": 0.5, "epochs": 500},
    {"name": "F2-lr.4-ep500", "lr": 0.4, "epochs": 500},
]


def main() -> None:
    t0 = time.perf_counter()
    df = load_reviews(config.TRAIN_CSV, is_train=True)
    feats = load_features(config.TRAIN_FEATURES)
    overlap = [c for c in feats.columns if c in df.columns and c != "id"]
    if overlap:
        feats = feats.drop(columns=overlap)
    df = df.merge(feats, on="id", how="left", validate="one_to_one")
    print(f"[i] merged: {len(df)} rows ({time.perf_counter() - t0:.0f}s)")

    val_mask = (df["is_val"] == 1).to_numpy()
    base_train_mask = (df["is_val"] == 0).to_numpy()
    if config.SAMPLE_FRAC:
        rng = np.random.default_rng(config.SEED)
        tr_pos = np.where(base_train_mask)[0]
        keep = rng.choice(tr_pos, size=int(len(tr_pos) * config.SAMPLE_FRAC),
                          replace=False)
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[keep] = True
        print(f"[i] pilot: {len(keep)} train rows")
    else:
        train_mask = base_train_mask

    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=float)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=float)

    # ---- shared representations (config-independent) ----
    vec = TfidfVectorizer(min_df=config.MIN_DF,
                          max_features=config.MAX_FEATURES,
                          ngram_range=config.NGRAM_RANGE)
    Xtr_s = vec.fit_transform(df.loc[train_mask, TEXT_COL])
    Xva_s = vec.transform(df.loc[val_mask, TEXT_COL])
    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    Xtr = AugmentedCSR(Xtr_s, apply_scaler(df.loc[train_mask], ALL_COLS, scaler))
    Xva = AugmentedCSR(Xva_s, apply_scaler(df.loc[val_mask], ALL_COLS, scaler))
    print(f"[i] tf-idf {Xtr_s.shape[1]} feats + {len(ALL_COLS)} dense cols "
          f"({time.perf_counter() - t0:.0f}s)")

    best = None   # (rmse, cfg, model)
    for cfg in CONFIGS:
        model = RidgeRegressionGD(lr=cfg["lr"], epochs=cfg["epochs"],
                                  l2=config.L2, lr_decay=0.995,
                                  patience=config.PATIENCE,
                                  verbose=False).fit(Xtr, ytr, Xva, yva)
        r = model.best_val_rmse_
        finite = np.isfinite(r) and np.all(np.isfinite(model.w_))
        print(f"[result] {cfg['name']}: val RMSE "
              f"{'nan/inf (skipped)' if not finite else f'{r:.4f}'} | "
              f"{len(model.epoch_times_)} epochs")
        if finite and (best is None or r < best[0]):
            best = (r, cfg, model)

    if best is None:
        print("[!] all configs diverged -- keeping previous artifacts")
        sys.exit(1)

    print(f"\nbest: {best[1]['name']} val RMSE {best[0]:.4f} "
          f"(run_ablation baseline: 1.4301 @ lr .3 / 300 ep)")
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    vec.save(config.VECTORIZER_FILE)
    best[2].save(config.MODEL_FILE)
    meta = {"group": "all", "cols": ALL_COLS, "scaler": scaler,
            "val_rmse": best[0], "tuning": best[1]}
    (config.ARTIFACT_DIR / "fusion_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("[i] artifacts saved -- `python -m src.predict` produces the "
          "v4 submission")
    print(f"[perf] offline time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
