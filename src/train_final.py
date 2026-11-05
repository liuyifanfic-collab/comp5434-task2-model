"""Final full-data training run (Plan A: local execution).

Trains the winning configuration on the FULL training fold (no sampling):

    TF-IDF (30k terms, min_df 3, uni+bigrams)  +  all 19 dense features
    fused as AugmentedCSR, trained by Ridge GD:
        lr 0.5, decay 0.995, 500 epochs, l2 1e-4, patience 15

Outputs:
    artifacts/tfidf.npz, artifacts/ridge_model.npz, fusion_meta.json
    artifacts/final_run.json  -- timing metrics for the report
        (per-epoch time, offline total; inference time is added by predict)

Usage: python -m src.train_final
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

# Tuned winning configuration (sweep 2026-11-03: F1-lr.5-ep500).
FINAL = {"lr": 0.5, "epochs": 500, "l2": 1e-4, "lr_decay": 0.995,
         "patience": 15}


def main() -> None:
    t0 = time.perf_counter()
    df = load_reviews(config.TRAIN_CSV, is_train=True)
    feats = load_features(config.TRAIN_FEATURES)
    overlap = [c for c in feats.columns if c in df.columns and c != "id"]
    if overlap:
        feats = feats.drop(columns=overlap)
    df = df.merge(feats, on="id", how="left", validate="one_to_one")
    print(f"[final] merged: {len(df)} rows")
    val_mask = (df["is_val"] == 1).to_numpy()
    train_mask = (df["is_val"] == 0).to_numpy()
    print(f"[final] FULL training fold: {train_mask.sum()} rows | "
          f"val fold: {val_mask.sum()}")
    t_load = time.perf_counter() - t0

    # ------------------------------------------------------ representations
    t1 = time.perf_counter()
    vec = TfidfVectorizer(min_df=config.MIN_DF,
                          max_features=config.MAX_FEATURES,
                          ngram_range=config.NGRAM_RANGE)
    Xtr_s = vec.fit_transform(df.loc[train_mask, TEXT_COL])
    Xva_s = vec.transform(df.loc[val_mask, TEXT_COL])
    t_tfidf = time.perf_counter() - t1
    print(f"[final] tf-idf: {Xtr_s.shape[1]} features "
          f"(fit on full train fold, {t_tfidf:.0f}s)")

    # Free the text columns -- not needed from here on (~2-3 GB).
    df.drop(columns=[c for c in ("title", "comment", TEXT_COL)
                     if c in df.columns], inplace=True)

    t2 = time.perf_counter()
    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    Xtr = AugmentedCSR(Xtr_s, apply_scaler(df.loc[train_mask], ALL_COLS, scaler))
    Xva = AugmentedCSR(Xva_s, apply_scaler(df.loc[val_mask], ALL_COLS, scaler))
    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=float)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=float)
    t_dense = time.perf_counter() - t2
    print(f"[final] dense block: {len(ALL_COLS)} cols ({t_dense:.0f}s)")

    # ------------------------------------------------------------- training
    model = RidgeRegressionGD(lr=FINAL["lr"], epochs=FINAL["epochs"],
                              l2=FINAL["l2"], lr_decay=FINAL["lr_decay"],
                              patience=FINAL["patience"],
                              verbose=True).fit(Xtr, ytr, Xva, yva)
    t_train = time.perf_counter() - t2 - t_dense
    print(f"[final] val RMSE: {model.best_val_rmse_:.4f}")

    # ------------------------------------------------------------- persist
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    vec.save(config.VECTORIZER_FILE)
    model.save(config.MODEL_FILE)
    meta = {"group": "all", "cols": ALL_COLS, "scaler": scaler,
            "val_rmse": model.best_val_rmse_, "config": FINAL,
            "run": "final-full-data"}
    (config.ARTIFACT_DIR / "fusion_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    ep = np.asarray(model.epoch_times_)
    report = {
        "run": "final-full-data (Plan A, local)",
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "tfidf_features": int(Xtr_s.shape[1]),
        "dense_features": len(ALL_COLS),
        "epochs_run": int(len(ep)),
        "load_merge_s": round(t_load, 1),
        "tfidf_s": round(t_tfidf, 1),
        "dense_prep_s": round(t_dense, 1),
        "train_s": round(t_train, 1),
        "epoch_avg_s": round(float(ep.mean()), 3),
        "epoch_min_s": round(float(ep.min()), 3),
        "epoch_max_s": round(float(ep.max()), 3),
        "offline_total_s": round(time.perf_counter() - t0, 1),
        "val_rmse": model.best_val_rmse_,
    }
    (config.ARTIFACT_DIR / "final_run.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("[final] artifacts saved -- run `python -m src.predict` for the "
          "v5 submission")


if __name__ == "__main__":
    main()
