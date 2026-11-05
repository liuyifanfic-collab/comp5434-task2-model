"""Ablation over C's v2 features: reference + four fusion groups.

Usage (needs data/ + features/ from C):
    python -m src.run_ablation

Rows produced:
    feats-only   : closed-form ridge on the 19 dense columns (no text)
    text         : TF-IDF only (our text baseline)
    text+user    : TF-IDF + user-side history features
    text+prod    : TF-IDF + product/parent-side features
    all          : TF-IDF + all 19 dense features

Eval-protocol compliant: split via C's is_val column (verified identical to
split/val_ids.csv); the dense scaler is fitted on the training rows only.

Learning rate is lowered to 0.3 for fusion runs: standardised dense columns
can push the Hessian's largest eigenvalue above 1, so we keep
lr * lambda_max safely below 2. Saves the best group's artifacts; predict.py
picks them up automatically via fusion_meta.json.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import config
from src.data_loader import TEXT_COL, load_reviews
from src.evaluate import rmse
from src.feature_fusion import (ALL_COLS, PROD_COLS, USER_COLS,
                                AugmentedCSR, apply_scaler, fit_scaler,
                                load_features)
from src.model import RidgeRegressionGD
from src.text_repr import TfidfVectorizer

GROUPS: dict[str, list[str]] = {
    "text": [],
    "text+user": USER_COLS,
    "text+prod": PROD_COLS,
    "all": ALL_COLS,
}


def dense_only_reference(df: pd.DataFrame, train_mask, val_mask) -> float:
    """Closed-form ridge on the 19 standardised dense features (no text).

    Cheap (19 columns -> 20x20 normal equations) and gives the ablation
    table a 'structured features alone' reference row.
    """
    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    Xtr = apply_scaler(df.loc[train_mask], ALL_COLS, scaler)
    Xva = apply_scaler(df.loc[val_mask], ALL_COLS, scaler)
    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=float)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=float)

    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])       # intercept column
    G = A.T @ A + 1.0 * np.eye(A.shape[1])
    G[-1, -1] -= 1.0                                    # don't penalise intercept
    w = np.linalg.solve(G, A.T @ ytr)
    pred = np.hstack([Xva, np.ones((len(Xva), 1))]) @ w
    return rmse(yva, pred)


def main() -> None:
    missing = [str(p) for p in [config.TRAIN_CSV, config.TRAIN_FEATURES]
               if not p.exists()]
    if missing:
        print("[!] missing inputs: " + "; ".join(missing))
        sys.exit(1)

    t0 = time.perf_counter()
    df = load_reviews(config.TRAIN_CSV, is_train=True)          # text + rating
    feats = load_features(config.TRAIN_FEATURES)                # parquet dir
    # votes/purchased already exist in train.csv -- drop C's passthrough
    # copies to avoid merge suffixes (values are identical anyway).
    overlap = [c for c in feats.columns if c in df.columns and c != "id"]
    if overlap:
        feats = feats.drop(columns=overlap)
    df = df.merge(feats, on="id", how="left", validate="one_to_one")
    if df[ALL_COLS].isna().any().any():
        raise ValueError("NaN after merge -- feature file does not cover "
                         "every train id; ask C for a fix.")
    print(f"[i] merged train + features: {len(df)} rows x "
          f"{df.shape[1]} cols ({time.perf_counter() - t0:.0f}s)")

    # C's is_val is verified identical to split/val_ids.csv (2026-11-03).
    val_mask = (df["is_val"] == 1).to_numpy()
    base_train_mask = (df["is_val"] == 0).to_numpy()
    print(f"[i] split via is_val: {base_train_mask.sum()} train / "
          f"{val_mask.sum()} val")

    if config.SAMPLE_FRAC:
        rng = np.random.default_rng(config.SEED)
        tr_pos = np.where(base_train_mask)[0]
        keep = rng.choice(tr_pos, size=int(len(tr_pos) * config.SAMPLE_FRAC),
                          replace=False)
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[keep] = True
        print(f"[i] pilot: train fold sampled to {len(keep)} rows")
    else:
        train_mask = base_train_mask

    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=float)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=float)

    # ---------------------------------------------------------- tf-idf --
    vec = TfidfVectorizer(min_df=config.MIN_DF,
                          max_features=config.MAX_FEATURES,
                          ngram_range=config.NGRAM_RANGE)
    Xtr_s = vec.fit_transform(df.loc[train_mask, TEXT_COL])
    Xva_s = vec.transform(df.loc[val_mask, TEXT_COL])
    print(f"[i] tf-idf: {Xtr_s.shape[1]} features "
          f"({time.perf_counter() - t0:.0f}s)")

    # ------------------------------------------------- reference + groups --
    ref = dense_only_reference(df, train_mask, val_mask)
    print(f"[ablation] feats-only(ridge-closed) val RMSE {ref:.4f}")

    results: list[tuple[str, float]] = [("feats-only", ref)]
    best = None   # (rmse, name, scaler, cols, model)
    for name, cols in GROUPS.items():
        if cols:
            scaler = fit_scaler(df.loc[train_mask], cols)
            Xtr = AugmentedCSR(Xtr_s, apply_scaler(df.loc[train_mask], cols, scaler))
            Xva = AugmentedCSR(Xva_s, apply_scaler(df.loc[val_mask], cols, scaler))
        else:
            scaler = None
            Xtr, Xva = Xtr_s, Xva_s

        model = RidgeRegressionGD(lr=0.3, epochs=config.EPOCHS, l2=config.L2,
                                  lr_decay=0.995, patience=config.PATIENCE,
                                  verbose=False).fit(Xtr, ytr, Xva, yva)
        rmse_val = model.best_val_rmse_
        results.append((name, rmse_val))
        print(f"[ablation] {name:<10} val RMSE {rmse_val:.4f}")
        if best is None or rmse_val < best[0]:
            best = (rmse_val, name, scaler, cols, model)

    print("\n=== ablation summary (val fold, pilot sample) ===")
    for name, r in results:
        print(f"  {name:<10} {r:.4f}")
    print(f"best fusion group: {best[1]} ({best[0]:.4f})")

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    vec.save(config.VECTORIZER_FILE)
    best[4].save(config.MODEL_FILE)
    meta = {"group": best[1], "cols": best[3], "scaler": best[2],
            "val_rmse": best[0], "reference_rows": dict(results)}
    (config.ARTIFACT_DIR / "fusion_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[i] best-group artifacts saved to {config.ARTIFACT_DIR}/ "
          f"(predict.py auto-loads fusion_meta.json)")
    print(f"[perf] offline time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
