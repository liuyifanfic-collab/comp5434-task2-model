"""Final full-data neural training (Plan A: local execution, no sampling).

Config = tune winner N1: EmbeddingBag(64) + 19 dense features -> Linear(256)
-> ReLU -> Linear(1). Vocabulary rebuilt on the FULL train fold (min_df=3,
cap 100k). Saves artifacts/ + neural_final_run.json (timing metrics).

Usage: python -m src.train_final_neural
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

import config
from src.data_loader import TEXT_COL, load_reviews
from src.feature_fusion import (ALL_COLS, apply_scaler, fit_scaler,
                                load_features)
from src.neural_model import train as train_net
from src.train_neural import build_vocab, encode

EPOCHS = 10          # 10x data per epoch vs pilot -> fewer epochs needed
PATIENCE = 3


def main() -> None:
    # Bit-reproducibility: multi-threaded CPU reductions are NOT
    # deterministic across machines (float summation order varies with
    # thread count/scheduling). Pinning to one thread makes the training
    # trajectory identical on any machine with the pinned library versions.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                       # already initialised -- harmless here
    t0 = time.perf_counter()
    df = load_reviews(config.TRAIN_CSV, is_train=True)
    feats = load_features(config.TRAIN_FEATURES)
    overlap = [c for c in feats.columns if c in df.columns and c != "id"]
    if overlap:
        feats = feats.drop(columns=overlap)
    df = df.merge(feats, on="id", how="left", validate="one_to_one")
    val_mask = (df["is_val"] == 1).to_numpy()
    train_mask = (df["is_val"] == 0).to_numpy()
    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=np.float64)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=np.float64)
    print(f"[final] FULL train fold {train_mask.sum()} | val {val_mask.sum()} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    t1 = time.perf_counter()
    vocab = build_vocab(df.loc[train_mask, TEXT_COL],
                        config.MIN_DF, config.NEURAL_VOCAB_MAX)
    t_vocab = time.perf_counter() - t1
    t2 = time.perf_counter()
    tok_tr, off_tr = encode(df.loc[train_mask, TEXT_COL], vocab)
    tok_va, off_va = encode(df.loc[val_mask, TEXT_COL], vocab)
    t_enc = time.perf_counter() - t2
    print(f"[final] vocab {len(vocab)} ({t_vocab:.0f}s) | tokens "
          f"{len(tok_tr)}/{len(tok_va)} ({t_enc:.0f}s)", flush=True)

    t3 = time.perf_counter()
    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    D_tr = apply_scaler(df.loc[train_mask], ALL_COLS, scaler).astype(np.float32)
    D_va = apply_scaler(df.loc[val_mask], ALL_COLS, scaler).astype(np.float32)
    t_dense = time.perf_counter() - t3

    model, best_rmse, epoch_times = train_net(
        tok_tr, off_tr, D_tr, ytr, tok_va, off_va, D_va, yva,
        len(vocab), n_dense=len(ALL_COLS), emb_dim=config.NEURAL_EMB_DIM,
        hidden=config.NEURAL_HIDDEN, lr=config.NEURAL_LR,
        batch=config.NEURAL_BATCH, epochs=EPOCHS, patience=PATIENCE,
        seed=config.SEED)
    t_train = time.perf_counter() - t3 - t_dense
    print(f"\n[final] val RMSE: {best_rmse:.4f} (pilot was 1.1914)")

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab),
                "n_dense": len(ALL_COLS), "emb_dim": config.NEURAL_EMB_DIM,
                "hidden": config.NEURAL_HIDDEN},
               config.ARTIFACT_DIR / "neural_model.pt")
    (config.ARTIFACT_DIR / "neural_vocab.json").write_text(
        json.dumps(vocab), encoding="utf-8")
    (config.ARTIFACT_DIR / "neural_meta.json").write_text(
        json.dumps({"val_rmse": best_rmse, "scaler": scaler, "cols": ALL_COLS,
                    "config": {"emb_dim": config.NEURAL_EMB_DIM,
                               "hidden": config.NEURAL_HIDDEN,
                               "lr": config.NEURAL_LR, "epochs": EPOCHS,
                               "batch": config.NEURAL_BATCH,
                               "seed": config.SEED}},
                   ensure_ascii=False), encoding="utf-8")

    ep = np.asarray(epoch_times)
    report = {"run": "final-full-data-neural (Plan A, local)",
              "train_rows": int(train_mask.sum()),
              "val_rows": int(val_mask.sum()),
              "vocab_size": len(vocab),
              "dense_features": len(ALL_COLS),
              "epochs_run": int(len(ep)),
              "load_merge_s": round(time.perf_counter() - t0 - t_vocab
                                    - t_enc - t_dense - t_train, 1),
              "vocab_s": round(t_vocab, 1), "encode_s": round(t_enc, 1),
              "dense_prep_s": round(t_dense, 1), "train_s": round(t_train, 1),
              "epoch_avg_s": round(float(ep.mean()), 2),
              "offline_total_s": round(time.perf_counter() - t0, 1),
              "val_rmse": best_rmse}
    (config.ARTIFACT_DIR / "neural_final_run.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("[final] artifacts saved -- run `python -m src.predict_neural` "
          "for the v6 submission")


if __name__ == "__main__":
    main()
