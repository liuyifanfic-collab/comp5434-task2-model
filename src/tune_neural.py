"""Quick tuning sweep for the neural text model (cheap after the one load).

Each config trains the FUSION variant (+19 dense features) on the same
pilot sample / canonical split as every other experiment:

    N1: emb 64,  hidden 256, lr 1e-3, 20 ep   (pilot used 8 ep)
    N2: emb 128, hidden 512, lr 1e-3, 20 ep
    N3: emb 128, hidden 512, lr 2e-3, 20 ep

Saves the overall best model/vocab/meta to artifacts/ (same file names as
train_neural) so predict_neural.py picks it up automatically.

Usage: python -m src.tune_neural
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
from src.train_neural import LINEAR_FUSION_RMSE, build_vocab, encode

CONFIGS = [
    {"name": "N1-e64-h256", "emb_dim": 64, "hidden": 256, "lr": 1e-3,
     "epochs": 20},
    {"name": "N2-e128-h512", "emb_dim": 128, "hidden": 512, "lr": 1e-3,
     "epochs": 20},
    {"name": "N3-e128-h512-lr2", "emb_dim": 128, "hidden": 512, "lr": 2e-3,
     "epochs": 20},
]


def main() -> None:
    t0 = time.perf_counter()
    df = load_reviews(config.TRAIN_CSV, is_train=True)
    feats = load_features(config.TRAIN_FEATURES)
    overlap = [c for c in feats.columns if c in df.columns and c != "id"]
    if overlap:
        feats = feats.drop(columns=overlap)
    df = df.merge(feats, on="id", how="left", validate="one_to_one")

    val_mask = (df["is_val"] == 1).to_numpy()
    base = (df["is_val"] == 0).to_numpy()
    rng = np.random.default_rng(config.SEED)
    tr_pos = np.where(base)[0]
    keep = rng.choice(tr_pos, size=int(len(tr_pos) * config.SAMPLE_FRAC),
                      replace=False)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[keep] = True
    ytr = df.loc[train_mask, "rating"].to_numpy(dtype=np.float64)
    yva = df.loc[val_mask, "rating"].to_numpy(dtype=np.float64)
    print(f"[i] pilot train {train_mask.sum()} | val {val_mask.sum()} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    vocab = build_vocab(df.loc[train_mask, TEXT_COL],
                        config.MIN_DF, config.NEURAL_VOCAB_MAX)
    tok_tr, off_tr = encode(df.loc[train_mask, TEXT_COL], vocab)
    tok_va, off_va = encode(df.loc[val_mask, TEXT_COL], vocab)
    print(f"[i] vocab {len(vocab)} | tokens {len(tok_tr)}/{len(tok_va)} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    D_tr = apply_scaler(df.loc[train_mask], ALL_COLS, scaler).astype(np.float32)
    D_va = apply_scaler(df.loc[val_mask], ALL_COLS, scaler).astype(np.float32)

    best = None   # (rmse, cfg, model)
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===", flush=True)
        model, r, _ = train_net(
            tok_tr, off_tr, D_tr, ytr, tok_va, off_va, D_va, yva,
            len(vocab), n_dense=len(ALL_COLS), emb_dim=cfg["emb_dim"],
            hidden=cfg["hidden"], lr=cfg["lr"], batch=config.NEURAL_BATCH,
            epochs=cfg["epochs"], patience=4, seed=config.SEED)
        print(f"[result] {cfg['name']} val RMSE {r:.4f}", flush=True)
        if best is None or r < best[0]:
            best = (r, cfg, model)

    print(f"\n=== sweep summary ===")
    print(f"best: {best[1]['name']} val RMSE {best[0]:.4f} "
          f"(pilot run was 1.1914, linear ref {LINEAR_FUSION_RMSE})")

    if best[0] < 1.1914:            # keep artifacts only if we improved
        config.ARTIFACT_DIR.mkdir(exist_ok=True)
        torch.save({"state_dict": best[2].state_dict(),
                    "vocab_size": len(vocab), "n_dense": len(ALL_COLS),
                    "emb_dim": best[1]["emb_dim"],
                    "hidden": best[1]["hidden"]},
                   config.ARTIFACT_DIR / "neural_model.pt")
        (config.ARTIFACT_DIR / "neural_vocab.json").write_text(
            json.dumps(vocab), encoding="utf-8")
        (config.ARTIFACT_DIR / "neural_meta.json").write_text(
            json.dumps({"val_rmse": best[0], "scaler": scaler,
                        "cols": ALL_COLS,
                        "config": {"emb_dim": best[1]["emb_dim"],
                                   "hidden": best[1]["hidden"],
                                   "lr": best[1]["lr"],
                                   "epochs": best[1]["epochs"],
                                   "batch": config.NEURAL_BATCH,
                                   "seed": config.SEED}},
                       ensure_ascii=False), encoding="utf-8")
        print("[i] artifacts updated -- run `python -m src.predict_neural` "
              "for the v5 submission")
    else:
        print("[i] sweep did not beat the pilot run; keeping pilot artifacts")
    print(f"[perf] total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
