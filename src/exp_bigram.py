"""Pilot experiment: bigram vocabulary for the neural model.

Same protocol as train_neural (10% pilot, canonical split, fusion variant,
N1 architecture). Only change: vocabulary includes uni+bigrams, cap 200k.

Reference: unigram pilot val RMSE 1.1914. If bigram wins, re-run the full
training with the bigram vocabulary.

Usage: python -m src.exp_bigram
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

NGRAM = (1, 2)
VOCAB_MAX = 200000
REF_UNIGRAM = 1.1914


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

    vocab = build_vocab(df.loc[train_mask, TEXT_COL], config.MIN_DF,
                        VOCAB_MAX, NGRAM)
    print(f"[i] vocab {len(vocab)} ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    tok_tr, off_tr = encode(df.loc[train_mask, TEXT_COL], vocab, NGRAM)
    tok_va, off_va = encode(df.loc[val_mask, TEXT_COL], vocab, NGRAM)
    print(f"[i] tokens {len(tok_tr)}/{len(tok_va)} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    D_tr = apply_scaler(df.loc[train_mask], ALL_COLS, scaler).astype(np.float32)
    D_va = apply_scaler(df.loc[val_mask], ALL_COLS, scaler).astype(np.float32)

    model, r, epoch_times = train_net(
        tok_tr, off_tr, D_tr, ytr, tok_va, off_va, D_va, yva,
        len(vocab), n_dense=len(ALL_COLS), emb_dim=config.NEURAL_EMB_DIM,
        hidden=config.NEURAL_HIDDEN, lr=config.NEURAL_LR,
        batch=config.NEURAL_BATCH, epochs=config.NEURAL_EPOCHS,
        patience=config.NEURAL_PATIENCE, seed=config.SEED)

    print(f"\n[bigram pilot] val RMSE {r:.4f} vs unigram {REF_UNIGRAM:.4f} "
          f"({'WIN' if r < REF_UNIGRAM else 'lose'})")
    print(f"[perf] total {time.perf_counter() - t0:.0f}s")

    if r < REF_UNIGRAM:
        config.ARTIFACT_DIR.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab),
                    "n_dense": len(ALL_COLS), "emb_dim": config.NEURAL_EMB_DIM,
                    "hidden": config.NEURAL_HIDDEN},
                   config.ARTIFACT_DIR / "neural_model_bigram.pt")
        (config.ARTIFACT_DIR / "neural_vocab_bigram.json").write_text(
            json.dumps(vocab), encoding="utf-8")
        print("[i] bigram artifacts saved (kept separate from the "
              "production unigram artifacts)")


if __name__ == "__main__":
    main()
