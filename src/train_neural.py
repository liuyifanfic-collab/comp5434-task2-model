"""Pilot comparison: self-implemented neural text model vs the linear ridge.

Same canonical split, same 10% pilot sample as every other experiment, so
the numbers are directly comparable:

    text-neural    (embedding bag only)              vs linear text  1.4403
    fusion-neural  (embedding bag + 19 dense feats)  vs linear fusion 1.4033

If fusion-neural beats the linear reference, its weights/vocab/meta are
saved to artifacts/ for later integration into predict.py.

Usage: python -m src.train_neural
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

import config
from src.data_loader import TEXT_COL, load_reviews
from src.feature_fusion import (ALL_COLS, apply_scaler, fit_scaler,
                                load_features)
from src.neural_model import train as train_net
from src.text_repr import tokenize

LINEAR_TEXT_RMSE = 1.4403     # sweep A-ep300 (lr 0.5, 300 ep)
LINEAR_FUSION_RMSE = 1.4033   # tune_fusion F1 (lr 0.5, 500 ep)


def build_vocab(texts, min_df: int, max_features: int,
                ngram_range: tuple = (1, 1)) -> dict:
    df: dict[str, int] = {}
    for t in texts:
        for tok in set(tokenize(t, ngram_range)):
            df[tok] = df.get(tok, 0) + 1
    cand = [t for t, c in df.items() if c >= min_df]
    cand.sort(key=lambda t: (-df[t], t))
    vocab = {"<unk>": 0}                       # id 0: OOV / empty fallback
    for t in cand[:max_features]:
        vocab[t] = len(vocab)
    return vocab


def encode(texts, vocab: dict, ngram_range: tuple = (1, 1)):
    """Corpus -> flat int32 token ids + int64 offsets (unk fallback)."""
    off = np.zeros(len(texts) + 1, dtype=np.int64)
    buf = []
    total = 0
    for i, t in enumerate(texts):
        ids = [vocab.get(x, 0) for x in tokenize(t, ngram_range)] or [0]
        buf.append(np.asarray(ids, dtype=np.int32))
        total += len(ids)
        off[i + 1] = total
    return np.concatenate(buf), off


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

    t1 = time.perf_counter()
    texts_tr = df.loc[train_mask, TEXT_COL]
    texts_va = df.loc[val_mask, TEXT_COL]
    vocab = build_vocab(texts_tr, config.MIN_DF, config.NEURAL_VOCAB_MAX)
    print(f"[i] vocab {len(vocab)} ({time.perf_counter() - t1:.0f}s)",
          flush=True)
    tok_tr, off_tr = encode(texts_tr, vocab)
    tok_va, off_va = encode(texts_va, vocab)
    print(f"[i] encoded: train {len(off_tr) - 1} docs / {len(tok_tr)} tokens"
          f" | val {len(off_va) - 1} docs / {len(tok_va)} tokens "
          f"({time.perf_counter() - t1:.0f}s)", flush=True)

    scaler = fit_scaler(df.loc[train_mask], ALL_COLS)
    D_tr = apply_scaler(df.loc[train_mask], ALL_COLS, scaler).astype(np.float32)
    D_va = apply_scaler(df.loc[val_mask], ALL_COLS, scaler).astype(np.float32)

    common = dict(vocab_size=len(vocab), emb_dim=config.NEURAL_EMB_DIM,
                  hidden=config.NEURAL_HIDDEN, lr=config.NEURAL_LR,
                  batch=config.NEURAL_BATCH, epochs=config.NEURAL_EPOCHS,
                  patience=config.NEURAL_PATIENCE, seed=config.SEED)

    print("\n=== variant 1: text-neural (embedding bag only) ===", flush=True)
    _, r_text, _ = train_net(tok_tr, off_tr, None, ytr,
                             tok_va, off_va, None, yva,
                             n_dense=0, **common)

    print("\n=== variant 2: fusion-neural (+19 dense features) ===", flush=True)
    model_f, r_fuse, ep_times = train_net(tok_tr, off_tr, D_tr, ytr,
                                          tok_va, off_va, D_va, yva,
                                          n_dense=len(ALL_COLS), **common)

    print("\n=== pilot comparison (same split / sample) ===")
    print(f"  text :  neural {r_text:.4f}  vs linear {LINEAR_TEXT_RMSE:.4f}"
          f"  ({'WIN' if r_text < LINEAR_TEXT_RMSE else 'lose'})")
    print(f"  fusion: neural {r_fuse:.4f}  vs linear {LINEAR_FUSION_RMSE:.4f}"
          f"  ({'WIN' if r_fuse < LINEAR_FUSION_RMSE else 'lose'})")
    print(f"[perf] total {time.perf_counter() - t0:.0f}s | "
          f"avg {sum(ep_times) / len(ep_times):.1f}s/epoch")

    if r_fuse < LINEAR_FUSION_RMSE:
        config.ARTIFACT_DIR.mkdir(exist_ok=True)
        torch.save({"state_dict": model_f.state_dict(),
                    "vocab_size": len(vocab),
                    "n_dense": len(ALL_COLS),
                    "emb_dim": config.NEURAL_EMB_DIM,
                    "hidden": config.NEURAL_HIDDEN},
                   config.ARTIFACT_DIR / "neural_model.pt")
        (config.ARTIFACT_DIR / "neural_vocab.json").write_text(
            json.dumps(vocab), encoding="utf-8")
        (config.ARTIFACT_DIR / "neural_meta.json").write_text(
            json.dumps({"val_rmse": r_fuse, "scaler": scaler,
                        "cols": ALL_COLS,
                        "config": {k: v for k, v in common.items()
                                   if k != "vocab_size"}},
                       ensure_ascii=False), encoding="utf-8")
        print("[i] neural artifacts saved to artifacts/ (integration into "
              "predict.py pending decision)")


if __name__ == "__main__":
    main()
