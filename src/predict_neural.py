"""Generate submissions/task2_submission.csv from the NEURAL model artifacts.

Mirrors src.predict.py for the neural regressor: loads
artifacts/neural_model.pt + neural_vocab.json + neural_meta.json,
encodes the test corpus, fuses the 19 dense features, predicts, clips
to [1, 5] and writes the Kaggle-format file. Reports total inference
time (efficiency hook).

Usage: python -m src.predict_neural
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import config
from src.data_loader import TEXT_COL, load_reviews
from src.feature_fusion import apply_scaler, load_features
from src.neural_model import NeuralRegressor, _gather
from src.predict import write_submission
from src.train_neural import encode


def main() -> None:
    torch.set_num_threads(1)       # deterministic forward pass
    model_file = config.ARTIFACT_DIR / "neural_model.pt"
    vocab_file = config.ARTIFACT_DIR / "neural_vocab.json"
    meta_file = config.ARTIFACT_DIR / "neural_meta.json"
    missing = [str(p) for p in [model_file, vocab_file, meta_file,
                                config.TEST_CSV, config.TEST_FEATURES]
               if not p.exists()]
    if missing:
        print("[!] missing inputs: " + "; ".join(missing))
        sys.exit(1)

    ckpt = torch.load(model_file, weights_only=True)
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    model = NeuralRegressor(ckpt["vocab_size"], n_dense=ckpt["n_dense"],
                            emb_dim=ckpt["emb_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = load_reviews(config.TEST_CSV, is_train=False)
    tok, off = encode(df[TEXT_COL], vocab)

    feats = load_features(config.TEST_FEATURES)
    overlap = [c for c in feats.columns if c in df.columns and c != "id"]
    if overlap:
        feats = feats.drop(columns=overlap)
    aligned = df[["id", "votes", "purchased"]].merge(
        feats, on="id", how="left", validate="one_to_one")
    dense = apply_scaler(aligned, meta["cols"], meta["scaler"]).astype(np.float32)

    t0 = time.perf_counter()
    preds = []
    with torch.no_grad():
        for s in range(0, len(df), 65536):
            e = min(s + 65536, len(df))
            bt, bo = _gather(tok, off, np.arange(s, e))
            x = torch.from_numpy(bt.astype(np.int64))
            o = torch.from_numpy(bo)
            d = torch.from_numpy(dense[s:e])
            preds.append(model(x, o, d).numpy())
    pred = np.clip(np.concatenate(preds), config.RATING_MIN, config.RATING_MAX)
    infer_time = time.perf_counter() - t0

    config.SUBMISSION_DIR.mkdir(exist_ok=True)
    out = config.SUBMISSION_DIR / "task2_submission.csv"
    write_submission(df["id"].to_numpy(), pred, out)
    print(f"[i] neural fusion prediction: {meta.get('val_rmse'):.4f} val | "
          f"wrote {out} ({len(pred)} rows)")
    print(f"[perf] total inference time for all test records: "
          f"{infer_time:.2f}s")


if __name__ == "__main__":
    main()
