"""Generate submissions/task2_submission.csv from saved artifacts.

Usage (run from the task2_model folder, after src.train or src.run_ablation):
    python -m src.predict

If artifacts/fusion_meta.json exists (written by run_ablation), the dense
feature block is rebuilt from features/test_features.csv and fused with the
TF-IDF block automatically. Otherwise falls back to the text-only model.

Reports the total inference time over all test records (an efficiency
metric the report requires).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import config
from src.data_loader import TEXT_COL, load_reviews
from src.feature_fusion import AugmentedCSR, apply_scaler, load_features
from src.model import RidgeRegressionGD
from src.text_repr import TfidfVectorizer

FUSION_META = config.ARTIFACT_DIR / "fusion_meta.json"


def write_submission(ids, ratings, path) -> None:
    """Write a Kaggle-format submission: columns (id, rating)."""
    out = pd.DataFrame({"id": ids, "rating": ratings})
    out.to_csv(path, index=False)


def main() -> None:
    if not config.MODEL_FILE.exists() or not config.VECTORIZER_FILE.exists():
        print("[!] artifacts not found -- run `python -m src.train` first.")
        sys.exit(1)
    if not config.TEST_CSV.exists():
        print(f"[!] {config.TEST_CSV} not found.")
        print("    Download test.csv from the Task 2 Kaggle page and put it "
              "in data/.")
        sys.exit(1)

    vec = TfidfVectorizer.load(config.VECTORIZER_FILE)
    model = RidgeRegressionGD.load(config.MODEL_FILE)

    df = load_reviews(config.TEST_CSV, is_train=False)
    Xte = vec.transform(df[TEXT_COL])

    # ------------------------------------------- optional dense fusion --
    if FUSION_META.exists():
        meta = json.loads(FUSION_META.read_text(encoding="utf-8"))
        if meta.get("cols"):
            if not config.TEST_FEATURES.exists():
                print(f"[!] {config.TEST_FEATURES} not found -- the saved "
                      "model needs C's test features.")
                sys.exit(1)
            feats = load_features(config.TEST_FEATURES)
            overlap = [c for c in feats.columns
                       if c in df.columns and c != "id"]
            if overlap:
                feats = feats.drop(columns=overlap)
            # carry votes/purchased from the original test csv (identical
            # to C's passthrough, avoids merge suffixes)
            aligned = df[["id", "votes", "purchased"]].merge(
                feats, on="id", how="left", validate="one_to_one")
            dense = apply_scaler(aligned, meta["cols"], meta["scaler"])
            Xte = AugmentedCSR(Xte, dense)
            print(f"[i] fusion active: group={meta['group']} "
                  f"(+{len(meta['cols'])} dense cols)")

    t0 = time.perf_counter()
    pred = model.predict(Xte)
    infer_time = time.perf_counter() - t0

    # Ratings live in [1, 5]; continuous predictions are kept (not rounded)
    # because RMSE is minimised by the conditional mean.
    pred = np.clip(pred, config.RATING_MIN, config.RATING_MAX)

    config.SUBMISSION_DIR.mkdir(exist_ok=True)
    out_path = config.SUBMISSION_DIR / "task2_submission.csv"
    write_submission(df["id"].to_numpy(), pred, out_path)

    print(f"[i] wrote {out_path} ({len(pred)} rows)")
    print(f"[perf] total inference time for all test records: "
          f"{infer_time:.2f}s")


if __name__ == "__main__":
    main()
