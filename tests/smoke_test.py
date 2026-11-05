"""End-to-end smoke test on SYNTHETIC data (no Kaggle data needed).

Validates the full B-side pipeline before the real dataset arrives:
    loader schema -> stratified split -> self-implemented TF-IDF ->
    self-implemented ridge GD -> RMSE beats the mean baseline ->
    submission writer produces the Kaggle format.

Run from the task2_model folder:
    python -m tests.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import TEXT_COL, load_reviews          # noqa: E402
from src.evaluate import rmse, stratified_split             # noqa: E402
from src.model import RidgeRegressionGD                     # noqa: E402
from src.predict import write_submission                    # noqa: E402
from src.text_repr import TfidfVectorizer                   # noqa: E402

POS = ["excellent", "amazing", "love", "perfect", "great"]
NEG = ["terrible", "awful", "hate", "broken", "worst"]
FILLER = ["the", "product", "item", "this", "day", "arrived", "box", "use"]

SYN_DIR = Path(__file__).resolve().parent / "_synthetic"


def make_row(rng: np.random.Generator) -> dict:
    """A synthetic review whose rating is driven by sentiment keywords."""
    n_words = int(rng.integers(4, 12))
    words = [FILLER[int(rng.integers(0, len(FILLER)))] for _ in range(n_words)]
    n_pos = int(rng.integers(0, 3))
    n_neg = int(rng.integers(0, 3))
    words += [POS[int(rng.integers(0, len(POS)))] for _ in range(n_pos)]
    words += [NEG[int(rng.integers(0, len(NEG)))] for _ in range(n_neg)]
    rng.shuffle(words)

    score = 3.0 + 0.9 * n_pos - 0.9 * n_neg + rng.normal(0.0, 0.25)
    rating = int(np.clip(round(score), 1, 5))

    prod = f"p{int(rng.integers(0, 40)):03d}"
    return {
        "id": 0,                                   # filled by caller
        "title": " ".join(words[:2]),
        "comment": " ".join(words),
        "user_id": f"u{int(rng.integers(0, 60)):03d}",
        "prod_id": prod,
        "parent_prod_id": f"pp{prod[1:]}",
        "time": int(rng.integers(1_600_000_000, 1_730_000_000)),
        "votes": int(rng.integers(0, 20)),
        "purchased": int(rng.integers(0, 2)),
        "rating": rating,
    }


def build_csv(path: Path, n_rows: int, start_id: int, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        r = make_row(rng)
        r["id"] = start_id + i
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    rng = np.random.default_rng(0)
    train_csv = SYN_DIR / "train.csv"
    test_csv = SYN_DIR / "test.csv"
    build_csv(train_csv, 800, 0, rng)
    build_csv(test_csv, 200, 10_000, rng)
    print(f"[i] synthetic data written to {SYN_DIR}")

    # --------------------------------------------- real pipeline, step by step
    train_df = load_reviews(train_csv, is_train=True)
    test_df = load_reviews(test_csv, is_train=False)

    tr_mask, va_mask = stratified_split(train_df["rating"].to_numpy(),
                                        0.1, seed=5434)
    ytr = train_df.loc[tr_mask, "rating"].to_numpy(dtype=float)
    yva = train_df.loc[va_mask, "rating"].to_numpy(dtype=float)

    vec = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(train_df.loc[tr_mask, TEXT_COL])
    Xva = vec.transform(train_df.loc[va_mask, TEXT_COL])
    print(f"[i] tf-idf fitted: {Xtr.shape[1]} features")

    baseline = rmse(yva, np.full(len(yva), ytr.mean()))
    model = RidgeRegressionGD(lr=0.5, epochs=120, l2=1e-4,
                              patience=20).fit(Xtr, ytr, Xva, yva)
    model_rmse = model.best_val_rmse_

    print(f"[i] baseline RMSE {baseline:.4f} | model RMSE {model_rmse:.4f}")
    assert model_rmse < baseline - 0.1, (
        f"model ({model_rmse:.4f}) should beat baseline ({baseline:.4f}) "
        "by a clear margin on synthetic data")

    # ------------------------------------------------------ submission format
    Xte = vec.transform(test_df[TEXT_COL])
    pred = np.clip(model.predict(Xte), 1, 5)
    sub_path = SYN_DIR / "submission.csv"
    write_submission(test_df["id"].to_numpy(), pred, sub_path)

    sub = pd.read_csv(sub_path)
    assert list(sub.columns) == ["id", "rating"], "wrong submission columns"
    assert len(sub) == len(test_df), "submission row count mismatch"
    assert sub["rating"].between(1, 5).all(), "predictions outside [1, 5]"
    assert sub["id"].tolist() == test_df["id"].tolist(), "id order changed"

    print("[ok] submission format valid (id, rating; 1 <= rating <= 5)")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
