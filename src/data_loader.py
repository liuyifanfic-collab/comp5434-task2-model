"""CSV loading + schema validation for Dataset 2 (review rating prediction)."""
from __future__ import annotations

import pandas as pd

BASE_COLS = ["id", "title", "comment", "user_id", "prod_id",
             "parent_prod_id", "time", "votes", "purchased"]
LABEL_COL = "rating"
TEXT_COL = "__text__"          # derived column: title + " " + comment


def load_reviews(path, is_train: bool) -> pd.DataFrame:
    """Read a Dataset 2 csv, validate the schema, derive the text column.

    Missing titles/comments become empty strings so downstream TF-IDF never
    sees NaN.
    """
    df = pd.read_csv(path)
    required = BASE_COLS + ([LABEL_COL] if is_train else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    title = df["title"].fillna("").astype(str)
    comment = df["comment"].fillna("").astype(str)
    df[TEXT_COL] = (title + " " + comment).str.strip()
    return df
