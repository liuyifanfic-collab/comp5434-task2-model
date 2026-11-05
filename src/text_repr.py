"""Self-implemented TF-IDF vectoriser (no sklearn).

The project forbids black-box model calls, and the text representation is
part of our own technical contribution. Standard smoothed formulation:

    tf(t, d)  : sub-linear    1 + ln(count(t, d))
    idf(t)    : smoothed      ln((1 + N) / (1 + df(t))) + 1
    weight    : tf * idf, then every document vector is L2-normalised.

Output is a CSRMatrix (see sparse.py) consumed by our own gradient-descent
kernels in model.py.
"""
from __future__ import annotations

import re

import numpy as np

from .sparse import CSRMatrix

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Lower-case word tokens, plus joined n-grams up to the given order."""
    tokens = _TOKEN_RE.findall(str(text).lower())
    lo, hi = ngram_range
    out = list(tokens)
    for n in range(max(2, lo), hi + 1):
        out.extend(
            "_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
        )
    return out


class TfidfVectorizer:
    """Fit on the TRAIN FOLD ONLY; transform never sees fitting data."""

    def __init__(self, min_df: int = 3, max_features: int = 30000,
                 ngram_range: tuple[int, int] = (1, 2)):
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    # ------------------------------------------------------------- fitting --
    def fit(self, corpus) -> "TfidfVectorizer":
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(tokenize(doc, self.ngram_range)):
                df[term] = df.get(term, 0) + 1

        # Prune rare terms, then keep the top-K by document frequency.
        # Tie-break on the term string itself -> deterministic vocabulary.
        candidates = [t for t, c in df.items() if c >= self.min_df]
        candidates.sort(key=lambda t: (-df[t], t))
        kept = sorted(candidates[: self.max_features])

        self.vocab = {t: i for i, t in enumerate(kept)}
        n_docs = len(corpus)
        self.idf = np.array(
            [np.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0 for t in kept],
            dtype=np.float64,
        )
        return self

    # -------------------------------------------------------- transform --
    def transform(self, corpus) -> CSRMatrix:
        """Vectorise a corpus into CSR.

        Memory-safe at full scale: per-doc arrays are concatenated in
        50k-doc chunks instead of one giant python list, and the row
        pointer array is pre-allocated as int64.
        """
        if self.idf is None:
            raise RuntimeError("call fit() before transform()")
        n_docs = len(corpus)
        indptr = np.zeros(n_docs + 1, dtype=np.int64)
        buf_d: list = []
        buf_i: list = []
        parts_d: list = []
        parts_i: list = []
        total = 0
        for i, doc in enumerate(corpus):
            counts: dict[int, int] = {}
            for tok in tokenize(doc, self.ngram_range):
                j = self.vocab.get(tok)
                if j is not None:                    # skip out-of-vocabulary
                    counts[j] = counts.get(j, 0) + 1
            if counts:
                js = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
                cs = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
                w = (1.0 + np.log(cs)) * self.idf[js]     # sub-linear tf * idf
                norm = np.sqrt(float(np.dot(w, w)))
                if norm > 0.0:
                    w = w / norm                          # L2-normalise the row
                buf_i.append(js)
                buf_d.append(w)
                total += len(js)
            indptr[i + 1] = total
            if (i + 1) % 50000 == 0 and buf_d:
                parts_d.append(np.concatenate(buf_d))
                parts_i.append(np.concatenate(buf_i))
                buf_d, buf_i = [], []
        if buf_d:
            parts_d.append(np.concatenate(buf_d))
            parts_i.append(np.concatenate(buf_i))
        data = (np.concatenate(parts_d) if parts_d
                else np.zeros(0, dtype=np.float64))
        indices = (np.concatenate(parts_i) if parts_i
                   else np.zeros(0, dtype=np.int32))
        return CSRMatrix(data, indices, indptr, len(self.vocab))

    def fit_transform(self, corpus) -> CSRMatrix:
        return self.fit(corpus).transform(corpus)

    # ---------------------------------------------------- persistence --
    def save(self, path) -> None:
        terms = sorted(self.vocab, key=self.vocab.get)
        np.savez(path, idf=self.idf, terms=np.array(terms))

    @classmethod
    def load(cls, path) -> "TfidfVectorizer":
        z = np.load(path, allow_pickle=False)
        vec = cls()
        terms = z["terms"].tolist()
        vec.vocab = {t: i for i, t in enumerate(terms)}
        vec.idf = z["idf"].astype(np.float64)
        return vec
