"""Self-implemented neural text regressor (fastText-style embedding bag + MLP).

PyTorch is used ONLY as a tensor / autograd engine. The architecture, the
loss, the batching and the training loop are all written from scratch -- no
ready-made estimator is called, keeping the project's no-blackbox rule
(PyTorch here plays the same role numpy plays for the ridge model).

Architecture:
    mean(token embeddings)  [+ 19 standardised dense features]
        -> Linear(emb + dense, hidden) -> ReLU
        -> Linear(hidden, 1)
Trained with MSE + Adam, mini-batches, early stopping on validation RMSE,
per-epoch wall time recorded (same efficiency hooks as the ridge model).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .evaluate import rmse


class NeuralRegressor(nn.Module):
    def __init__(self, vocab_size: int, n_dense: int = 0, emb_dim: int = 64,
                 hidden: int = 256):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.fc1 = nn.Linear(emb_dim + n_dense, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.n_dense = n_dense

    def forward(self, tokens, offsets, dense=None):
        x = self.emb(tokens, offsets)                  # (B, emb_dim)
        if dense is not None:
            x = torch.cat([x, dense], dim=1)
        return self.fc2(torch.relu(self.fc1(x))).squeeze(1)


def _gather(tok: np.ndarray, off: np.ndarray, idx: np.ndarray):
    """Slice a flat token-id batch for arbitrary document indices.

    Returns (tokens, bag_start_offsets) where len(offsets) == len(idx):
    EmbeddingBag opens one bag per offset and extends the last bag to the
    end of the token array, so the trailing end-offset must NOT be passed.
    """
    starts = off[idx].astype(np.int64)
    counts = (off[idx + 1] - off[idx]).astype(np.int64)
    offsets_b = np.zeros(len(idx) + 1, dtype=np.int64)
    offsets_b[1:] = np.cumsum(counts)
    total = int(counts.sum())
    # Every doc has at least the <unk> token, so total >= len(idx) > 0.
    r = np.arange(total)
    within = r - np.repeat(np.cumsum(counts) - counts, counts)
    gather = np.repeat(starts, counts) + within
    return tok[gather], offsets_b[:-1]


def evaluate(model, tok, off, dense, y, chunk=65536) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(y), chunk):
            e = min(s + chunk, len(y))
            bt, bo = _gather(tok, off, np.arange(s, e))
            x = torch.from_numpy(bt.astype(np.int64))
            o = torch.from_numpy(bo)
            d = (torch.from_numpy(dense[s:e]).float()
                 if dense is not None else None)
            preds.append(model(x, o, d).numpy())
    return rmse(y, np.concatenate(preds))


def train(tok, off, dense, y, tok_va, off_va, dense_va, y_va,
          vocab_size, *, n_dense, emb_dim, hidden, lr, batch, epochs,
          patience, seed, verbose=True):
    """Hand-written training loop: Adam + MSE + early stopping on val RMSE."""
    torch.manual_seed(seed)
    model = NeuralRegressor(vocab_size, n_dense=n_dense, emb_dim=emb_dim,
                            hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    n = len(y)
    y_t = torch.from_numpy(y.astype(np.float32))

    best_rmse, best_state, bad = np.inf, None, 0
    epoch_times: list[float] = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            bt, bo = _gather(tok, off, idx)
            x = torch.from_numpy(bt.astype(np.int64))
            o = torch.from_numpy(bo)
            d = (torch.from_numpy(dense[idx]).float()
                 if dense is not None else None)
            pred = model(x, o, d)
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        epoch_times.append(time.perf_counter() - t0)
        val_rmse = evaluate(model, tok_va, off_va, dense_va, y_va)
        if verbose:
            print(f"epoch {epoch + 1:>2d} | val RMSE {val_rmse:.4f} | "
                  f"{epoch_times[-1]:.1f}s", flush=True)
        if val_rmse < best_rmse - 1e-5:
            best_rmse, bad = val_rmse, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse, epoch_times
