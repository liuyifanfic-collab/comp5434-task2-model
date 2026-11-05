"""Self-implemented ridge regression trained by full-batch gradient descent.

Objective:      L(w) = mean((X w - yc)^2) + l2 * ||w||^2
with yc = y - mean(y) (the bias is absorbed by the mean, so w stays sparse-
friendly).

Gradient:       X^T (X w - yc) / n + 2 * l2 * w

Every row of X is L2-normalised (TF-IDF contract), so the Hessian of the
data term is bounded by the identity -> a learning rate <= 1.0 is
unconditionally stable; we start at 0.5 and decay.

Why this matters for the team: the gradient above is computed with exactly
two sparse map-reduce style reductions (CSRMatrix.dot / CSRMatrix.T_dot).
Member C parallelises the SAME formula on Spark RDDs, so local and
distributed training stay mathematically identical and the reported
efficiency metrics transfer.
"""
from __future__ import annotations

import time

import numpy as np

from .evaluate import rmse
from .sparse import CSRMatrix


class RidgeRegressionGD:
    def __init__(self, lr: float = 0.5, epochs: int = 80, l2: float = 1e-4,
                 lr_decay: float = 0.99, patience: int = 8,
                 verbose: bool = True):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.lr_decay = lr_decay
        self.patience = patience
        self.verbose = verbose

        self.w_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.best_val_rmse_: float | None = None
        self.history_: list[dict] = []
        self.epoch_times_: list[float] = []   # efficiency hook: sec per epoch

    # ------------------------------------------------------------ training --
    def fit(self, X: CSRMatrix, y, Xval: CSRMatrix | None = None,
            yval=None) -> "RidgeRegressionGD":
        n, d = X.shape
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_

        w = np.zeros(d, dtype=np.float64)
        lr = self.lr
        best_val = np.inf
        best_w = w.copy()
        bad_epochs = 0
        self.history_ = []
        self.epoch_times_ = []

        for epoch in range(self.epochs):
            t0 = time.perf_counter()
            resid = X.dot(w) - yc                          # (n,)
            grad = X.T_dot(resid) / n + 2.0 * self.l2 * w  # (d,)
            w = w - lr * grad
            lr *= self.lr_decay
            self.epoch_times_.append(time.perf_counter() - t0)

            entry = {
                "epoch": epoch + 1,
                "train_rmse": float(np.sqrt(np.mean(resid ** 2))),
                "sec": round(self.epoch_times_[-1], 4),
            }
            if Xval is not None and yval is not None:
                entry["val_rmse"] = rmse(yval, self._predict(w, Xval))
                if entry["val_rmse"] < best_val - 1e-6:
                    best_val, best_w, bad_epochs = entry["val_rmse"], w.copy(), 0
                else:
                    bad_epochs += 1
            self.history_.append(entry)

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                msg = (f"epoch {entry['epoch']:>3d} | "
                       f"train {entry['train_rmse']:.4f}")
                if "val_rmse" in entry:
                    msg += f" | val {entry['val_rmse']:.4f}"
                print(msg + f" | {entry['sec']}s")

            if Xval is not None and bad_epochs >= self.patience:
                if self.verbose:
                    print(f"early stop at epoch {epoch + 1} "
                          f"(best val RMSE {best_val:.4f})")
                break

        self.w_ = best_w if Xval is not None else w
        self.best_val_rmse_ = best_val if Xval is not None else None
        return self

    # ----------------------------------------------------------- inference --
    def _predict(self, w: np.ndarray, X: CSRMatrix) -> np.ndarray:
        return X.dot(w) + self.y_mean_

    def predict(self, X: CSRMatrix) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("model is not fitted")
        return self._predict(self.w_, X)

    # -------------------------------------------------------- persistence --
    def save(self, path) -> None:
        np.savez(path, w=self.w_, y_mean=np.array(self.y_mean_))

    @classmethod
    def load(cls, path) -> "RidgeRegressionGD":
        z = np.load(path, allow_pickle=False)
        model = cls(verbose=False)
        model.w_ = z["w"].astype(np.float64)
        model.y_mean_ = float(z["y_mean"])
        return model
