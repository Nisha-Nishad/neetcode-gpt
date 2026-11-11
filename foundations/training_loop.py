import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        scale = 2.0 / n

        for _ in range(epochs):
            err = X @ w + b - y          # (n,)
            w -= lr * scale * (X.T @ err)
            b -= lr * scale * err.sum()

        return np.round(w, 5), round(float(b), 5)