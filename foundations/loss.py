import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return round(float(loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-7, 1 - 1e-7)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round(float(loss), 4)