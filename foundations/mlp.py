import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = np.asarray(x, dtype=np.float64)
        last = len(weights) - 1
        for i, (W, b) in enumerate(zip(weights, biases)):
            h = h @ np.asarray(W, dtype=np.float64) + np.asarray(b, dtype=np.float64)
            if i != last:
                np.maximum(h, 0, out=h)  # in-place ReLU on hidden layers only
        return np.round(h, 5)