import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eps = 1e-5
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean()
        var = x.var()  # population variance (divides by n), matches the formula
        x_hat = (x - mean) / np.sqrt(var + eps)
        return np.round(gamma * x_hat + beta, 5)