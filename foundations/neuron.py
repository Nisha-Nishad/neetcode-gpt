import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # Pre-activation: z = dot(x, w) + b
        z = float(np.dot(x, w) + b)

        if activation == "sigmoid":
            out = 1.0 / (1.0 + np.exp(-z))
        elif activation == "relu":
            out = max(0.0, z)
        else:
            raise ValueError("activation must be 'sigmoid' or 'relu'")

        return round(float(out), 5)