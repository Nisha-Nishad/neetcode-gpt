import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.asarray(x, dtype=np.float64)
        W1 = np.asarray(W1, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)
        W2 = np.asarray(W2, dtype=np.float64)
        b2 = np.asarray(b2, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)

        # ---------- Forward ----------
        z1 = W1 @ x + b1              # (hidden,)
        a1 = np.maximum(0.0, z1)      # ReLU
        z2 = W2 @ a1 + b2             # (output,)  == y_hat

        diff = z2 - y
        n = y.size
        loss = np.mean(diff ** 2)

        # ---------- Backward ----------
        dz2 = (2.0 / n) * diff                 # dL/dz2        (output,)
        dW2 = np.outer(dz2, a1)                # (output, hidden)
        db2 = dz2                              # (output,)

        da1 = W2.T @ dz2                       # (hidden,)
        dz1 = da1 * (z1 > 0)                   # ReLU mask
        dW1 = np.outer(dz1, x)                 # (hidden, input)
        db1 = dz1                              # (hidden,)

        def r(a):
            # +0.0 converts any -0.0 into 0.0
            return (np.round(a, 4) + 0.0).tolist()

        return {
            'loss': round(float(loss), 4) + 0.0,
            'dW1': r(dW1),
            'db1': r(db1),
            'dW2': r(dW2),
            'db2': r(db2),
        }