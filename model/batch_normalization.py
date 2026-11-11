import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.asarray(x, dtype=np.float64)
        gamma = np.asarray(gamma, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        r_mean = np.asarray(running_mean, dtype=np.float64)
        r_var = np.asarray(running_var, dtype=np.float64)

        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)  # biased (1/N), matches the problem's formula
            r_mean = (1 - momentum) * r_mean + momentum * mean
            r_var = (1 - momentum) * r_var + momentum * var
        else:
            mean, var = r_mean, r_var

        x_hat = (x - mean) / np.sqrt(var + eps)
        y = gamma * x_hat + beta

        return (np.round(y, 4).tolist(),
                np.round(r_mean, 4).tolist(),
                np.round(r_var, 4).tolist())