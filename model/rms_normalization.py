import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.asarray(x, dtype=np.float64)
        gamma = np.asarray(gamma, dtype=np.float64)

        # RMS = sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.mean(x * x) + eps)

        # Normalize, then scale by gamma
        out = gamma * (x / rms)

        return np.round(out, 4).tolist()