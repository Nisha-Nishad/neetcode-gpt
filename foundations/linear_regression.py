import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> (n,) predictions
        return np.round(np.asarray(X) @ np.asarray(weights), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        diff = np.asarray(model_prediction) - np.asarray(ground_truth)
        return round(float(np.mean(diff ** 2)), 5)