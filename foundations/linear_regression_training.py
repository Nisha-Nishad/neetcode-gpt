import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        weights = np.array(initial_weights, dtype=np.float64)  # copy, so the input isn't mutated
        N = len(X)

        for _ in range(num_iterations):
            preds = self.get_model_prediction(X, weights)
            # Compute all gradients from the same predictions, then update together
            grads = np.array([
                self.get_derivative(preds, Y, N, X, j) for j in range(3)
            ])
            weights -= self.learning_rate * grads

        return np.round(weights, 5)