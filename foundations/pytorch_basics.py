import torch
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # -1 lets PyTorch infer M*N//2 rows; reshape returns a view when possible (no copy)
        return torch.round(to_reshape.reshape(-1, 2), decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Reduce along rows -> one mean per column
        return torch.round(to_avg.mean(dim=0), decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Side-by-side join: (M, N) + (M, M) -> (M, M+N)
        return torch.round(torch.cat((cat_one, cat_two), dim=1), decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Mean of squared differences
        return torch.round(F.mse_loss(prediction, target), decimals=4)