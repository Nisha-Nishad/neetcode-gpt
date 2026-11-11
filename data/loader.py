import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)

        # Valid starts: 0 .. len(data) - context_length - 1 (need one extra token for Y)
        high = len(data) - context_length
        starts = torch.randint(0, high, (batch_size,))

        # (batch_size, context_length) index matrix: starts[:, None] + [0..context_length-1]
        idx = starts.unsqueeze(1) + torch.arange(context_length)

        X = data[idx]
        Y = data[idx + 1]
        return X, Y