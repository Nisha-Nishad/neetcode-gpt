import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        tokens = raw_dataset.split()

        # Valid start indices: [0, len(tokens) - context_length - 1]
        # torch.randint's high is exclusive, so high = len(tokens) - context_length
        starts = torch.randint(0, len(tokens) - context_length, (batch_size,)).tolist()

        X = [tokens[i : i + context_length] for i in starts]
        Y = [tokens[i + 1 : i + 1 + context_length] for i in starts]
        return X, Y