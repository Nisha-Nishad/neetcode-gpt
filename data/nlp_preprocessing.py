import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        sentences = [s.split() for s in positive + negative]

        # 1. Vocabulary: unique words sorted lexicographically, IDs start at 1
        vocab = {w: i for i, w in enumerate(sorted({w for s in sentences for w in s}), 1)}

        # 2. Encode each sentence (float dtype to match expected output)
        tensors = [torch.tensor([vocab[w] for w in s], dtype=torch.float) for s in sentences]

        # 3 + 4. Pad with 0s into a (2N, T) tensor
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)