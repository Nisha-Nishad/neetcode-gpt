import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V
        k = self.key_gen(embedded)      # (B, T, A)
        q = self.query_gen(embedded)    # (B, T, A)
        v = self.value_gen(embedded)    # (B, T, A)

        # 2. Scaled dot-product scores: (B, T, T)
        scores = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)

        # 3. Causal mask: each token attends only to itself and earlier tokens
        T = embedded.shape[1]
        mask = torch.tril(torch.ones(T, T, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax over the key dimension
        weights = torch.softmax(scores, dim=2)

        # 5. Weighted sum of values, rounded to 4 decimals
        return torch.round(weights @ v, decimals=4)