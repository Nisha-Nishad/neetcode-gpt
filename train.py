import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int,
              context_length: int, batch_size: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

        # Valid start indices: [0, len(data) - context_length - 1]
        # randint's high is exclusive, so high = len(data) - context_length
        high = len(data) - context_length
        offsets = torch.arange(context_length)  # reused every epoch

        loss = None
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ix = torch.randint(0, high, (batch_size,))

            # Vectorized batch construction (no Python loop / torch.stack)
            idx = ix.unsqueeze(1) + offsets            # (B, T)
            X = data[idx]                              # inputs
            Y = data[idx + 1]                          # targets (X shifted by 1)

            logits = model(X)                          # (B, T, C)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),      # (B*T, C)
                Y.reshape(-1)                          # (B*T,)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)