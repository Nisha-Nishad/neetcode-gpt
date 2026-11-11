import torch
import math
from typing import List


class Solution:

    @staticmethod
    def _xavier_std(fan_in: int, fan_out: int) -> float:
        return math.sqrt(2.0 / (fan_in + fan_out))

    @staticmethod
    def _kaiming_std(fan_in: int, fan_out: int) -> float:
        return math.sqrt(2.0 / fan_in)

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        w = torch.randn(fan_out, fan_in) * self._xavier_std(fan_in, fan_out)
        return torch.round(w * 10000).div(10000).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        w = torch.randn(fan_out, fan_in) * self._kaiming_std(fan_in, fan_out)
        return torch.round(w * 10000).div(10000).tolist()

    def check_activations(self, num_layers: int, input_dim: int,
                          hidden_dim: int, init_type: str) -> List[float]:
        if init_type not in ('xavier', 'kaiming', 'random'):
            raise ValueError(f"Unknown init_type: {init_type}")

        torch.manual_seed(0)

        # 1) Build ALL weight matrices first
        weights = []
        in_dim = input_dim
        for _ in range(num_layers):
            w = torch.randn(hidden_dim, in_dim)
            if init_type == 'xavier':
                w = w * self._xavier_std(in_dim, hidden_dim)
            elif init_type == 'kaiming':
                w = w * self._kaiming_std(in_dim, hidden_dim)
            weights.append(w)
            in_dim = hidden_dim

        # 2) Then draw the input
        x = torch.randn(input_dim)

        # 3) Forward pass
        stds = []
        with torch.no_grad():
            for w in weights:
                x = torch.relu(w @ x)
                stds.append(round(x.std().item(), 2))
        return stds