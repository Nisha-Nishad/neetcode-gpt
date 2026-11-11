import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions: List[float] = []

        def hook(module, inputs, output):
            # output: (batch, features, ...) -> a neuron is dead if it is 0 for all samples
            # (output <= 0).all(dim=0) -> per-neuron dead mask; mean gives the fraction
            dead = (output == 0).all(dim=0).float().mean().item()
            dead_fractions.append(round(dead, 4))

        handles = [m.register_forward_hook(hook)
                   for m in model.modules() if isinstance(m, nn.ReLU)]

        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return 'healthy'

        # 1. Severe: any layer more than half dead
        if max(dead_fractions) > 0.5:
            return 'use_leaky_relu'

        # 2. Early-layer death
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Strictly increasing with depth and last layer > 0.1
        #    (a single layer has no "increase", so require at least 2 layers)
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if len(dead_fractions) > 1 and strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4 & 5. Healthy either way
        return 'healthy'