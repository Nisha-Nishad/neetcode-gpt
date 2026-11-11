import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        activations = []
        hooks = []

        def hook_fn(module, input, output):
            out = output.detach()
            mean = round(float(out.mean().item()), 4)
            std = round(float(out.std().item()), 4) if out.numel() > 1 else 0.0

            # A neuron is "dead" if its output is <= 0 for all samples in the batch.
            if out.ndim > 2:
                out_reshaped = out.reshape(-1, out.shape[-1])
                dead = (out_reshaped <= 0).all(dim=0)
            else:
                dead = (out <= 0).all(dim=0) if out.ndim == 2 else (out <= 0).all()

            dead_fraction = round(float(dead.float().mean().item()), 4) if dead.numel() > 0 else 0.0

            activations.append({
                "mean": mean,
                "std": std,
                "dead_fraction": dead_fraction
            })

        for module in model.modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = model(x)

        for h in hooks:
            h.remove()

        return activations

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.train()
        model.zero_grad()

        preds = model(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds, y)
        loss.backward()

        gradient_stats = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    grad = module.weight.grad.detach()
                    mean = round(float(grad.mean().item()), 4)
                    std = round(float(grad.std().item()), 4) if grad.numel() > 1 else 0.0
                    norm = round(float(torch.norm(grad).item()), 4)
                else:
                    mean = 0.0
                    std = 0.0
                    norm = 0.0
                gradient_stats.append({
                    "mean": mean,
                    "std": std,
                    "norm": norm
                })

        return gradient_stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # 1. 'dead_neurons' if any layer has dead_fraction > 0.5
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. 'exploding_gradients' if any layer gradient norm > 1000
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        # 3. 'vanishing_gradients' if last layer gradient norm < 1e-5
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Check activation std for all layers
        for stat in activation_stats:
            if stat["std"] < 0.1:
                return "vanishing_gradients"
            if stat["std"] > 10.0:
                return "exploding_gradients"

        # 5. 'healthy' if none of the above
        return "healthy"