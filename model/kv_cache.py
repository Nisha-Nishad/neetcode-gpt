import torch
import torch.nn as nn
from typing import Tuple, Optional


class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None:
            self.cache_k, self.cache_v = new_k, new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.model_dim = model_dim
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        if kv_cache is None:
            kv_cache = KVCache()

        # Number of tokens already cached (before this call)
        past_len = 0 if kv_cache.cache_k is None else kv_cache.cache_k.size(1)

        q = self.q_proj(x)                      # (B, S, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k_all, v_all = kv_cache.update(k, v)    # (B, past+S, D)

        S, T = q.size(1), k_all.size(1)

        scores = q @ k_all.transpose(-2, -1) / (self.model_dim ** 0.5)   # (B, S, T)

        # Causal mask offset by past_len: query i (absolute pos past_len+i)
        # may attend to key j iff j <= past_len + i
        mask = torch.ones(S, T, dtype=torch.bool, device=x.device).triu(diagonal=past_len + 1)
        scores = scores.masked_fill(mask, float('-inf'))

        out = torch.softmax(scores, dim=-1) @ v_all                      # (B, S, D)

        return torch.round(out * 10000) / 10000, kv_cache