"""Multi-head self-attention layer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input token.
    n_heads : int
        Number of attention heads.  ``input_dim`` must be divisible by
        ``n_heads``.
    dropout : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, input_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim % n_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, seq_len, input_dim)``

        Returns
        -------
        Tensor of the same shape.
        """
        B, S, _ = x.size()

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)
        attn = self.dropout(F.softmax(scores, dim=-1))
        weighted = torch.matmul(attn, v)  # (B, H, S, head_dim)

        # Concatenate heads
        weighted = weighted.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(weighted)
