"""Temporal Fusion Transformer: BiLSTM encoder + multi-head attention + dual output heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from mefai_risk.models.attention import MultiHeadAttention


class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal-Fusion-Transformer architecture for risk scoring.

    Architecture
    ------------
    1. BiLSTM encoder
    2. Temporal self-attention
    3. Feature self-attention
    4. Unidirectional LSTM decoder
    5. Two output heads:
       - **risk_head** -> sigmoid in [0, 1]
       - **volatility_head** -> ReLU >= 0

    Parameters
    ----------
    input_size : int
        Number of input features per time-step.
    hidden_size : int
        Hidden dimensionality of LSTM / attention layers.
    output_size : int
        Output dimensionality of the risk head (default 1).
    num_layers : int
        Number of stacked BiLSTM layers in the encoder.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Encoder (Bidirectional LSTM) ---------------------------------
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        enc_dim = hidden_size * 2  # bidirectional doubles the output dim

        # --- Attention layers ---------------------------------------------
        self.temporal_attention = MultiHeadAttention(enc_dim, n_heads=n_heads, dropout=dropout)
        self.feature_attention = MultiHeadAttention(enc_dim, n_heads=n_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(enc_dim)

        # --- Decoder (unidirectional LSTM) --------------------------------
        self.decoder_lstm = nn.LSTM(
            input_size=enc_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # --- Output heads -------------------------------------------------
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid(),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU(),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(batch, seq_len, input_size)``

        Returns
        -------
        risk_score : Tensor ``(batch, output_size)`` in [0, 1]
        volatility : Tensor ``(batch, 1)``  >= 0
        """
        # Encoder
        enc_out, (h_n, c_n) = self.encoder_lstm(x)

        # Attention (with residual + layer-norm)
        attn = self.temporal_attention(enc_out) + enc_out
        attn = self.layer_norm(attn)
        attn = self.feature_attention(attn) + attn
        attn = self.layer_norm(attn)

        # Decoder ··· initialise with the last-layer forward hidden state
        dec_out, _ = self.decoder_lstm(attn, (h_n[:1], c_n[:1]))

        last_hidden = dec_out[:, -1, :]  # (batch, hidden_size)

        risk_score = self.risk_head(last_hidden)
        volatility = self.volatility_head(last_hidden)
        return risk_score, volatility
