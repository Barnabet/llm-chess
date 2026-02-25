"""Transformer-based chess position evaluator.

The architecture follows a ViT-style design:

1. Each of the 64 board squares becomes a token with 21 input features.
2. A linear projection maps them to ``d_model`` dimensions.
3. Learnable positional encodings + a prepended [CLS] token give the
   Transformer encoder a global aggregation point.
4. The [CLS] output feeds an **Eval head** (scalar in [-1, 1]).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ChessEvalTransformer(nn.Module):
    """Transformer-based chess position evaluator.

    Parameters
    ----------
    d_model:
        Hidden dimension of the Transformer.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of ``TransformerEncoderLayer`` blocks.
    d_ff:
        Feed-forward inner dimension (defaults to ``4 * d_model``).
    dropout:
        Dropout rate used throughout.
    input_channels:
        Number of input channels per square (21 for the eval observation).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int | None = None,
        dropout: float = 0.1,
        input_channels: int = 21,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model

        # --- 1. Spatial projection ---
        # Input shape: [B, 21, 8, 8] -> flatten spatial -> [B, 64, 21]
        # then project -> [B, 64, d_model].
        self.input_proj = nn.Linear(input_channels, d_model)

        # --- 2. Positional encoding + [CLS] token ---
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- 3. Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- 4. Eval head: [CLS] -> scalar in [-1, 1] ---
        self.eval_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Initialise linear layers with scaled Xavier and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(3))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:  Observation tensor of shape ``[B, 21, 8, 8]``.

        Returns
        -------
        eval:  ``[B, 1]`` board evaluation in [-1, 1].
        """
        B = x.size(0)

        # Flatten spatial dims: [B, 21, 8, 8] -> [B, 21, 64] -> [B, 64, 21]
        x = x.view(B, x.size(1), 64).permute(0, 2, 1)  # [B, 64, 21]

        # Project to d_model.
        x = self.input_proj(x)  # [B, 64, d_model]

        # Add positional encoding.
        x = x + self.pos_embed

        # Prepend [CLS] token.
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 65, d_model]

        # Transformer encoder.
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Extract [CLS] output (position 0).
        cls_out = x[:, 0]  # [B, d_model]

        return self.eval_head(cls_out)  # [B, 1]
