"""
LeWM Encoder Module

Implements a ViT-Tiny style encoder for the LeWM-VC video codec.
Converts YUV420 frames into latent representations with optional
semantic surprise detection for physics implausibility.
"""


import torch
import torch.nn as nn


class LeWMEncoder(nn.Module):
    """
    Vision Transformer (ViT)-Tiny style encoder for video compression.

    Architecture:
        - Patchify layer: 16x16 patches, projects to hidden_dim=192
        - 6-layer Transformer encoder (Tiny variant)
        - Final projection to latent_dim=192
        - Optional semantic surprise branch (MLP head)

    Args:
        latent_dim: Output latent dimension (default: 192)
        patch_size: Size of patches for patchification (default: 16)
        hidden_dim: Hidden dimension for transformer (default: 192)
        num_layers: Number of transformer encoder layers (default: 6)
        num_heads: Number of attention heads (default: 3)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        semantic_surprise: Whether to include surprise detection head (default: False)

    Input:
        x: YUV420 frame tensor of shape [B, 3, H, W] normalized [0, 1]

    Output:
        latent: Latent tensor of shape [B, latent_dim, H//16, W//16]
        surprise: (optional) Physics implausibility score [B, 1] if semantic_surprise=True
    """

    def __init__(
        self,
        latent_dim: int = 192,
        patch_size: int = 16,
        hidden_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        semantic_surprise: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.semantic_surprise = semantic_surprise

        self.patch_embed = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=int(hidden_dim * mlp_ratio),
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.latent_proj = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)

        if semantic_surprise:
            self.surprise_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        x: torch.Tensor,
        return_surprise: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder network.

        Args:
            x: Input YUV tensor [B, 3, H, W] normalized [0, 1]
            return_surprise: Whether to return surprise scores (requires semantic_surprise=True)

        Returns:
            latent: Encoded latent [B, latent_dim, H//16, W//16]
            surprise: (optional) Physics implausibility [B, 1]
        """
        b, c, h, w = x.shape

        x = self.patch_embed(x)

        x_flat = x.flatten(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x_flat], dim=1)

        x_with_cls = x_with_cls + self.pos_embed

        for layer in self.encoder_layers:
            x_with_cls = layer(x_with_cls)

        x_with_cls = self.norm(x_with_cls)

        cls_output = x_with_cls[:, 0]
        patch_output = x_with_cls[:, 1:]

        patch_output = patch_output.permute(0, 2, 1).reshape(b, self.hidden_dim, h // self.patch_size, w // self.patch_size)

        latent = self.latent_proj(patch_output)

        if self.semantic_surprise and return_surprise:
            surprise = self.surprise_head(cls_output)
            return latent, surprise

        return latent


class TransformerEncoderLayer(nn.Module):
    """
    Standard transformer encoder layer with pre-norm architecture.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0]

        x2 = self.norm2(x)
        x = x + self.dropout(self.linear2(self.activation(self.linear1(x2))))

        return x
