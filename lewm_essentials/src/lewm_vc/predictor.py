"""
LeWM Temporal Predictor Module

Implements the temporal predictor for the LeWM-VC video codec.
Predicts future latents from past latents using a transformer encoder,
outputting isotropic Gaussian parameters (mean and std) per SIGReg.
"""

import torch
import torch.nn as nn


class LeWMPredictor(nn.Module):
    """
    Temporal predictor network using transformer encoder.

    Predicts future latent representations from a context of 1-4 previous
    latents. Outputs isotropic Gaussian parameters for stochastic prediction
    following the SIGReg (Split Implicit Gradient Regularization) approach.

    Architecture:
        - 8-layer Transformer encoder
        - 256-dim hidden, 4 attention heads
        - Output heads for mean and log_std (isotropic Gaussian)

    Args:
        latent_dim: Latent dimension (default: 192)
        hidden_dim: Transformer hidden dimension (default: 256)
        num_layers: Number of transformer layers (default: 8)
        num_heads: Number of attention heads (default: 4)
        context_len: Maximum context length (default: 4)

    Input:
        context: List of 1-4 previous latent tensors, each [B, latent_dim, H//16, W//16]

    Output:
        mean: Predicted latent mean [B, latent_dim, H//16, W//16]
        std: Predicted latent std [B, latent_dim, H//16, W//16]
    """

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        context_len: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_len = context_len

        self.input_proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.frame_tokens = nn.Parameter(torch.zeros(1, context_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        self.mean_head = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)
        self.log_std_head = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)

    def forward(
        self,
        context: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the predictor network.

        Args:
            context: List of 1-4 previous latent tensors,
                     each [B, latent_dim, H//16, W//16]

        Returns:
            mean: Predicted latent mean [B, latent_dim, H//16, W//16]
            std: Predicted latent std [B, latent_dim, H//16, W//16]
        """
        if len(context) == 0:
            raise ValueError("Context cannot be empty")
        if len(context) > self.context_len:
            raise ValueError(f"Context length {len(context)} exceeds maximum {self.context_len}")

        b = context[0].shape[0]
        h, w = context[0].shape[2], context[0].shape[3]

        projected = [self.input_proj(latent) for latent in context]

        pooled = []
        for p in projected:
            pooled_frame = p.mean(dim=[2, 3])
            pooled.append(pooled_frame)

        temporal_input = torch.stack(pooled, dim=1)

        if temporal_input.shape[1] < self.context_len:
            padding = torch.zeros(b, self.context_len - temporal_input.shape[1], self.hidden_dim, device=temporal_input.device)
            temporal_input = torch.cat([temporal_input, padding], dim=1)

        temporal_input = temporal_input + self.frame_tokens

        temporal_output = self.transformer(temporal_input)

        temporal_output = self.temporal_norm(temporal_output)

        last_frame_idx = len(context) - 1
        last_temporal = temporal_output[:, last_frame_idx]

        last_frame_proj = projected[last_frame_idx]

        combined = torch.cat([last_frame_proj, last_temporal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)], dim=1)

        spatial_features = self.spatial_conv(combined)

        mean = self.mean_head(spatial_features)
        log_std = self.log_std_head(spatial_features)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)

        return mean, std

    def predict(
        self,
        context: list[torch.Tensor],
        sample: bool = True
    ) -> torch.Tensor:
        """
        Generate predicted latent with optional sampling.

        Args:
            context: List of previous latent tensors
            sample: Whether to sample from predicted distribution (default: True)

        Returns:
            Predicted latent tensor [B, latent_dim, H//16, W//16]
        """
        mean, std = self.forward(context)

        if sample:
            noise = torch.randn_like(mean)
            predicted = mean + std * noise
        else:
            predicted = mean

        return predicted

    def nll_loss(
        self,
        context: list[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for training.

        Args:
            context: List of previous latent tensors
            target: Target latent tensor [B, latent_dim, H//16, W//16]

        Returns:
            NLL loss (scalar)
        """
        mean, std = self.forward(context)

        var = std ** 2
        nll = 0.5 * (
            ((target - mean) ** 2) / var +
            torch.log(var) +
            torch.log(torch.tensor(2 * torch.pi))
        )

        return nll.mean()
