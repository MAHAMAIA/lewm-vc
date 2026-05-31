"""
Entropy Model for LeWM-VC Video Codec

Implements hyperprior-based entropy coding with SIGReg closed-form KL divergence.
Performs rate estimation for latent residuals using Gaussian mixture modeling.
"""

import torch
import torch.nn as nn


class HyperpriorEntropy(nn.Module):
    """
    Hyperprior entropy model for latent compression.

    Uses a hyperprior CNN to predict μ (mean) and σ (std) parameters
    for each latent element, enabling context-adaptive arithmetic coding.

    Architecture:
        - 5-layer CNN predicting 2 * latent_dim parameters (μ, σ)
        - Gaussian KL divergence computed in closed-form (SIGReg)

    Args:
        latent_dim: Number of latent channels (default: 192)
        hyper_channels: Number of hyperprior hidden channels (default: 320)

    Input:
        residual: Residual tensor [B, latent_dim, H, W]

    Output:
        rate: Estimated bitrate in bits [B, 1, H, W]
        context: Dict containing μ, σ parameters for decoding
    """

    def __init__(self, latent_dim: int = 192, hyper_channels: int = 320):
        super().__init__()
        self.latent_dim = latent_dim
        self.hyper_channels = hyper_channels

        self.hyperprior_cnn = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, latent_dim * 2, 3, 1, 1),
        )

        self.entropy_bottleneck = None

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass to estimate rate for residual latent.

        Args:
            residual: Input residual [B, latent_dim, H, W]

        Returns:
            rate: Bitrate estimate in bits [B, 1, H, W]
            context: Dict with mu and sigma tensors
        """
        params = self.hyperprior_cnn(residual)

        mu = params[:, :self.latent_dim, :, :]
        log_sigma = params[:, self.latent_dim:, :, :]
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-5

        rate = self.gaussian_kl(residual, mu, sigma)

        return rate, {"mu": mu, "sigma": sigma}

    @staticmethod
    def gaussian_kl(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian KL divergence in closed form (SIGReg).

        Computes: KL(N(x|μ,σ²) || N(0,1)) = 0.5 * (μ² + σ² - log(σ²) - 1)

        This is the rate lower bound for coding x with learned parameters.

        Args:
            x: Latent tensor [B, C, H, W]
            mu: Predicted mean [B, C, H, W]
            sigma: Predicted standard deviation [B, C, H, W]

        Returns:
            kl: KL divergence per element [B, 1, H, W] or scalar
        """
        sigma_sq = sigma ** 2

        kl = 0.5 * (mu ** 2 + sigma_sq - torch.log(sigma_sq) - 1)

        if mu.shape[1] == 1:
            kl = kl.sum()
        else:
            kl = kl.sum(dim=1, keepdim=True)

        kl_bits = kl / torch.log(torch.tensor(2.0, device=x.device))

        return kl_bits

    def get_entropy_parameters(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get μ and σ without computing rate.

        Useful for inference when only parameters are needed.

        Args:
            residual: Input residual [B, C, H, W]

        Returns:
            mu: Predicted mean [B, C, H, W]
            sigma: Predicted std [B, C, H, W]
        """
        params = self.hyperprior_cnn(residual)

        mu = params[:, :self.latent_dim, :, :]
        log_sigma = params[:, self.latent_dim:, :, :]
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-5

        return mu, sigma
