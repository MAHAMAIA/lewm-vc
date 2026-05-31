"""
LeWM Decoder & Post-Filter Module

Implements the decoder component of the LeWM-VC video codec,
featuring 4-layer ConvTranspose upsampling with residual blocks
and a learned post-filter trained with LPIPS loss.
"""

import torch
import torch.nn as nn


class LeWMDecoder(nn.Module):
    """
    Decoder network for LeWM-VC video codec.

    Upsamples quantized latent representations to full-resolution YUV frames
    using transposed convolutions with residual connections, followed by
    a learned post-filter for perceptual quality improvement.

    Architecture:
        - 4-layer ConvTranspose2d upsampling (192→128→64→32→16)
        - Residual blocks after first three upsample layers
        - Final conv to 16 channels, then to 3-channel RGB
        - 3-layer ConvNet post-filter (LPIPS-trained)

    Args:
        latent_dim: Number of channels in the latent representation (default: 192)

    Input:
        quant_latent: Quantized latent tensor of shape [B, 192, H//16, W//16]
        residual: Optional residual tensor to add to output (default: None)

    Output:
        YUV tensor of shape [B, 3, H, W]
    """

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1)
        self.res1 = self._res_block(128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.res2 = self._res_block(64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.res3 = self._res_block(32)

        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

        self.final = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.post_filter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

    @staticmethod
    def _res_block(ch: int) -> nn.Sequential:
        """
        Create a residual block with two 3x3 convolutions.

        Args:
            ch: Number of input/output channels

        Returns:
            Sequential module containing the residual block
        """
        return nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(
        self,
        quant_latent: torch.Tensor,
        residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the decoder network.

        Args:
            quant_latent: Quantized latent tensor [B, 192, H//16, W//16]
            residual: Optional residual to add to output [B, 3, H, W]

        Returns:
            Reconstructed YUV tensor [B, 3, H, W]
        """
        x = self.up1(quant_latent)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        x = self.up3(x)
        x = self.res3(x)

        x = self.up4(x)

        x = self.final(x)

        if residual is not None:
            x = x + residual

        return self.post_filter(x)
