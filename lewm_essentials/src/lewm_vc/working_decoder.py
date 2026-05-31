"""
LeWM-VC Video Decoder

Decoder that reconstructs RGB frames from latent representations.
Uses transposed convolutions with residual blocks for better quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class LeWMDecoder(nn.Module):
    """
    Video decoder with improved architecture:
        - Project latent_dim -> hidden_dim
        - 4 upsampling blocks with residual connections
        - Deeper processing for better reconstruction

    Args:
        latent_dim: Input latent dimension (default: 192)
        hidden_dim: Hidden dimension (default: 512)
        output_channels: Output channels (default: 3 for RGB)
    """

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 512,
        output_channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(hidden_dim // 2)

        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualBlock(hidden_dim // 4)

        self.up3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1)
        self.res3 = ResidualBlock(hidden_dim // 8)

        self.up4 = nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=4, stride=2, padding=1)
        self.res4 = ResidualBlock(hidden_dim // 16)

        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 32, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim // 32),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 32, output_channels, 3, padding=1),
        )

    def forward(self, latent: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Decode latent to RGB frame.

        Args:
            latent: [B, latent_dim, H, W] latent tensor
            target_size: Optional (H, W) target output size

        Returns:
            [B, 3, H*16, W*16] or [B, 3, target_size[0], target_size[1]] RGB tensor in [0, 1] range
        """
        x = self.proj(latent)

        x = self.up1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        x = self.up3(x)
        x = self.res3(x)

        x = self.up4(x)
        x = self.res4(x)

        x = self.final(x)
        x = torch.sigmoid(x)

        if target_size is not None:
            x = functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class SimpleWorkingDecoder(nn.Module):
    """
    Simpler decoder for quick testing (without full upsampling).
    Uses bilinear interpolation for upsampling.
    """

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.latent_dim = latent_dim

        self.rgb_proj = nn.Conv2d(latent_dim, 3, kernel_size=1)
        nn.init.xavier_normal_(self.rgb_proj.weight)
        nn.init.zeros_(self.rgb_proj.bias)

    def forward(self, latent: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        if target_size is not None:
            latent = functional.interpolate(latent, size=target_size, mode='bilinear', align_corners=False)

        x = self.rgb_proj(latent)
        x = torch.tanh(x) * 0.5 + 0.5

        return x


class WorkingVideoDecoder:
    """
    Full video decoder with temporal processing.
    """

    def __init__(self, latent_dim: int = 192, use_trained: bool = True):
        if use_trained:
            self.decoder = LeWMDecoder(latent_dim=latent_dim)
        else:
            self.decoder = SimpleWorkingDecoder(latent_dim=latent_dim)
        self.decoder.eval()

    def decode_frame(self, latent: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """Decode a single frame."""
        with torch.no_grad():
            return self.decoder(latent, target_size)

    def decode_video(self, latents: list, target_size: tuple = None) -> list:
        """Decode a list of latents to RGB frames."""
        frames = []
        for latent in latents:
            frame = self.decode_frame(latent, target_size)
            frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype('uint8')
            frames.append(frame_np)
        return frames
