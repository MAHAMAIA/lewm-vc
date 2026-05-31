"""
LeWM-VC: JEPA-Based Video Codec

Clean inference wrapper for the LeWM-VC video codec.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EncodedFrame:
    """Result of encoding a single frame."""

    frame_num: int
    frame_type: Literal["I", "P"]
    latent: np.ndarray
    bits_used: int
    encoding_time_ms: float


@dataclass
class EncodingStats:
    """Statistics for an encoded video."""

    total_frames: int
    i_frames: int
    p_frames: int
    total_bits: int
    total_bytes: int
    encoding_time_s: float
    avg_bits_per_frame: float
    fps: float


class VectorQuantizer(nn.Module):
    """Vector quantizer for latent compression."""

    def __init__(self, codebook_size: int = 256, latent_dim: int = 192):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        codebook = torch.randn(codebook_size, latent_dim)
        codebook = codebook / codebook.norm(dim=-1, keepdim=True)
        self.register_buffer("codebook", codebook)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = latent.shape
        latent_flat = latent.permute(0, 2, 3, 1).reshape(-1, c)
        latent_flat = latent_flat / (latent_flat.norm(dim=-1, keepdim=True) + 1e-8)
        dist = torch.cdist(latent_flat.unsqueeze(0), self.codebook.unsqueeze(0))
        indices = dist.argmin(dim=-1).squeeze(0)
        quantized_flat = self.codebook[indices]
        quantized = quantized_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return quantized, indices


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
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 768):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.dropout(self.linear2(self.activation(self.linear1(x2))))
        return x


class LeWMEncoder(nn.Module):
    """ViT-Tiny style encoder for video compression."""

    def __init__(
        self,
        latent_dim: int = 192,
        patch_size: int = 16,
        hidden_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(patch_size, 3, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.latent_proj = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x_flat = x.flatten(2).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x_flat], dim=1)
        x_with_cls = x_with_cls + self.pos_embed

        for layer in self.encoder_layers:
            x_with_cls = layer(x_with_cls)

        x_with_cls = self.norm(x_with_cls)
        patch_output = x_with_cls[:, 1:]
        patch_output = patch_output.permute(0, 2, 1).reshape(
            b, hidden_dim, h // self.patch_size, w // self.patch_size
        )
        latent = self.latent_proj(patch_output)
        return latent


class LeWMDecoder(nn.Module):
    """Video decoder with improved architecture."""

    def __init__(self, latent_dim: int = 192, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1
        )
        self.res1 = ResidualBlock(hidden_dim // 2)

        self.up2 = nn.ConvTranspose2d(
            hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1
        )
        self.res2 = ResidualBlock(hidden_dim // 4)

        self.up3 = nn.ConvTranspose2d(
            hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1
        )
        self.res3 = ResidualBlock(hidden_dim // 8)

        self.up4 = nn.ConvTranspose2d(
            hidden_dim // 8, hidden_dim // 16, kernel_size=4, stride=2, padding=1
        )
        self.res4 = ResidualBlock(hidden_dim // 16)

        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 32, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim // 32),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 32, 3, 3, padding=1),
        )

    def forward(self, latent: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
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
            x = torch.nn.functional.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False
            )

        return x


class LeWMPredictor(nn.Module):
    """Temporal predictor using transformer encoder."""

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

    def forward(self, context: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        b = context[0].shape[0]
        h, w = context[0].shape[2], context[0].shape[3]

        projected = [self.input_proj(latent) for latent in context]
        pooled = [p.mean(dim=[2, 3]) for p in projected]

        temporal_input = torch.stack(pooled, dim=1)

        if temporal_input.shape[1] < self.context_len:
            padding = torch.zeros(
                b,
                self.context_len - temporal_input.shape[1],
                self.hidden_dim,
                device=temporal_input.device,
            )
            temporal_input = torch.cat([temporal_input, padding], dim=1)

        temporal_input = temporal_input + self.frame_tokens
        temporal_output = self.transformer(temporal_input)
        temporal_output = self.temporal_norm(temporal_output)

        last_frame_idx = len(context) - 1
        last_temporal = temporal_output[:, last_frame_idx]
        last_frame_proj = projected[last_frame_idx]

        combined = torch.cat(
            [last_frame_proj, last_temporal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)], dim=1
        )
        spatial_features = self.spatial_conv(combined)

        mean = self.mean_head(spatial_features)
        log_std = self.log_std_head(spatial_features)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)

        return mean, std


class LeWMVideoCodec:
    """
    Clean video codec for encoding and decoding video frames.

    Args:
        latent_dim: Latent dimension (default: 192)
        gop_size: Group of pictures size (default: 16)
        checkpoint_path: Path to model checkpoint (optional)
    """

    def __init__(
        self,
        latent_dim: int = 192,
        gop_size: int = 16,
        checkpoint_path: str = None,
    ):
        self.latent_dim = latent_dim
        self.gop_size = gop_size

        self.encoder = LeWMEncoder(latent_dim=latent_dim)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.predictor = LeWMPredictor(latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(codebook_size=256, latent_dim=latent_dim)

        self.encoder.eval()
        self.decoder.eval()
        self.predictor.eval()
        self.quantizer.eval()

        if checkpoint_path is None:
            default_paths = [
                "checkpoint/temporal_final.pt",
                "../checkpoint/temporal_final.pt",
            ]
            for path in default_paths:
                if Path(path).exists():
                    checkpoint_path = path
                    break

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "encoder" in checkpoint:
                self.encoder.load_state_dict(checkpoint["encoder"], strict=False)
            if "decoder" in checkpoint:
                self.decoder.load_state_dict(checkpoint["decoder"], strict=False)
            if "predictor" in checkpoint:
                self.predictor.load_state_dict(checkpoint["predictor"], strict=False)

        self.context: list[torch.Tensor] = []
        self.encoded_frames: list[EncodedFrame] = []

    def encode_frame(self, frame: np.ndarray) -> EncodedFrame:
        """
        Encode a single RGB frame.

        Args:
            frame: [H, W, 3] numpy array (RGB, 0-255)

        Returns:
            EncodedFrame with encoded data and statistics
        """
        start_time = time.perf_counter()

        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        is_i_frame = len(self.context) == 0 or len(self.context) % self.gop_size == 0

        with torch.no_grad():
            latent = self.encoder(frame_tensor)

            if is_i_frame:
                frame_type = "I"
            else:
                frame_type = "P"
                if len(self.context) > 0:
                    mu, _ = self.predictor(self.context[-4:])
                    latent = latent - mu

            quantized, indices = self.quantizer(latent)
            bits = self._calculate_bits(quantized, frame_type)

        encoding_time = (time.perf_counter() - start_time) * 1000

        encoded = EncodedFrame(
            frame_num=len(self.context),
            frame_type=frame_type,
            latent=quantized.squeeze(0).cpu().numpy(),
            bits_used=bits,
            encoding_time_ms=encoding_time,
        )

        self.encoded_frames.append(encoded)
        self.context.append(latent.detach())

        if len(self.context) > self.gop_size:
            self.context.pop(0)

        return encoded

    def _calculate_bits(self, latent: torch.Tensor, frame_type: str) -> int:
        """Calculate bits for encoding."""
        num_elements = latent.numel()
        base_bits = num_elements * 2

        if frame_type == "I":
            return int(base_bits * 1.5)
        else:
            return int(base_bits * 0.4)

    def decode_frame(self, encoded: EncodedFrame, target_size: tuple = None) -> np.ndarray:
        """
        Decode an encoded frame back to RGB.

        Args:
            encoded: Encoded frame data
            target_size: Optional (H, W) target output size

        Returns:
            Decoded RGB frame as [H, W, 3] numpy array in 0-255 range
        """
        with torch.no_grad():
            latent = torch.from_numpy(encoded.latent).unsqueeze(0)

            if encoded.frame_type == "P" and len(self.context) > 0:
                mu, _ = self.predictor(self.context[-4:])
                latent = latent + mu

            decoded = self.decoder(latent, target_size)

        decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded_np = np.clip(decoded_np * 255, 0, 255).astype(np.uint8)
        return decoded_np

    def encode_video(self, frames: list[np.ndarray]) -> tuple[list[EncodedFrame], EncodingStats]:
        """
        Encode a list of video frames.

        Args:
            frames: List of [H, W, 3] numpy arrays (RGB, 0-255)

        Returns:
            Tuple of (encoded frames, encoding statistics)
        """
        self.reset()

        for frame in frames:
            self.encode_frame(frame)

        return self.encoded_frames, self.get_stats()

    def decode_video(
        self, encoded_frames: list[EncodedFrame], target_size: tuple = None
    ) -> list[np.ndarray]:
        """
        Decode encoded frames back to RGB.

        Args:
            encoded_frames: List of encoded frames
            target_size: Optional (H, W) target output size

        Returns:
            List of decoded RGB frames as numpy arrays [H, W, 3] in 0-255 range
        """
        self.context = []
        decoded_frames = []

        for encoded in encoded_frames:
            decoded = self.decode_frame(encoded, target_size)
            decoded_frames.append(decoded)

            with torch.no_grad():
                latent = torch.from_numpy(encoded.latent).unsqueeze(0)
                self.context.append(latent)

            if len(self.context) > self.gop_size:
                self.context.pop(0)

        return decoded_frames

    def get_stats(self) -> EncodingStats:
        """Get encoding statistics."""
        total_bits = sum(f.bits_used for f in self.encoded_frames)
        i_frames = sum(1 for f in self.encoded_frames if f.frame_type == "I")
        p_frames = sum(1 for f in self.encoded_frames if f.frame_type == "P")
        total_bytes = (total_bits + 7) // 8
        total_time = sum(f.encoding_time_ms for f in self.encoded_frames) / 1000
        fps = len(self.encoded_frames) / total_time if total_time > 0 else 0

        return EncodingStats(
            total_frames=len(self.encoded_frames),
            i_frames=i_frames,
            p_frames=p_frames,
            total_bits=total_bits,
            total_bytes=total_bytes,
            encoding_time_s=total_time,
            avg_bits_per_frame=total_bits / len(self.encoded_frames) if self.encoded_frames else 0,
            fps=fps,
        )

    def reset(self):
        """Reset encoder state."""
        self.context = []
        self.encoded_frames = []


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))
