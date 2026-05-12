"""
LeWM-VC: JEPA-Based Video Codec

Clean inference wrapper for the LeWM-VC video codec.
Implements temporal coding with JEPA-based prediction.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RESOLUTION = 256
QUANT_STEP = 2.0 / 255


@dataclass
class EncodedFrame:
    """Result of encoding a single frame."""

    frame_num: int
    frame_type: Literal["I", "P"]
    quantized: np.ndarray
    bits_used: float
    encoding_time_ms: float


@dataclass
class EncodingStats:
    """Statistics for an encoded video."""

    total_frames: int
    i_frames: int
    p_frames: int
    total_bits: float
    total_bytes: float
    encoding_time_s: float
    avg_bpp: float
    fps: float


# --- Architecture Classes (from milestone2_temporal.py) ---


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(x))
        x = self.conv1(x)
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.res1 = ResidualBlock(hidden_dim // 2)
        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.res2 = ResidualBlock(hidden_dim // 4)
        self.up3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1)
        self.res3 = ResidualBlock(hidden_dim // 8)
        self.up4 = nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, 4, 2, 1)
        self.res4 = ResidualBlock(hidden_dim // 16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 32, 3, 1, 1),
            nn.InstanceNorm2d(hidden_dim // 32),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 32, 3, 3, 1, 1),
        )

    def forward(self, latent, target_size=None):
        x = self.proj(latent)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        x = self.up3(x)
        x = self.res3(x)
        x = self.up4(x)
        x = self.res4(x)
        x = torch.sigmoid(self.final(x))
        if target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class AffineNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=768):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0]
        x2 = self.norm2(x)
        x = x + self.dropout(self.linear2(self.activation(self.linear1(x2))))
        return x


class LeWMEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=192,
        patch_size=16,
        hidden_dim=192,
        num_layers=6,
        num_heads=3,
        semantic_surprise=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
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

    def forward(self, x, return_surprise=False):
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
            b, self.hidden_dim, h // self.patch_size, w // self.patch_size
        )
        latent = self.latent_proj(patch_output)
        return latent


class LeWMPredictor(nn.Module):
    def __init__(
        self,
        latent_dim=192,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        context_len=4,
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

    def forward(self, context):
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


class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, num_components=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.hyperprior = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, latent_dim * num_components * 3, 3, padding=1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        params = self.hyperprior(x)
        B, C, H, W = params.shape
        cp = C // self.num_components
        params = params.view(B, self.num_components, cp, H, W)
        mu = params[:, :, : self.latent_dim, :, :]
        log_scale = params[:, :, self.latent_dim : 2 * self.latent_dim, :, :]
        log_weight = params[:, :, 2 * self.latent_dim : 3 * self.latent_dim, :, :]
        scale = self.softplus(log_scale) + 1e-5
        weight = torch.softmax(log_weight, dim=1)
        return mu, scale, weight


class TemporalCodec(nn.Module):
    """Full codec with JEPA predictor for temporal residual coding."""

    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
        self.predictor = LeWMPredictor(latent_dim=latent_dim)

    def encode_frame(self, frame, prev_latents=None):
        latent = self.affine(self.encoder(frame, return_surprise=False))

        if prev_latents is None or len(prev_latents) == 0:
            xq = torch.round(latent / QUANT_STEP) * QUANT_STEP
            q = xq + (latent - xq.detach()) * 0.5
            return q, latent, True
        else:
            pred_mean, _ = self.predictor(prev_latents)
            residual = latent - pred_mean
            xq_res = torch.round(residual / QUANT_STEP) * QUANT_STEP
            q_res = xq_res + (residual - xq_res.detach()) * 0.5
            decoded_latent = pred_mean + q_res
            return q_res, decoded_latent, False

    def decode_frame(self, q, prev_latents=None, is_i_frame=True):
        if is_i_frame:
            decoded_latent = q
        else:
            pred_mean, _ = self.predictor(prev_latents)
            decoded_latent = pred_mean + q
        return self.decoder(decoded_latent, target_size=(RESOLUTION, RESOLUTION))


# --- Main Codec Class ---


class LeWMVideoCodec:
    """
    Video codec for encoding and decoding video frames using JEPA temporal prediction.

    Args:
        checkpoint_path: Path to trained checkpoint (temporal_final.pt)
        device: Device to use ('cuda' or 'cpu')
    """

    def __init__(self, checkpoint_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.codec = TemporalCodec(latent_dim=192).to(self.device)
        self.entropy = GMMEntropyModel(latent_dim=192).to(self.device)

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
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "codec" in checkpoint:
                self.codec.load_state_dict(checkpoint["codec"], strict=False)
            if "entropy" in checkpoint:
                self.entropy.load_state_dict(checkpoint["entropy"], strict=False)

        self.codec.eval()
        self.entropy.eval()

        self.context = []
        self.encoded_frames = []
        self.total_bits = 0.0
        self.frame_count = 0
        self.gop_size = 8
        self.quant_step = QUANT_STEP

    def encode_frame(self, frame: np.ndarray) -> EncodedFrame:
        """
        Encode a single RGB frame.

        Args:
            frame: [H, W, 3] numpy array (RGB, 0-255)

        Returns:
            EncodedFrame with encoded data and statistics
        """
        start_time = time.perf_counter()

        frame_tensor = (
            torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        )

        is_i_frame = len(self.context) == 0 or self.frame_count % self.gop_size == 0

        with torch.no_grad():
            q, decoded_latent, is_i = self.codec.encode_frame(
                frame_tensor,
                self.context[-self.codec.predictor.context_len :] if self.context else None,
            )

            mu, scale, weight = self.entropy(q)
            from torch.distributions.normal import Normal

            nc = mu.shape[1]
            ye = q.unsqueeze(1).expand(-1, nc, -1, -1, -1)
            n = Normal(mu, scale)
            pmf = torch.clamp(
                n.cdf(ye + 0.5 * self.quant_step) - n.cdf(ye - 0.5 * self.quant_step),
                min=1e-12,
                max=1.0,
            )
            nll = -torch.log((weight * pmf).sum(dim=1)).mean()
            bits = (nll.item() / np.log(2)) * q.numel()
            self.total_bits += bits

        encoding_time = (time.perf_counter() - start_time) * 1000

        encoded = EncodedFrame(
            frame_num=self.frame_count,
            frame_type="I" if is_i else "P",
            quantized=q.squeeze(0).cpu().numpy(),
            bits_used=bits,
            encoding_time_ms=encoding_time,
        )

        self.encoded_frames.append(encoded)
        self.context.append(decoded_latent.detach())
        if len(self.context) > self.codec.predictor.context_len:
            self.context.pop(0)
        self.frame_count += 1

        return encoded

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
            q = torch.from_numpy(encoded.quantized).unsqueeze(0).to(self.device)
            is_i_frame = encoded.frame_type == "I"

            decoded = self.codec.decode_frame(
                q,
                self.context[-self.codec.predictor.context_len :] if self.context else None,
                is_i_frame,
            )

            if target_size:
                decoded = F.interpolate(
                    decoded, size=target_size, mode="bilinear", align_corners=False
                )

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
        self.frame_count = 0
        decoded_frames = []

        for encoded in encoded_frames:
            decoded = self.decode_frame(encoded, target_size)
            decoded_frames.append(decoded)

            q = torch.from_numpy(encoded.quantized).unsqueeze(0).to(self.device)
            is_i_frame = encoded.frame_type == "I"

            if is_i_frame:
                decoded_latent = q
            else:
                pred_mean, _ = self.codec.predictor(
                    self.context[-self.codec.predictor.context_len :]
                )
                decoded_latent = pred_mean + q

            self.context.append(decoded_latent.detach())
            if len(self.context) > self.codec.predictor.context_len:
                self.context.pop(0)
            self.frame_count += 1

        return decoded_frames

    def get_stats(self) -> EncodingStats:
        """Get encoding statistics."""
        total_frames = len(self.encoded_frames)
        if total_frames == 0:
            return EncodingStats(
                total_frames=0,
                i_frames=0,
                p_frames=0,
                total_bits=0,
                total_bytes=0,
                encoding_time_s=0,
                avg_bpp=0,
                fps=0,
            )

        i_frames = sum(1 for f in self.encoded_frames if f.frame_type == "I")
        p_frames = sum(1 for f in self.encoded_frames if f.frame_type == "P")
        total_time = sum(f.encoding_time_ms for f in self.encoded_frames) / 1000
        fps = total_frames / total_time if total_time > 0 else 0

        total_pixels = total_frames * 3 * RESOLUTION * RESOLUTION
        avg_bpp = self.total_bits / total_pixels if total_pixels > 0 else 0

        return EncodingStats(
            total_frames=total_frames,
            i_frames=i_frames,
            p_frames=p_frames,
            total_bits=self.total_bits,
            total_bytes=self.total_bits / 8,
            encoding_time_s=total_time,
            avg_bpp=avg_bpp,
            fps=fps,
        )

    def reset(self):
        """Reset encoder state."""
        self.context = []
        self.encoded_frames = []
        self.total_bits = 0.0
        self.frame_count = 0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))
