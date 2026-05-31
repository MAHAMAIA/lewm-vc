"""
Scalable Video Coding (SVC) for LeWM-VC.

Dual-layer architecture:
  - Base Layer (BL): 64-ch latent, aggressively quantized, streamed continuously to cloud
  - Enhancement Layer (EL): 128-ch residual, lightly quantized, stored on edge NVMe

The existing LeWMDecoder (192-ch input) reconstructs full-quality frames when both layers are
fused. For AI inference, only BL is needed — zero-pad from 64 to 192 and decode at lower quality.

Usage:
    splitter = LatentSplitter()
    bl, el = splitter(latent)           # [B,192,H,W] -> [B,64,H,W], [B,128,H,W]

    fuser = LatentFuser()
    full = fuser(bl, el)                # [B,64,H,W] + [B,128,H,W] -> [B,192,H,W]

    quant = MultiRateQuantizer()
    bl_q = quant(bl, num_bits=4)        # aggressive
    el_q = quant(el, num_bits=8)        # standard

    decoder = SVCDecoder(base_decoder)
    ai_frame = decoder.decode_bl(bl)    # zero-padded, AI quality
    hq_frame = decoder.decode_full(bl, el)  # full quality, human-viewable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class LatentSplitter(nn.Module):
    """
    Splits a 192-channel latent into Base Layer (64ch) and Enhancement Layer (128ch).

    The split is channel-wise: first 64 channels go to BL, remaining 128 to EL.
    An optional learned projection can be trained end-to-end for optimal split.

    Args:
        latent_dim: Full latent dimension (default: 192)
        base_dim: Base layer dimension (default: 64)
        use_learned_split: Whether to use learned projections (default: False)
    """

    def __init__(
        self,
        latent_dim: int = 192,
        base_dim: int = 64,
        use_learned_split: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.enhance_dim = latent_dim - base_dim

        if use_learned_split:
            self.bl_proj = nn.Conv2d(latent_dim, base_dim, kernel_size=1)
            self.el_proj = nn.Conv2d(latent_dim, self.enhance_dim, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split latent into base and enhancement layers.

        Args:
            latent: [B, latent_dim, H, W] latent tensor

        Returns:
            bl: [B, base_dim, H, W] base layer
            el: [B, enh_dim, H, W] enhancement layer
        """
        if hasattr(self, "bl_proj"):
            bl = self.bl_proj(latent)
            el = self.el_proj(latent)
        else:
            bl = latent[:, : self.base_dim]
            el = latent[:, self.base_dim :]

        return bl, el


class LatentFuser(nn.Module):
    """
    Fuses Base Layer and Enhancement Layer back into full 192-ch latent.

    Supports zero-pad mode (for AI-only inference) and full fusion
    (for human-viewable reconstruction).

    Args:
        latent_dim: Full latent dimension (default: 192)
        base_dim: Base layer dimension (default: 64)
        use_learned_fusion: Whether to use learned fusion (default: False)
    """

    def __init__(
        self,
        latent_dim: int = 192,
        base_dim: int = 64,
        use_learned_fusion: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_dim = base_dim

        if use_learned_fusion:
            self.fusion = nn.Conv2d(base_dim + (latent_dim - base_dim), latent_dim, kernel_size=1)

    def forward(self, bl: torch.Tensor, el: torch.Tensor | None = None) -> torch.Tensor:
        """
        Fuse layers back to full latent.

        Args:
            bl: [B, base_dim, H, W] base layer
            el: [B, enh_dim, H, W] enhancement layer, or None for zero-pad

        Returns:
            [B, latent_dim, H, W] fused latent
        """
        if el is None:
            pad = self.latent_dim - bl.shape[1]
            return F.pad(bl, (0, 0, 0, 0, 0, pad))

        if hasattr(self, "fusion"):
            return self.fusion(torch.cat([bl, el], dim=1))

        return torch.cat([bl, el], dim=1)


class MultiRateQuantizer(nn.Module):
    """
    Quantizer with per-layer bit-depth control.

    Base Layer gets fewer bits (higher compression, lower quality).
    Enhancement Layer gets more bits (lower compression, higher quality).

    Args:
        num_levels_bl: Quantization levels for base layer (default: 16 = 4-bit)
        num_levels_el: Quantization levels for enhancement layer (default: 256 = 8-bit)
    """

    def __init__(
        self,
        num_levels_bl: int = 16,
        num_levels_el: int = 256,
    ):
        super().__init__()
        self.bl_quant = _LayerQuantizer(num_levels=num_levels_bl)
        self.el_quant = _LayerQuantizer(num_levels=num_levels_el)

    def forward(
        self,
        bl: torch.Tensor,
        el: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Quantize both layers at different bit depths.

        Args:
            bl: Base layer tensor [B, C_bl, H, W]
            el: Enhancement layer tensor [B, C_el, H, W] or None

        Returns:
            bl_q: Quantized base layer
            el_q: Quantized enhancement layer (or None)
        """
        bl_q = self.bl_quant(bl)
        el_q = self.el_quant(el) if el is not None else None
        return bl_q, el_q

    def get_bl_bits_per_element(self) -> int:
        return self.bl_quant.get_num_bits()

    def get_el_bits_per_element(self) -> int:
        return self.el_quant.get_num_bits()

    def get_total_bits_per_frame(self, bl_shape: tuple, el_shape: tuple | None = None) -> int:
        bl_bits = self.bl_quant.get_num_bits() * _num_elements(bl_shape)
        el_bits = 0
        if el_shape is not None:
            el_bits = self.el_quant.get_num_bits() * _num_elements(el_shape)
        return bl_bits + el_bits


class _LayerQuantizer(nn.Module):
    """
    Internal uniform quantizer with straight-through estimator.

    Matches the existing Quantizer API but with configurable bit depth.
    """

    def __init__(self, num_levels: int = 256):
        super().__init__()
        self.num_levels = num_levels
        self.register_buffer("step_size", torch.tensor(2.0 / num_levels))
        self.register_buffer("max_val", torch.tensor((num_levels - 1) / 2 * (2.0 / num_levels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.empty_like(x).uniform_(-self.step_size.item(), self.step_size.item())
            x = x + noise
        x_q = torch.round(x / self.step_size) * self.step_size
        x_q = torch.clamp(x_q, -self.max_val.to(x.device), self.max_val.to(x.device))
        if self.training:
            return x_q.detach() + x - x.detach()
        return x_q

    def get_num_bits(self) -> int:
        return int(torch.log2(torch.tensor(self.num_levels, dtype=torch.float32)).item())


class SVCDecoder(nn.Module):
    """
    SVC-aware decoder wrapping the existing LeWMDecoder.

    Two modes:
      - decode_bl(bl): Zero-pad base layer to 192-ch, pass through base decoder.
        Produces lower-quality frames suitable for AI pipeline consumption.
      - decode_full(bl, el): Fuse both layers, pass through base decoder.
        Produces full-quality frames for human review.

    Args:
        base_decoder: Existing LeWMDecoder instance
        latent_dim: Full latent dimension (default: 192)
        base_dim: Base layer dimension (default: 64)
        fuser: Optional LatentFuser instance (created if None)
    """

    def __init__(
        self,
        base_decoder: nn.Module,
        latent_dim: int = 192,
        base_dim: int = 64,
        fuser: LatentFuser | None = None,
    ):
        super().__init__()
        self.base = base_decoder
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.fuser = fuser or LatentFuser(latent_dim=latent_dim, base_dim=base_dim)

    @torch.no_grad()
    def decode_bl(
        self, bl: torch.Tensor, target_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        Decode base layer only — for AI pipeline consumption.

        Zero-pads BL from {base_dim}ch to {latent_dim}ch before passing to base decoder.
        Quality is degraded but sufficient for machine perception tasks.

        Args:
            bl: [B, base_dim, H, W] base layer latent
            target_size: Optional output resolution

        Returns:
            [B, 3, H_out, W_out] RGB frame in [0, 1]
        """
        full = self.fuser(bl, el=None)
        return self.base(full, target_size)

    @torch.no_grad()
    def decode_full(
        self,
        bl: torch.Tensor,
        el: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Decode base + enhancement layers — for human-viewable reconstruction.

        Fuses BL and EL into full 192-ch latent, passes through base decoder.

        Args:
            bl: [B, base_dim, H, W] base layer latent
            el: [B, enh_dim, H, W] enhancement layer latent
            target_size: Optional output resolution

        Returns:
            [B, 3, H_out, W_out] RGB frame in [0, 1]
        """
        full = self.fuser(bl, el)
        return self.base(full, target_size)


class SVCEncoder:
    """
    High-level SVC encoding controller.

    Manages the split → quantize → dispatch pipeline:
      - Continuous stream: BL only → cloud
      - On-demand: BL + EL → local storage (NVMe ring buffer)

    Args:
        splitter: LatentSplitter instance
        quantizer: MultiRateQuantizer instance
        base_dim: Base layer channel count
    """

    def __init__(
        self,
        splitter: LatentSplitter | None = None,
        quantizer: MultiRateQuantizer | None = None,
        base_dim: int = 64,
    ):
        self.splitter = splitter or LatentSplitter(base_dim=base_dim)
        self.quantizer = quantizer or MultiRateQuantizer()
        self.base_dim = base_dim

    def encode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode a full latent into SVC layers.

        Args:
            latent: [B, 192, H, W] full latent from encoder

        Returns:
            bl_q: Quantized base layer [B, base_dim, H, W] — send to cloud
            el_q: Quantized enhancement layer [B, 128, H, W] — store locally
        """
        bl, el = self.splitter(latent)
        bl_q, el_q = self.quantizer(bl, el)
        return bl_q, el_q

    def get_bl_bit_budget(self, bl_shape: tuple) -> int:
        """Get bit budget for one base layer frame."""
        bits_per_elem = self.quantizer.get_bl_bits_per_element()
        return bits_per_elem * _num_elements(bl_shape)

    def get_el_bit_budget(self, el_shape: tuple) -> int:
        """Get bit budget for one enhancement layer frame."""
        bits_per_elem = self.quantizer.get_el_bits_per_element()
        return bits_per_elem * _num_elements(el_shape)

    def get_total_bit_budget(self, bl_shape: tuple, el_shape: tuple) -> int:
        return self.quantizer.get_total_bits_per_frame(bl_shape, el_shape)


def _num_elements(shape: tuple) -> int:
    result = 1
    for d in shape:
        result *= d
    return result
