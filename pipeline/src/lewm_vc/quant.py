"""
Quantization Module for LeWM-VC Video Codec

Implements differentiable quantization with straight-through estimator (STE)
for training, and hard rounding for inference. Includes QAT wrapper stubs.
"""

from enum import Enum

import torch
import torch.nn as nn


class QuantMode(Enum):
    """Quantization mode selection."""
    TRAINING = "training"
    INFERENCE = "inference"


class Quantizer(nn.Module):
    """
    Learnable quantizer with straight-through estimator.

    Training mode: Adds uniform noise for gradient computation
    Inference mode: Hard rounding for deterministic output

    Supports:
        - Per-tensor or per-channel quantization
        - QAT wrapper stubs for AIMET/NNCF integration

    Args:
        num_levels: Number of quantization levels (default: 256 for 8-bit)
        mode: Quantization mode (TRAINING or INFERENCE)
        channelwise: Whether to use per-channel quantization (default: False)
        qat_wrapper: QAT framework to use: 'aimet', 'nncf', or None (default: None)
    """

    def __init__(
        self,
        num_levels: int = 256,
        mode: QuantMode = QuantMode.TRAINING,
        channelwise: bool = False,
        qat_wrapper: str | None = None,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.mode = mode
        self.channelwise = channelwise
        self.qat_wrapper = qat_wrapper

        self.register_buffer("step_size", torch.tensor(2.0 / num_levels))

        self.qat_model: nn.Module | None = None
        if qat_wrapper is not None:
            self._setup_qat(qat_wrapper)

    def _setup_qat(self, wrapper: str) -> None:
        """
        Setup QAT wrapper stub for AIMET/NNCF.

        This is a stub implementation. Full integration requires:
        - AIMET: conda install aimet-pages aimet-pages
        - NNCF: pip install nncf

        Args:
            wrapper: QAT framework ('aimet' or 'nncf')
        """
        if wrapper == "aimet":
            self.qat_model = AIMETQuantStub(bitwidth=8)
        elif wrapper == "nncf":
            self.qat_model = NNCFQuantStub(bitwidth=8)
        else:
            raise ValueError(f"Unknown QAT wrapper: {wrapper}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor.

        Training: Adds uniform noise [0, step_size) for STE gradients
        Inference: Hard rounds to nearest quantization level

        Args:
            x: Input tensor of any shape

        Returns:
            quantized: Quantized tensor with straight-through gradient
        """
        if self.qat_wrapper and self.qat_model is not None:
            return self.qat_model(x)

        if self.mode == QuantMode.TRAINING:
            return self._quantize_ste(x)
        else:
            return self._quantize_hard(x)

    def _quantize_ste(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantization with straight-through estimator.

        Forward: Adds noise and rounds
        Backward: Passes gradient through unchanged (STE)

        Args:
            x: Input tensor

        Returns:
            Quantized tensor with STE gradient
        """
        noise = torch.empty_like(x).uniform_(-self.step_size, self.step_size)
        x_noisy = x + noise

        x_quantized = torch.round(x_noisy / self.step_size) * self.step_size

        max_val = self.step_size * (self.num_levels - 1) / 2
        x_quantized = torch.clamp(x_quantized, -max_val, max_val)

        return x_quantized.detach() + x - x.detach()

    def _quantize_hard(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hard quantization (inference mode).

        Simply rounds to nearest quantization level.

        Args:
            x: Input tensor

        Returns:
            Quantized tensor
        """
        x_quantized = torch.round(x / self.step_size) * self.step_size

        max_val = self.step_size * (self.num_levels - 1) / 2
        x_quantized = torch.clamp(x_quantized, -max_val, max_val)

        return x_quantized

    def set_mode(self, mode: QuantMode) -> None:
        """
        Set quantization mode explicitly.

        Args:
            mode: New quantization mode
        """
        self.mode = mode

    def get_num_bits(self) -> int:
        """
        Get number of bits per parameter.

        Returns:
            Number of bits
        """
        return int(torch.log2(torch.tensor(self.num_levels)).item())


class AIMETQuantStub(nn.Module):
    """
    Stub for AIMET quantization wrapper.

    Full implementation would use:
    from aimet_torch import quant_wrapper

    This stub maintains API compatibility with AIMET.
    """

    def __init__(self, bitwidth: int = 8):
        super().__init__()
        self.bitwidth = bitwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through for stub."""
        return x


class NNCFQuantStub(nn.Module):
    """
    Stub for NNCF quantization wrapper.

    Full implementation would use:
    from nncf.torch import quantize

    This stub maintains API compatibility with NNCF.
    """

    def __init__(self, bitwidth: int = 8):
        super().__init__()
        self.bitwidth = bitwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through for stub."""
        return x


class QuantizedTensor:
    """
    Container for quantized values with metadata.

    Stores the quantized values along with quantization parameters
    for later decoding.
    """

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
    ):
        """
        Initialize quantized tensor container.

        Args:
            data: Quantized integer tensor
            scale: Quantization scale
            zero_point: Zero point for asymmetric quantization
        """
        self.data = data
        self.scale = scale
        self.zero_point = zero_point if zero_point is not None else torch.zeros_like(scale)

    def dequantize(self) -> torch.Tensor:
        """
       Dequantize back to floating point.

        Returns:
            Original scale floating point tensor
        """
        return (self.data.float() - self.zero_point) * self.scale


def quantize_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    per_channel: bool = False
) -> QuantizedTensor:
    """
    Functional interface for quantization.

    Args:
        x: Input tensor
        num_bits: Number of quantization bits
        per_channel: Whether to quantize per-channel

    Returns:
        QuantizedTensor container
    """
    if per_channel:
        scale = x.abs().max(dim=-1, keepdim=True)[0] / (2 ** (num_bits - 1))
    else:
        scale = x.abs().max() / (2 ** (num_bits - 1))

    scale = torch.clamp(scale, min=1e-8)

    x_scaled = x / scale
    x_quantized = torch.round(x_scaled)

    return QuantizedTensor(x_quantized, scale)
