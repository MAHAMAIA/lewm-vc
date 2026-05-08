"""
Unit tests for Quantization Module

Tests quantizer with straight-through estimator,
different modes, and QAT wrapper stubs.
"""

import pytest
import torch

from src.lewm_vc.quant import (
    AIMETQuantStub,
    NNCFQuantStub,
    QuantizedTensor,
    Quantizer,
    QuantMode,
    quantize_tensor,
)


class TestQuantizer:
    """Test suite for Quantizer class."""

    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance."""
        return Quantizer(num_levels=256, mode=QuantMode.TRAINING)

    def test_initialization(self, quantizer):
        """Test quantizer initializes correctly."""
        assert quantizer.num_levels == 256
        assert quantizer.mode == QuantMode.TRAINING

    def test_forward_shape(self, quantizer):
        """Test forward pass preserves shape."""
        x = torch.randn(2, 192, 16, 16)

        x_quantized = quantizer(x)

        assert x_quantized.shape == x.shape

    def test_training_mode_adds_noise(self, quantizer):
        """Test training mode adds noise."""
        x = torch.zeros(1, 192, 16, 16)

        x_quant1 = quantizer(x)
        x_quant2 = quantizer(x)

        assert not torch.allclose(x_quant1, x_quant2)

    def test_inference_mode_deterministic(self):
        """Test inference mode is deterministic."""
        quantizer = Quantizer(num_levels=256, mode=QuantMode.INFERENCE)
        x = torch.randn(1, 192, 16, 16)

        x_quant1 = quantizer(x)
        x_quant2 = quantizer(x)

        assert torch.allclose(x_quant1, x_quant2)

    def test_quantization_bounded(self, quantizer):
        """Test quantized values are within expected range."""
        x = torch.randn(1, 192, 16, 16) * 10

        x_quantized = quantizer(x)

        max_val = quantizer.step_size * (quantizer.num_levels / 2)
        assert torch.all(x_quantized >= -max_val)
        assert torch.all(x_quantized <= max_val)

    def test_differentiable_ste(self, quantizer):
        """Test straight-through estimator is differentiable."""
        x = torch.randn(1, 192, 16, 16, requires_grad=True)

        x_quantized = quantizer(x)

        loss = x_quantized.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_set_mode(self, quantizer):
        """Test setting quantization mode."""
        quantizer.set_mode(QuantMode.INFERENCE)
        assert quantizer.mode == QuantMode.INFERENCE

        quantizer.set_mode(QuantMode.TRAINING)
        assert quantizer.mode == QuantMode.TRAINING

    def test_get_num_bits(self, quantizer):
        """Test getting number of bits."""
        assert quantizer.get_num_bits() == 8

    @pytest.mark.parametrize("num_levels", [128, 256, 512, 1024])
    def test_different_num_levels(self, num_levels):
        """Test quantizer with different number of levels."""
        quantizer = Quantizer(num_levels=num_levels)
        x = torch.randn(1, 192, 16, 16)

        x_quantized = quantizer(x)

        assert x_quantized.shape == x.shape
        assert quantizer.get_num_bits() == int(torch.log2(torch.tensor(num_levels)).item())

    def test_qat_wrapper_aimet(self):
        """Test AIMET QAT wrapper stub."""
        quantizer = Quantizer(num_levels=256, qat_wrapper="aimet")
        x = torch.randn(1, 192, 16, 16)

        x_quantized = quantizer(x)

        assert x_quantized.shape == x.shape

    def test_qat_wrapper_nncf(self):
        """Test NNCF QAT wrapper stub."""
        quantizer = Quantizer(num_levels=256, qat_wrapper="nncf")
        x = torch.randn(1, 192, 16, 16)

        x_quantized = quantizer(x)

        assert x_quantized.shape == x.shape


class TestAIMETQuantStub:
    """Test suite for AIMET quantizer stub."""

    def test_initialization(self):
        """Test AIMET stub initializes."""
        stub = AIMETQuantStub(bitwidth=8)
        assert stub.bitwidth == 8

    def test_forward_passthrough(self):
        """Test stub passes through input."""
        stub = AIMETQuantStub(bitwidth=8)
        x = torch.randn(1, 192, 16, 16)

        output = stub(x)

        assert torch.equal(output, x)


class TestNNCFQuantStub:
    """Test suite for NNCF quantizer stub."""

    def test_initialization(self):
        """Test NNCF stub initializes."""
        stub = NNCFQuantStub(bitwidth=8)
        assert stub.bitwidth == 8

    def test_forward_passthrough(self):
        """Test stub passes through input."""
        stub = NNCFQuantStub(bitwidth=8)
        x = torch.randn(1, 192, 16, 16)

        output = stub(x)

        assert torch.equal(output, x)


class TestQuantizedTensor:
    """Test suite for QuantizedTensor container."""

    def test_initialization(self):
        """Test quantized tensor initializes."""
        data = torch.zeros(1, 192, 16, 16, dtype=torch.int32)
        scale = torch.tensor(0.1)

        qt = QuantizedTensor(data, scale)

        assert qt.data.shape == data.shape
        assert qt.scale == scale

    def test_dequantize(self):
        """Test dequantization."""
        data = torch.tensor([[[[5]]]]).int()
        scale = torch.tensor(0.1)

        qt = QuantizedTensor(data, scale)
        dequantized = qt.dequantize()

        expected = torch.tensor([[[[0.5]]]])
        assert torch.allclose(dequantized, expected, atol=1e-5)


class TestQuantizeTensor:
    """Test suite for functional quantize_tensor."""

    def test_basic_quantization(self):
        """Test basic tensor quantization."""
        x = torch.randn(1, 192, 16, 16)

        qt = quantize_tensor(x, num_bits=8)

        assert isinstance(qt, QuantizedTensor)

    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        x = torch.randn(1, 192, 16, 16)

        qt = quantize_tensor(x, num_bits=8, per_channel=True)

        assert isinstance(qt, QuantizedTensor)

    def test_dequantize_roundtrip(self):
        """Test dequantization produces reasonable output."""
        x = torch.randn(1, 3, 8, 8)

        qt = quantize_tensor(x, num_bits=8)
        x_dequant = qt.dequantize()

        assert x_dequant.shape == x.shape


class TestQuantMode:
    """Test suite for QuantMode enum."""

    def test_mode_values(self):
        """Test mode enum values."""
        assert QuantMode.TRAINING.value == "training"
        assert QuantMode.INFERENCE.value == "inference"
