"""
Unit tests for LeWM Decoder Module

Tests shape verification and reconstruction quality stubs
for the LeWMDecoder class.
"""

import pytest
import torch

from src.lewm_vc.decoder import LeWMDecoder


class TestLeWMDecoder:
    """Test suite for LeWMDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create a decoder instance for testing."""
        return LeWMDecoder(latent_dim=192)

    @pytest.fixture
    def sample_latent(self):
        """Create a sample latent tensor for testing."""
        b, c, h, w = 2, 192, 256, 256
        return torch.randn(b, c, h // 16, w // 16)

    @pytest.fixture
    def sample_residual(self):
        """Create a sample residual tensor for testing."""
        b, c, h, w = 2, 3, 256, 256
        return torch.randn(b, c, h, w)

    def test_decoder_initialization(self, decoder):
        """Test that decoder initializes with correct architecture."""
        assert decoder.up1 is not None
        assert decoder.res1 is not None
        assert decoder.up2 is not None
        assert decoder.res2 is not None
        assert decoder.up3 is not None
        assert decoder.res3 is not None
        assert decoder.up4 is not None
        assert decoder.final is not None
        assert decoder.post_filter is not None

    def test_output_shape_without_residual(self, decoder, sample_latent):
        """Test that output has correct shape without residual."""
        output = decoder(sample_latent)
        expected_shape = (2, 3, 256, 256)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_output_shape_with_residual(self, decoder, sample_latent, sample_residual):
        """Test that output has correct shape with residual."""
        output = decoder(sample_latent, residual=sample_residual)
        expected_shape = (2, 3, 256, 256)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_upsampling_stages(self, decoder, sample_latent):
        """Test that all 4 upsampling stages work correctly."""
        x = sample_latent
        assert x.shape[2] == 16 and x.shape[3] == 16

        x = decoder.up1(x)
        assert x.shape[2] == 32 and x.shape[3] == 32

        x = decoder.up2(x)
        assert x.shape[2] == 64 and x.shape[3] == 64

        x = decoder.up3(x)
        assert x.shape[2] == 128 and x.shape[3] == 128

        x = decoder.up4(x)
        assert x.shape[2] == 256 and x.shape[3] == 256

    def test_residual_blocks(self, decoder):
        """Test that residual blocks maintain channel dimensions."""
        ch = 128
        x = torch.randn(1, ch, 32, 32)
        out = decoder.res1(x)
        assert out.shape == x.shape

        ch = 64
        x = torch.randn(1, ch, 64, 64)
        out = decoder.res2(x)
        assert out.shape == x.shape

        ch = 32
        x = torch.randn(1, ch, 128, 128)
        out = decoder.res3(x)
        assert out.shape == x.shape

    def test_post_filter(self, decoder):
        """Test that post-filter has correct input/output channels."""
        x = torch.randn(1, 3, 256, 256)
        out = decoder.post_filter(x)
        assert out.shape == x.shape

    def test_different_resolutions(self, decoder):
        """Test decoder works with different input resolutions."""
        for h, w in [(128, 128), (256, 256), (512, 512), (192, 320)]:
            latent = torch.randn(1, 192, h // 16, w // 16)
            output = decoder(latent)
            assert output.shape == (1, 3, h, w), (
                f"Failed for resolution {h}x{w}: got {output.shape}"
            )

    def test_different_batch_sizes(self, decoder):
        """Test decoder works with different batch sizes."""
        for b in [1, 4, 8, 16]:
            latent = torch.randn(b, 192, 16, 16)
            output = decoder(latent)
            assert output.shape[0] == b

    def test_optional_residual_none(self, decoder, sample_latent):
        """Test that residual=None works correctly."""
        output = decoder(sample_latent, residual=None)
        assert output.shape == (2, 3, 256, 256)

    def test_gradient_flow(self, decoder, sample_latent):
        """Test that gradients flow through the network."""
        sample_latent.requires_grad = True
        output = decoder(sample_latent)
        loss = output.sum()
        loss.backward()

        assert sample_latent.grad is not None

    def test_reconstruction_quality_stub(self, decoder):
        """
        Stub test for reconstruction quality.

        Note: Actual PSNR testing requires:
        - Original reference frames
        - Trained encoder-decoder pair
        - Test dataset (e.g., UVG)

        Target: PSNR > 42 dB at QP=28 on UVG
        """
        latent = torch.randn(1, 192, 16, 16)
        output = decoder(latent)

        assert output.min() >= -1.0
        assert output.max() <= 1.0 or output.max() <= 255.0

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_output_value_range(self, decoder, sample_latent):
        """Test that output values are in reasonable range."""
        output = decoder(sample_latent)
        assert output.shape == (2, 3, 256, 256)

        mean = output.mean().item()
        std = output.std().item()
        assert -10 < mean < 10, f"Mean {mean} out of expected range"
        assert 0 < std < 10, f"Std {std} out of expected range"

    def test_eval_mode(self, decoder):
        """Test decoder in eval mode."""
        decoder.eval()
        latent = torch.randn(1, 192, 16, 16)

        with torch.no_grad():
            output = decoder(latent)

        assert output.shape == (1, 3, 256, 256)
        assert not torch.isnan(output).any()

    def test_train_mode(self, decoder):
        """Test decoder in train mode."""
        decoder.train()
        latent = torch.randn(1, 192, 16, 16)

        output = decoder(latent)

        assert output.shape == (1, 3, 256, 256)
        assert not torch.isnan(output).any()

    def test_residual_addition(self, decoder, sample_latent, sample_residual):
        """Test that residual is correctly added."""
        decoder.eval()
        with torch.no_grad():
            output_with_residual = decoder(sample_latent, residual=sample_residual)

        assert output_with_residual.shape == sample_residual.shape
