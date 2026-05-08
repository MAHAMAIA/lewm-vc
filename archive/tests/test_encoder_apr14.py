"""
Unit tests for LeWM Encoder Module

Tests shape verification, round-trip latent test,
and semantic surprise branch for the LeWMEncoder class.
"""

import pytest
import torch

from src.lewm_vc.encoder import LeWMEncoder, TransformerEncoderLayer


class TestLeWMEncoder:
    """Test suite for LeWMEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create an encoder instance for testing."""
        return LeWMEncoder(latent_dim=192)

    @pytest.fixture
    def encoder_with_surprise(self):
        """Create an encoder instance with surprise detection."""
        return LeWMEncoder(latent_dim=192, semantic_surprise=True)

    @pytest.fixture
    def sample_input(self):
        """Create a sample input tensor for testing."""
        b, c, h, w = 2, 3, 256, 256
        return torch.rand(b, c, h, w)

    def test_encoder_initialization(self, encoder):
        """Test that encoder initializes with correct architecture."""
        assert encoder.patch_embed is not None
        assert encoder.cls_token is not None
        assert encoder.pos_embed is not None
        assert len(encoder.encoder_layers) == 6
        assert encoder.norm is not None
        assert encoder.latent_proj is not None

    def test_output_shape(self, encoder, sample_input):
        """Test that output has correct shape."""
        output = encoder(sample_input)
        expected_shape = (2, 192, 16, 16)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_different_resolutions(self, encoder):
        """Test encoder works with different input resolutions."""
        for h, w in [(128, 128), (256, 256), (512, 512), (192, 320)]:
            x = torch.rand(1, 3, h, w)
            output = encoder(x)
            expected_shape = (1, 192, h // 16, w // 16)
            assert output.shape == expected_shape, (
                f"Failed for resolution {h}x{w}: got {output.shape}"
            )

    def test_different_batch_sizes(self, encoder):
        """Test encoder works with different batch sizes."""
        for b in [1, 4, 8]:
            x = torch.rand(b, 3, 256, 256)
            output = encoder(x)
            assert output.shape[0] == b
            assert output.shape[1] == 192
            assert output.shape[2] == 16
            assert output.shape[3] == 16

    def test_gradient_flow(self, encoder, sample_input):
        """Test that gradients flow through the network."""
        sample_input.requires_grad = True
        output = encoder(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None

    def test_gradient_flow_through_layers(self, encoder, sample_input):
        """Test that gradients propagate through all transformer layers."""
        output = encoder(sample_input)
        loss = output.sum()
        loss.backward()

        assert encoder.patch_embed.weight.grad is not None
        assert encoder.latent_proj.weight.grad is not None

    def test_round_trip_latent(self, encoder, sample_input):
        """Test round-trip: encode then decode."""
        from src.lewm_vc.decoder import LeWMDecoder

        decoder = LeWMDecoder(latent_dim=192)
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            latent = encoder(sample_input)
            reconstructed = decoder(latent)

        assert reconstructed.shape == sample_input.shape

    def test_round_trip_quality_stub(self, encoder, sample_input):
        """
        Stub test for round-trip quality.

        Note: Actual quality testing requires:
        - Trained encoder-decoder pair
        - Test dataset
        - PSNR/SSIM metrics
        """
        from src.lewm_vc.decoder import LeWMDecoder

        encoder.eval()
        decoder = LeWMDecoder(latent_dim=192)
        decoder.eval()

        with torch.no_grad():
            latent = encoder(sample_input)
            reconstructed = decoder(latent)

        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()

    def test_semantic_surprise_branch(self, encoder_with_surprise, sample_input):
        """Test that semantic surprise branch works."""
        encoder_with_surprise.eval()
        with torch.no_grad():
            latent, surprise = encoder_with_surprise(sample_input, return_surprise=True)

        assert latent.shape == (2, 192, 16, 16)
        assert surprise.shape == (2, 1)
        assert (surprise >= 0).all() and (surprise <= 1).all()

    def test_semantic_surprise_training_mode(self, encoder_with_surprise, sample_input):
        """Test surprise output in training mode."""
        encoder_with_surprise.train()
        latent, surprise = encoder_with_surprise(sample_input, return_surprise=True)

        assert latent.shape == (2, 192, 16, 16)
        assert surprise.shape == (2, 1)

    def test_no_surprise_without_flag(self, encoder_with_surprise, sample_input):
        """Test that surprise is not returned without flag."""
        encoder_with_surprise.eval()
        output = encoder_with_surprise(sample_input, return_surprise=False)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 192, 16, 16)

    def test_without_surprise_branch(self, encoder, sample_input):
        """Test encoder without surprise branch."""
        output = encoder(sample_input, return_surprise=False)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 192, 16, 16)

    def test_pos_embed_shape(self, encoder):
        """Test that positional embeddings have correct shape."""
        assert encoder.pos_embed.shape[1] == 1

    def test_transformer_layers_count(self, encoder):
        """Test that correct number of transformer layers are created."""
        assert len(encoder.encoder_layers) == 6

    def test_eval_mode(self, encoder, sample_input):
        """Test encoder in eval mode."""
        encoder.eval()
        with torch.no_grad():
            output = encoder(sample_input)

        assert output.shape == (2, 192, 16, 16)
        assert not torch.isnan(output).any()

    def test_train_mode(self, encoder, sample_input):
        """Test encoder in train mode."""
        encoder.train()
        output = encoder(sample_input)

        assert output.shape == (2, 192, 16, 16)
        assert not torch.isnan(output).any()

    def test_output_value_range(self, encoder, sample_input):
        """Test that output values are in reasonable range."""
        output = encoder(sample_input)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_patch_sizes(self):
        """Test encoder with different patch sizes."""
        for patch_size in [8, 16]:
            encoder = LeWMEncoder(latent_dim=192, patch_size=patch_size)
            x = torch.rand(1, 3, 256, 256)
            output = encoder(x)
            expected_shape = (1, 192, 256 // patch_size, 256 // patch_size)
            assert output.shape == expected_shape


class TestTransformerEncoderLayer:
    """Test suite for TransformerEncoderLayer."""

    @pytest.fixture
    def layer(self):
        """Create a transformer layer for testing."""
        return TransformerEncoderLayer(d_model=192, nhead=3)

    def test_forward(self, layer):
        """Test basic forward pass."""
        b, n, d = 2, 16, 192
        x = torch.randn(b, n, d)
        out = layer(x)
        assert out.shape == x.shape

    def test_pre_norm_architecture(self, layer):
        """Test that pre-norm is used."""
        assert hasattr(layer, 'norm1')
        assert hasattr(layer, 'norm2')

    def test_gradient_flow(self, layer):
        """Test gradient flow through layer."""
        x = torch.randn(2, 16, 192, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None

    def test_different_seq_lengths(self, layer):
        """Test with different sequence lengths."""
        for n in [1, 8, 32, 64]:
            x = torch.randn(1, n, 192)
            out = layer(x)
            assert out.shape == (1, n, 192)
