"""
Unit tests for LeWM Predictor Module

Tests Gaussian output validation, shape checks,
and temporal prediction for the LeWMPredictor class.
"""

import pytest
import torch

from src.lewm_vc.predictor import LeWMPredictor


class TestLeWMPredictor:
    """Test suite for LeWMPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing."""
        return LeWMPredictor(latent_dim=192, hidden_dim=256, num_layers=8, num_heads=4)

    @pytest.fixture
    def sample_context(self):
        """Create a sample context of latents."""
        b, c, h, w = 2, 192, 16, 16
        return [torch.randn(b, c, h, w) for _ in range(4)]

    @pytest.fixture
    def single_context(self):
        """Create a single latent context."""
        b, c, h, w = 2, 192, 16, 16
        return [torch.randn(b, c, h, w)]

    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes with correct architecture."""
        assert predictor.input_proj is not None
        assert predictor.frame_tokens is not None
        assert predictor.transformer is not None
        assert predictor.temporal_norm is not None
        assert predictor.mean_head is not None
        assert predictor.log_std_head is not None

    def test_output_shapes(self, predictor, sample_context):
        """Test that mean and std have correct shapes."""
        mean, std = predictor(sample_context)
        expected_shape = (2, 192, 16, 16)
        assert mean.shape == expected_shape, f"Expected mean {expected_shape}, got {mean.shape}"
        assert std.shape == expected_shape, f"Expected std {expected_shape}, got {std.shape}"

    def test_std_positive(self, predictor, sample_context):
        """Test that std is always positive."""
        mean, std = predictor(sample_context)
        assert (std > 0).all(), "Standard deviation must be positive"

    def test_std_reasonable_range(self, predictor, sample_context):
        """Test that std is in reasonable range."""
        mean, std = predictor(sample_context)
        assert (std > 0.01).all(), "Standard deviation too small"
        assert (std < 10).all(), "Standard deviation too large"

    def test_different_context_lengths(self, predictor):
        """Test predictor with different context lengths."""
        b, c, h, w = 2, 192, 16, 16

        for ctx_len in [1, 2, 3, 4]:
            context = [torch.randn(b, c, h, w) for _ in range(ctx_len)]
            mean, std = predictor(context)
            assert mean.shape == (b, c, h, w)
            assert std.shape == (b, c, h, w)

    def test_context_length_limit(self, predictor):
        """Test that context length exceeding limit raises error."""
        b, c, h, w = 2, 192, 16, 16
        context = [torch.randn(b, c, h, w) for _ in range(5)]

        with pytest.raises(ValueError):
            predictor(context)

    def test_empty_context_error(self, predictor):
        """Test that empty context raises error."""
        with pytest.raises(ValueError):
            predictor([])

    def test_gaussian_output_validation(self, predictor, sample_context):
        """Test that output forms valid isotropic Gaussian."""
        mean, std = predictor(sample_context)

        assert not torch.isnan(mean).any(), "Mean contains NaN"
        assert not torch.isnan(std).any(), "Std contains NaN"
        assert not torch.isinf(mean).any(), "Mean contains Inf"
        assert not torch.isinf(std).any(), "Std contains Inf"

        assert (std > 0).all(), "Std must be positive"

    def test_mean_differs_from_input(self, predictor, sample_context):
        """Test that predicted mean differs from context."""
        mean, std = predictor(sample_context)

        for ctx_tensor in sample_context:
            diff = (mean - ctx_tensor).abs().mean()
            assert diff > 0.01, "Mean too similar to context"

    def test_predict_deterministic_mode(self, predictor, sample_context):
        """Test predict function without sampling."""
        predictor.eval()
        pred1 = predictor.predict(sample_context, sample=False)
        pred2 = predictor.predict(sample_context, sample=False)

        assert torch.allclose(pred1, pred2), "Deterministic prediction should be identical"

    def test_predict_sampling_mode(self, predictor, sample_context):
        """Test predict function with sampling."""
        pred = predictor.predict(sample_context, sample=True)

        assert pred.shape == (2, 192, 16, 16)
        assert not torch.isnan(pred).any()

    def test_nll_loss_computation(self, predictor, sample_context):
        """Test NLL loss computation."""
        b, c, h, w = 2, 192, 16, 16
        target = torch.randn(b, c, h, w)

        loss = predictor.nll_loss(sample_context, target)

        assert isinstance(loss.item(), float)
        assert loss.item() > 0, "NLL should be positive"
        assert not torch.isnan(loss), "NLL contains NaN"

    def test_gradient_flow(self, predictor, sample_context):
        """Test that gradients flow through the network."""
        for ctx_tensor in sample_context:
            ctx_tensor.requires_grad = True

        mean, std = predictor(sample_context)
        loss = mean.sum() + std.sum()
        loss.backward()

        for ctx_tensor in sample_context:
            assert ctx_tensor.grad is not None

    def test_gradient_flow_predict(self, predictor, sample_context):
        """Test gradient flow through predict function."""
        predictor.train()
        pred = predictor.predict(sample_context, sample=True)
        loss = pred.sum()
        loss.backward()

        assert predictor.input_proj.weight.grad is not None
        assert predictor.frame_tokens.grad is not None

    def test_different_batch_sizes(self, predictor):
        """Test predictor with different batch sizes."""
        c, h, w = 192, 16, 16
        for b in [1, 4, 8]:
            context = [torch.randn(b, c, h, w)]
            mean, std = predictor(context)
            assert mean.shape[0] == b
            assert std.shape[0] == b

    def test_different_resolutions(self, predictor):
        """Test predictor with different spatial resolutions."""
        b, c = 2, 192
        for h, w in [(8, 8), (16, 16), (32, 32), (12, 20)]:
            context = [torch.randn(b, c, h, w)]
            mean, std = predictor(context)
            assert mean.shape == (b, c, h, w)
            assert std.shape == (b, c, h, w)

    def test_eval_mode(self, predictor, sample_context):
        """Test predictor in eval mode."""
        predictor.eval()
        with torch.no_grad():
            mean, std = predictor(sample_context)

        assert mean.shape == (2, 192, 16, 16)
        assert std.shape == (2, 192, 16, 16)
        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

    def test_train_mode(self, predictor, sample_context):
        """Test predictor in train mode."""
        predictor.train()
        mean, std = predictor(sample_context)

        assert mean.shape == (2, 192, 16, 16)
        assert std.shape == (2, 192, 16, 16)
        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

    def test_single_context_element(self, predictor, single_context):
        """Test predictor with single context element."""
        mean, std = predictor(single_context)
        assert mean.shape == (2, 192, 16, 16)
        assert std.shape == (2, 192, 16, 16)

    def test_log_std_clamping(self, predictor, sample_context):
        """Test that log_std is properly clamped."""
        mean, std = predictor(sample_context)

        log_std = torch.log(std)
        assert (log_std > -10).all(), "log_std underflow"
        assert (log_std < 2).all(), "log_std overflow"

    def test_reproducibility_with_seed(self, predictor, sample_context):
        """Test that same seed gives same results for mean."""
        predictor.eval()
        torch.manual_seed(42)
        mean1, std1 = predictor.predict(sample_context, sample=False)

        torch.manual_seed(42)
        mean2, std2 = predictor.predict(sample_context, sample=False)

        assert torch.allclose(mean1, mean2), "Mean should be reproducible"
        assert torch.allclose(std1, std2), "Std should be reproducible"


class TestLeWMPredictorIntegration:
    """Integration tests for predictor with encoder."""

    def test_encoder_predictor_roundtrip(self):
        """Test encoding then predicting."""
        from src.lewm_vc.decoder import LeWMDecoder
        from src.lewm_vc.encoder import LeWMEncoder

        encoder = LeWMEncoder(latent_dim=192)
        predictor = LeWMPredictor(latent_dim=192)
        decoder = LeWMDecoder(latent_dim=192)

        encoder.eval()
        predictor.eval()
        decoder.eval()

        x = torch.rand(1, 3, 256, 256)

        with torch.no_grad():
            latent = encoder(x)

            context = [latent]
            pred_mean, pred_std = predictor(context)

            sampled = predictor.predict(context, sample=True)

            reconstructed = decoder(sampled)

        assert reconstructed.shape == x.shape
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()

    def test_multi_frame_prediction(self):
        """Test predicting multiple frames in sequence."""
        from src.lewm_vc.encoder import LeWMEncoder

        encoder = LeWMEncoder(latent_dim=192)
        predictor = LeWMPredictor(latent_dim=192)

        encoder.eval()
        predictor.eval()

        frames = [torch.rand(1, 3, 256, 256) for _ in range(5)]

        with torch.no_grad():
            latents = [encoder(f) for f in frames]

            context = [latents[0]]
            pred1, _ = predictor(context)

            context = [latents[0], latents[1]]
            pred2, _ = predictor(context)

            context = latents[:3]
            pred3, _ = predictor(context)

        assert pred1.shape == pred2.shape == pred3.shape == (1, 192, 16, 16)
