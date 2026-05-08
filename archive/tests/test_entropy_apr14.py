"""
Unit tests for Entropy Model Module

Tests hyperprior entropy model, Gaussian KL computation,
and rate estimation.
"""

import pytest
import torch

from src.lewm_vc.entropy import HyperpriorEntropy


class TestHyperpriorEntropy:
    """Test suite for HyperpriorEntropy class."""

    @pytest.fixture
    def entropy_model(self):
        """Create entropy model instance."""
        return HyperpriorEntropy(latent_dim=192, hyper_channels=320)

    def test_initialization(self, entropy_model):
        """Test model initializes correctly."""
        assert entropy_model.latent_dim == 192
        assert entropy_model.hyper_channels == 320

    def test_forward_shape(self, entropy_model):
        """Test forward pass produces correct shapes."""
        batch_size = 2
        height, width = 16, 16

        residual = torch.randn(batch_size, 192, height, width)

        rate, context = entropy_model(residual)

        assert rate.shape == (batch_size, 1, height, width)
        assert "mu" in context
        assert "sigma" in context
        assert context["mu"].shape == residual.shape
        assert context["sigma"].shape == residual.shape

    def test_gaussian_kl_positive(self, entropy_model):
        """Test KL divergence is always non-negative."""
        residual = torch.randn(1, 192, 16, 16)

        rate, _ = entropy_model(residual)

        assert torch.all(rate >= 0)

    def test_differentiable(self, entropy_model):
        """Test model is differentiable."""
        residual = torch.randn(1, 192, 16, 16, requires_grad=True)

        rate, _ = entropy_model(residual)

        loss = rate.sum()
        loss.backward()

        assert residual.grad is not None
        assert not torch.isnan(residual.grad).any()

    def test_sigma_positive(self, entropy_model):
        """Test sigma is always positive."""
        residual = torch.randn(1, 192, 16, 16)

        _, context = entropy_model(residual)

        assert torch.all(context["sigma"] > 0)

    def test_get_entropy_parameters(self, entropy_model):
        """Test getting entropy parameters without rate."""
        residual = torch.randn(1, 192, 16, 16)

        mu, sigma = entropy_model.get_entropy_parameters(residual)

        assert mu.shape == residual.shape
        assert sigma.shape == residual.shape
        assert torch.all(sigma > 0)

    def test_rate_scales_with_batch(self, entropy_model):
        """Test rate estimation scales with batch size."""
        residual_1 = torch.randn(1, 192, 16, 16)
        residual_2 = torch.randn(2, 192, 16, 16)

        rate_1, _ = entropy_model(residual_1)
        rate_2, _ = entropy_model(residual_2)

        assert rate_1.numel() * 2 == rate_2.numel()

    @pytest.mark.parametrize("latent_dim", [64, 128, 192, 256])
    def test_different_latent_dims(self, latent_dim):
        """Test model works with different latent dimensions."""
        model = HyperpriorEntropy(latent_dim=latent_dim)
        residual = torch.randn(1, latent_dim, 16, 16)

        rate, context = model(residual)

        assert context["mu"].shape[1] == latent_dim
        assert context["sigma"].shape[1] == latent_dim

    def test_deterministic(self, entropy_model):
        """Test model is deterministic with same input."""
        residual = torch.randn(1, 192, 16, 16)

        rate1, _ = entropy_model(residual)
        rate2, _ = entropy_model(residual)

        assert torch.allclose(rate1, rate2, atol=1e-6)


class TestGaussianKL:
    """Test suite for Gaussian KL divergence computation."""

    def test_kl_zero_for_standard_normal(self):
        """Test KL is zero when x ~ N(0,1) and mu=0, sigma=1."""
        x = torch.randn(1000)
        mu = torch.zeros_like(x)
        sigma = torch.ones_like(x)

        kl = HyperpriorEntropy.gaussian_kl(x.unsqueeze(0).unsqueeze(0), mu.unsqueeze(0).unsqueeze(0), sigma.unsqueeze(0).unsqueeze(0))

        assert kl.item() < 0.1

    def test_kl_increases_with_sigma(self):
        """Test KL increases when sigma increases."""
        x = torch.zeros(100)
        mu = torch.zeros_like(x)

        sigma_1 = torch.ones_like(x) * 0.5
        sigma_2 = torch.ones_like(x) * 2.0

        kl_1 = HyperpriorEntropy.gaussian_kl(
            x.unsqueeze(0).unsqueeze(0),
            mu.unsqueeze(0).unsqueeze(0),
            sigma_1.unsqueeze(0).unsqueeze(0)
        )
        kl_2 = HyperpriorEntropy.gaussian_kl(
            x.unsqueeze(0).unsqueeze(0),
            mu.unsqueeze(0).unsqueeze(0),
            sigma_2.unsqueeze(0).unsqueeze(0)
        )

        assert kl_2 > kl_1

    def test_kl_increases_with_mu_squared(self):
        """Test KL increases with mu squared."""
        x = torch.zeros(100)
        sigma = torch.ones_like(x)

        mu_1 = torch.zeros_like(x)
        mu_2 = torch.ones_like(x) * 3.0

        kl_1 = HyperpriorEntropy.gaussian_kl(
            x.unsqueeze(0).unsqueeze(0),
            mu_1.unsqueeze(0).unsqueeze(0),
            sigma.unsqueeze(0).unsqueeze(0)
        )
        kl_2 = HyperpriorEntropy.gaussian_kl(
            x.unsqueeze(0).unsqueeze(0),
            mu_2.unsqueeze(0).unsqueeze(0),
            sigma.unsqueeze(0).unsqueeze(0)
        )

        assert kl_2 > kl_1
