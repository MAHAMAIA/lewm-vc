"""
Rate Control Module for LeWM-VC Video Codec

Implements learned rate control combining MLP-based complexity estimation
with CRF table lookup for ABR ladder compliance.
Target: bitrate ±3% on ABR ladder per modules.md line 207.
"""

import torch
import torch.nn as nn


class RateController(nn.Module):
    """
    Learned rate controller for LeWM-VC video codec.

    Combines:
    - MLP-based complexity estimation from latent statistics
    - CRF table lookup for ABR ladder compliance
    - Learned lambda prediction for RD optimization

    Architecture:
        - Complexity estimator: 3-layer MLP (256→128→64→1)
        - Lambda predictor: 2-layer MLP from complexity + target_bpp
        - CRF table: Precomputed QP values for standard ABR ladder

    Args:
        latent_dim: Latent dimension (default: 192)
        hidden_dim: Hidden dimension for MLPs (default: 256)
        crf_table: CRF values for ABR ladder tiers (default: standard ladder)
        enable_mlp: Whether to use learned MLP (default: True)
    """

    CRF_TABLE = {
        "1080p": {
            "ultra_low": 28,
            "low": 32,
            "medium": 36,
            "high": 40,
            "ultra_high": 44,
        },
        "720p": {
            "ultra_low": 26,
            "low": 30,
            "medium": 34,
            "high": 38,
            "ultra_high": 42,
        },
        "480p": {
            "ultra_low": 24,
            "low": 28,
            "medium": 32,
            "high": 36,
            "ultra_high": 40,
        },
    }

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 256,
        crf_table: dict | None = None,
        enable_mlp: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.enable_mlp = enable_mlp
        self.crf_table = crf_table if crf_table is not None else self.CRF_TABLE

        if enable_mlp:
            self.complexity_estimator = nn.Sequential(
                nn.Linear(latent_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

            self.lambda_predictor = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Softplus(),
            )

    def select_qp(
        self,
        complexity: float,
        target_bpp: float,
        resolution: str = "1080p",
    ) -> int:
        """
        Select QP value based on complexity and target bits-per-pixel.

        Args:
            complexity: Frame complexity score [0, 1] from latent analysis
            target_bpp: Target bits-per-pixel for the frame
            resolution: Video resolution tier ('1080p', '720p', '480p')

        Returns:
            QP value (integer, typically 0-51)
        """
        if resolution not in self.crf_table:
            resolution = "1080p"

        tier_map = self.crf_table[resolution]

        if target_bpp < 0.05:
            tier = "ultra_low"
        elif target_bpp < 0.1:
            tier = "low"
        elif target_bpp < 0.2:
            tier = "medium"
        elif target_bpp < 0.4:
            tier = "high"
        else:
            tier = "ultra_high"

        base_qp = tier_map[tier]

        if self.enable_mlp and self.training:
            complexity_tensor = torch.tensor(
                [complexity, target_bpp], dtype=torch.float32
            ).unsqueeze(0)
            adjustment = self.lambda_predictor(complexity_tensor)
            qp_adjustment = int(adjustment.item() * 5)
            base_qp = max(0, min(51, base_qp + qp_adjustment))

        if not self.training:
            complexity_factor = 1.0 - (complexity - 0.5) * 0.2
            complexity_factor = max(0.8, min(1.2, complexity_factor))
            base_qp = int(base_qp * complexity_factor)
            base_qp = max(0, min(51, base_qp))

        return base_qp

    def estimate_complexity(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Estimate frame complexity from latent representation.

        Uses statistical features:
        - Mean and std of latent values
        - Energy (L2 norm)
        - Sparsity (zero gradient regions)

        Args:
            latent: Latent tensor [B, latent_dim, H, W]

        Returns:
            Complexity score [B, 1]
        """
        if not self.enable_mlp:
            mean = latent.mean(dim=[1, 2, 3], keepdim=True)
            std = latent.std(dim=[1, 2, 3], keepdim=True)
            energy = latent.pow(2).mean(dim=[1, 2, 3], keepdim=True)
            sparsity = (latent.abs() < 0.01).float().mean(dim=[1, 2, 3], keepdim=True)

            complexity = 0.3 * mean + 0.3 * std + 0.2 * energy + 0.2 * sparsity
            return complexity.clamp(0, 1).squeeze(-1).squeeze(-1)

        b = latent.shape[0]

        mean = latent.mean(dim=[1, 2, 3])
        std = latent.std(dim=[1, 2, 3])
        energy = latent.pow(2).mean(dim=[1, 2, 3])
        max_val = latent.abs().max(dim=1)[0].mean(dim=[1, 2])
        sparsity = (latent.abs() < 0.01).float().mean(dim=[1, 2, 3])

        features = torch.stack([mean, std, energy, max_val, sparsity], dim=1)

        if features.shape[1] < self.latent_dim * 4:
            padding = torch.zeros(
                b, self.latent_dim * 4 - features.shape[1],
                device=features.device
            )
            features = torch.cat([features, padding], dim=1)

        complexity = self.complexity_estimator(features)

        return complexity

    def predict_lambda(
        self,
        complexity: float,
        target_bpp: float,
    ) -> float:
        """
        Predict Lagrange multiplier (lambda) for RD optimization.

        Args:
            complexity: Frame complexity [0, 1]
            target_bpp: Target bits-per-pixel

        Returns:
            lambda_value: Lagrange multiplier for rate-distortion
        """
        if not self.enable_mlp:
            base_lambda = target_bpp * 10.0
            complexity_penalty = (1.0 - complexity) * 5.0
            return base_lambda + complexity_penalty

        inputs = torch.tensor(
            [[complexity, target_bpp]], dtype=torch.float32
        )
        lambda_val = self.lambda_predictor(inputs)

        return lambda_val.item()

    def get_qp_for_bitrate(
        self,
        current_bpp: float,
        target_bpp: float,
        current_qp: int,
        resolution: str = "1080p",
    ) -> int:
        """
        Adjust QP based on current vs target bitrate (feedback loop).

        Args:
            current_bpp: Actual achieved bits-per-pixel
            target_bpp: Target bits-per-pixel
            current_qp: Current QP value
            resolution: Video resolution tier

        Returns:
            Adjusted QP value
        """
        error = (target_bpp - current_bpp) / max(target_bpp, 1e-6)

        if abs(error) < 0.03:
            return current_qp

        if error > 0:
            qp_delta = int(error * 10)
            new_qp = max(0, current_qp - qp_delta)
        else:
            qp_delta = int(-error * 10)
            new_qp = min(51, current_qp + qp_delta)

        return new_qp


class CRFSchedule:
    """
    CRF (Constant Rate Factor) schedule for quality-based encoding.

    Provides smooth QP transitions between frames while maintaining
    target quality level.
    """

    def __init__(
        self,
        base_crf: int = 28,
        min_crf: int = 15,
        max_crf: int = 51,
        scd_threshold: float = 0.3,
    ):
        """
        Initialize CRF schedule.

        Args:
            base_crf: Base CRF value for normal scenes
            min_crf: Minimum CRF (highest quality)
            max_crf: Maximum CRF (lowest quality)
            scd_threshold: Scene change detection threshold
        """
        self.base_crf = base_crf
        self.min_crf = min_crf
        self.max_crf = max_crf
        self.scd_threshold = scd_threshold

        self.prev_frame_complexity: float | None = None

    def compute_crf(
        self,
        complexity: float,
        is_scene_change: bool = False,
    ) -> int:
        """
        Compute CRF for current frame.

        Args:
            complexity: Frame complexity [0, 1]
            is_scene_change: Whether this is a scene change frame

        Returns:
            CRF value
        """
        if is_scene_change:
            return max(self.min_crf, self.base_crf - 5)

        if self.prev_frame_complexity is not None:
            complexity_delta = abs(complexity - self.prev_frame_complexity)
            if complexity_delta > self.scd_threshold:
                return max(self.min_crf, self.base_crf - 3)

        qp = int(self.base_crf + (complexity - 0.5) * 10)
        qp = max(self.min_crf, min(self.max_crf, qp))

        self.prev_frame_complexity = complexity

        return qp

    def reset(self) -> None:
        """Reset state for new sequence."""
        self.prev_frame_complexity = None


def compute_bpp(latent_bits: int, height: int, width: int) -> float:
    """
    Compute bits-per-pixel from bit count and resolution.

    Args:
        latent_bits: Number of bits for latent
        height: Frame height in pixels
        width: Frame width in pixels

    Returns:
        Bits-per-pixel value
    """
    pixel_count = height * width
    return latent_bits / pixel_count


def estimate_frame_bits(
    qp: int,
    latent_dim: int,
    num_patches: int,
    motion_complexity: float = 0.5,
) -> int:
    """
    Estimate frame bit usage based on QP and complexity.

    Args:
        qp: Quantization parameter
        latent_dim: Latent dimension
        num_patches: Number of latent patches
        motion_complexity: Motion complexity [0, 1]

    Returns:
        Estimated bit count
    """
    base_bits_per_patch = latent_dim * 8

    qp_factor = (51 - qp) / 51.0

    complexity_factor = 0.5 + 0.5 * motion_complexity

    bits = int(base_bits_per_patch * num_patches * qp_factor * complexity_factor)

    return bits
