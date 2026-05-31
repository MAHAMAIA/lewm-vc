"""
Task probe for SVC training.

A frozen feature extractor that provides task-aware gradient signal during
dual-layer SVC training. Forces the Base Layer to preserve features relevant
for detection/recognition, not just pixel-level reconstruction.

Usage:
    from lewm_vc.utils.task_probe import create_task_probe

    probe = create_task_probe("resnet18", device="cuda")
    probe.eval()

    # During SVC training:
    feat_recon = probe(bl_decoded_frame)
    feat_orig = probe(original_frame)
    task_loss = F.mse_loss(feat_recon, feat_orig)
"""

from typing import Literal

import torch
import torch.nn as nn


class _MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts multi-scale features from a backbone network.

    Hooks into intermediate layers to get feature maps at multiple spatial
    resolutions. Returns the sum of MSE losses across all scales.

    Args:
        backbone: Pretrained nn.Module (e.g. resnet18).
        layers: Names of modules to extract features from.
        weights: Pretrained weight enum (e.g. ResNet18_Weights.DEFAULT).
    """

    def __init__(
        self,
        backbone: nn.Module,
        layers: list[str] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.layers = layers or ["layer1", "layer2", "layer3"]
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        for name in self.layers:
            module = dict([*self.backbone.named_modules()]).get(name)
            if module is not None:
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = output.detach()

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns concatenated multi-scale features."""
        self._activations = {}
        self.backbone(x)

        if not self._activations:
            return torch.zeros(1, device=x.device)

        features = []
        for name in self.layers:
            if name in self._activations:
                f = self._activations[name]
                f = f.mean(dim=(2, 3))
                features.append(f)

        return torch.cat(features, dim=-1)

    def compute_task_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale feature MSE between reconstructed and target frames.

        Args:
            recon: [B, 3, H, W] reconstructed frame in [0, 1]
            target: [B, 3, H, W] original frame in [0, 1]

        Returns:
            Scalar task loss
        """
        self._activations = {}
        _ = self.backbone(recon)
        recon_feats = dict(self._activations)

        self._activations = {}
        _ = self.backbone(target)
        target_feats = dict(self._activations)

        total_loss = torch.tensor(0.0, device=recon.device)
        count = 0
        for name in self.layers:
            if name in recon_feats and name in target_feats:
                total_loss = total_loss + nn.functional.mse_loss(
                    recon_feats[name], target_feats[name]
                )
                count += 1

        return total_loss / max(1, count)


class _SingleFeatureExtractor(nn.Module):
    """
    Extracts a single feature vector using a backbone's final avg pool.

    Simpler than multi-scale, good for quick experiments.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_task_probe(
    backbone_name: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
    multi_scale: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Create a frozen task probe for SVC training.

    Args:
        backbone_name: Which backbone to use (default: resnet18, 11M params).
        multi_scale: If True, extract features from layer1/2/3 for multi-scale loss.
        device: Device to place the probe on.

    Returns:
        Frozen nn.Module that takes [B, 3, H, W] in [0, 1] and returns features.

    Usage:
        probe = create_task_probe("resnet18")
        features = probe(frames)  # [B, feature_dim]
        task_loss = F.mse_loss(probe(recon), probe(original))
    """
    try:
        import torchvision.models as tv_models
    except ImportError:
        print("  [task_probe] torchvision not available, using identity probe")
        return nn.Identity()

    # Build weights enum
    weights_enum = {
        "resnet18": tv_models.ResNet18_Weights.DEFAULT,
        "resnet34": tv_models.ResNet34_Weights.DEFAULT,
        "resnet50": tv_models.ResNet50_Weights.DEFAULT,
    }

    weights = weights_enum.get(backbone_name)
    if weights is None:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. Choose from {list(weights_enum.keys())}"
        )

    # Create backbone with pretrained weights
    backbone = getattr(tv_models, backbone_name)(weights=weights)
    backbone.requires_grad_(False)
    backbone.eval()

    # Remove the classification head
    backbone.fc = nn.Identity()

    # Wrap in feature extractor
    if multi_scale:
        probe = _MultiScaleFeatureExtractor(backbone)
    else:
        probe = _SingleFeatureExtractor(backbone)

    probe = probe.to(device)
    probe.requires_grad_(False)
    probe.eval()

    # Compute feature dimension
    dummy = torch.rand(1, 3, 256, 256).to(device)
    with torch.no_grad():
        feat_dim = probe(dummy).shape[-1]
    print(f"  Task probe: {backbone_name} (multi_scale={multi_scale}, feat_dim={feat_dim})")

    return probe
