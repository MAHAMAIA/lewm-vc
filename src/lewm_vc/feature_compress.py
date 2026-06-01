import torch
import torch.nn as nn
import torchvision.models as tv_models


CHANNELS = {
    "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256},
    "resnet34": {"layer1": 64, "layer2": 128, "layer3": 256},
    "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024},
}


class ResNetFeatureExtractor(nn.Module):
    """Frozen ResNet backbone up to layer3 for feature extraction."""

    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()
        weights_enum = {
            "resnet18": tv_models.ResNet18_Weights.DEFAULT,
            "resnet34": tv_models.ResNet34_Weights.DEFAULT,
            "resnet50": tv_models.ResNet50_Weights.DEFAULT,
        }
        self._backbone_name = backbone_name
        weights = weights_enum.get(backbone_name)
        backbone = getattr(tv_models, backbone_name)(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

    @property
    def feature_channels(self) -> int:
        return CHANNELS.get(self._backbone_name, {}).get("layer3", 256)


class FeatureCompressor(nn.Module):
    """Compress detector features to compact latent representation.

    Input:  [B, in_channels, H, W] features from detector backbone
    Output: [B, latent_dim, H, W] compressed features
    """

    def __init__(self, in_channels: int = 256, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDecompressor(nn.Module):
    """Decompress latent back to detector feature space.

    Input:  [B, latent_dim, H, W] quantized latent
    Output: [B, out_channels, H, W] reconstructed features
    """

    def __init__(self, latent_dim: int = 8, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
