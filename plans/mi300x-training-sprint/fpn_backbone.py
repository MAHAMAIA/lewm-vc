import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ResNet50FPN(nn.Module):
    """ResNet50 backbone with FPN for multi-scale feature extraction.

    Outputs P3, P4, P5 pyramid levels:
      P3: 1/8  resolution, 256 channels
      P4: 1/16 resolution, 256 channels
      P5: 1/32 resolution, 256 channels
    """

    def __init__(self):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT
        backbone = tv_models.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # C2: 1/4
        self.layer2 = backbone.layer2  # C3: 1/8, 512ch
        self.layer3 = backbone.layer3  # C4: 1/16, 1024ch
        self.layer4 = backbone.layer4  # C5: 1/32, 2048ch

        lateral_C3 = nn.Conv2d(512, 256, 1)
        lateral_C4 = nn.Conv2d(1024, 256, 1)
        lateral_C5 = nn.Conv2d(2048, 256, 1)
        self.lateral = nn.ModuleDict(
            {
                "c3": lateral_C3,
                "c4": lateral_C4,
                "c5": lateral_C5,
            }
        )

        smooth_P3 = nn.Conv2d(256, 256, 3, 1, 1)
        smooth_P4 = nn.Conv2d(256, 256, 3, 1, 1)
        smooth_P5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth = nn.ModuleDict(
            {
                "p3": smooth_P3,
                "p4": smooth_P4,
                "p5": smooth_P5,
            }
        )

        self.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.stem(x)  # 1/4
        c2 = self.layer1(c1)  # 1/4, 256ch
        c3 = self.layer2(c2)  # 1/8, 512ch
        c4 = self.layer3(c3)  # 1/16, 1024ch
        c5 = self.layer4(c4)  # 1/32, 2048ch

        m5 = self.lateral["c5"](c5)
        p5 = self.smooth["p5"](m5)

        m4 = self.lateral["c4"](c4) + F.interpolate(m5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth["p4"](m4)

        m3 = self.lateral["c3"](c3) + F.interpolate(m4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth["p3"](m3)

        return {"P3": p3, "P4": p4, "P5": p5}

    @property
    def pyramid_channels(self) -> dict[str, int]:
        return {"P3": 256, "P4": 256, "P5": 256}
