import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (Cramér-Wold random projections).

    Measures divergence from an isotropic Gaussian via the Epps-Pulley
    statistic applied to random projections of the latent embeddings.
    Prevents representational collapse in JEPA training.

    Reference: LeJEPA (arXiv:2511.08544) and stable-worldmodel implementation.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (T, B, D)  —  latents stacked across time, batch, and features.

        Returns:
            scalar loss averaged over projections and time.
        """
        dev = proj.device
        D = proj.size(-1)

        A = torch.randn(D, self.num_proj, device=dev, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True))

        t = self.t.to(dev)
        phi = self.phi.to(dev)
        weights = self.weights.to(dev)

        x_t = (proj @ A).unsqueeze(-1) * t  # (T, B, P, K)
        err = (x_t.cos().mean(dim=-3) - phi).square() + (
            x_t.sin().mean(dim=-3) - 0.0
        ).square()  # (T, P, K)

        statistic = (err @ weights) * proj.size(0)  # (T, P)
        return statistic.mean()
