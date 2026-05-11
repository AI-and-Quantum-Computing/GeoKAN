"""
model_geokan_gamma.py

GeoKAN with a separable RBF metric and engineered γ feature

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "GeoKAN-gamma"
COLOR = "#2ca02c"

HP = dict(
    in_dim   = 1,
    out_dim  = 1,
    width    = 22,
    depth    = 2,
    K        = 12,        

    epochs       = 600,
    lr           = 2e-3,
    weight_decay = 1e-5,
    patience     = 120,
    min_epochs   = 200,
)


# ---------------- separable RBF metric ----------------
class SeparableRBFMetric(nn.Module):
    """log g_i(x) = sum_k c_{ik} * exp(-0.5 ((x_i - mu_{ik}) / sigma_{ik})^2)."""
    def __init__(self, in_dim, K=12):
        super().__init__()
        grid = torch.linspace(-2.5, 2.5, K).view(1, K).repeat(in_dim, 1)
        self.centers = nn.Parameter(grid + 0.03 * torch.randn(in_dim, K))
        init_w = max(0.35, 5.0 / max(K, 1))
        self.log_widths = nn.Parameter(torch.log(init_w * torch.ones(in_dim, K)))
        self.coeffs = nn.Parameter(0.1 * torch.randn(in_dim, K))

    def forward(self, x):
        widths = torch.exp(self.log_widths).unsqueeze(0) + 1e-12
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)
        u = diff / widths
        phi = torch.exp(-0.5 * u * u)
        coeffs = self.coeffs.unsqueeze(0)
        log_g = (coeffs * phi).sum(dim=-1)
        g = torch.exp(log_g)
        dphi_dx = phi * (-diff / (widths * widths))
        dlogg_dx = (coeffs * dphi_dx).sum(dim=-1)
        gamma = 0.5 * dlogg_dx
        return g, gamma


class GeoKANGammaLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K=12):
        super().__init__()
        self.metric = SeparableRBFMetric(in_dim, K=K)
        self.alpha = nn.Parameter(torch.ones(in_dim))
        self.beta  = nn.Parameter(torch.ones(in_dim))
        self.delta = nn.Parameter(torch.ones(in_dim))
        self.out = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x):
        g, gamma = self.metric(x)
        sqrt_g = torch.sqrt(g)
        warp = x * sqrt_g
        sigma_branch = F.silu(x)
        geo = (self.alpha.unsqueeze(0) * warp
               + self.beta.unsqueeze(0) * gamma
               + self.delta.unsqueeze(0) * sqrt_g)
        return self.out(torch.cat([sigma_branch, geo], dim=1))


class GeoKANGamma(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, width=22, depth=2, K=12):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [GeoKANGammaLayer(d, width, K=K), nn.Tanh()]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


def build_model():
    return GeoKANGamma(
        in_dim=HP["in_dim"], out_dim=HP["out_dim"],
        width=HP["width"], depth=HP["depth"], K=HP["K"],
    )
