
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# This configuration gives ~700 trainable parameters.
LMKAN_WIDTH = 5
LMKAN_DEPTH = 2
LMKAN_K = 7
LMKAN_METRIC_HIDDEN = 9
LMKAN_GAMMA = 0.5  #changed gamma from 2 to 0.5
LMKAN_BASIS = "rbf"     # choose "wavelet" or "rbf"


# ============================================================
# 3) LM-KAN model (same model, only smaller config)
# ============================================================
class MetricNet(nn.Module):
    def __init__(self, d_in, hidden=64, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x):
        return F.softplus(self.net(x)) + self.eps


def mexican_hat(u):
    return (1.0 - u * u) * torch.exp(-0.5 * u * u)


def rbf_gamma(u, gamma):
    return torch.exp(-gamma * (u * u))


class LearnedMetricBasisLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K=16, basis="wavelet", gamma=2.0, metric_hidden=64):
        super().__init__()
        assert basis in ["wavelet", "rbf"]
        self.in_dim = in_dim
        self.K = K
        self.basis = basis
        self.gamma = float(gamma)
        self.metric = MetricNet(in_dim, hidden=metric_hidden)

        grid = torch.linspace(-2.5, 2.5, K).view(1, K).repeat(in_dim, 1)
        self.centers = nn.Parameter(grid + 0.05 * torch.randn(in_dim, K))
        self.log_widths = nn.Parameter(torch.log(0.8 * torch.ones(in_dim, K)))
        self.out = nn.Linear(in_dim * K + 1, out_dim)

    def forward(self, x):
        g = self.metric(x)
        z = x * torch.sqrt(g)
        widths = torch.exp(self.log_widths).unsqueeze(0) + 1e-12
        u = (z.unsqueeze(-1) - self.centers.unsqueeze(0)) / widths

        if self.basis == "wavelet":
            phi = mexican_hat(u)
        else:
            phi = rbf_gamma(u, gamma=self.gamma)

        phi = phi.reshape(x.shape[0], -1)
        logdet = torch.log(g).sum(dim=1, keepdim=True)
        feats = torch.cat([phi, logdet], dim=1)
        return self.out(feats)


class LMKAN(nn.Module):
    def __init__(self, in_dim, width=32, depth=2, K=16, basis="wavelet", gamma=2.0, metric_hidden=64):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(LearnedMetricBasisLayer(d, width, K=K, basis=basis, gamma=gamma, metric_hidden=metric_hidden))
            layers.append(nn.Tanh())
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x))



# lmkan_model = LMKAN(
#     in_dim=2,
#     width=LMKAN_WIDTH,
#     depth=LMKAN_DEPTH,
#     K=LMKAN_K,
#     basis=LMKAN_BASIS,
#     gamma=LMKAN_GAMMA,
#     metric_hidden=LMKAN_METRIC_HIDDEN,
# ).to(device=device, dtype=LMKAN_DTYPE)