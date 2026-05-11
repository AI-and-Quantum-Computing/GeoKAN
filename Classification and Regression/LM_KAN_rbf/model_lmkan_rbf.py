"""
model_lmkan_rbf.py

Learned-Metric KAN with Gaussian RBF basis.
Same metric architecture as LM-KAN-Wav but with positive RBF kernels.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "LMKAN-RBF"
COLOR = "#9467bd"

HP = dict(
    in_dim         = 1,
    out_dim        = 1,
    width          = 10,
    depth          = 2,
    K              = 12,
    metric_hidden  = 8,
    gamma          = 2.0,    # RBF bandwidth multiplier

    epochs       = 600,
    lr           = 2e-3,
    weight_decay = 1e-5,
    patience     = 120,
    min_epochs   = 200,
)


def rbf_gamma(u, gamma=2.0):
    return torch.exp(-gamma * (u * u))


class MetricNet(nn.Module):
    def __init__(self, d_in, hidden=8, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x):
        return F.softplus(self.net(x)) + self.eps


class LMKANRBFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K=12, gamma=2.0, metric_hidden=8):
        super().__init__()
        self.gamma = float(gamma)
        self.metric = MetricNet(in_dim, hidden=metric_hidden)
        grid = torch.linspace(-2.5, 2.5, K).view(1, K).repeat(in_dim, 1)
        self.centers = nn.Parameter(grid + 0.03 * torch.randn(in_dim, K))
        init_w = max(0.35, 5.0 / max(K, 1))
        self.log_widths = nn.Parameter(torch.log(init_w * torch.ones(in_dim, K)))
        self.out = nn.Linear(in_dim * K + 1, out_dim)

    def forward(self, x):
        g = self.metric(x)
        z = x * torch.sqrt(g)
        widths = torch.exp(self.log_widths).unsqueeze(0) + 1e-12
        u = (z.unsqueeze(-1) - self.centers.unsqueeze(0)) / widths
        phi = rbf_gamma(u, gamma=self.gamma).reshape(x.shape[0], -1)
        logdet = torch.log(g).sum(dim=1, keepdim=True)
        return self.out(torch.cat([phi, logdet], dim=1))


class LMKANRBF(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, width=10, depth=2,
                 K=12, gamma=2.0, metric_hidden=8):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [LMKANRBFLayer(d, width, K=K, gamma=gamma,
                                     metric_hidden=metric_hidden),
                       nn.Tanh()]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


def build_model():
    return LMKANRBF(
        in_dim=HP["in_dim"], out_dim=HP["out_dim"],
        width=HP["width"], depth=HP["depth"],
        K=HP["K"], gamma=HP["gamma"], metric_hidden=HP["metric_hidden"],
    )
