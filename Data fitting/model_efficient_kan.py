"""
model_efficient_kan.py

EfficientKAN (B-spline KAN) baseline. Stronger than MLP but still loses to
GeoKAN variants on highly oscillatory / discontinuous targets.

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "EfficientKAN"
COLOR = "#ff7f0e"

HP = dict(
    layers       = [1, 17, 17, 1],   # ~1938 params
    grid_size    = 3,
    spline_order = 1,
    base_act     = "silu",

    # training
    epochs       = 600,
    lr           = 2e-3,
    weight_decay = 1e-5,
    patience     = 120,
    min_epochs   = 200,

    # KAN-specific
    enable_grid_update = False,       # set True if you want adaptive grids
    reg_activation     = 5e-5,
    reg_entropy        = 0.1,
)


# ---------------- KANLinear (B-spline layer) ----------------
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=3, spline_order=1,
                 scale_noise=0.05, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h
                 + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / max(self.grid_size, 1))
            self.spline_weight.data.copy_(self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order], noise))
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            d1 = grid[:, k:-1] - grid[:, :-(k + 1)]
            d2 = grid[:, k + 1:] - grid[:, 1:(-k)]
            bases = ((x - grid[:, :-(k + 1)]) / (d1 + 1e-12) * bases[:, :, :-1]
                     + (grid[:, k + 1:] - x) / (d2 + 1e-12) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        try:
            sol = torch.linalg.lstsq(A, B).solution
        except Exception:
            sol = torch.matmul(torch.linalg.pinv(A), B)
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features)
        base = F.linear(self.base_activation(x), self.base_weight)
        spl  = F.linear(self.b_splines(x).reshape(x.size(0), -1),
                        self.scaled_spline_weight.reshape(self.out_features, -1))
        return (base + spl).reshape(*shape[:-1], self.out_features)

    def regularization_loss(self, ra=1.0, re=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        s = l1.sum()
        p = l1 / (s + 1e-12)
        return ra * s + re * (-(p * (p + 1e-12).log()).sum())


# ---------------- Stacked EfficientKAN ----------------
class EfficientKAN(nn.Module):
    def __init__(self, layers, grid_size=3, spline_order=1, base_activation=nn.SiLU):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(a, b, grid_size=grid_size, spline_order=spline_order,
                      base_activation=base_activation)
            for a, b in zip(layers, layers[1:])
        ])

    def forward(self, x, update_grid=False):
        for l in self.layers:
            x = l(x)
        return x

    def regularization_loss(self, ra=1.0, re=1.0):
        return sum(l.regularization_loss(ra, re) for l in self.layers)


def build_model():
    act = nn.SiLU if HP["base_act"] == "silu" else nn.Tanh
    return EfficientKAN(layers=HP["layers"], grid_size=HP["grid_size"],
                        spline_order=HP["spline_order"], base_activation=act)
