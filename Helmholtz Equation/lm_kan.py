# --- 2. LMKAN Architecture ---
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
    def __init__(self, in_dim, out_dim=2, width=32, depth=2, K=16, basis="wavelet", gamma=2.0, metric_hidden=64):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(LearnedMetricBasisLayer(d, width, K=K, basis=basis, gamma=gamma, metric_hidden=metric_hidden))
            layers.append(nn.Tanh())
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.head(self.body(x))
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper Basis Functions ---
def fourier_basis(u, K):
    # u shape: (batch, in_dim)
    # Create frequencies [1, ..., K]
    k = torch.arange(1, K + 1, device=u.device).view(1, 1, K).float()
    u_expanded = u.unsqueeze(-1) # (batch, in_dim, 1)
    
    # Generate sin and cos components
    sin_feats = torch.sin(np.pi * k * u_expanded)
    cos_feats = torch.cos(np.pi * k * u_expanded)
    
    # Concatenate sin and cos: (batch, in_dim, 2*K)
    return torch.cat([sin_feats, cos_feats], dim=-1)

def mexican_hat(u):
    return (1.0 - u * u) * torch.exp(-0.5 * u * u)

def rbf_gamma(u, gamma):
    return torch.exp(-gamma * (u * u))

# --- LM-KAN Components ---
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

class LearnedMetricBasisLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K=16, basis="wavelet", gamma=2.0, metric_hidden=64):
        super().__init__()
        # 1. Update the assertion to allow "fourier"
        assert basis in ["wavelet", "rbf", "fourier_basis"]
        self.in_dim = in_dim
        self.K = K
        self.basis = basis
        self.gamma = float(gamma)
        self.metric = MetricNet(in_dim, hidden=metric_hidden)

        # 2. Adjust Linear Output Dimension
        # Fourier has sin AND cos, so it creates 2*K features per input dimension
        n_features = (2 * K if basis == "fourier_basis" else K)
        self.out = nn.Linear(in_dim * n_features + 1, out_dim)

        # Grid parameters (only used for local basis like wavelet/rbf)
        grid = torch.linspace(-2.5, 2.5, K).view(1, K).repeat(in_dim, 1)
        self.centers = nn.Parameter(grid + 0.05 * torch.randn(in_dim, K))
        self.log_widths = nn.Parameter(torch.log(0.8 * torch.ones(in_dim, K)))

    def forward(self, x):
        g = self.metric(x)
        z = x * torch.sqrt(g)
        
        # Calculate Basis Functions
        if self.basis == "fourier_basis":
            # Global oscillations
            phi = fourier_basis(z, self.K)
        else:
            # Localized oscillations (needs u-transform)
            widths = torch.exp(self.log_widths).unsqueeze(0) + 1e-12
            u = (z.unsqueeze(-1) - self.centers.unsqueeze(0)) / widths
            if self.basis == "wavelet":
                phi = mexican_hat(u)
            else:
                phi = rbf_gamma(u, gamma=self.gamma)

        # Flatten features: (batch, in_dim * n_features)
        phi = phi.reshape(x.shape[0], -1)
        
        # Geometric correction
        logdet = torch.log(g).sum(dim=1, keepdim=True)
        feats = torch.cat([phi, logdet], dim=1)
        return self.out(feats)

class LMKAN(nn.Module):
    def __init__(self, in_dim, out_dim=2, width=32, depth=2, K=16, basis="wavelet", gamma=2.0, metric_hidden=64):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(LearnedMetricBasisLayer(d, width, K=K, basis=basis, gamma=gamma, metric_hidden=metric_hidden))
            layers.append(nn.Tanh())
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.head(self.body(x))