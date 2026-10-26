# Extended Chaotic Lorenz System Benchmark ($t \in [0, 10.0]$)

This folder contains the complete benchmark, training pipelines, publication-grade visualizations, and Jupyter notebook for the **extended chaotic Lorenz system** ($t \in [0, 10.0]$).

> **"Geometric Kolmogorov–Arnold Network (GeoKAN)"**  
> *Abhijit Sen, Bikram Keshari Parida, Giridas Maiti, Mahima Arya, Denys I. Bondar*  
> arXiv: [https://arxiv.org/abs/2605.06740](https://arxiv.org/abs/2605.06740)

---


### Key Regime Comparison:
| Parameter / Property | Non-Chaotic Regime (Manuscript Fig. 17) | Extended Chaotic Regime  |
| :--- | :--- | :--- |
| **Prandtl Number $\sigma$** | $0.1$ | **$10.0$** |
| **Rayleigh Number $\rho$** | $0.2$ | **$28.0$** |
| **Geometric Factor $\beta$** | $0.3$ | **$8/3 \approx 2.6667$** |
| **Initial State $(x_0, y_0, z_0)$** | $(1.0, 1.0, 1.0)$ | $(1.0, 1.0, 1.0)$ |
| **Time Domain $t$** | $t \in [0, 20]$ | **$t \in [0, 10.0]$** |
| **Dynamics Classification** | Monotonic asymptotic decay to $(0,0,0)$ | **Fully Chaotic Strange Attractor** |
| **Lyapunov Exponent $\lambda_{\max}$** | $\lambda \le 0$ (Stable point attractor) | **$\lambda \approx +0.9056$** ($\tau_L \approx 1.104$ s) |
| **Lyapunov Times Horizon** | $0$ (No chaos) | **$\approx 9.06$ Lyapunov times (~15 oscillations)** |

---

## 2.  Parameter Constraints & Architecture Budgets

1. **Comparable Budgets with LM-KAN Having FEWEST Parameters:**
   - The parameter counts are comparable with slight variation, but **LM-KAN's parameters are kept strictly lower** than PIKAN and Standard PINN.
   - This directly confirms that LM-KAN's superior performance is derived from its **learned Riemannian space-time metric**, rather than sheer parameter capacity.
2. **Untouched Module Integrity:**
   - `lm_kan.py`, `wavekan.py`, and `efficientkan.py` are imported directly without altering underlying class definitions.
3. **All Models Compared Simultaneously:**
   - Every figure evaluates LM-KAN, PIKAN, and Standard PINN together on identical axes.

### Model Parameter Budgets:
| Model | Architecture Details | Parameters | Relative Capacity |
| :--- | :--- | :---: | :---: |
| **LM-KAN (GeoKAN)** | `width=11, depth=2, K=24, metric_hidden=11` (Wavelet basis) | **4,363** | **Baseline (Smallest)** |
| **PIKAN (Wavelet KAN)** | `[1, 26, 40, 3]` (Mexican hat wavelets) | **4,744** | **+8.7% larger** |
| **Standard PINN (MLP)** | 4 hidden layers of width 40 with Tanh activations | **5,123** | **+17.4% larger** |

---

## 3. Comprehensive Experimental Results (10,000 Epochs, $t \in [0, 10.0]$)

```
==============================================================================================
EXTENDED CHAOTIC LORENZ BENCHMARK RESULTS (t in [0, 10], 10,000 EPOCHS, VERSION 4)
==============================================================================================
Model              | Parameters | Final Loss     | Trajectory MSE   | Time (s)   | Peak RAM (MB)
----------------------------------------------------------------------------------------------
LM-KAN             | 4363       | 0.00000034     | 0.00035440       | 422.75     | 393.9       
PIKAN              | 4744       | 0.00019380     | 0.37945058       | 2693.90    | 455.8       
Standard PINN      | 5123       | 0.00160450     | 3.68958772       | 138.61     | 308.5       
==============================================================================================
```

### Key Performance Highlights:
- **Optimization Convergence (Final Loss):**
  - LM-KAN achieved a final loss of **$3.4 \times 10^{-7}$**.
  - LM-KAN is **570$\times$ lower loss** than PIKAN ($1.94 \times 10^{-4}$).
  - LM-KAN is **4,719$\times$ lower loss** than Standard PINN ($1.60 \times 10^{-3}$).
- **Reconstruction Accuracy (Trajectory MSE):**
  - LM-KAN reached **$3.54 \times 10^{-4}$ MSE**, reconstructing all 15 chaotic oscillations with exceptional fidelity.
  - LM-KAN is **1,070$\times$ more accurate** than PIKAN (MSE $= 0.379$).
  - LM-KAN is **10,410$\times$ more accurate** than Standard PINN (MSE $= 3.69$).
- **Computational Efficiency:**
  - LM-KAN completed 10,000 epochs in **422.75 s (~7.0 min)** with peak memory of **393.9 MB**, running **6.4$\times$ faster** than PIKAN (2,693.90 s / 44.9 min).

---

## 4. Visualizations (`outputs/`)

All figures present simultaneous comparisons across all three models:

1. **`loss_comparison_v4.png` / `.pdf`**: Physics-informed relative loss convergence across 10,000 epochs on logarithmic scale (without arbitrary benchmark line).
2. **`trajectories_v4.png` / `.pdf`**: Time-domain state trajectories $x(t), y(t), z(t)$ across $t \in [0, 10.0]$ against high-precision RK45 reference.
3. **`attractor_3d_v4.png` / `.pdf`**: $2 \times 2$ phase-space portrait comparing Exact RK45, LM-KAN, PIKAN, and Standard PINN on identical coordinate limits.
4. **`pointwise_error_v4.png` / `.pdf`**: Pointwise absolute reconstruction errors $|x_{\mathrm{pred}} - x_{\mathrm{exact}}|$, $|y_{\mathrm{pred}} - y_{\mathrm{exact}}|$, $|z_{\mathrm{pred}} - z_{\mathrm{exact}}|$.
5. **`parameter_scaling_progression_v4.png` / `.pdf`**: Trajectory MSE across 3 distinct parameter capacity tiers.

---

## 5. Directory Structure

```
Lorenz_system_chaotic/
├── outputs/
│   ├── loss_comparison_v4.pdf
│   ├── loss_comparison_v4.png
│   ├── trajectories_v4.pdf
│   ├── trajectories_v4.png
│   ├── attractor_3d_v4.pdf
│   ├── attractor_3d_v4.png
│   ├── pointwise_error_v4.pdf
│   ├── pointwise_error_v4.png
│   ├── parameter_scaling_progression_v4.pdf
│   └── parameter_scaling_progression_v4.png
├── lm_kan.py                  # Untouched native module
├── wavekan.py                 # Untouched native module
├── efficientkan.py            # Untouched native module
├── train_chaotic_v4.py        # 10,000-epoch benchmark training pipeline
├── create_notebook_v4.py      # Notebook compilation script
├── chaotic_lorenz_v4.ipynb    # Executed self-contained Jupyter notebook
└── README.md                  # Comprehensive benchmark documentation
```
