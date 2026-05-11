# GeoKAN: Geometric Kolmogorov–Arnold Networks 
([https://arxiv.org/abs/2605.06740](https://arxiv.org/abs/2605.06740))

We introduce **Geometric Kolmogorov–Arnold Networks (GeoKANs)**, a family of geometry-aware KAN-type models in which approximation is carried out in learned, geometry-adapted coordinates rather than in fixed Euclidean input coordinates.

## Abstract
GeoKAN achieves its performance by learning a **diagonal Riemannian metric** that warps the input before basis expansion and feature mixing. The learned metric provides a geometric inductive bias through local length scaling and volume distortion, and in physics-informed settings, it also affects the differential structure seen by the model.

By stretching regions with rapid variation and compressing smoother regions, GeoKAN reallocates representational resolution in a task-dependent manner, allowing the model to place capacity where it is most needed. As a result, GeoKAN is well suited to sharp, stiff, localized, and strongly non-uniform regimes arising in scientific machine learning and differential-equation problems.

## Key Frameworks & Variants
Within this framework, we develop three main variants:
* **GeoKAN-NNMetric**: Metric parameterization via neural networks.
* **GeoKAN-$\gamma$**: A specialized geometric scaling variant.
* **LM-KAN (Learned Metric KAN)**: Specifically designed for basis-specific optimization.
    * **LM-KAN-RBF**: Using Radial Basis Functions.
    * **LM-KAN-Wav**: Using Wavelets.
    * **LM-KAN-Fourier**: Using Fourier bases.

## Code
The implementation for GeoKAN is available at: [https://github.com/AI-and-Quantum-Computing/GeoKAN](https://github.com/AI-and-Quantum-Computing/GeoKAN).

The proposed model was implemented and evaluated on a Linux-based system (6.8.0-57-generic) powered by a 56-core x86\_64 CPU (112 logical processors) and 754.53 GB of system memory.

## Citation
If you find GeoKAN useful in your research, please cite our work using the following format:

```bibtex
@misc{geokan_2026,
      title={Geometric Kolmogorov--Arnold Network (GeoKAN)}, 
      author={Abhijit Sen and Bikram Keshari Parida and Giridas Maiti and Mahima Arya and Denys I. Bondar},
      year={2026},
      eprint={2605.06740},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.06740}, 
}
