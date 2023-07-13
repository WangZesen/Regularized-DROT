# Regularized DROT

This repo is the official implementation for the paper, **Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs**, which is available on [arxiv](https://arxiv.org/abs/2305.18483).

## Benchmark

Some of the benchmarks included in the paper are shown here. For more detail, please refer to the [paper](https://arxiv.org/abs/2305.18483).

### Quadratic Regularized Optimal Transport

<img src="./figures/quadratic_reg_drot_1000.png" width="40%" alt="Compare with LBFGS-based method with quadratic regularizer with n=m=1000"/> <img src="./figures/quadratic_reg_drot_3000.png" width="40%" alt="Compare with LBFGS-based method with quadratic regularizer with n=m=3000"/>

### Group-lasso Regularized Optimal Transport

| Method | Ent | GL | Median | $q10$ | $q90$ | Median | $q10$ | $q90$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLSK | 1e-3 | 1e-6 | 3.82 | 3.75 | 9.03 | 0.311 | 0.0657 | 6.48 |
|  | 1e-3 | 5e-4 | 3.8 | 3.75 | 9 | 0.311 | 0.0657 | 6.48 |
|  | 1e-3 | 5e-2 | 3.8 | 3.77 | 10.6 | 0.345 | 0.0665 | 6.48 |
|  | 1e-2 | 1e-6 | 3.78 | 3.73 | 6.8 | 1.21 | 0.69 | 5.47 |
|  | 1e-2 | 5e-4 | 3.8 | 3.76 | 6.82 | 1.21 | 0.69 | 5.47 |
|  | 1e-2 | 5e-2 | 3.82 | 3.79 | 6.88 | 1.21 | 0.69 | 5.47 |
|  | 1e-1 | 1e-6 | 3.7 | 3.67 | 3.72 | 8.24 | 3.47 | 31.6 |
|  | 1e-1 | 5e-4 | 3.75 | 3.73 | 3.78 | 8.24 | 3.47 | 31.6 |
|  | 1e-1 | 5e-2 | 3.77 | 3.73 | 3.81 | 8.24 | 3.47 | 31.6 |
|  | 1. | 1e-6 | 3.73 | 1.86 | 3.77 | 45.5 | 9.92 | 218 |
|  | 1. | 5e-4 | 3.69 | 1.85 | 3.73 | 45.5 | 9.92 | 218 |
|  | 1. | 5e-2 | 3.75 | 1.88 | 3.78 | 45.5 | 9.92 | 218 |
| GLDROT |  | 1e-6 | 0.113 | 0.0758 | 0.147 | 0.0475 | 0.0215 | 0.137 |
|  |  | 5e-4 | 0.0745 | 0.0549 | 0.0951 | 0.0529 | 0.0245 | 0.163 |
|  |  | 5e-2 | 0.0232 | 0.0178 | 0.0288 | 0.331 | 0.154 | 1.38 |

## Folder Structure

- `/core`: core CUDA code for Regularized DROT GPU kernels
- `/tensorflow`: tensorflow CUDA extension. See [`/tensorflow/README.md`](./tensorflow/README.md) for how to proceed.
- `/pytorch`: pytorch CUDA extension. See [`/pytorch/README.md`](./pytorch/README.md) for how to proceed.
- `/benchmark`: benchmarking for Quadratic Regularized DROT and Group-Lasso Regularized DROT
- `/cpu`: cpu version of the Regularized DROT implementation
- `/demo`: [POT](https://pythonot.github.io/)-compatible demos


## Citation
To cite this repo, please include
```
@article{lindback2023bringing,
  title={Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs},
  author={Lindb{\"a}ck, Jacob and Wang, Zesen and Johansson, Mikael},
  journal={arXiv preprint arXiv:2305.18483},
  year={2023}
}
```
