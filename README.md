# Regularized DROT

This repo is the official implementation for the paper, **Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs**, which is available on [arxiv](https://arxiv.org/abs/2305.18483).

## GPU Benchmark

Some of the benchmarks included in the paper are shown here. For more details, please refer to the [paper](https://arxiv.org/abs/2305.18483).

The benchmarks were run on one Tesla V100 GPU.

### Quadratic Regularized Optimal Transport

**Problem Setting for Quadratic Regularized Optimal Transport**

```math
\begin{align}
    \begin{array}{ll}
        \underset{X \in \mathbb{R}_+^{m \times n}}{\text{minimize}} &\left\langle C, X\right\rangle + \frac{(m+n)\alpha}{2}\Vert X\Vert_F^2 \\
        \text{subject to} & X \mathbf{1}_n = p, X^\top \mathbf{1}_m = q
    \end{array}
\end{align}
```

Comparison between our method and the method solving the smooth relaxed dual based on L-BFGS [[1](#references)] (using [PyTorch's L-BFGS optimizer implementation](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html)) with problem sizes `1000x1000` (left) and `3000x3000` (right).


50 random data ($C$ matrix) are generated for each problem size, and both algorithm run with same weights of the regularization and same error tolerances (for stopping criterion). The averaged runtime under various weights of the regularization of the two methods are compared below.

<img src="./figures/quadratic_reg_drot_1000.png" width="49%" alt="Compare with LBFGS-based method with quadratic regularizer with n=m=1000"/> <img src="./figures/quadratic_reg_drot_3000.png" width="49%" alt="Compare with LBFGS-based method with quadratic regularizer with n=m=3000"/>

- QDROT: Our method.
- LBFGS: The method solving the smooth relaxed dual based on L-BFGS [[1](#references)].

It is clear that our method better exploits the parallelization from the GPU, as it consistently leads to a speedup of at least one order of magnitude compared to the L-BFGS method.

### Group-lasso Regularized Optimal Transport

**Problem Setting for Group-Lasso Regularized Optimal Transport (our method)**

```math
\begin{align}
    \begin{array}{ll}
        \underset{X \in \mathbb{R}_+^{m \times n}}{\text{minimize}} &\left\langle C, X\right\rangle + \lambda\sum_{g\in\mathcal{G}} \Vert X_g\Vert_F\\
        \text{subject to} & X \mathbf{1}_n = p, X^\top \mathbf{1}_m = q &
    \end{array}
\end{align}
```

**Problem Setting for Group-Lasso Regularized Optimal Transport (Based On Sinkhorn's Algorithm [[2](#references)])**

```math
\begin{align}
    \begin{array}{ll}
        \underset{X \in \mathbb{R}_+^{m \times n}}{\text{minimize}} & \left\langle C, X\right\rangle + \lambda\sum_{g\in\mathcal{G}} \Vert X_g\Vert_F + \text{reg}\cdot\sum_{i,j} X_{i,j}\log(X_{i,j})\\
        \text{subject to} & X \mathbf{1}_n = p, X^\top \mathbf{1}_m = q &
    \end{array}
\end{align}
```

We simulated 50 datasets with two features, `1500` training samples, and `1000` test samples. The transformed domain of the test set is achieved by a random affine transformation of the source domain. Each set had 2 unique labels that were uniformly distributed among the instances. The labels in the test set were only used for validation.

To assess the quality of the adaptation, we compute the Wasserstein distance between the adapted samples of each label and the corresponding samples of the test set. This means that the better the alignment, the lower the aggregated distances. For the method using entropic regularization, we varied the regularization parameters $0.001$ to $100$, and the group lasso regularization was set to $0.001$ and $0.005$.


| Method | Ent | GL | Median <br/> (RT) | q10 <br/> (RT) | q90 <br/> (RT) | Median <br/> (Dist) | q10 <br/> (Dist) | q90 <br/> (Dist) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLSK [[2](#references)] | `1e-3` | `1e-3` | 3.77 | 3.68 | 8.90 | 0.311 | 0.0657 | 6.48 |
| GLSK [[2](#references)] | `1e-1` | `1e-3` | 3.73 | 3.71 | 3.77 | 8.24 | 3.47 | 31.6 |
| GLSK [[2](#references)] | `1e+2` | `1e-3` | 1.86 | 1.86 | 1.88 | 52.8 | 10.9 | 311 |
| GLDROT (ours) | - | `1e-3` | 0.0619 | 0.0475 | 0.0771 | **0.0576** | 0.0262 | 0.182 |
| GLDROT (ours) | - | `5e-3` | **0.0384** | 0.0306 | 0.0474 | 0.0879 | 0.0358 | 0.274 |

- GLSK: Group-lasso regularized Sinkhorn's algorithm [[2](#references)]. The GPU implementation is available in [POT](https://pythonot.github.io/gen_modules/ot.da.html#ot.da.sinkhorn_l1l2_gl).
- GLDROT: Our method.

*RT: Runtime in second, Dist: Wasserstein distance, q10: 10th percentile, q90: 90th percentile*.

The results are presented in the table above, in which it is clear that our
method consistently outperforms the alternative methods, both in terms of adaptation quality, but also
time to reach convergence

## Folder Structure

- `/core`: core CUDA code for Regularized DROT GPU kernels
- `/tensorflow`: tensorflow CUDA extension. See [`/tensorflow/README.md`](./tensorflow/README.md) for how to proceed.
- `/pytorch`: pytorch CUDA extension. See [`/pytorch/README.md`](./pytorch/README.md) for how to proceed.
- `/benchmark`: benchmarking for Quadratic Regularized DROT and Group-Lasso Regularized DROT
- `/numpy`: cpu version based on numpy (with numba accelleration) of the Regularized DROT implementation
- `/numpy/examples`: notebook demos with comparisons between regdrot and state-of-the-art implemented in [POT](https://pythonot.github.io/).

## Reproducing the Examples

To reproduce the examples provided in the `/numpy/examples` directory, we recommend setting up a conda environment using either **conda** or **miniconda**. Please ensure that you have either of these installed before proceeding.

Follow the steps below to set up and activate the virtual environment:

```
conda env create
conda activate regdrot
```
This will activate the `regdrot` environment and ensure that the required dependencies are available.

You are now ready to reproduce the examples in the `/numpy/examples` directory. Simply navigate to the desired example file and run it using your preferred  Jupyter notebook environment. :)

## Citation
To cite this repo and our paper, please include
```
@article{lindback2023bringing,
  title={Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs},
  author={Lindb{\"a}ck, Jacob and Wang, Zesen and Johansson, Mikael},
  journal={arXiv preprint arXiv:2305.18483},
  year={2023}
}
```

## References
[1] Mathieu Blondel, Vivien Seguy, and Antoine Rolet. Smooth and sparse optimal transport. In *International conference on artificial intelligence and statistics*, pages 880–889. PMLR, 2018.

[2] Nicolas Courty, Flamary Rémi, and Tuia Devis. Domain adaptation with regularized optimal transport. In *Machine Learning and Knowledge Discovery in Databases: European Conference*, Springer Berlin Heidelberg, 2014.