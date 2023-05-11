# Regularized DROT GPU Benchmark

## Prerequisite

- CUDA 11.7+

    Make sure `nvcc` can be found in `$PATH`.
    ```
    xxx@xxx:~/xxx$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Tue_May__3_18:49:52_PDT_2022
    Cuda compilation tools, release 11.7, V11.7.64
    Build cuda_11.7.r11.7/compiler.31294372_0
    ```
- Activated Python environment with
    - `POT==0.9.0`
    - `scikit-learn==1.2.2`
    - `numpy==1.24.2`
    - `torch==2.0.0`
    - `cupy-cuda11x==11.5.0` (for sinkhorn benchmark)
    - `reg-drot==1.0.0` (Pytorch wrapper for Regularized DROT, See [here](../pytorch/) for installation)
    - `seaborn==0.12.2` (for plotting)

## Run benchmark

- Genenerate test data
    ```shell
    make gendata DATA_DIR=<path-for-generated-data>
    ```

- Run benchmark
    ```shell
    # Optimal Transport Benchmark for Quadratically Regularized DROT
    #   Output: ./log/quad_drot.log
    make benchmark-drot DATA_DIR=<path-for-generated-data>
    # Domain Adaptation Benchmark for Group-Lasso Regularized DROT
    #   Output: ./log/gl_drot.log
    make benchmark-drot-da DATA_DIR=<path-for-generated-data>
    # Compute Wasserstein-2 Distance for Domain Adaptation
    #   Output: ./log/gl_drot_amend.log
    python utils/compute_w2_dist.py <path-for-generated-data>/data/da <path-for-generated-data>/data/da-out
    ```

The results are generated at
- `./log/qrad_drot.log`: results for quadratically regularized DROT
- `./log/gl_drot_amend.log`: results for group-lasso regularized DROT
- `./log/gl_drot_amend.log`: results for group-lasso regularized DROT with computed W-2 distance

### Benchmark for Other Methods

#### Sinkhorn's Algorithm with Entropic Reg. [1,2]

```shell
# Optimal Transport Benchmark for Sinkhorn-Knopp algorithm
#   Output: ./log/sk_entropic.log
make benchmark-sk DATA_DIR=<path-for-generated-data>
```

#### Sinkhorn's Algorithm with Entropic Reg. + Group-Lasso Reg. [1,2]

```shell
# Optimal Transport Benchmark for Sinkhorn-Knopp algorithm
#   Output: ./log/sk_entropic_gl.log
make benchmark-sk-da DATA_DIR=<path-for-generated-data>
```

#### L-BFGS Solver for Smooth Relaxed Dual (based on PyTorch) [3,4]

```shell
# Optimal Transport Benchmark for L-BFGS solver for the smooth relaxed dual
#   Output: ./log/lbfgs.log
make benchmark-lbfgs DATA_DIR=<path-for-generated-data>
```

### Clean Up and Rerun
```shell
make clean DATA_DIR=<path-for-generated-data>
```

## Results

The following benchmark results are generated on a Linux server with one `Tesla V100-SXM2-32GB` GPU.
- Quadratically Regularized DROT: [./result/quad_drot.log](./result/quad_drot.log)
- Group-Lasso Regularized DROT: [./result/gl_drot_amend.log](./result/gl_drot_amend.log)
- Entropic Regularized Sinkhorn-Knopp Algorithm: [./result/sk_entropic.log](./result/sk_entropic.log)
- Entropic and Group-Lasso Regularized Sinkhorn-Knopp Algorithm: [./result/sk_entropic_gl_amend.log](./result/sk_entropic_gl_amend.log)
- L-BFGS Solver for the Smooth Relaxed Dual: [./result/lbfgs.log](./result/lbfgs.log)

### Graphs

> To be added

## Reference

[1] POT: Python Optimal Transport. https://pythonot.github.io/

[2] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." Advances in neural information processing systems 26 (2013).

[3] PyTorch L-BFGS Optimizer. https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html

[4] Blondel, Mathieu, Vivien Seguy, and Antoine Rolet. "Smooth and sparse optimal transport." International conference on artificial intelligence and statistics. PMLR, 2018.
