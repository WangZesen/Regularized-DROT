# Regularized DROT Benchmark

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

## How to run

- Genenerate test data
    ```
    make gendata DATADIR=<path-for-generated-data>
    ```

- Run benchmark
    ```
    make benchmark DATADIR=<path-for-generated-data>
    ```

The results are generated at
- `./log/qrad_drot.log`: results for quadratically regularized DROT
- `./log/gl_drot.log`: results for group-lasso regularized DROT


## Results

The following results are generated on Linux server with one `Nvidia V100` GPU.
- [Quadratically Regularized DROT](./result/quad_drot.log)
- [Group-Lasso Regularized DROT](./result/gl_drot.log)

