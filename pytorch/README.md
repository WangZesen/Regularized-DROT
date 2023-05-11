# PyTorch Extension for Regularized DROT

## Prerequisite

- Python 3.8+
- CUDA 11.7+
- PyTorch 2.0.0 with GPU support

> This is the setup of the environment where the code is tested. Other versions of prerequisites may work, but they're not tested.

## Build & Install
```
python setup.py install
```

## Test

```
python test.py
```
A green `PASS` should show up in the end if it passes all tests.

## How to use

Import it in Python code
```python
import torch
# Import Reg-DROT extension
import reg_drot
```

Both quadratically regularized DROT and group-lasso regularized DROT is ported to PyTorch, and it's **GPU-only** with datatype `float32`. Here is the API:
```
x = reg_drot.quadratic_drot(cost, p, q, rho, r_weight, max_iter, eps)
```
- Input:
    - `cost`: cost matrix, a non-negative float32 tensor with shape `[M,N]`, which should be normalized to `[0,1]` by `cost=cost/cost.max()`. (Must be on GPU)
    - `p`: target distribution, a float32 tensor with shape `[N]`. The elements should be positive and sum up to `1`. (Must be on GPU)
    - `q`: source distribution, a float32 tensor with shape `[M]`. The elements should be positive and sum up to `1`. (Must be on GPU)
    - `rho`: unnormalized step size, a positive float scalar. Recommended default value is `2.0`. After normalization, the step size is
    $$
        \text{step size}=\frac{\rho}{N+M}
    $$
    - `r_weight`: unnormalized weight for the quadratic regularization, a non-negative float scalar. If it's `0`, it means solving for unregularized OT. After normalization, the regularizer's weight is
    $$
        \text{scaled\_reg\_weight}=\text{r\_weight}\cdot (N+M)
    $$
    - `max_iter`: maximal number of iterations, a positive integer scalar.
    - `eps`: tolerance error for the primal residual, a positive float scalar. Recommended default value is `1e-4`.
- Output:
    - `x`: the transportation plan, a float32 tensor with the same  shape as `cost`.

```
x = reg_drot.group_lasso_drot(cost, p, q, n_class, rho, r_weight, max_iter, eps)
```
- Input:
    - `cost`: cost matrix, a non-negative float32 tensor with shape `[M,N]`, which should be normalized to `[0,1]` by `cost=cost/cost.max()`. Note that the currrent version only supports classes with same number of instances, and the target distribution should be aligned with class, which means `[round(i*N/n_class),round((i+1)*N/n_class)]` columns should belong to `(i+1)`-th class. (Must be on GPU)
    - `p`: target distribution, a float32 tensor with shape `[N]`. The elements should be positive and sum up to `1`. (Must be on GPU)
    - `q`: source distribution, a float32 tensor with shape `[M]`. The elements should be positive and sum up to `1`. (Must be on GPU)
    - `n_class`: number of classes, a `>1` integer scalar.
    - `rho`: unnormalized step size, a positive float scalar. Recommended default value is `2.0`. After normalization, the step size is
    $$
        \text{step size}=\frac{\rho}{N+M}
    $$
    - `r_weight`: unnormalized weight for the group-lasso regularization, a non-negative float scalar. If it's `0`, it means solving for unregularized OT. After normalization, the weight is
    $$
        \text{scaled\_reg\_weight}=\text{r\_weight}\cdot \sqrt{\frac{N}{\text{n\_class}}}
    $$
    - `max_iter`: maximal number of iterations, a positive integer scalar.
    - `eps`: tolerance error for the primal residual, a positive float scalar. Recommended default value is `1e-4`.
- Output:
    - `x`: the transportation plan, a float32 tensor with the same  shape as `cost`.


To see an example, `./test.py` can be a reference.

## Appendix

### How to setup Python environment
The following commands are from [PyTorch official tutorial](https://pytorch.org/get-started/locally/).

Please create a conda virtual environment first before preceeding.

```shell
pip3 install torch torchvision torchaudio
```


