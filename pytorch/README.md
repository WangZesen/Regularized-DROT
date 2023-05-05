# PyTorch Extension for Regularized DROT

## Prerequisite

- Python 3.9+
- CUDA 11.7+
- PyTorch 2.0.0

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

For now, only quadratically regularized DROT is ported to PyTorch, and it's **GPU-only**. Here is the API:
```
x = reg_drot.quadratic_drot(cost, p, q, rho, r_weight, max_iter, eps)
```
- Input:
    - `cost`: cost matrix, a float32 tensor with shape `[M,N]`. (Should be on GPU)
    - `p`: source distribution, a float32 tensor with shape `[N]`. The elements should be positive and sum up to `1`. (Should be on GPU)
    - `q`: source distribution, a float32 tensor with shape `[M]`. The elements should be positive and sum up to `1`. (Should be on GPU)
    - `rho`: unnormalized step size, a positive float scalar. Recommended default value is `2.0`.
    - `r_weight`: weight for the quadratic regularization, a non-negative float scalar. If it's `0`, it means solving for unregularized OT.
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


