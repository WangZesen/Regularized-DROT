# Tensorflow Extension for Regularized DROT

## Prerequisite

- Python 3.9+
- CUDA 11.7+
- Tensorflow 2.12.0

> This is the setup of the environment where the code is tested. Other versions of prerequisites may work, but they're not tested.

## Build
```
make build
```

## Test

```
make test
```
A green `PASS` should show up in the end if it passes all tests.

## How to use

After building the extension, it will output the path for the extension in the last line of the build message, which is something like `xxxxx/drot_extend.so`.

Then import it in Python code
```python
import tensorflow as tf
REG_DROT_EXTEND_MODULE_DIR = <path-to-extension>
reg_drot = tf.load_op_library(REG_DROT_EXTEND_MODULE_DIR)
```

For now, only quadratically regularized DROT is ported to Tensorflow, and it's **GPU-only**. Here is the API:
```
x = reg_drot.quadratic_drot(cost, rho, r_weight, p, q, eps, max_iter)
```
- Input:
    - `cost`: cost matrix, a float32 tensor with shape `[M,N]`. (Should be on GPU)
    - `rho`: unnormalized step size, a positive float scalar. Recommended default value is `2.0`.
    - `r_weight`: weight for the quadratic regularization, a non-negative float scalar. If it's `0`, it means solving for unregularized OT.
    - `p`: source distribution, a float32 tensor with shape `[N]`. The elements should be positive and sum up to `1`. (Should be on GPU)
    - `q`: source distribution, a float32 tensor with shape `[M]`. The elements should be positive and sum up to `1`. (Should be on GPU)
    - `eps`: tolerance error for the primal residual, a positive float scalar. Recommended default value is `1e-4`.
    - `max_iter`: maximal number of iterations, a positive integer scalar.
- Output:
    - `x`: the transportation plan, a float32 tensor with the same  shape as `cost`.

To see an example, `./test.py` can be a reference.


## Appendix

### How to setup Python environment
The following commands are from [Tensorflow official tutorial](https://www.tensorflow.org/install/pip).

Please create a conda virtual environment first before preceeding.

```shell
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


