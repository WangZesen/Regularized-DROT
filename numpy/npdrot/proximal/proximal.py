import numpy as np
import scipy.linalg as splinalg
import numba as nb

# PROXIMAL OPERATORS

## UNREGULARIZED OT
@nb.jit(cache=True, nopython=True, fastmath=True, parallel=True)
def trace_nonnegative_prox_nb(x, C, step=1.0):
    """
    Proximal operator for unregularized optimal transport problem with non-negativity constraint. 
    Code is accelerated with numba.
    
    Arguments:
    - x: Variable to be optimized
    - C: Cost matrix
    - step: Step size parameter (default: 1.0)
    
    Returns:
    - reg: Value of regularization term, which is zero
    
    """
    for i in nb.prange(x.size):
        x[i] = max(x[i] - step * C[i], 0.0)
    return 0.

def trace_nonnegative_prox(x, C, step=1.0):
    """
    Proximal operator for unregularized optimal transport problem with non-negativity constraint.
    
    Arguments:
    - x: Variable to be optimized
    - C: Cost matrix
    - step: Step size parameter (default: 1.0)
    
    Returns:
    - quad_reg: Quadratic regularization term
    
    """
    assert C.flags['F_CONTIGUOUS']
    assert x.flags['F_CONTIGUOUS']
    if np.isscalar(x):
        x = np.array([[x]])
    return np.maximum(x - step * C, 0.0, out=x, order='F')


## QUADRATICALLY REGULARIZED OT
@nb.jit(cache=True, nopython=True, fastmath=True, parallel=True)
def trace_nonnegative_quad_prox_nb(x, C, step=1.0, reg=0., **kwargs):
    """
    Proximal operator for quadratically regularized optimal transport problem with non-negativity constraint.
    Code is accelerated with numba.
    
    Arguments:
    - x: Variable to be optimized
    - C: Cost matrix
    - step: Step size parameter (default: 1.0)
    - reg: Regularization parameter (default: 0.0)
    - **kwargs: Additional optional arguments
    
    Returns:
    - quad_reg: Quadratic regularization term
    
    """
    quad_reg = 0.

    for i in nb.prange(x.size):
        x[i] = max(x[i] - step * C[i], 0.0)/(1+step*reg)
        quad_reg += 0.5*reg*x[i]**2
    
    quad_reg *= reg
    return quad_reg

def trace_nonnegative_quad_prox(x, C, step=1.0, reg=0., **kwargs):
    """
    Proximal operator for quadratically regularized optimal transport problem with non-negativity constraint.
    
    Arguments:
    - x: Variable to be optimized
    - C: Cost matrix
    - step: Step size parameter (default: 1.0)
    - reg: Regularization parameter (default: 0.0)
    - **kwargs: Additional optional arguments
    
    Returns:
    - quad_reg: Quadratic regularization term
    
    """
    assert C.flags['F_CONTIGUOUS']
    assert x.flags['F_CONTIGUOUS']
    
    trace_nonnegative_prox(x, C, step=step)
    x /= 1. + step*reg

    quad_reg = 0.5 * reg * np.linalg.norm(x) ** 2
    return quad_reg


## GROUP LASSO REGULARIZED OT
@nb.jit(cache=True, nopython=True, fastmath=True, parallel=True)
def trace_nonnegative_l1l2_prox_nb(x, C, groups, n_cols=1, n_rows=1, step=1.0, reg=1., **kwargs):
    """
    Proximal operator for group lasso regularized optimal transport problem with non-negativity constraint.
    Code is accelerated with numba.
    
    Arguments:
    - x: Variable to be optimized
    - C: Cost matrix
    - groups: Group information for group lasso regularization
    - n_cols: Number of columns in x (default: 1)
    - n_rows: Number of rows in x (default: 1)
    - step: Step size parameter (default: 1.0)
    - reg: Regularization parameter (default: 1.0)
    - **kwargs: Additional optional arguments
    
    Returns:
    - gl_reg: Group lasso regularization term
    
    """
    gl_reg = 0.
    for i in nb.prange(n_cols):
        for group in groups:
            i1, i2 = group
            i1 += i*n_rows
            i2 += i*n_rows

            group_norm2 = 0.

            for j in range(i1, i2):
                  xtemp = max(x[j] - step * C[j], 0.0)
                  group_norm2 += xtemp*xtemp
                  x[j] = xtemp
    
            group_norm = np.sqrt(group_norm2)

            if group_norm > 0:
                scale = max(0., 1-reg*step/group_norm)
                x[i1:i2] *= scale
                gl_reg += scale*group_norm
    
    gl_reg *= reg*gl_reg
    return gl_reg

def trace_nonnegative_l1l2_prox(x, C, groups=None, n_cols=1, n_rows=1, step=1.0, reg=1., **kwargs):
    """
    Proximal operator for group lasso regularized optimal transport problem with non-negativity constraint.
    Code is accelerated with numba.
    
    Arguments:
    - x: Variable to be optimized
    - C: Cost matrix
    - groups: Group information for group lasso regularization
    - n_cols: Number of columns in x (default: 1)
    - n_rows: Number of rows in x (default: 1)
    - step: Step size parameter (default: 1.0)
    - reg: Regularization parameter (default: 1.0)
    - **kwargs: Additional optional arguments
    
    Returns:
    - gl_reg: Group lasso regularization term
    
    """
    assert C.flags['F_CONTIGUOUS']
    assert x.flags['F_CONTIGUOUS']

    
    trace_nonnegative_prox(x, C, step=step)
    gl_reg = 0.

    for i in range(n_cols):
        for group in groups:
            i1, i2 = group
            
            group_norm = np.linalg.norm(x[(i1+i*n_rows):(i2+i*n_rows)])

            if group_norm > 0:
                scale = max(0., 1-reg*step/group_norm)
                x[(i1+i*n_rows):(i2+i*n_rows)] *= scale
                gl_reg += scale*group_norm

    gl_reg *= reg*gl_reg
    return gl_reg

# UTILITIES

def apply_adjoint_operator_and_override(e, f, y, x, T, alpha=1.0, beta=None):
    """
    Apply adjoint operator and override values.
    
    Arguments:
    - e: Vector e
    - f: Vector f
    - y: Vector y
    - x: Vector x
    - T: Target vector
    - alpha: Alpha parameter (default: 1.0)
    - beta: Beta parameter (default: None)
    
    Returns:
    - T: Resulting vector
    
    """
    assert T.flags['F_CONTIGUOUS']
    assert e.size == x.size, "Dimension mismatch"
    assert f.size == y.size, "Dimension mismatch"
    if beta is None:
        beta = alpha
    splinalg.blas.dger(alpha, y, e, a=T, overwrite_a=1)
    splinalg.blas.dger(beta, f, x, a=T, overwrite_a=1)
    return T