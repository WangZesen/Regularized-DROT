import numpy as np
import numpy.linalg as nla
import numba.typed as nbtyped
from time import time
from .proximal import *

def regdrot(init, C, p, q, prox, **kwargs):
    """
    Regularized OT solver for optimal transport problem via DR-splitting.
    
    Arguments:
    - init: Initial solution matrix
    - C: Cost matrix
    - p: Marginal distribution vector for rows
    - q: Marginal distribution vector for columns
    - prox: Proximal operator for the regularization
    - **kwargs: Additional optional arguments
    
    Optional Arguments:
    - max_iters: Maximum number of iterations (default: 100)
    - eps_abs: Absolute tolerance for convergence (default: 1e-6)
    - step: Stepsize parameter (default: 2. / (len(p) + len(q)))
    - verbose: Flag to enable verbose output (default: False)
    - print_every: Frequency of printing progress (default: 1)
    - compute_r_primal: Flag to compute primal residual (default: False)
    - compute_r_dual: Flag to compute dual residual (default: False)
    - return_log: Flag to return optimization log (default: False)
    - callback: Callback function (default: None)
    - init_strategy: Flag to use initialization strategy (default: False)
    - dual_offset: Offset function for dual residual (default: None)
    - dual_resid: Dual residual function (default: None)
    
    Returns:
    - x: Solution matrix
    - log (optional): Dictionary containing optimization log
    
    """
    # Stopping parameters
    max_iters = kwargs.pop("max_iters", 100)
    eps_abs = kwargs.pop("eps_abs", 1e-6)

    # Stepsize parameters
    step = kwargs.pop("step", 2. / (len(p) + len(q)))

    # Printing parameters
    verbose = kwargs.pop("verbose", False)
    print_every = kwargs.pop("print_every", 1)

    # Compute residuals
    compute_r_primal = kwargs.pop("compute_r_primal", False)
    compute_r_dual = kwargs.pop("compute_r_dual", False)

    # Logging settings
    return_log = kwargs.pop("return_log", False)

    # Callback
    callbacks = kwargs.pop("callbacks", None)

    # Initialization strategy
    init_strategy = kwargs.pop("init_strategy", False)

    # Offsets for the residuals
    dual_gap_offset_fun = kwargs.pop("dual_offset", None)
    dual_resid_fun = kwargs.pop("dual_resid", None)

    assert C.flags['F_CONTIGUOUS']

    if verbose:
        print("----------------------------------------------------")
        print(" iter | total res | primal res | dual res | time (s)")
        print("------------------------------------------------ ----")

    k = 0
    x = np.array(init, order = 'F')
    m, n = x.shape
    e = np.ones(n)
    f = np.ones(m)

    if init_strategy:
        w = 4*m*n/(m+n)
        t1 = - w * p
        t2 = - w * q
        c1 = w / (m + n)
        c2 = w / (m + n)

        yy = t1 - c1
        xx = t2 - c2

        apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)

        a1 = x.dot(e)
        b1 = x.T.dot(f)

    else:
        a1 = x.dot(e)
        b1 = x.T.dot(f)

    b = np.hstack((p, q))
    r_primal = np.zeros(max_iters)
    r_gap = np.zeros(max_iters)
    r_dual = np.zeros(max_iters)


    if callbacks is not None:
        assert isinstance(callbacks, dict)
        callback_log = {callback_name: np.zeros(max_iters) for callback_name in callbacks.keys()}
    else:
        callback_log = dict()
    
    done = False

    start = time()
    while not done:
        # Implicit F-order for Numba
        reg_value = prox(x.T.reshape(-1), C.T.reshape(-1), step=step, **kwargs)
        a2 = x.dot(e)
        b2 = x.T.dot(f)
        t1 = 2 * a2 - a1 - p
        t2 = 2 * b2 - b1 - q
        c1 = f.dot(t1) / (m + n)
        c2 = e.dot(t2) / (m + n)

        # Broadcasting
        yy = t1 - c1
        xx = t2 - c2
        a1 = a2 - yy - c2
        b1 = b2 - xx - c1

        if compute_r_primal:
            Ax = np.hstack((a2, b2))
            r_primal[k] = nla.norm(Ax - b)
    
            
        if compute_r_dual:
            dual_gap_offset = 0.
            dual_resid = 0.

            if dual_gap_offset_fun is not None:
                dual_gap_offset = 0.
            
            if dual_resid_fun is not None:
                dual_resid = 0.

            r_gap[k] = abs(np.sum(x * C) + reg_value - (-yy.dot(p)/n - xx.dot(q)/m) / step + dual_gap_offset)
            r_dual[k] = dual_resid

        if compute_r_primal or compute_r_dual:
            r_full = max([r_primal[k], r_dual[k]])

        if callbacks is not None:
            for callback_name, callback in callbacks.items():
                callback_log[callback_name][k] = callback(x, -yy.dot(p)/(n*step), -xx.dot(q)/(m*step), **kwargs)


        apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)           

        if (k % print_every == 0 or k == max_iters-1) and verbose:
            print("{}| {}  {}  {}  {}".format(str(k).rjust(6),
                                        format(r_full, ".5e").ljust(10),
                                        format(r_primal[k], ".5e").ljust(11),
                                        format(r_gap[k], ".5e").ljust(9),
                                        format(r_dual[k], ".5e").ljust(8),
                                        format(time() - start, ".2e").ljust(7)))

        k += 1
        done = (k >= max_iters) or (r_full <= eps_abs)

        
    end = time()
    if verbose: print("Drot terminated at iteration ", k-1)

    prox(x.T.reshape(-1), C.T.reshape(-1), step=step, **kwargs)

    if return_log:
        log =   {"dual_sol":     (-yy/n, -xx/m),
                "primal":       np.array(r_primal[:k]),
                "gap":          np.array(r_primal[:k]),
                "dual":         np.array(r_dual[:k]),
                "num_iters":    k,
                "solve_time":   (end - start)}

        for name, logged_callbacks in callback_log.items():
            callback_log[name] = logged_callbacks[:k]

        log.update(callback_log)
        
        return x, log
    
    return x

def drot(init, C, p, q, numba=False, **kwargs):
    """
    OT solver for unregularized optimal transport based on regDROT.
    
    Arguments:
    - init: Initial solution matrix
    - C: Cost matrix
    - p: Marginal distribution vector for rows
    - q: Marginal distribution vector for columns
    - numba: Flag to enable Numba acceleration (default: False)
    - **kwargs: Additional optional arguments
    
    Returns:
    - x: Solution matrix
    
    """
    if numba:
        prox = trace_nonnegative_prox_nb
    else:
        prox = trace_nonnegative_prox

    return regdrot(init, C, p, q, prox, **kwargs)

def l1l2drot(init, C, p, q, reg, groups, numba=False, **kwargs):
    """
    L1-L2 Regularized OT solver for optimal transport via regDROT.
    
    Arguments:
    - init: Initial solution matrix
    - C: Cost matrix
    - p: Marginal distribution vector for rows
    - q: Marginal distribution vector for columns
    - reg: Regularization parameter
    - groups: Group information for L1-L2 regularization
    - numba: Flag to enable Numba acceleration (default: False)
    - **kwargs: Additional optional arguments
    
    Returns:
    - x: Solution matrix
    
    """
    if numba:
        prox = trace_nonnegative_l1l2_prox_nb
        groups_nb = nbtyped.List()
        [groups_nb.append(group) for group in groups]
        groups = groups_nb
    else:
        prox = trace_nonnegative_l1l2_prox

    n_rows, n_cols = len(p), len(q)
    return regdrot(init, C, p, q, prox, reg=reg, groups=groups_nb, n_rows=n_rows, n_cols=n_cols, **kwargs)

def dual_offset_quadot(u, v, C, reg):
    """
    Offset of the duality gap function for quadratic regularization.
    
    Arguments:
    - u: Dual variable u
    - v: Dual variable v
    - C: Cost matrix
    - reg: Regularization parameter
    
    Returns:
    - offset: Offset value
    
    """
    if reg == 0: return 0.

    offset = -(0.5 / reg) * ((np.maximum(0, u.reshape(-1, 1) + v.reshape(1, -1) - C)) ** 2).sum()
    return offset

def quaddrot(init, C, p, q, reg, numba=False, incl_dual_offset=False, **kwargs):
    """
    Quadratic Regularized OT solver via regDROT.
    
    Arguments:
    - init: Initial solution matrix
    - C: Cost matrix
    - p: Marginal distribution vector for rows
    - q: Marginal distribution vector for columns
    - reg: Regularization parameter
    - numba: Flag to enable Numba acceleration (default: False)
    - incl_dual_offset: Flag to include dual offset (default: False)
    - **kwargs: Additional optional arguments
    
    Returns:
    - x: Solution matrix
    
    """
    if numba:
        prox = trace_nonnegative_quad_prox_nb
    else:
        prox = trace_nonnegative_quad_prox

    if incl_dual_offset:
        dual_offset = dual_offset_quadot
    else:
        dual_offset = None

    return regdrot(init, C, p, q, prox, reg=reg, dual_offset=dual_offset, **kwargs)