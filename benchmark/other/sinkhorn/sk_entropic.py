import numpy as np
import cupy as cp
import time
import sys
import ot

try:
    DATA_DIR = sys.argv[1]
    N_ROWS = int(sys.argv[2])
    N_COLS = int(sys.argv[3])
    EPS = float(sys.argv[4])
    MAX_ITER = int(sys.argv[5])
    R_WEIGHT = float(sys.argv[6])
    N_TEST = int(sys.argv[7])
    IS_BENCHMARK = int(sys.argv[8])
except:
    DATA_DIR = "./data/"
    N_COLS = 512
    N_ROWS = 512
    MAX_ITER = 10000
    R_WEIGHT = 1e-4
    N_TEST = 50
    IS_BENCHMARK = 0
    EPS = 1e-4

def load_c(n_test):
    c = np.fromfile(f"{DATA_DIR}/cmatrix_{N_ROWS}_{N_COLS}_{n_test}", dtype=np.float32)
    c = c.reshape((N_ROWS, N_COLS))
    return c

def benchmark(c, p, q, reg, max_iter, stop_thres):
    c_cp = cp.asarray(c)
    p_cp = cp.asarray(p)
    q_cp = cp.asarray(q)
    start = time.perf_counter_ns()
    x, log = ot.bregman.sinkhorn_log(p_cp, q_cp, c_cp, reg=reg, numItermax=max_iter, stopThr=stop_thres, log=True)
    end = time.perf_counter_ns()
    x = cp.asnumpy(x)
    obj = np.sum(np.multiply(x, c))
    res = np.sqrt(np.sum(np.square(np.sum(x, axis=1) - p)) + np.sum(np.square(np.sum(x, axis=0) - q)))
    return end - start, res, obj, log['niter']

p = np.ones((N_ROWS,), dtype=np.float32) / N_ROWS
q = np.ones((N_COLS,), dtype=np.float32) / N_COLS

ts = []
objs = []
ress = []
n_iters = []
for i in range(N_TEST):
    c = load_c(i)
    t, res, obj, n_iter = benchmark(c, p, q, R_WEIGHT, MAX_ITER, EPS)
    ts.append(t / 1e6)
    objs.append(obj)
    ress.append(res)
    n_iters.append(n_iter)

ts = np.array(ts[1:])
objs = np.array(objs[1:])
ress = np.array(ress[1:])
n_iters = np.array(n_iters[1:], dtype=np.double)

if IS_BENCHMARK == 1:
    print("test_index\tmethod\tn\tm\teps\tr_weight\tmax_iters\truntime_ms\tobjective\tn_iteration\truntime_ms_per_iteration\tresidual")
    for i in range(ts.shape[0]):
        print(f"{i+1}\tsinkhorn\t{N_ROWS}\t{N_COLS}\t{EPS:.8f}\t{R_WEIGHT:.8f}\t{MAX_ITER}\t{ts[i]:.8f}\t{objs[i]:.8f}\t{int(n_iters[i])}\t{ts[i]/max(1, n_iters[i]):.8f}\t{ress[i]:.8f}")
else:    
    print(f"R Weight = {R_WEIGHT:.4f}")
    print(f"\tTime (ms) = {np.mean(ts):12.8f} (std = {np.std(ts):.8f})")
    print(f"\tObjective = {np.mean(objs):12.8f} (std = {np.std(objs):.8f})")
    print(f"\tResidual  = {np.mean(ress):12.8f} (std = {np.std(ress):.8f})")
    print(f"\t# Iter    = {np.mean(n_iters):12.8f} (std = {np.std(n_iters):.8f})")
