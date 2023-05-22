import math
import numpy as np
import cupy as cp
import time
import sys
import ot
import os

try:
    DATA_DIR = sys.argv[1]
    OUT_DIR = sys.argv[2]
    N_ROWS = int(sys.argv[3])
    N_COLS = int(sys.argv[4])
    N_CLASS = int(sys.argv[5])
    EPS = float(sys.argv[6])
    MAX_ITER = int(sys.argv[7])
    R_WEIGHT = float(sys.argv[8])
    E_WEIGHT = float(sys.argv[9])
    N_TEST = int(sys.argv[10])
    IS_BENCHMARK = int(sys.argv[11])
except:
    DATA_DIR = "./class/"
    OUT_DIR = os.environ["TMPDIR"] if len(os.environ["TMPDIR"]) else "./tmp/"
    N_COLS = 512
    N_ROWS = 512
    N_CLASS = 2
    MAX_ITER = 20000
    R_WEIGHT = 1e-2
    E_WEIGHT = 1e-4
    N_TEST = 50
    IS_BENCHMARK = 0
    EPS = 1e-4

def load_c(n_test):
    c = np.fromfile(f"{DATA_DIR}/cmatrix_{N_ROWS}_{N_COLS}_{N_CLASS}_{n_test}", dtype=np.float32)
    c = c.reshape((N_ROWS, N_COLS))
    return c

def benchmark(c, p, q, label, reg, eta, max_iter, stop_thres):
    c_cp = cp.asarray(c.astype(np.float64))
    p_cp = cp.asarray(p.astype(np.float64))
    q_cp = cp.asarray(q.astype(np.float64))
    label_cp = cp.asarray(label)
    start = time.perf_counter_ns()
    x = ot.da.sinkhorn_l1l2_gl(p_cp, label_cp, q_cp, c_cp, reg=reg, eta=eta / math.sqrt(N_ROWS/N_CLASS), numItermax=max_iter, stopInnerThr=stop_thres)
    end = time.perf_counter_ns()
    x = cp.asnumpy(x)
    obj = np.sum(np.multiply(x, c))
    res = np.sqrt(np.sum(np.square(np.sum(x, axis=1) - p)) + np.sum(np.square(np.sum(x, axis=0) - q)))
    return end - start, res, obj, x

p = np.ones((N_ROWS,), dtype=np.float32) / N_ROWS
q = np.ones((N_COLS,), dtype=np.float32) / N_COLS
label = np.zeros((N_ROWS,), dtype=np.int32)
for i in range(N_CLASS):
    label[round(N_ROWS / N_CLASS * i):round(N_ROWS / N_CLASS * (i + 1))] = i

ts = []
objs = []
ress = []
n_iters = []
for i in range(N_TEST):
    c = load_c(i)
    t, res, obj, x = benchmark(c, p, q, label, R_WEIGHT, E_WEIGHT, MAX_ITER, EPS)
    ts.append(t / 1e6)
    objs.append(obj)
    ress.append(res)
    filename = os.path.join(OUT_DIR, f"sinkhorn-gl_{N_ROWS}_{N_COLS}_{N_CLASS}_{i}")
    np.save(filename, x.T, allow_pickle=False)

ts = np.array(ts[1:])
objs = np.array(objs[1:])
ress = np.array(ress[1:])

if IS_BENCHMARK == 1:
    print("test_index\tmethod\tn\tm\tn_class\teps\tr_weight\tgl_weight\tmax_iters\truntime_ms\tobjective\tresidual")
    for i in range(ts.shape[0]):
        print(f"{i+1}\tsinkhorn\t{N_ROWS}\t{N_COLS}\t{N_CLASS}\t{EPS:.8f}\t{R_WEIGHT:.8f}\t{E_WEIGHT:.8f}\t{MAX_ITER}\t{ts[i]:.8f}\t{objs[i]:.8f}\t{ress[i]:.8f}")
else:    
    print(f"R Weight = {R_WEIGHT:.4f}")
    print(f"\tTime (ms) = {np.mean(ts):12.8f} (std = {np.std(ts):.8f})")
    print(f"\tObjective = {np.mean(objs):12.8f} (std = {np.std(objs):.8f})")
    print(f"\tResidual  = {np.mean(ress):12.8f} (std = {np.std(ress):.8f})")
