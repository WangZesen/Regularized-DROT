import sys, os
import numpy as np
import ot
import csv

def save(C, nrows, ncols, filename):
    assert C.flags['F_CONTIGUOUS']
    output_file = open(filename, 'wb')
    C.tofile(output_file)
    output_file.close()

def two_dimensional_gaussian_ot(m, n, filename=None):
    d = 2
    mu_s = np.random.normal(0.0, 1.0, (d,)) # Gaussian mean
    A_s = np.random.rand(d, d)
    cov_s = np.dot(A_s, A_s.transpose()) # Gaussian covariance matrix
    mu_t = np.random.normal(5.0, 10.0, (d,))
    A_t = np.random.rand(d, d)
    cov_t = np.dot(A_t, A_t.transpose())
    xs = ot.datasets.make_2D_samples_gauss(m, mu_s, cov_s).astype(np.float32)
    xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t).astype(np.float32)
    p, q = np.ones((m,)).astype(np.float32) / m, np.ones((n,)).astype(np.float32) / n  
    C = np.array(ot.dist(xs, xt)).astype(np.float32)
    C /= C.max()
    if filename is not None:
        save(C, m, n, filename)
    return m, n, C, p, q

try:
    output_dir = sys.argv[1]
    n_rows = int(sys.argv[2])
    n_cols = int(sys.argv[3])
    n_tests = int(sys.argv[4])
    assert n_rows >= 1
    assert n_cols >= 1
    assert n_tests >= 1
    assert os.path.exists(output_dir)
except:
    print(f"Expect Arguments: <output_dir> <n_rows> <n_cols> <n_tests>")
    exit(1)

np.random.seed(123321)

for i in range(n_tests):
    two_dimensional_gaussian_ot(n_rows, n_cols, f"{output_dir}/cmatrix_{n_rows}_{n_cols}_{i}")
print(f"Generated {n_tests} C Matrix ({n_rows}, {n_cols}) at {output_dir}")
