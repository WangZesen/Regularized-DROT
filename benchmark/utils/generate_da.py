import os
import ot
import sys
import numpy as np
from sklearn.datasets import make_blobs

def save(C, nrows, ncols, filename):
    output_file = open(filename, 'wb')
    C.tofile(output_file)
    output_file.close()

def save_points(X, filename):
    np.save(filename, X, allow_pickle=False)

def make_dataset(n_class, n_src, n_tgt, seed=None, dim=2, std=2):
    Xs, ys, centers = make_blobs(n_samples=n_src, n_features=dim, centers=n_class, 
                              shuffle=False, return_centers=True, random_state=seed, cluster_std=std/np.sqrt(n_class))
    
    try:
        seed *= 7
    except:
        pass
    
    Xt, yt = make_blobs(n_samples=n_tgt, n_features=dim, centers=centers, 
                              shuffle=False, return_centers=False, random_state=seed, cluster_std=std/np.sqrt(n_class))
    
    np.random.seed(seed)
    k = 5
    n_rows_A = min([k, dim])
    A = np.random.normal(size=(n_rows_A, dim))
    transform = A.T @ A / n_rows_A
    
    alpha = 0.3
    transform = alpha*np.eye(dim) + (1-alpha)*transform

    Xt = Xt @ transform
    translation = 5*np.random.normal(size=(1, dim))
    Xt += translation

    C = np.array(ot.dist(Xs, Xt)).T.astype(np.float32) # [N, M]
    C /= C.max()
    
    return Xs, ys, Xt, yt, C

try:
    output_dir = sys.argv[1]
    n_tgt = int(sys.argv[2]) # N
    n_src = int(sys.argv[3]) # M
    n_class = int(sys.argv[4])
    n_tests = int(sys.argv[5])
    assert n_src >= 1
    assert n_tgt >= 1
    assert n_class >= 2
    assert n_tests >= 1
    assert os.path.exists(output_dir)
except:
    print(f"Expect Arguments: <output_dir> <n_src> <n_tgt> <n_class> <n_tests>")
    exit(1)

np.random.seed(123321)

for i in range(n_tests):
    _seed = int(np.random.uniform(1, 1e8))
    Xs, ys, Xt, yt, C = make_dataset(n_class, n_src, n_tgt, _seed)
    filename = os.path.join(output_dir, f"cmatrix_{n_tgt}_{n_src}_{n_class}_{i}")
    save(C, n_src, n_tgt, filename)
    filename = os.path.join(output_dir, f"cmatrix_{n_tgt}_{n_src}_{n_class}_{i}_Xs")
    save_points(Xs, filename)
    filename = os.path.join(output_dir, f"cmatrix_{n_tgt}_{n_src}_{n_class}_{i}_Xt")
    save_points(Xt, filename)

print(f"Generated {n_tests} C Matrix ({n_tgt}, {n_src}) with {n_class} classes at {output_dir}")