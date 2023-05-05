import os
import tensorflow as tf

# Import Reg-DROT extension
REG_DROT_EXTEND_MODULE_DIR = os.path.join(os.path.dirname(__file__), "bin", "drot_extend.so")
reg_drot = tf.load_op_library(REG_DROT_EXTEND_MODULE_DIR)

# Auxiliary function
def GREEN(s):
    return "\033[1;32m" + s + "\033[0m"

# Test sets
testsets = [
    (2, 3, 2.0, 1e-2, 1e-4, 10000),
    (10, 20, 2.0, 1e-3, 1e-4, 10000),
    (100, 100, 2.0, 1e-2, 1e-4, 10000),
    (20, 10, 2.0, 10., 1e-4, 10000),
    (5000, 5000, 2.0, 1e-3, 1e-4, 10000),
    (1000, 5000, 2.0, 1e-2, 1e-4, 10000),
    (5000, 1000, 2.0, 0, 1e-4, 10000),
]

tf.random.set_seed(12345)

for testset in testsets:
    n, m, rho, r_weight, eps, max_iter = testset
    
    # Randomly generate p, q and cost
    p = tf.ones([n], dtype=tf.float32) / n
    q = tf.ones([m], dtype=tf.float32) / m
    c = tf.random.uniform([m, n], 1e-3, 1, dtype=tf.float32)
    c = c / tf.reduce_max(c)

    # Call module for computing transportation plan
    x = reg_drot.quadratic_drot(c, rho, r_weight, p, q, eps, max_iter)

    # Compute residual
    residual = tf.math.sqrt(
        tf.reduce_sum(tf.square(p - tf.reduce_sum(x, axis=0))) +
        tf.reduce_sum(tf.square(q - tf.reduce_sum(x, axis=1)))
    ).numpy()

    print(f"[Testset] N={n}, M={m}")
    assert residual <= eps, "ERROR: residual doesn't fulfill"
    print(f"\tResidual = {residual:.6f} (<= EPS = {eps:.6f})")
    print(f"\tObjective <C,X> = {tf.reduce_sum(tf.multiply(x, c)).numpy():.6f}")

print(GREEN("PASS"))