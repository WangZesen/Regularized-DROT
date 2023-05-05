import torch

# Import Reg-DROT extension
import reg_drot

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


assert torch.cuda.is_available(), "can not run without GPU"
torch.manual_seed(12345)

for testset in testsets:
    n, m, rho, r_weight, eps, max_iter = testset
    
    # Randomly generate p, q and cost
    p = torch.ones([n], dtype=torch.float32) / n
    q = torch.ones([m], dtype=torch.float32) / m
    c = torch.rand([m, n], dtype=torch.float32) + 1e-3
    c = c / torch.max(c)

    # Put it on GPU
    p = p.to("cuda")
    q = q.to("cuda")
    c = c.to("cuda")

    # Call module for computing transportation plan
    x = reg_drot.quadratic_drot(c, p, q, rho, r_weight, max_iter, eps)

    # Compute residual
    residual = torch.sqrt(
        torch.sum(torch.square(p - torch.sum(x, dim=0))) +
        torch.sum(torch.square(q - torch.sum(x, dim=1)))
    ).numpy(force=True)

    print(f"[Testset] N={n}, M={m}")
    assert residual <= eps, "ERROR: residual doesn't fulfill"
    print(f"\tResidual = {residual:.6f} (<= EPS = {eps:.6f})")
    print(f"\tObjective <C,X> = {torch.sum(torch.multiply(x, c)).numpy(force=True):.6f}")

print(GREEN("PASS"))