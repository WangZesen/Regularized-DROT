import sys
import time
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype = torch.float64

assert torch.cuda.is_available(), "No GPU Support"
device = torch.device("cuda:0")

DATA_DIR = sys.argv[1]
N_ROWS = int(sys.argv[2])
N_COLS = int(sys.argv[3])
R_WEIGHT = float(sys.argv[4])
LR = float(sys.argv[5])
EPS = float(sys.argv[6])
HISTORY_SIZE = int(sys.argv[7])
N_TESTS = int(sys.argv[8])
MAX_ITERS = 20000

if abs(R_WEIGHT) < 1e-9:
    exit(0)

SCALED_R_WEIGHT = R_WEIGHT * (N_ROWS + N_COLS)

def load_c(n_test):
    c = np.fromfile(f"{DATA_DIR}/cmatrix_{N_ROWS}_{N_COLS}_{n_test}", dtype=np.float32)
    c = c.reshape((N_ROWS, N_COLS))
    return c

class LossModule(torch.nn.Module):
    def __init__(self) -> None:
        super(LossModule, self).__init__()
        self._x = torch.nn.Parameter(torch.ones((1, N_ROWS)).double() / N_ROWS)
        self._y = torch.nn.Parameter(torch.ones((1, N_COLS)).double() / N_COLS)
    
    def forward(self, p, q, c, alpha):
        out = - torch.sum(torch.multiply(self._x, p))
        out += - torch.sum(torch.multiply(self._y, q))
        _x = torch.tile(self._x.view((-1, 1)), (1, N_COLS))
        _y = torch.tile(self._y, (N_ROWS, 1))
        out += 0.5 / alpha * torch.sum(torch.square(torch.nn.functional.relu(_x + _y - c)))
        return out

def closure():
    optim.zero_grad()
    loss = module(p, q, c, SCALED_R_WEIGHT)
    loss.backward()
    return loss

torch.manual_seed(123)
np.random.seed(123)

print("test_index\tmethod\tn\tm\teps\tlearning_rate\thistory_size\tr_weight\tmax_iters\truntime_ms\tobjective\tresidual\tobjective+L2", flush=True)

for n_test in range(N_TESTS):
    _c = load_c(n_test)
    _p = np.ones([_c.shape[0], ]) / _c.shape[0]
    _q = np.ones([_c.shape[1], ]) / _c.shape[1]

    c = torch.Tensor(_c).double().to(device)
    p = torch.Tensor(_p).double().to(device)
    q = torch.Tensor(_q).double().to(device)

    module = LossModule()
    optim = torch.optim.LBFGS(module.parameters(),
                              lr=LR,
                              history_size=HISTORY_SIZE,
                              line_search_fn="strong_wolfe",
                              max_iter=MAX_ITERS,
                              tolerance_grad=EPS/5.,
                              tolerance_change=0.)
    module = module.to(device)

    
    start = time.time_ns()
    optim.step(closure=closure)
    end = time.time_ns()

    with torch.no_grad():
        _x = torch.tile(module._x.view((-1, 1)), (1, N_COLS))
        _y = torch.tile(module._y, (N_ROWS, 1))
        x = (torch.nn.functional.relu(_x + _y - c) / SCALED_R_WEIGHT).numpy(force=True)

        if n_test > 0:
            residual = np.sqrt(np.sum(np.square(_p - np.sum(x, axis=1))) + np.sum(np.square(_q - np.sum(x, axis=0))))

            objective = np.sum(np.multiply(x, _c))
            objective_L2 = np.sum(np.multiply(x, _c)) + SCALED_R_WEIGHT / 2. * np.sum(np.square(x))
            print(f"{n_test}\tl-bfgs\t{N_ROWS}\t{N_COLS}\t{EPS:.8f}\t{LR:.8f}\t{HISTORY_SIZE}\t{R_WEIGHT:.8f}\t{MAX_ITERS}\t{(end-start)/1e6:.8f}\t{objective:.8f}\t{residual:.8f}\t{objective_L2:.8f}", flush=True)
    
    del module
    del optim
