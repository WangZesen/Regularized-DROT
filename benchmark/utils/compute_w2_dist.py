# Compute Wasserstain-2 Distance

import os
import ot
import sys
import glob
import numpy as np
import torch
import reg_drot

DA_DATA_DIR = sys.argv[1]
DA_OUT_DATA_DIR = sys.argv[2]

DROT_RHO = 2.0
DROT_MAX_ITERS = 50000
DROT_EPS = 2e-6

def get_all_drot_output(DA_OUT_DATA_DIR):
    drot = glob.glob(os.path.join(DA_OUT_DATA_DIR, "drot", "*", "drot-gl_*"))
    def get_r_weight(item):
        return item.split("/")[-2]
    def get_n_row(item):
        return int(item.split("/")[-1].split("_")[1])
    def get_n_col(item):
        return int(item.split("/")[-1].split("_")[2])
    def get_n_class(item):
        return int(item.split("/")[-1].split("_")[3])
    def get_index(item):
        return int(item.split("/")[-1].split("_")[4])
    drot = [[item, get_n_row(item), get_n_col(item), get_n_class(item), get_index(item), get_r_weight(item)] for item in drot]
    drot.sort(key=lambda x: x[-2] * 100 + float(x[-1]))
    return drot

def get_all_sk_output(DA_OUT_DATA_DIR):
    sk = glob.glob(os.path.join(DA_OUT_DATA_DIR, "sk", "*", "*", "sinkhorn-gl_*"))
    def get_r_weight(item):
        return item.split("/")[-3]
    def get_gl_weight(item):
        return item.split("/")[-2]
    def get_n_row(item):
        return int(item[:-4].split("/")[-1].split("_")[1])
    def get_n_col(item):
        return int(item[:-4].split("/")[-1].split("_")[2])
    def get_n_class(item):
        return int(item[:-4].split("/")[-1].split("_")[3])
    def get_index(item):
        return int(item[:-4].split("/")[-1].split("_")[4])
    sk = [[item, get_n_row(item), get_n_col(item), get_n_class(item), get_index(item), get_r_weight(item), get_gl_weight(item)] for item in sk]
    sk.sort(key=lambda x: x[-3] * 100 + float(x[-2]))
    return sk

def load_x(filename):
    mat = []
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            content = [float(x) for x in line.strip(" \n").split(" ")]
            mat.append(content)
            line = f.readline()
    return np.array(mat, dtype=np.float32) # [N, M]

def adapt_target_domain(Xt, transportation_plan, p):
    scale = transportation_plan.sum(axis=0)
    scale += p
    scale /= 2
    return transportation_plan.T @ Xt / scale.reshape(-1, 1)

def compute_group_wasserstain_2_dist(Xt, trans, n_group):
    _p = np.ones((trans.shape[1],), dtype=np.float32) / trans.shape[1]
    adapted_Xt = adapt_target_domain(Xt, trans, _p)

    total_obj = 0.
    for i in range(n_group):
        s = round(Xt.shape[0] / n_group * i)
        t = round(Xt.shape[0] / n_group * (i + 1))
        adapted_s = round(adapted_Xt.shape[0] / n_group * i)
        adapted_t = round(adapted_Xt.shape[0] / n_group * (i+1))
        c = np.array(ot.dist(adapted_Xt[adapted_s:adapted_t], Xt[s:t]))
        c_max = np.max(c)
        c = torch.Tensor(c / c_max).to("cuda:0")

        p = torch.ones((t - s,), dtype=torch.float32).to("cuda:0") / (t - s)
        q = torch.ones((adapted_t - adapted_s,), dtype=torch.float32).to("cuda:0") / (adapted_t - adapted_s)

        x = reg_drot.quadratic_drot(c, p, q, DROT_RHO, 0., DROT_MAX_ITERS, DROT_EPS)
        residual = torch.sqrt(torch.sum(torch.square(torch.sum(x, dim=0) - p)) + torch.sum(torch.square(torch.sum(x, dim=1) - q)))
        assert residual < DROT_EPS*2, f"{residual}"
        total_obj += torch.sum(torch.multiply(c, x)).numpy(force=True) * c_max
    return total_obj

def find_r_weight(data, r_weight):
    for item in data:
        if (abs(float(item[5]) - float(r_weight)) < 1e-9):
            return item[5]
    raise Exception()

def find_gl_weight(data, gl_weight):
    for item in data:
        if (abs(float(item[6]) - float(gl_weight)) < 1e-9):
            return item[6]
    raise Exception()

drot = get_all_drot_output(DA_OUT_DATA_DIR)

# Add results to log
content = []
with open("./log/gl_drot.log", "r") as f:
    content = [x.strip("\n") for x in f.readlines()]

i = 1
with open("./log/gl_drot_amend.log", "w") as f:
    while i < len(content):
        assert content[i][0].isalpha()
        if "within_class_w2_dist" in content[i]:
            print(content[i], file=f)
            i += 1
            while (i < len(content)) and (not content[i][0].isalpha()):
                print(content[i], file=f)
                i += 1
            continue
        else:
            print(content[i]+"\twithin_class_w2_dist", file=f)
            test_index_ind = content[i].split("\t").index("test_index")
            n_ind = content[i].split("\t").index("n")
            m_ind = content[i].split("\t").index("m")
            n_class_ind = content[i].split("\t").index("n_class")
            r_weight_ind = content[i].split("\t").index("r_weight")
            i += 1
            while (i < len(content)) and (not content[i][0].isalpha()):
                test_index = int(content[i].split("\t")[test_index_ind])
                n = int(content[i].split("\t")[n_ind])
                m = int(content[i].split("\t")[m_ind])
                n_class = int(content[i].split("\t")[n_class_ind])
                r_weight = content[i].split("\t")[r_weight_ind]
                r_weight_str = find_r_weight(drot, r_weight)

                Xt = np.load(os.path.join(DA_DATA_DIR, f"cmatrix_{n}_{m}_{n_class}_{test_index}_Xt.npy"), allow_pickle=False)
                trans_file = os.path.join(DA_OUT_DATA_DIR, "drot", r_weight_str, f"drot-gl_{n}_{m}_{n_class}_{test_index}")
                trans = load_x(trans_file)
                obj = compute_group_wasserstain_2_dist(Xt, trans, n_class)
                print(content[i]+f"\t{obj:.8f}", file=f, flush=True)
                i += 1
                print(f"{i:4d}/{len(content)}")
                
sk = get_all_sk_output(DA_OUT_DATA_DIR)

# Add results to log
content = []
with open("./log/sk_entropic_gl.log", "r") as f:
    content = [x.strip("\n") for x in f.readlines()]

i = 1
with open("./log/sk_entropic_gl_amend.log", "w") as f:
    while i < len(content):
        assert content[i][0].isalpha()
        if "within_class_w2_dist" in content[i]:
            print(content[i], file=f)
            i += 1
            while (i < len(content)) and (not content[i][0].isalpha()):
                print(content[i], file=f)
                i += 1
            continue
        else:
            print(content[i]+"\twithin_class_w2_dist", file=f)
            test_index_ind = content[i].split("\t").index("test_index")
            n_ind = content[i].split("\t").index("n")
            m_ind = content[i].split("\t").index("m")
            n_class_ind = content[i].split("\t").index("n_class")
            r_weight_ind = content[i].split("\t").index("r_weight")
            gl_weight_ind = content[i].split("\t").index("gl_weight")
            i += 1
            while (i < len(content)) and (not content[i][0].isalpha()):
                test_index = int(content[i].split("\t")[test_index_ind])
                n = int(content[i].split("\t")[n_ind])
                m = int(content[i].split("\t")[m_ind])
                n_class = int(content[i].split("\t")[n_class_ind])
                r_weight = content[i].split("\t")[r_weight_ind]
                gl_weight = content[i].split("\t")[gl_weight_ind]
                
                r_weight_str = find_r_weight(sk, r_weight)
                gl_weight_str = find_gl_weight(sk, gl_weight)

                Xt = np.load(os.path.join(DA_DATA_DIR, f"cmatrix_{n}_{m}_{n_class}_{test_index}_Xt.npy"), allow_pickle=False)

                trans_file = os.path.join(DA_OUT_DATA_DIR, "sk", r_weight_str, gl_weight_str, f"sinkhorn-gl_{n}_{m}_{n_class}_{test_index}.npy")
                trans = np.load(trans_file, allow_pickle=False)
                obj = compute_group_wasserstain_2_dist(Xt, trans, n_class)
                print(content[i]+f"\t{obj:.8f}", file=f, flush=True)
                print(f"{i:4d}/{len(content)}")
                i += 1
