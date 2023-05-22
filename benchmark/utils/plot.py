import os
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

CONVERT_MAP = {
    "int": ["test_index", "n", "m", "use_warmup_init", "max_iters", "n_iteration", "n_class"],
    "str": ["method", "history_size"],
    "float": ["eps", "rho", "r_weight", "runtime_ms", "objective", "runtime_ms_per_iteration",
              "residual", "gl_weight", "within_class_w2_dist", "learning_rate", "objective+L2",
              "normed_within_class_w2_dist"],
}

def _convert(columns, content):
    ret = []
    for index in range(len(columns)):
        if columns[index] in CONVERT_MAP["int"]:
            ret.append(int(content[index]))
        elif columns[index] in CONVERT_MAP["str"]:
            ret.append(str(content[index]))
        elif columns[index] in CONVERT_MAP["float"]:
            ret.append(float(content[index]))
        else:
            raise NotImplementedError(f"Unknown Column: {columns[index]}")
    return ret

def load_log(log_dir):
    log = []
    with open(log_dir) as f:
        line = f.readline().strip("\n")
        if len(line) == 0:
            line = f.readline().strip("\n")
        columns = line.split("\t")
        
        line = f.readline()
        while line:
            if line[0].isalpha():
                line = f.readline()
                continue
            content = _convert(columns, line.strip("\n").split("\t"))
            log.append(content)
            line = f.readline()
    df = pd.DataFrame(log)
    df.columns = columns
    return df

def compare_ot(df: pd.DataFrame):
    testsets = {}
    for i in range(len(df)):
        testsets[(df.iloc[i]["n"], df.iloc[i]["m"])] = True
    testsets = [item for item in testsets.keys()]
    testsets.sort(key=lambda x: x[0] * 1e5 + x[1])
    
    for (n, m) in testsets:
        sub_df = df.loc[(df.n == n) & (df.m == m)]
        sub_df = sub_df.groupby(["method", "r_weight", "rho"], dropna=False).mean(numeric_only=True)

        sfig = seaborn.scatterplot(sub_df, x="runtime_ms", y="objective", hue="method", alpha=0.4)
        plt.ylabel("Objective <C,X>")
        plt.xlabel("Runtime (ms)")
        plt.title(f"Optimal Transport (Log Scale). N = {n}, M = {m}")
        plt.xscale("log")
        plt.tight_layout()
        fig = sfig.get_figure()
        fig.savefig(f"./plot/OT_{n}-{m}.png", dpi=300)
        plt.clf()


def compare_reg(df: pd.DataFrame):
    testsets = {}
    for i in range(len(df)):
        testsets[(df.iloc[i]["n"], df.iloc[i]["m"])] = True
    testsets = [item for item in testsets.keys()]
    testsets.sort(key=lambda x: x[0] * 1e5 + x[1])

    for (n, m) in testsets:
        sub_df = df.loc[(df.n == n) & (df.m == m)]

        plt.figure(figsize=(10,6))
        sfig = seaborn.boxplot(sub_df, x="r_weight", y="runtime_ms", orient="v", hue="method", fliersize=0.5, linewidth=1, width=0.5)
        plt.ylabel("Runtime in ms")
        plt.xlabel("Unnormed Weight for Quadratic Regularizer")
        plt.title(f"Runtime Comparison (Log Scale). N={n}, M={m}")
        plt.yscale("log")
        plt.tight_layout()
        fig = sfig.get_figure()
        fig.savefig(f"./plot/REG_{n}-{m}.png", dpi=300)
        plt.clf()

def compare_da(df: pd.DataFrame):
    testsets = {}
    for i in range(len(df)):
        testsets[(df.iloc[i]["n"], df.iloc[i]["m"], df.iloc[i]["n_class"])] = True
    testsets = [item for item in testsets.keys()]
    testsets.sort(key=lambda x: x[0] * 1e5 + x[1])
    
    for (n, m, n_class) in testsets:
        sub_df = df.loc[(df.n == n) & (df.m == m)]
        sub_df = sub_df.groupby(["method", "r_weight", "gl_weight", "rho"], dropna=False).mean(numeric_only=True)

        sfig = seaborn.scatterplot(sub_df, x="runtime_ms", y="normed_within_class_w2_dist", hue="method", alpha=0.4)
        plt.ylabel("Objective")
        plt.xlabel("Runtime (ms)")
        plt.title(f"Domain Adaptation (Log Scale). Target={n}, Source={m}, Class={n_class}")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        fig = sfig.get_figure()
        fig.savefig(f"./plot/DA_{n}-{m}.png", dpi=300)
        plt.clf()

LOG_DIR = "./log/"
os.makedirs("./plot", exist_ok=True)

# Compare Optimal Transport
quad_drot_df = load_log(os.path.join(LOG_DIR, "quad_drot.log"))
sk_df = load_log(os.path.join(LOG_DIR, "sk_entropic.log"))
lbfgs_df = load_log(os.path.join(LOG_DIR, "lbfgs.log"))
ot_df = pd.concat([quad_drot_df, sk_df, lbfgs_df])
compare_ot(ot_df)

# Compare Domain Adaptation
gl_drot_df = load_log(os.path.join(LOG_DIR, "gl_drot_amend.log"))
gl_sk_df = load_log(os.path.join(LOG_DIR, "sk_entropic_gl_amend.log"))
da_df = pd.concat([gl_drot_df, gl_sk_df])
compare_da(da_df)

# Compare Reg and Runtime between DROT and L-BFGS
quad_drot_df = load_log(os.path.join(LOG_DIR, "quad_drot.log"))
lbfgs_df = load_log(os.path.join(LOG_DIR, "lbfgs.log"))
reg_df = pd.concat([quad_drot_df, lbfgs_df])
compare_reg(reg_df)
