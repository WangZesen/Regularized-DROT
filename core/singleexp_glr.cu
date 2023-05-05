#include <vector>
#include <string>
#include "drot_glr.hpp"
#include "utils.hpp"

const float EPS = 1E-4;
const float ALPHA = 2.0;
const float R_WEIGHT = 0.0;
const int MAX_ITERS = 20000;
const int DEFAULT_N_ROWS = 2000;
const int DEFAULT_N_COLS = 2000;
const int N_GROUPS = 2;
const bool USE_WARM_UP = true;

template <typename T, int NGROUPS>
void multi_experiment(const std::string filedir,
        const int n_rows,
        const int n_cols,
        const T step_size,
        const T r_weight,
        const bool use_warmup_init) {
    
    std::vector<T> prep_duration, run_duration, n_iteration, objectives;

    // test with uniform distribution
    std::vector<T> p(n_rows, 1/(T) n_rows);
    std::vector<T> q(n_cols, 1/(T) n_cols);

    // load C matrix
    auto cost = utils::load<T>(filedir, n_rows, n_cols);

    float test_run_dur_in_ms = 0;
    float test_prep_dur_in_ms = 0;
    int test_n_iter = 0;
    T objective = 0;
    
    T *x = group_lasso_regularizer_drot_wrapper<T, NGROUPS>(&cost[0], &p[0], &q[0], n_rows,
        n_cols, step_size, r_weight, MAX_ITERS, EPS, &test_run_dur_in_ms,
        &test_prep_dur_in_ms, &test_n_iter, &objective, use_warmup_init, true);
    
    for (int j = 0; j < n_cols; j++) {
        for (int i = 0; i < n_rows; i++) {
            printf("%.10f ", x[i + j * n_rows]);
        }
        printf("\n");
    }
    printf("%.10f\n", objective);
    printf("%d\n", test_n_iter);

    free(x);
}

int main(int argc, char *argv[]) {
    int n_rows = DEFAULT_N_ROWS;
    int n_cols = DEFAULT_N_COLS;
    int n_groups = N_GROUPS;
    float step_size = ALPHA / (float) (n_rows + n_cols);
    float r_weight = R_WEIGHT;
    bool use_warmup_init = USE_WARM_UP;
    std::string filedir = "./tmp/cmatrix.tmp";
    if (argc == 8) {
        filedir = argv[1];
        n_rows = atoi(argv[2]);
        n_cols = atoi(argv[3]);
        n_groups = atoi(argv[4]);
        step_size = atof(argv[5]) / (float) (n_rows + n_cols);
        r_weight = atof(argv[6]);
        use_warmup_init = (atoi(argv[7]) > 0);
    } else {
        printf("Expected arguments: <cmat_dir> <n_rows> <n_cols> <n_groups> <rho> <r_weight> <use_warmup_init> <is_benchmark>\n");
        printf("Using default values...\n");
    }
    
    if (n_groups == 2) {
        multi_experiment<float, 2>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    } else
    if (n_groups == 3) {
        multi_experiment<float, 3>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    } else
    if (n_groups == 4) {
        multi_experiment<float, 4>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    } else
    if (n_groups == 5) {
        multi_experiment<float, 5>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    } else
    if (n_groups == 6) {
        multi_experiment<float, 6>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    } else
    if (n_groups == 7) {
        multi_experiment<float, 7>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    } else
    if (n_groups == 8) {
        multi_experiment<float, 8>(filedir, n_rows, n_cols, step_size, r_weight, use_warmup_init);
    }

    return 0;
}

