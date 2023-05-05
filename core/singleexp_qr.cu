// single experiment for correctness check

#include <vector>
#include <string>
#include "drot_qr.hpp"
#include "utils.hpp"

const float EPS = 1E-4;
const float ALPHA = 2.0;
const float R_WEIGHT = 0.0;
const int MAX_ITERS = 20000;
const int DEFAULT_N_ROWS = 512;
const int DEFAULT_N_COLS = 512;
const int DEFAULT_N_TESTS = 1;
const bool USE_WARM_UP = true;

template <typename T>
void run(const std::string filedir,
        const int n_rows,
        const int n_cols,
        const int n_tests,
        const T step_size,
        const T r_weight,
        const bool use_warmup_init,
        const int benchmark) {
    
    // test with uniform distribution
    std::vector<T> p(n_rows, 1/(T) n_rows);
    std::vector<T> q(n_cols, 1/(T) n_cols);

    std::string filename;

    // load C matrix
    auto cost = utils::load<T>(filedir, n_rows, n_cols);

    float test_run_dur_in_ms = 0;
    float test_prep_dur_in_ms = 0;
    T *x = NULL;
    int test_n_iter = 0;
    T objective = 0;
    
    x = quadratic_regularizer_drot_wrapper(&cost[0], &p[0], &q[0], n_rows,
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
    int n_tests = DEFAULT_N_TESTS;
    float step_size = ALPHA / (float) (n_rows + n_cols);
    float r_weight = R_WEIGHT;
    int benchmark = 0;
    bool use_warmup_init = USE_WARM_UP;
    std::string filedir = "./data/";
    if (argc == 7) {
        filedir = argv[1];
        n_rows = atoi(argv[2]);
        n_cols = atoi(argv[3]);
        step_size = atof(argv[4]) / (float) (n_rows + n_cols);
        r_weight = atof(argv[5]);
        use_warmup_init = (atoi(argv[6]) > 0);
    } else {
        printf("Expected arguments: <cmat_dir> <n_rows> <n_cols> <alpha> <q_weight>\n");
        printf("Using default values...\n");
    }

    run<float>(filedir, n_rows, n_cols, n_tests, step_size, r_weight, use_warmup_init, benchmark);
    return 0;
}

