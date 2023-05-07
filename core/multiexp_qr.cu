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
void multi_experiment(const std::string filedir,
        const int n_rows,
        const int n_cols,
        const int n_tests,
        const T step_size,
        const T r_weight,
        const bool use_warmup_init,
        const T eps,
        const int max_iters,
        const int benchmark) {
    
    std::vector<T> prep_duration, run_duration, n_iteration, objectives;

    // test with uniform distribution
    std::vector<T> p(n_rows, 1/(T) n_rows);
    std::vector<T> q(n_cols, 1/(T) n_cols);

    std::string filename;

    for (int idx = 0; idx < n_tests; idx++) {
        filename = filedir + "/cmatrix_" + std::to_string(n_rows) + "_" + std::to_string(n_cols) + "_" +
            std::to_string(idx);
        // load C matrix
        auto cost = utils::load<T>(filename, n_rows, n_cols);

        float test_run_dur_in_ms = 0;
        float test_prep_dur_in_ms = 0;
        T *x = NULL;
        int test_n_iter = 0;
        T objective = 0;
        
        x = quadratic_regularizer_drot_wrapper(&cost[0], &p[0], &q[0], n_rows,
            n_cols, step_size, r_weight, max_iters, eps, &test_run_dur_in_ms,
            &test_prep_dur_in_ms, &test_n_iter, &objective, use_warmup_init, false);

        prep_duration.push_back(test_prep_dur_in_ms);
        run_duration.push_back(test_run_dur_in_ms);
        n_iteration.push_back((float) test_n_iter);
        objectives.push_back(objective);
    }

    if (benchmark == 0) {
        printf("R Weight = %.4f, # Tests = %d\n", (T) r_weight, n_tests - 1);

        float mean_dur, std_dur;
        utils::calc_mean_std(run_duration, &mean_dur, &std_dur);
        printf("\tTime (ms)               = %12.8f (std = %.8f)\n", mean_dur, std_dur);
        
        float mean_n_iter, std_n_iter;
        utils::calc_mean_std(n_iteration, &mean_n_iter, &std_n_iter);
        printf("\t# Iteration             = %12.8f (std = %.8f)\n", mean_n_iter, std_n_iter);

        T mean_obj, std_obj;
        utils::calc_mean_std<T>(objectives, &mean_obj, &std_obj);
        printf("\tObjective               = %12.8f (std = %.8f)\n", mean_obj, std_obj);
        printf("\tTime Per Iteration (ms) = %12.8f\n", mean_dur / mean_n_iter);

        float mean_prep_dur, std_prep_dur;
        utils::calc_mean_std(prep_duration, &mean_prep_dur, &std_prep_dur);
        printf("\tPreperation Time (ms)   = %12.8f (std = %.8f)\n", mean_prep_dur, std_prep_dur);
    } else {
        float mean_dur, std_dur;
        utils::calc_mean_std(run_duration, &mean_dur, &std_dur);
        
        float mean_n_iter, std_n_iter;
        utils::calc_mean_std(n_iteration, &mean_n_iter, &std_n_iter);

        T mean_obj, std_obj;
        utils::calc_mean_std<T>(objectives, &mean_obj, &std_obj);
        printf("test_index\tmethod\tn\tm\teps\tuse_warmup_init\trho\tr_weight\tmax_iters\truntime_ms\tobjective\tn_iteration\truntime_ms_per_iteration\n");
        for (int idx = 1; idx < n_tests; idx++) {
            printf("%d\tquad_drot\t%d\t%d\t%.8f\t%d\t%.8f\t%.8f\t%d\t%.8f\t%.8f\t%d\t%.8f\n",
                idx, n_rows, n_cols, eps, use_warmup_init, step_size * (n_rows + n_cols), r_weight, max_iters,
                run_duration[idx], objectives[idx], int(n_iteration[idx]), run_duration[idx] / float(n_iteration[idx]));
        }
    }
}

int main(int argc, char *argv[]) {
    int n_rows = DEFAULT_N_ROWS;
    int n_cols = DEFAULT_N_COLS;
    int n_tests = DEFAULT_N_TESTS;
    float step_size = ALPHA / (float) (n_rows + n_cols);
    float r_weight = R_WEIGHT;
    int benchmark = 0;
    bool use_warmup_init = USE_WARM_UP;
    float eps = EPS;
    float max_iters = MAX_ITERS;
    std::string filedir = "./data/";
    if (argc == 11) {
        filedir = argv[1];
        n_rows = atoi(argv[2]);
        n_cols = atoi(argv[3]);
        n_tests = atoi(argv[4]);
        step_size = atof(argv[5]) / (float) (n_rows + n_cols);
        r_weight = atof(argv[6]);
        use_warmup_init = (atoi(argv[7]) > 0);
        eps = atof(argv[8]);
        max_iters = atoi(argv[9]);
        benchmark = atoi(argv[10]);
    } else {
        printf("Expected arguments: <cmat_dir> <n_rows> <n_cols> <n_tests> <alpha> <q_weight> <use_warmup_init> <eps> <max_iters> <is_benchmark>\n");
        printf("Using default values...\n");
    }
    
    if (benchmark == 0) {
        printf("n_rows: %d, n_cols: %d, n_tests: %d\n", n_rows, n_cols, n_tests);
        printf("step_size: %.8f, q_weight: %.8f\n", step_size, r_weight);
    }

    multi_experiment<float>(filedir, n_rows, n_cols, n_tests, step_size, r_weight, use_warmup_init, eps, max_iters, benchmark);

    return 0;
}

