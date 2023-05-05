#include "drot_qr.hpp"

void quadratic_regularizer_drot_tf_float32(
        const float *c,
        const float *p,
        const float *q,
        const int n_rows,
        const int n_cols,
        const float* rho,
        const float* r_weight,
        const int64_t* max_iters,
        const float* eps,
        const int work_size_update_x,
        float *x,
        float *a,
        float *row_sum,
        float *row_sum_1,
        float *row_sum_2,
        float *b,
        float *col_sum,
        float *col_sum_1,
        float *col_sum_2,
        float *phi1,
        float *phi2,
        float *aux) {
    
    float _eps, step_size, _r_weight, objective;
    int _max_iters, _n_iters, row_size, col_size, mat_size;
    int64_t max_iters_t;

    cudaMemcpy(&_eps, eps, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&_r_weight, r_weight, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&step_size, rho, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_iters_t, max_iters, sizeof(int64_t), cudaMemcpyDeviceToHost);
    step_size = step_size / (float(n_rows) + float(n_cols));
    _max_iters = static_cast<int>(max_iters_t);

    const float scale = 1. / (1 + step_size * (n_rows + n_cols) * _r_weight);
    row_size = n_rows * sizeof(float);
    col_size = n_cols * sizeof(float);
    mat_size = n_rows * n_cols * sizeof(float);

    // initialization
    const float _n = float(n_rows);
    const float _m = float(n_cols);
    const float _k = float(1.0) * float(2.) * _n * _n * _m * _m / (_n * _n * _n + _m * _m * _m) - float(2.);

    const float v_phi1 = _m * (_k + float(2)) / (_n * _n) / (_m + _n);
    const float v_phi2 = _n * (_k + float(2)) / (_m * _m) / (_m + _n);
    const float v_a = _k / _n;
    const float v_b = _k / _m;
    const float v_alpha = _k;
    const float v_beta = 0;

    std::vector<float> c_phi1(n_rows, v_phi1);
    std::vector<float> c_phi2(n_cols, v_phi2);
    std::vector<float> c_a1(n_rows, v_a);
    std::vector<float> c_b1(n_cols, v_b);
    std::vector<float> c_alpha_gamma(2);
    c_alpha_gamma[0] = v_alpha;
    c_alpha_gamma[1] = v_beta;

    cudaMemcpy(phi1, &c_phi1[0], row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(phi2, &c_phi2[0], col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a, &c_a1[0], row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, &c_b1[0], col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&aux[4], &c_alpha_gamma[0], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(x, 0, mat_size);

    // cudaMemset(a, 0, row_size);
    // cudaMemset(row_sum, 0, row_size);
    // cudaMemset(b, 0, col_size);
    // cudaMemset(col_sum, 0, col_size);
    // cudaMemset(phi1, 0, row_size);
    // cudaMemset(phi2, 0, col_size);

    // dim3 grid_init_x(
    //         (n_rows + INIT_X_BLOCK_SIZE_X - 1) / INIT_X_BLOCK_SIZE_X,
    //         (n_cols + INIT_X_BLOCK_SIZE_Y - 1) / INIT_X_BLOCK_SIZE_Y);
    // dim3 block_init_x(INIT_X_THREAD_SIZE_X);
    // init_x<<<grid_init_x, block_init_x>>>(x, p, q, n_rows, n_cols);

    quadratic_regularizer_drot<float>(
            c, p, q, n_rows, n_cols, step_size, scale, _r_weight,
            _max_iters, _eps, work_size_update_x, x, a, row_sum,
            row_sum_1, row_sum_2, b, col_sum, col_sum_1, col_sum_2,
            phi1, phi2, aux, &_n_iters, &objective);
    // printf("%d\n", _n_iters);
}


// void quadratic_regularizer_drot_tf_double(
//         const double *c,
//         const double *p,
//         const double *q,
//         const int n_rows,
//         const int n_cols,
//         const double* rho,
//         const double* r_weight,
//         const int64_t* max_iters,
//         const double* eps,
//         const int work_size_update_x,
//         double *x,
//         double *a,
//         double *row_sum,
//         double *row_sum_1,
//         double *row_sum_2,
//         double *b,
//         double *col_sum,
//         double *col_sum_1,
//         double *col_sum_2,
//         double *phi1,
//         double *phi2,
//         double *aux) {
    
//     double _eps, step_size, _r_weight, objective;
//     int _max_iters, _n_iters, row_size, col_size, mat_size;
//     int64_t max_iters_t;

//     cudaMemcpy(&_eps, eps, sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&_r_weight, r_weight, sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&step_size, rho, sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&max_iters_t, max_iters, sizeof(int64_t), cudaMemcpyDeviceToHost);
//     step_size = step_size / (double(n_rows) + double(n_cols));
//     _max_iters = static_cast<int>(max_iters_t);

//     const double scale = 1. / (1 + step_size * (n_rows + n_cols) * _r_weight);
//     row_size = n_rows * sizeof(double);
//     col_size = n_cols * sizeof(double);
//     mat_size = n_rows * n_cols * sizeof(double);

//     // initialization
//     const double _n = double(n_rows);
//     const double _m = double(n_cols);
//     const double _k = double(1.) * double(2.) * _n * _n * _m * _m / (_n * _n * _n + _m * _m * _m) - double(2.);

//     const double v_phi1 = _m * (_k + double(2)) / (_n * _n) / (_m + _n);
//     const double v_phi2 = _n * (_k + double(2)) / (_m * _m) / (_m + _n);
//     const double v_a = _k / _n;
//     const double v_b = _k / _m;
//     const double v_alpha = _k;
//     const double v_beta = 0;

//     std::vector<double> c_phi1(n_rows, v_phi1);
//     std::vector<double> c_phi2(n_cols, v_phi2);
//     std::vector<double> c_a1(n_rows, v_a);
//     std::vector<double> c_b1(n_cols, v_b);
//     std::vector<double> c_alpha_gamma(2);
//     c_alpha_gamma[0] = v_alpha;
//     c_alpha_gamma[1] = v_beta;

//     cudaMemcpy(phi1, &c_phi1[0], row_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(phi2, &c_phi2[0], col_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(a, &c_a1[0], row_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(b, &c_b1[0], col_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(&aux[4], &c_alpha_gamma[0], sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemset(x, 0, mat_size);

//     quadratic_regularizer_drot<double>(
//             c, p, q, n_rows, n_cols, step_size, scale, _r_weight,
//             _max_iters, _eps, work_size_update_x, x, a, row_sum,
//             row_sum_1, row_sum_2, b, col_sum, col_sum_1, col_sum_2,
//             phi1, phi2, aux, &_n_iters, &objective);
// }


