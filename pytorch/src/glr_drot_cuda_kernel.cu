#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// #include "param_qr.cuh"
#include "drot_glr.hpp"

#define CEILDIV(x, y) ((x+y-1)/y)

// auxiliary function

void group_lasso_regularizer_drot_torch_float32(
        const float *c,
        const float *p,
        const float *q,
        const int n_rows,
        const int n_cols,
        const int NGROUPS,
        const float rho,
        const float r_weight,
        const int max_iters,
        const float eps,
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
        float *col_obj_sum,
        float *col_norm_sum,
        float *group_sum,
        float *group_obj_sum,
        float *group_norm_sum,
        float *phi1,
        float *phi2,
        float *aux) {
    
    float step_size, objective;
    int n_iters, row_size, col_size, mat_size;
    step_size = rho / (float(n_rows) + float(n_cols));

    const float lambda = r_weight * sqrt(n_rows / NGROUPS);
    row_size = n_rows * sizeof(float);
    col_size = n_cols * sizeof(float);
    mat_size = n_rows * n_cols * sizeof(float);

    // initialization
    const float _n = float(n_rows);
    const float _m = float(n_cols);
    const float _k = float(1.0) * step_size * _n * _m - float(2.);

    const float v_phi1 = (_k + float(2)) / _n / (_m + _n);
    const float v_phi2 = (_k + float(2)) / _m / (_m + _n);
    const float v_a = (_k + float(1)) / _n;
    const float v_b = (_k + float(1)) / _m;
    const float v_alpha = _k + float(1);
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
    cudaMemset(aux, 0, 5*sizeof(float));
    cudaMemcpy(&aux[4], &c_alpha_gamma[0], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(x, 0, mat_size);

    group_lasso_regularizer_drot<float>(
            c, p, q, n_rows, n_cols, NGROUPS, step_size, lambda,
            max_iters, eps, work_size_update_x, x, a, row_sum,
            row_sum_1, row_sum_2, b, col_sum, col_sum_1, col_sum_2,
            col_obj_sum, col_norm_sum, group_sum, group_obj_sum,
            group_norm_sum, phi1, phi2, aux, &n_iters, &objective);
}


torch::Tensor group_lasso_drot_cuda(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    int n_class,
    float rho,
    float r_weight,
    int max_iters,
    float eps) {
    
    const int NGROUPS = n_class;
    const int n_rows = (int) p.numel();
    const int n_cols = (int) q.numel();

    TORCH_CHECK((int) c.size(0) == n_cols, "C.size(0) does not match with q.numel()");
    TORCH_CHECK((int) c.size(1) == n_rows, "C.size(1) does not match with p.numel()");

    // tensor options
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA)
        .requires_grad(false);

    // allocate output: transportation plan
    auto x = torch::zeros_like(c, options);

    // allocate temporary variables
    const int work_size_update_x = _gl_get_work_size_update_x(n_rows, n_cols);

    auto a = torch::empty({n_rows}, options);
    auto row_sum = torch::empty({n_rows}, options);
    auto row_sum_1 = torch::empty({n_rows * CEILDIV(n_cols, work_size_update_x)}, options);
    auto row_sum_2 = torch::empty({n_rows * CEILDIV(n_cols, work_size_update_x)}, options);
    auto b = torch::empty({n_cols}, options);
    auto col_sum = torch::empty({n_cols}, options);
    auto col_sum_1 = torch::empty({n_cols * CEILDIV(CEILDIV(n_rows, NGROUPS), UPDATE_X_BLOCK_SIZE_X) * NGROUPS}, options);
    auto col_sum_2 = torch::empty({n_cols * CEILDIV(CEILDIV(n_rows, NGROUPS), UPDATE_X_BLOCK_SIZE_X) * NGROUPS}, options);
    auto col_obj_sum = torch::empty({n_cols * CEILDIV(CEILDIV(n_rows, NGROUPS), UPDATE_X_BLOCK_SIZE_X) * NGROUPS}, options);
    auto col_norm_sum = torch::empty({n_cols * CEILDIV(CEILDIV(n_rows, NGROUPS), UPDATE_X_BLOCK_SIZE_X) * NGROUPS}, options);
    auto group_sum = torch::empty({n_cols * NGROUPS}, options);
    auto group_obj_sum = torch::empty({n_cols * NGROUPS}, options);
    auto group_norm_sum = torch::empty({n_cols * NGROUPS}, options);

    auto phi1 = torch::empty({n_rows}, options);
    auto phi2 = torch::empty({n_cols}, options);

    auto aux = torch::empty({5}, options);

    // run

    group_lasso_regularizer_drot_torch_float32(
        c.data_ptr<float>(),
        p.data_ptr<float>(),
        q.data_ptr<float>(),
        n_rows,
        n_cols,
        NGROUPS,
        rho,
        r_weight,
        max_iters,
        eps,
        work_size_update_x,
        x.data_ptr<float>(),
        a.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        row_sum_1.data_ptr<float>(),
        row_sum_2.data_ptr<float>(),
        b.data_ptr<float>(),
        col_sum.data_ptr<float>(),
        col_sum_1.data_ptr<float>(),
        col_sum_2.data_ptr<float>(),
        col_obj_sum.data_ptr<float>(),
        col_norm_sum.data_ptr<float>(),
        group_sum.data_ptr<float>(),
        group_obj_sum.data_ptr<float>(),
        group_norm_sum.data_ptr<float>(),
        phi1.data_ptr<float>(),
        phi2.data_ptr<float>(),
        aux.data_ptr<float>());

    // return transportation plan
    return x;
}
