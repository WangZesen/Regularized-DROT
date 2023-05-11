#ifndef DROT_KERNEL_H_
#define DROT_KERNEL_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct QuadraticDrotFuntor {
    void operator() (const Device& d,
            const T *c,
            const T *p,
            const T *q,
            const int n_rows,
            const int n_cols,
            const T* step_size,
            const T* r_weight,
            const int64_t* max_iters,
            const T* eps,
            const int work_size_update_x,
            T *x,
            T *a,
            T *row_sum,
            T *row_sum_1,
            T *row_sum_2,
            T *b,
            T *col_sum,
            T *col_sum_1,
            T *col_sum_2,
            T *phi1,
            T *phi2,
            T *aux
    );
};

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
        float *aux
);

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
//         double *aux
// );

int _q_get_work_size_update_x(int n_rows, int n_cols);

#endif // DROT_KERNEL_H_