#include <stdio.h>
#include <math.h>
#include <vector>
#include "kernel_qr.hpp"
#include "param_qr.hpp"

#define CEILDIV(x, y) ((x+y-1)/y)

template <typename T>
void reduce_axis_sum(const int n,
        int n_block_left,
        T *p_sum_1,
        T *p_sum_2,
        T *sum) {
    
    dim3 reduce_grid(CEILDIV(n_block_left, REDUCE_AXIS_SUM_BLOCK_SIZE_X), n);
    dim3 reduce_block(REDUCE_AXIS_SUM_THREAD_SIZE_X);
    int o = 0;
    while (n_block_left > REDUCE_AXIS_SUM_BLOCK_SIZE_X) {
        if (o % 2 == 0) {
            reduce_axis_sum_step<<<reduce_grid, reduce_block>>>(n_block_left, p_sum_1, p_sum_2);
        } else {
            reduce_axis_sum_step<<<reduce_grid, reduce_block>>>(n_block_left, p_sum_2, p_sum_1);
        }
        n_block_left = CEILDIV(n_block_left, REDUCE_AXIS_SUM_BLOCK_SIZE_X);
        reduce_grid.x = CEILDIV(n_block_left, REDUCE_AXIS_SUM_BLOCK_SIZE_X);
        o += 1;
    }
    
    if (o % 2 == 0) {
        reduce_axis_sum_step<<<reduce_grid, reduce_block>>>(n_block_left, p_sum_1, sum);
    } else {
        reduce_axis_sum_step<<<reduce_grid, reduce_block>>>(n_block_left, p_sum_2, sum);
    }
}

template <typename T>
void reduce_all_sum(int n_block_left,
    const T *col_sum,
    T *p_sum_1,
    T *p_sum_2,
    T *sum) {
    
    dim3 reduce_grid((n_block_left + REDUCE_AXIS_SUM_BLOCK_SIZE_X - 1)/ REDUCE_AXIS_SUM_BLOCK_SIZE_X, 1);
    dim3 reduce_block(REDUCE_AXIS_SUM_THREAD_SIZE_X);

    reduce_axis_sum_step<<<reduce_grid, reduce_block>>>(n_block_left, col_sum, p_sum_1);
    n_block_left = CEILDIV(n_block_left, REDUCE_AXIS_SUM_BLOCK_SIZE_X);
    reduce_axis_sum(1, n_block_left, p_sum_1, p_sum_2, sum);
}

template <typename T>
void quadratic_regularizer_step(T *x,
        const T step_size,
        const T scale,
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
        const T *c,
        const T *p,
        const T *q,
        const int n_rows,
        const int n_cols,
        const int work_size_update_x,
        const int n_iter,
        T *aux) {
    // aux[0]: objective value, aux[1]: sum of matrix X
    // aux[2]: row residual,    aux[3]: col residual
    // aux[4]: alpha

    dim3 grid_update_x(CEILDIV(n_rows, UPDATE_X_BLOCK_SIZE_X),
            CEILDIV(n_cols, work_size_update_x));
    dim3 block_update_x(UPDATE_X_THREAD_SIZE_X);
    
    if (n_iter % 2 == 0) {
        if (min(n_cols, n_rows) > 2 * G_BLOCK_SIZE) {
            quadratic_regularizer_update_x_even<T><<<grid_update_x, block_update_x>>>(
                x,
                phi1,
                phi2,
                c,
                step_size,
                scale,
                n_rows,
                n_cols,
                n_rows * n_cols,
                work_size_update_x,
                row_sum_1,
                col_sum_1,
                &aux[0]
            );
        } else {
            quadratic_regularizer_update_x_even_old<T><<<grid_update_x, block_update_x>>>(
                x,
                phi1,
                phi2,
                c,
                step_size,
                scale,
                n_rows,
                n_cols,
                work_size_update_x,
                row_sum_1,
                col_sum_1,
                &aux[0]
            );
        }
    } else {
        if (min(n_cols, n_rows) > 2 * G_BLOCK_SIZE) {
            quadratic_regularizer_update_x_odd<T><<<grid_update_x, block_update_x>>>(
                x,
                phi1,
                phi2,
                step_size,
                scale,
                n_rows,
                n_cols,
                n_rows * n_cols,
                work_size_update_x,
                row_sum_1,
                col_sum_1
            );
        } else {
            quadratic_regularizer_update_x_odd_old<T><<<grid_update_x, block_update_x>>>(
                x,
                phi1,
                phi2,
                step_size,
                scale,
                n_rows,
                n_cols,
                work_size_update_x,
                row_sum_1,
                col_sum_1
            );
        }
    }
    reduce_axis_sum<T>(n_rows, grid_update_x.y, row_sum_1, row_sum_2, row_sum);
    reduce_axis_sum<T>(n_cols, grid_update_x.x, col_sum_1, col_sum_2, col_sum);
    reduce_all_sum<T>(n_cols, col_sum, col_sum_1, col_sum_2, &aux[1]);

    dim3 grid_update_aux(CEILDIV(max(n_rows, n_cols), UPDATE_AUX_BLOCK_SIZE_X), 2);
    dim3 block_update_aux(UPDATE_AUX_BLOCK_SIZE_X);
    quadratic_regularizer_update_aux<T><<<grid_update_aux, block_update_aux>>>(
            n_rows, n_cols, phi1, phi2, p, q, row_sum, col_sum, &aux[1], row_sum_1, col_sum_1, a, b, &aux[4]);
    
    update_alpha<T><<<1, 1>>>(&aux[1], &aux[4]);
    reduce_axis_sum<T>(1, CEILDIV(n_rows, UPDATE_AUX_BLOCK_SIZE_X) * 2, row_sum_1, row_sum_2, &aux[2]);
    reduce_axis_sum<T>(1, CEILDIV(n_cols, UPDATE_AUX_BLOCK_SIZE_X) * 2, col_sum_1, col_sum_2, &aux[3]);
}

template <typename T>
void quadratic_regularizer_drot(
        const T *c,
        const T *p,
        const T *q,
        const int n_rows,
        const int n_cols,
        const T step_size,
        const T scale,
        const T r_weight,
        const int max_iters,
        const T eps,
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
        T *aux,
        int *_n_iter,
        T *_objective) {
    T _aux[5];
    T residual = 1e9;
    int n_iter = 0;
    
    while ((n_iter < max_iters) /*&& (residual > eps)*/) {
        cudaMemset(aux, 0, 4 * sizeof(T));
        cudaMemset(row_sum_1, 0, n_rows * CEILDIV(n_cols, work_size_update_x) * sizeof(T));
        cudaMemset(col_sum_1, 0, n_cols * CEILDIV(n_rows, UPDATE_X_BLOCK_SIZE_X) * sizeof(T));
        quadratic_regularizer_step(x,
                step_size,
                scale,
                a,
                row_sum,
                row_sum_1,
                row_sum_2,
                b,
                col_sum,
                col_sum_1,
                col_sum_2,
                phi1,
                phi2,
                c,
                p,
                q,
                n_rows,
                n_cols,
                work_size_update_x,
                n_iter,
                aux
        );

        if (n_iter % 2 == 0) {
            cudaMemcpy(_aux, aux, 5 * sizeof(T), cudaMemcpyDeviceToHost);
            residual = sqrt(_aux[2] + _aux[3]);
            *_objective = _aux[0];
            // aux[0]: objective value, aux[1]: sum of matrix X
            // aux[2]: row residual,    aux[3]: col residual
            // aux[4]: alpha
            // printf("%4d %.10f %.10f %.10f %.10f\n", n_iter, residual, _aux[4], _aux[1], _aux[0]);
            if (residual < eps) {
                fix_x<T><<<CEILDIV(n_rows * n_cols, G_BLOCK_SIZE * FIX_X_WORK_SIZE), G_BLOCK_SIZE>>>(x, c, step_size, n_rows * n_cols);
                break;
            }
        }
        
        n_iter++;
    }
    *_n_iter = n_iter + 1;
}

int _q_get_work_size_update_x(int n_rows, int n_cols) {
    int work_size_log2 = static_cast<int>(round(log2(max(n_rows, n_cols) * UPDATE_X_WORK_SIZE_SLOPE + UPDATE_X_WORK_SIZE_Y_INTERCEPT)));
    return exp2(min(max(work_size_log2, 2), 6));
}

template <typename T>
T* quadratic_regularizer_drot_wrapper(const T *_c, // cost
        const T *_p, // distribution: p
        const T *_q, // distribution: q
        const int n_rows, // number of rows
        const int n_cols, // number of columns
        const T step_size, // rho
        const T r_weight, // weight for quadratic regularizer
        const int max_iters, // maximal number of iter
        const T eps, // error for stopping criterior
        float *dur_in_ms, // time elapsed in running
        float *prep_dur_in_ms, // time elapsed in data preperation
        int *_n_iter, // number of iteration in test
        T *_objective, // objective value
        bool use_warmup_init, // whether use warmup init
        bool return_x // whether return transportation plan
        ) {
    
    const size_t mat_size = n_rows * n_cols * sizeof(T);
    const size_t row_size = n_rows * sizeof(T);
    const size_t col_size = n_cols * sizeof(T);
    const T scale = (1 + step_size * T(n_rows + n_cols) * r_weight);
    const int work_size_update_x = _q_get_work_size_update_x(n_rows, n_cols);

    T *c, *p, *q, *x;
    T *a, *row_sum, *b, *col_sum;
    T *row_sum_1, *row_sum_2, *col_sum_1, *col_sum_2;
    T *phi1, *phi2;
    T *aux;

    cudaEvent_t start, prep_end, end;
    cudaEventCreate(&start);
    cudaEventCreate(&prep_end);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    // allocate memery on GPU
    cudaMalloc((void**)&c, mat_size);   // cost matrix: C
    cudaMalloc((void**)&p, row_size);   // distribution: p
    cudaMalloc((void**)&q, col_size);   // distribution: q
    cudaMalloc((void**)&x, mat_size);   // transport plan: X

    cudaMalloc((void**)&a, row_size);  // vector: a (same size as p)
    cudaMalloc((void**)&row_sum, row_size);  // vector: sum of rows (same size as p)
    cudaMalloc((void**)&row_sum_1, row_size * CEILDIV(n_cols, work_size_update_x));
    cudaMalloc((void**)&row_sum_2, row_size * CEILDIV(n_cols, work_size_update_x));
    cudaMalloc((void**)&b, col_size);  // vector: a (same size as q)
    cudaMalloc((void**)&col_sum, col_size);  // vector: sum of cols (same size as q)
    cudaMalloc((void**)&col_sum_1, col_size * CEILDIV(n_rows, UPDATE_X_BLOCK_SIZE_X));
    cudaMalloc((void**)&col_sum_2, col_size * CEILDIV(n_rows, UPDATE_X_BLOCK_SIZE_X));

    cudaMalloc((void**)&phi1, row_size); // vector: phi 1 (same size as p)
    cudaMalloc((void**)&phi2, col_size); // vector: phi 2 (same size as q)

    // aux[0]: objective value, aux[1]: sum of matrix X
    // aux[2]: row residual,    aux[3]: col residual
    // aux[4]: alpha
    cudaMalloc((void**)&aux, 5*sizeof(T));

    // copy data from CPU to GPU
    cudaMemcpy(c, _c, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(p, _p, row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(q, _q, col_size, cudaMemcpyHostToDevice);

    // initialization
    if (!use_warmup_init) {
        cudaMemset(a, 0, row_size);
        cudaMemset(row_sum, 0, row_size);
        cudaMemset(b, 0, col_size);
        cudaMemset(col_sum, 0, col_size);
        cudaMemset(phi1, 0, row_size);
        cudaMemset(phi2, 0, col_size);

        dim3 grid_init_x(
                (n_rows + INIT_X_BLOCK_SIZE_X - 1) / INIT_X_BLOCK_SIZE_X,
                (n_cols + INIT_X_BLOCK_SIZE_Y - 1) / INIT_X_BLOCK_SIZE_Y);
        dim3 block_init_x(INIT_X_THREAD_SIZE_X);
        init_x<<<grid_init_x, block_init_x>>>(x, p, q, n_rows, n_cols);
    } else {
        const T _n = T(n_rows);
        const T _m = T(n_cols);
        const T _k = T(1.0) * step_size * _n * _m - T(2.);

        const T v_phi1 = (_k + T(2)) / _n / (_m + _n);
        const T v_phi2 = (_k + T(2)) / _m / (_m + _n);
        const T v_a = (_k + T(1)) / _n;
        const T v_b = (_k + T(1)) / _m;
        const T v_alpha = _k + T(1);
        const T v_beta = 0;

        std::vector<T> c_phi1(n_rows, v_phi1);
        std::vector<T> c_phi2(n_cols, v_phi2);
        std::vector<T> c_a1(n_rows, v_a);
        std::vector<T> c_b1(n_cols, v_b);
        std::vector<T> c_alpha_gamma(2);
        c_alpha_gamma[0] = v_alpha;
        c_alpha_gamma[1] = v_beta;

        cudaMemcpy(phi1, &c_phi1[0], row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(phi2, &c_phi2[0], col_size, cudaMemcpyHostToDevice);
        cudaMemcpy(a, &c_a1[0], row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(b, &c_b1[0], col_size, cudaMemcpyHostToDevice);
        cudaMemcpy(&aux[4], &c_alpha_gamma[0], sizeof(T), cudaMemcpyHostToDevice);
        cudaMemset(x, 0, mat_size);
    }
    
    // record initialization done
    cudaEventRecord(prep_end);
    cudaEventSynchronize(prep_end);

    // main starts
    quadratic_regularizer_drot<T>(
                c, p, q, n_rows, n_cols, step_size, scale, r_weight,
                max_iters, eps, work_size_update_x, x, a, row_sum,
                row_sum_1, row_sum_2, b, col_sum, col_sum_1, col_sum_2,
                phi1, phi2, aux, _n_iter, _objective);

    // record all done
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // log
    cudaEventElapsedTime(dur_in_ms, prep_end, end);
    cudaEventElapsedTime(prep_dur_in_ms, start, prep_end);

    // free memory
    cudaFree(c);
    cudaFree(p);
    cudaFree(q);
    cudaFree(a);
    cudaFree(row_sum);
    cudaFree(row_sum_1);
    cudaFree(row_sum_2);
    cudaFree(b);
    cudaFree(col_sum);
    cudaFree(col_sum_1);
    cudaFree(col_sum_2);
    cudaFree(phi1);
    cudaFree(phi2);
    cudaFree(aux);

    if (!return_x) {
        cudaFree(x);
        return NULL;
    } else {
        T* tmp;
        tmp = (T*) malloc(mat_size);
        cudaMemcpy(tmp, x, mat_size, cudaMemcpyDeviceToHost);
        return tmp;
    }
}
