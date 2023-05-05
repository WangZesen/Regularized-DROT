#include "param_qr.hpp"

#define OFFSET(row, col, ld) ((col) * (ld) + (row))
#define CEILDIV(x, y) ((x+y-1)/y)
#define ALIGN(offset) ((offset + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE * G_BLOCK_SIZE)

template <typename T>
__global__ void init_x(T *x,
        const T* __restrict__ p,
        const T* __restrict__ q,
        const int n_rows,
        const int n_cols) {

    const int n = INIT_X_BLOCK_SIZE_X * blockIdx.x + threadIdx.x;
    const int m = INIT_X_BLOCK_SIZE_Y * blockIdx.y;
    const int tm = min(INIT_X_WORK_SIZE, n_cols - m);

    // load p to register
    T r_p;
    if (n < n_rows) {
        r_p = p[n];
    }
    __syncthreads();

    // load q to shared memory
    __shared__ T s_q[INIT_X_WORK_SIZE];
    if ((threadIdx.x < INIT_X_WORK_SIZE) && (m + threadIdx.x < n_cols)) {
        s_q[threadIdx.x] = q[m + threadIdx.x];
    }
    __syncthreads();

    // initialize x
    int offset = OFFSET(n, m, n_rows);
    if (n < n_rows) {
        for (int idx = 0; idx < tm; idx++) {
            x[offset] = r_p * s_q[idx];
            offset += n_rows;
        }
    }
}

template <typename T>
__inline__ __device__ T _warp_reduce_sum(T val) {
    #pragma unroll
    for (int w = 16; w > 0; w /= 2)
        val += __shfl_down_sync(0xffffffff, val, w);
    return val;
}

template <typename T>
__global__ void quadratic_regularizer_update_x_even_old(
        T *x,
        const T *phi1,
        const T *phi2,
        const T * __restrict__ c,
        const T step_size,
        const T scale,
        const int n_rows,
        const int n_cols,
        const int work_size,
        T *block_row_sum,
        T *block_col_sum,
        T *obj
) {
    const int n = UPDATE_X_BLOCK_SIZE_X * blockIdx.x + threadIdx.x;
    const int m = work_size * blockIdx.y;
    const int tm = min(work_size, n_cols - m);
    T t_row_sum = 0, t_col_sum, x_val, c_val;
    T t_obj_sum = 0;

    // load phi_1 to register
    T r_phi1;
    if (n < n_rows) {
        r_phi1 = phi1[n];
    }

    // load phi2 to shared memory
    __shared__ T s_phi2[UPDATE_X_BLOCK_SIZE_X];
    if (threadIdx.x < tm) {
        s_phi2[threadIdx.x] = phi2[threadIdx.x + m];
    }
    __syncthreads();

    // update x
    int offset = OFFSET(n, m, n_rows);
    if (n < n_rows) {
        for (int idx = 0; idx < tm; idx++) {
            x_val = x[offset];
            c_val = c[offset];
            x_val = max(x_val + r_phi1 + s_phi2[idx] - step_size * c_val, 0.) * scale;
            x[offset] = x_val - step_size * c_val;
            t_row_sum += x_val;
            t_obj_sum += x_val * c_val;
            t_col_sum = _warp_reduce_sum(x_val);
            if (threadIdx.x % 32 == 0) {
                atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
            }
            offset += n_rows;
        }
        block_row_sum[n * gridDim.y + blockIdx.y] = t_row_sum;
    } else {
        for (int idx = 0; idx < tm; idx++) {
            t_col_sum = _warp_reduce_sum(0.);
            if (threadIdx.x % 32 == 0) {
                atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
            }
        }
    }
    t_obj_sum = _warp_reduce_sum(t_obj_sum);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(obj, t_obj_sum);
    }
}

template <typename T>
__global__ void quadratic_regularizer_update_x_odd_old(
        T *x,
        const T *phi1,
        const T *phi2,
        const T step_size,
        const T scale,
        const int n_rows,
        const int n_cols,
        const int work_size,
        T *block_row_sum,
        T *block_col_sum
) {
    const int n = UPDATE_X_BLOCK_SIZE_X * blockIdx.x + threadIdx.x;
    const int m = work_size * blockIdx.y;
    const int tm = min(work_size, n_cols - m);
    T t_row_sum = 0, t_col_sum, x_val;

    // load phi_1 to register
    T r_phi1;
    if (n < n_rows) {
        r_phi1 = phi1[n];
    }

    // load phi2 to shared memory
    __shared__ T s_phi2[UPDATE_X_BLOCK_SIZE_X];
    if (threadIdx.x < tm) {
        s_phi2[threadIdx.x] = phi2[m + threadIdx.x];
    }
    __syncthreads();

    // update x
    int offset = OFFSET(n, m, n_rows);
    if (n < n_rows) {
        for (int idx = 0; idx < tm; idx++) {
            x_val = x[offset];
            x_val = max(x_val + r_phi1 + s_phi2[idx], 0.) * scale;
            x[offset] = x_val;
            t_row_sum += x_val;
            t_col_sum = _warp_reduce_sum(x_val);
            if (threadIdx.x % 32 == 0) {
                atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
            }
            offset += n_rows;
        }
        block_row_sum[n * gridDim.y + blockIdx.y] = t_row_sum;
    } else {
        for (int idx = 0; idx < tm; idx++) {
            t_col_sum = _warp_reduce_sum(0.);
            if (threadIdx.x % 32 == 0) {
                atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
            }
        }
    }
}

template <typename T>
__global__ void quadratic_regularizer_update_x_even(
        T *x,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ c,
        const T step_size,
        const T scale,
        const int n_rows,
        const int n_cols,
        const int n_total,
        const int work_size,
        T *block_row_sum,
        T *block_col_sum,
        T *obj
) {
    const int n_start = UPDATE_X_BLOCK_SIZE_X * blockIdx.x;
    const int n = n_start + threadIdx.x;
    const int m = work_size * blockIdx.y;
    const int tm = min(work_size, n_cols - m);
    T t_col_sum, x_val, c_val;
    T t_obj_sum = 0, leftover = 0;
    bool have_leftover = false;

    // load phi1 to shared memory
    __shared__ T r_phi1[G_BLOCK_SIZE * 2];
    r_phi1[threadIdx.x] = (n < n_rows)?phi1[n]:phi1[n - n_rows];
    r_phi1[threadIdx.x + G_BLOCK_SIZE] = (n + G_BLOCK_SIZE < n_rows)?
            phi1[n + G_BLOCK_SIZE]:
            phi1[n - n_rows + G_BLOCK_SIZE];

    // load phi2 to shared memory
    __shared__ T s_phi2[G_BLOCK_SIZE + 1];
    s_phi2[threadIdx.x] = (threadIdx.x < min(tm + 1, n_cols - m))?phi2[m + threadIdx.x]:0;

    // sum of rows
    __shared__ T row_sum[G_BLOCK_SIZE * 2];
    row_sum[threadIdx.x] = 0;
    row_sum[threadIdx.x + G_BLOCK_SIZE] = 0;
    __syncthreads();

    // update x
    int offset = OFFSET(n, m, n_rows);
    int shift = OFFSET(n_start, m, n_rows) % G_BLOCK_SIZE;
    shift = (shift > 0)?(G_BLOCK_SIZE - shift):0;
    
    for (int idx = 0; idx < tm; idx++) {
        // get shift for first whole block
        shift = OFFSET(n_start, m + idx, n_rows) % G_BLOCK_SIZE;
        shift = (shift > 0)?(G_BLOCK_SIZE - shift):0;

        if (n_start + shift >= n_rows) {
            if (have_leftover) {
                // accumulate sum of col
                t_col_sum = _warp_reduce_sum(leftover);
                if (threadIdx.x % 32 == 0) {
                    atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
                }
                leftover = 0;
                have_leftover = false;
            }
            offset += n_rows;
            continue;
        }
        __syncthreads();
        // load values of x and c (make sure x_val is 0 after update if out of bound)
        x_val = (offset + shift < n_total)?x[offset + shift]:-1e10;
        c_val = (offset + shift < n_total)?c[offset + shift]:1e10;
        // compute update
        x_val = (n + shift < n_rows)?
                max(x_val + r_phi1[threadIdx.x + shift] + s_phi2[idx] - step_size * c_val, 0.) * scale:
                max(x_val + r_phi1[threadIdx.x + shift] + s_phi2[idx + 1] - step_size * c_val, 0.) * scale;
        // update x (may diverge in warp...)
        if (offset + shift < n_total) {
            x[offset + shift] = x_val - step_size * c_val;
        }
        // accumulate sum of row
        row_sum[threadIdx.x + shift] += x_val;
        
        // accumulate objective value
        t_obj_sum += x_val * c_val;
        // accumulate sum of col
        t_col_sum = _warp_reduce_sum((n + shift < n_rows)?(x_val + leftover):leftover);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
        }
        // leftover for next col
        leftover = ((n + shift >= n_rows) && (offset + shift < n_total))?x_val:0;
        have_leftover = (n_start + shift + G_BLOCK_SIZE >= n_rows);
        // work on next col
        offset += n_rows;
    }
    if (have_leftover && (m + tm < n_cols)) { // there is leftover
        t_col_sum = _warp_reduce_sum(leftover);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&block_col_sum[(m + tm) * gridDim.x + blockIdx.x], t_col_sum);
        }
        leftover = 0;
    }
    __syncthreads();
    atomicAdd(&block_row_sum[(n % n_rows) * gridDim.y + blockIdx.y], row_sum[threadIdx.x]);
    atomicAdd(&block_row_sum[((n + G_BLOCK_SIZE) % n_rows) * gridDim.y + blockIdx.y], row_sum[threadIdx.x + G_BLOCK_SIZE]);

    t_obj_sum = _warp_reduce_sum(t_obj_sum);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(obj, t_obj_sum);
    }
}

template <typename T>
__global__ void quadratic_regularizer_update_x_odd(
        T *x,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T step_size,
        const T scale,
        const int n_rows,
        const int n_cols,
        const int n_total,
        const int work_size,
        T *block_row_sum,
        T *block_col_sum
) {
    const int n_start = UPDATE_X_BLOCK_SIZE_X * blockIdx.x;
    const int n = n_start + threadIdx.x;
    const int m = work_size * blockIdx.y;
    const int tm = min(work_size, n_cols - m);
    T t_col_sum, x_val;
    T leftover = 0;
    bool have_leftover = false;

    // load phi1 to shared memory
    __shared__ T r_phi1[G_BLOCK_SIZE * 2];
    r_phi1[threadIdx.x] = (n < n_rows)?phi1[n]:phi1[n - n_rows];
    r_phi1[threadIdx.x + G_BLOCK_SIZE] = (n + G_BLOCK_SIZE < n_rows)?
            phi1[n + G_BLOCK_SIZE]:
            phi1[n - n_rows + G_BLOCK_SIZE];

    // load phi2 to shared memory
    __shared__ T s_phi2[G_BLOCK_SIZE + 1];
    s_phi2[threadIdx.x] = (threadIdx.x < min(tm + 1, n_cols - m))?phi2[m + threadIdx.x]:0;

    // sum of rows
    __shared__ T row_sum[G_BLOCK_SIZE * 2];
    row_sum[threadIdx.x] = 0;
    row_sum[threadIdx.x + G_BLOCK_SIZE] = 0;
    __syncthreads();

    // update x
    int offset = OFFSET(n, m, n_rows);
    int shift = OFFSET(n_start, m, n_rows) % G_BLOCK_SIZE;
    shift = (shift > 0)?(G_BLOCK_SIZE - shift):0;
    
    for (int idx = 0; idx < tm; idx++) {
        // get shift for first whole block
        shift = OFFSET(n_start, m + idx, n_rows) % G_BLOCK_SIZE;
        shift = (shift > 0)?(G_BLOCK_SIZE - shift):0;

        if (n_start + shift >= n_rows) {
            if (have_leftover) {
                // accumulate sum of col
                t_col_sum = _warp_reduce_sum(leftover);
                if (threadIdx.x % 32 == 0) {
                    atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
                }
                leftover = 0;
                have_leftover = false;
            }
            offset += n_rows;
            continue;
        }
        __syncthreads();
        // load values of x and c (make sure x_val is 0 after update if out of bound)
        x_val = (offset + shift < n_total)?x[offset + shift]:-1e10;
        // compute update
        x_val = (n + shift < n_rows)?
                max(x_val + r_phi1[threadIdx.x + shift] + s_phi2[idx], 0.) * scale:
                max(x_val + r_phi1[threadIdx.x + shift] + s_phi2[idx + 1], 0.) * scale;
        // update x (may diverge in warp...)
        if (offset + shift < n_total) {
            x[offset + shift] = x_val;
        }
        // accumulate sum of row
        row_sum[threadIdx.x + shift] += x_val;
        // accumulate sum of col
        t_col_sum = _warp_reduce_sum((n + shift < n_rows)?(x_val + leftover):leftover);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&block_col_sum[(m + idx) * gridDim.x + blockIdx.x], t_col_sum);
        }
        // leftover for next col
        leftover = ((n + shift >= n_rows) && (offset + shift < n_total))?x_val:0;
        have_leftover = (n_start + shift + G_BLOCK_SIZE >= n_rows);
        // work on next col
        offset += n_rows;
    }
    if (have_leftover && (m + tm < n_cols)) { // there is leftover
        t_col_sum = _warp_reduce_sum(leftover);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&block_col_sum[(m + tm) * gridDim.x + blockIdx.x], t_col_sum);
        }
        leftover = 0;
    }
    __syncthreads();
    atomicAdd(&block_row_sum[(n % n_rows) * gridDim.y + blockIdx.y], row_sum[threadIdx.x]);
    atomicAdd(&block_row_sum[((n + G_BLOCK_SIZE) % n_rows) * gridDim.y + blockIdx.y], row_sum[threadIdx.x + G_BLOCK_SIZE]);
}

template <typename T>
__global__ void reduce_axis_sum_step(const int n_block,
        const T * __restrict__ sum_in,
        T *sum_out) {
    
    const int offset_out = blockIdx.y * gridDim.x + blockIdx.x;
    const int work_size = min((n_block - (blockIdx.x * REDUCE_AXIS_SUM_BLOCK_SIZE_X + threadIdx.x) + REDUCE_AXIS_SUM_THREAD_SIZE_X - 1) / REDUCE_AXIS_SUM_THREAD_SIZE_X,
            REDUCE_AXIS_SUM_WORK_SIZE);
    int offset_in = blockIdx.y * n_block + blockIdx.x * REDUCE_AXIS_SUM_BLOCK_SIZE_X + threadIdx.x;
    T sum = 0;
    
    if (threadIdx.x == 0) {
        sum_out[offset_out] = 0;
    }
    __syncthreads();

    for (int idx = 0; idx < work_size; idx++) {
        sum += sum_in[offset_in];
        offset_in += REDUCE_AXIS_SUM_THREAD_SIZE_X;
    }
    sum = _warp_reduce_sum(sum);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&sum_out[offset_out], sum);
    }
}

template <typename T>
__global__ void quadratic_regularizer_update_aux(const int n_rows,
        const int n_cols,
        T *phi1,
        T *phi2,
        const T * __restrict__ p,
        const T * __restrict__ q,
        const T * __restrict__ row_sum,
        const T * __restrict__ col_sum,
        const T * __restrict__ x_sum,
        T *row_residual,
        T *col_residual,
        T *a,
        T *b,
        T *alpha) {
    
    T gamma = (2 * ((*x_sum) - 1.) - (*alpha)) / (n_rows + n_cols);
    T ab_val, rs_val, residual = 0;
    int offset = UPDATE_AUX_BLOCK_SIZE_X * blockIdx.x + threadIdx.x;
    int work_size;

    if (blockIdx.y == 0) { // row
        work_size = min(CEILDIV(n_rows - offset, G_BLOCK_SIZE), UPDATE_AUX_WORK_SIZE);
        for (int idx = 0; idx < work_size; idx++) {
            rs_val = row_sum[offset] - p[offset];
            ab_val = a[offset] - rs_val;
            phi1[offset] = (ab_val - rs_val + gamma) / T(n_cols);
            a[offset] = ab_val;
            residual += (rs_val * rs_val);
            offset += G_BLOCK_SIZE;
        }
        residual = _warp_reduce_sum(residual);
        if (threadIdx.x == 0) {
            row_residual[blockIdx.x * 2] = residual;
        }
        if (threadIdx.x == 32) {
            row_residual[blockIdx.x * 2 + 1] = residual;
        }
    } else { // col
        work_size = min(CEILDIV(n_cols - offset, G_BLOCK_SIZE), UPDATE_AUX_WORK_SIZE);
        for (int idx = 0; idx < work_size; idx++) {
            rs_val = col_sum[offset] - q[offset];
            ab_val = b[offset] - rs_val;
            phi2[offset] = (ab_val - rs_val + gamma) / T(n_rows);
            b[offset] = ab_val;
            residual += (rs_val * rs_val);
            offset += G_BLOCK_SIZE;
        }
        residual = _warp_reduce_sum(residual);
        if (threadIdx.x == 0) {
            col_residual[blockIdx.x * 2] = residual;
        }
        if (threadIdx.x == 32) {
            col_residual[blockIdx.x * 2 + 1] = residual;
        }
    }
}

template <typename T>
__global__ void update_alpha(const T *x_sum, T *alpha) {
    T _alpha = ((*alpha) + 1.) - (*x_sum);
    *alpha = _alpha;
}

template <typename T>
__global__ void fix_x(
        T *x,
        const T *c,
        const T step_size,
        const int total) {
    
    int offset = blockIdx.x * G_BLOCK_SIZE * FIX_X_WORK_SIZE + threadIdx.x;
    for (int i = 0; i < FIX_X_WORK_SIZE; i++) {
        if (offset < total) {
            x[offset] = max(x[offset] + step_size * c[offset], 0.);
        }
        offset += G_BLOCK_SIZE;
    }
}