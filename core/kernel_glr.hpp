#define OFFSET(row, col, ld) ((col) * (ld) + (row))
#define CEILDIV(x, y) ((x+y-1)/y)

#include "param_glr.hpp"

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
__inline__ __device__ T _warp_reduce_sum_bc(T val) {
    #pragma unroll
    for (int w = 16; w > 0; w /= 2)
        val += __shfl_down_sync(0xffffffff, val, w);
    val = __shfl_sync(0xffffffff, val, 0);
    return val;
}

template <typename T, int NGROUPS>
__global__ void group_lasso_regularizer_update_x_for_small(T *x,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ c,
        const T step_size,
        const T lambda,
        const int n_rows,
        const int n_cols,
        T *block_row_sum,
        T *group_col_sum,
        T *obj) {
    
    __shared__ T x_val[GROUP_THREASHOLD];
    __shared__ T row_sum[GROUP_THREASHOLD];
    __shared__ T s_phi1[GROUP_THREASHOLD];
    __shared__ T s_phi2[UPDATE_X_SMALL_WORK_SIZE_Y];
    __shared__ T share[2];
    
    const int n_start = int(T(n_rows) / T(NGROUPS) * T(blockIdx.x) + 0.49);
    const int n_end = int(T(n_rows) / T(NGROUPS) * T(blockIdx.x + 1) + 0.49);
    const int m = UPDATE_X_SMALL_WORK_SIZE_Y * blockIdx.y;

    const int tn = CEILDIV(n_end - n_start, UPDATE_X_SMALL_THREAD_SIZE);
    const int tm = min(UPDATE_X_SMALL_WORK_SIZE_Y, n_cols - m);

    int offset, row_offset;
    T c_val, obj_sum = 0, col_obj_sum, col_sum, norm;

    offset = threadIdx.x;
    for (int idx = 0; idx < tn; idx++) {
        row_sum[offset] = 0;
        s_phi1[offset] = (offset + n_start < n_end)?phi1[offset + n_start]:0.;
        offset += UPDATE_X_SMALL_THREAD_SIZE;
    }
    if (threadIdx.x < tm) {
        s_phi2[threadIdx.x] = phi2[threadIdx.x + m];
    }
    __syncthreads();

    for (int idxm = 0; idxm < tm; idxm++) {
        offset = OFFSET(n_start + threadIdx.x, m + idxm, n_rows);
        row_offset = threadIdx.x;
        col_sum = 0;
        col_obj_sum = 0;
        norm = 0;
        for (int idxn = 0; idxn < tn; idxn++) {
            x_val[row_offset] = (n_start + row_offset < n_end)?x[offset]:-1e10;
            c_val = (n_start + row_offset < n_end)?c[offset]:1e10;
            x_val[row_offset] = max(x_val[row_offset] + s_phi1[row_offset] + s_phi2[idxm] - step_size * c_val, 0.);
            col_obj_sum += x_val[row_offset] * c_val;
            col_sum += x_val[row_offset];
            norm += x_val[row_offset] * x_val[row_offset];

            offset += UPDATE_X_SMALL_THREAD_SIZE;
            row_offset += UPDATE_X_SMALL_THREAD_SIZE;
        }
        
        // broadcast & compute norm scale
        norm = _warp_reduce_sum_bc<T>(norm);
        if (threadIdx.x % 32 == 0) {
            share[threadIdx.x / 32] = norm;
        }
        __syncthreads();
        norm += (threadIdx.x < 32)?share[1]:share[0];
        norm = max(T(1) - lambda / (sqrt(norm) + 1e-9), 0.);

        // apply norm
        offset = OFFSET(n_start + threadIdx.x, m + idxm, n_rows);
        row_offset = threadIdx.x;
        for (int idxn = 0; idxn < tn; idxn++) {
            x_val[row_offset] *= norm;
            row_sum[row_offset] += x_val[row_offset];
            if (n_start + row_offset < n_end) {
                x[offset] = x_val[row_offset];
            }
            offset += UPDATE_X_SMALL_THREAD_SIZE;
            row_offset += UPDATE_X_SMALL_THREAD_SIZE;
        }
        obj_sum += col_obj_sum * norm;
        col_sum *= norm;

        // accumulate sum of col
        col_sum = _warp_reduce_sum<T>(col_sum);
        
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&group_col_sum[(m + idxm) * gridDim.x + blockIdx.x], col_sum);
        }
    }
    
    // accumulate sum of row
    row_offset = threadIdx.x;
    for (int idxn = 0; idxn < tn; idxn++) {
        if (n_start + row_offset < n_end) {
            block_row_sum[(n_start + row_offset) * gridDim.y + blockIdx.y] = row_sum[row_offset];
        }
        row_offset += UPDATE_X_SMALL_THREAD_SIZE;
    }

    // accumulate sum of obj
    atomicAdd(obj, obj_sum);
}

template<typename T, int NGROUPS>
__global__ void group_lasso_regularizer_update_x(
        T *x,
        const T * __restrict__ phi1,
        const T * __restrict__ phi2,
        const T * __restrict__ c,
        const T step_size,
        const int n_rows,
        const int n_cols,
        const int n_total,
        const int work_size,
        T *group_col_sum,
        T *group_obj_col_sum,
        T *group_norm_col_sum) {
    
    // ASSERT: group size is larger than UPDATE_X_THREAD_SIZE_X (G_BLOCK_SIZE)

    // blockIdx.x: block idx within group
    // blockIdx.y: group idx
    // blockIdx.z: block idx along cols

    const int n_group_end = int(T(n_rows) / NGROUPS * T(blockIdx.y + 1) + 0.49999);
    const int n_start = int(T(n_rows) / NGROUPS * T(blockIdx.y) + 0.49999) + G_BLOCK_SIZE * blockIdx.x;
    const int n = n_start + threadIdx.x;
    const int m = work_size * blockIdx.z;
    const int tm = min(work_size, n_cols - m);

    __shared__ T s_phi1[G_BLOCK_SIZE * 2];
    __shared__ T s_phi2[G_BLOCK_SIZE + 1];

    // if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (threadIdx.x == 0)) {
    //     printf("321\n");
    //     // printf("!! %.10f %.10f %.10f\n", x_val, s_phi1[threadIdx.x + shift], s_phi2[idx]);
    // }

    // load phi1 to shared memory
    s_phi1[threadIdx.x] = (n < n_rows)?phi1[n]:phi1[n - n_rows];
    s_phi1[threadIdx.x + G_BLOCK_SIZE] = (n + G_BLOCK_SIZE < n_rows)?
            phi1[n + G_BLOCK_SIZE]:
            phi1[n - n_rows + G_BLOCK_SIZE];
    
    // load phi2 to shared memory
    s_phi2[threadIdx.x] = (threadIdx.x < min(tm + 1, n_cols - m))?phi2[m + threadIdx.x]:0;
    
    __syncthreads();
    
    
    int shift, offset = OFFSET(n, m, n_rows);
    T t_col_obj_sum, t_col_sum, t_col_sqr_sum, x_val, c_val;

    // update x
    for (int idx = 0; idx < tm; idx++) {
        // get shift for first whole block
        shift = OFFSET(n_start, m + idx, n_rows) % G_BLOCK_SIZE;
        shift = (shift > 0)?(G_BLOCK_SIZE - shift):0;
        t_col_sum = 0;
        t_col_obj_sum = 0;
        t_col_sqr_sum = 0;

        if (n_start + shift >= n_group_end) { // it's in next group, then skip
            offset += n_rows;
            continue;
        }

        __syncthreads();

        

        // load values of x and c (make sure x_val is 0 after update if out of bound)
        x_val = (offset + shift < n_total)?x[offset + shift]:-1e10;
        c_val = (offset + shift < n_total)?c[offset + shift]:1e10;

        // compute update
        x_val = (n + shift < n_rows)?
                max(x_val + s_phi1[threadIdx.x + shift] + s_phi2[idx] - step_size * c_val, 0.):
                max(x_val + s_phi1[threadIdx.x + shift] + s_phi2[idx + 1] - step_size * c_val, 0.);
        
        // update x (may diverge in warp...)
        if (offset + shift < n_total) {
            x[offset + shift] = x_val;
        }

        // accumulate (unnormed) sum of col of objective value
        t_col_obj_sum = _warp_reduce_sum<T>((n + shift < n_group_end)?c_val * x_val:0);
        if (threadIdx.x % 32 == 0) {
            // group_obj_col_sum[block_idx_in_group, group_idx, col_idx]
            atomicAdd(&group_obj_col_sum[(m + idx) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x], t_col_obj_sum);
        }

        // leftover for next group
        if (n_start + shift + G_BLOCK_SIZE >= n_group_end) {
            t_col_obj_sum = _warp_reduce_sum<T>((n + shift >= n_group_end)?x_val * c_val:0);
            // if it's last group in this column (but not for the whole matrix), leftover will be for next col
            if (threadIdx.x % 32 == 0) {
                if ((blockIdx.y == NGROUPS - 1) && ((blockIdx.z < gridDim.z - 1) || (idx < tm - 1))) {
                    atomicAdd(&group_obj_col_sum[(m + idx + 1) * gridDim.x * gridDim.y], t_col_obj_sum);
                } else if (blockIdx.y < NGROUPS - 1) { // leftover still in this column, but different group
                    atomicAdd(&group_obj_col_sum[(m + idx) * gridDim.x * gridDim.y + (blockIdx.y + 1) * gridDim.x], t_col_obj_sum);
                }
            }
        }
        __syncthreads();

        // accumulate (unnormed) sum of col in current group
        t_col_sum = _warp_reduce_sum<T>((n + shift < n_group_end)?x_val:0);
        if (threadIdx.x % 32 == 0) {
            // group_col_sum[block_idx_in_group, group_idx, col_idx]
            atomicAdd(&group_col_sum[(m + idx) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x], t_col_sum);
        }

        // leftover for next group
        if (n_start + shift + G_BLOCK_SIZE >= n_group_end) {
            t_col_sum = _warp_reduce_sum<T>((n + shift >= n_group_end)?x_val:0);
            // if it's last group in this column (but not for the whole matrix), leftover will be for next col
            if (threadIdx.x % 32 == 0) {
                if ((blockIdx.y == NGROUPS - 1) && ((blockIdx.z < gridDim.z - 1) || (idx < tm - 1))) {
                    atomicAdd(&group_col_sum[(m + idx + 1) * gridDim.x * gridDim.y], t_col_sum);
                } else if (blockIdx.y < NGROUPS - 1) { // leftover still in this column, but different group
                    atomicAdd(&group_col_sum[(m + idx) * gridDim.x * gridDim.y + (blockIdx.y + 1) * gridDim.x], t_col_sum);
                }
            }
        }
        __syncthreads();

        // accumulate sum of col of squared elements in current group
        t_col_sqr_sum = _warp_reduce_sum<T>((n + shift < n_group_end)?x_val * x_val:0);
        if (threadIdx.x % 32 == 0) {
            // group_col_sum[block_idx_in_group, group_idx, col_idx]
            atomicAdd(&group_norm_col_sum[(m + idx) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x], t_col_sqr_sum);
        }

        // leftover for next group
        if (n_start + shift + G_BLOCK_SIZE >= n_group_end) {
            t_col_sqr_sum = _warp_reduce_sum<T>((n + shift >= n_group_end)?x_val * x_val:0);
            // if it's last group in this column (but not for the whole matrix), leftover will be for next col
            if (threadIdx.x % 32 == 0) {
                if ((blockIdx.y == NGROUPS - 1) && ((blockIdx.z < gridDim.z - 1) || (idx < tm - 1))) {
                    atomicAdd(&group_norm_col_sum[(m + idx + 1) * gridDim.x * gridDim.y], t_col_sqr_sum);
                } else if (blockIdx.y < NGROUPS - 1) { // leftover still in this column, but different group
                    atomicAdd(&group_norm_col_sum[(m + idx) * gridDim.x * gridDim.y + (blockIdx.y + 1) * gridDim.x], t_col_sqr_sum);
                }
            }
        }

        offset += n_rows;
    }
}

template <typename T>
__global__ void group_lasso_regularizer_update_norm(
        T *g_norm_sum,
        T *g_obj_sum,
        T *g_sum,
        const T lambda,
        const int n_total) {
    int offset = blockIdx.x * UPDATE_NORM_BLOCK_SIZE + threadIdx.x;
    T scale;
    for (int i = 0; i < UPDATE_NORM_WORK_SIZE; i++) {
        if (offset >= n_total) {
            break;
        }
        scale = max(1. - lambda / (sqrt(g_norm_sum[offset]) + 1e-9), 0.);
        g_norm_sum[offset] = scale;
        g_obj_sum[offset] *= scale;
        g_sum[offset] *= scale;
        offset += UPDATE_NORM_THREAD_SIZE;
    }
}

template<typename T, int NGROUPS>
__global__ void group_lasso_regularizer_apply_norm(
        T *x,
        const int n_rows,
        const int n_cols,
        const int n_total,
        const T * __restrict__ group_scale,
        T *block_row_sum) {
    
    // ASSERT: group size is larger than UPDATE_X_THREAD_SIZE_X (G_BLOCK_SIZE)

    // blockIdx.x: block idx within group
    // blockIdx.y: group idx
    // blockIdx.z: block idx along cols

    const int n_group_end = int(T(n_rows) / NGROUPS * T(blockIdx.y + 1) + 0.49999);
    const int n_start = int(T(n_rows) / NGROUPS * T(blockIdx.y) + 0.49999) + G_BLOCK_SIZE * blockIdx.x;
    const int n = n_start + threadIdx.x;
    const int m = APPLY_NORM_WORK_SIZE_Y * blockIdx.z;
    const int tm = min(APPLY_NORM_WORK_SIZE_Y, n_cols - m);

    __shared__ T row_sum[APPLY_NORM_BLOCK_SIZE_X * 2];
    __shared__ T s_group_scale[2 * (APPLY_NORM_WORK_SIZE_Y + 1)];

    // init row_sum
    row_sum[threadIdx.x] = 0;
    row_sum[threadIdx.x + G_BLOCK_SIZE] = 0;

    // load group scale
    if (threadIdx.x < min(tm + 1, n_cols - m)) {
        s_group_scale[threadIdx.x * 2] = group_scale[(m + threadIdx.x) * NGROUPS + blockIdx.y];
        s_group_scale[threadIdx.x * 2 + 1] = ((m + threadIdx.x) * NGROUPS + blockIdx.y + 1 < n_cols * NGROUPS)?group_scale[(m + threadIdx.x) * NGROUPS + blockIdx.y + 1]:0;
    }
    
    __syncthreads();

    int shift, offset = OFFSET(n, m, n_rows);
    T x_val;

    // update x
    for (int idx = 0; idx < tm; idx++) {
        // get shift for first whole block
        shift = OFFSET(n_start, m + idx, n_rows) % G_BLOCK_SIZE;
        shift = (shift > 0)?(G_BLOCK_SIZE - shift):0;

        if (n_start + shift >= n_group_end) { // it's in next group, then skip
            offset += n_rows;
            continue;
        }
        __syncthreads();

        // load value of x
        x_val = (offset + shift < n_total)?x[offset + shift]:0;
        
        // apply norm & update x
        x_val = x_val * ((n + shift < n_group_end)?s_group_scale[idx * 2]:s_group_scale[idx * 2 + 1]);
        if (offset + shift < n_total) {
            x[offset + shift] = x_val;
        }
        __syncthreads();

        // accumulate normed sum of row
        row_sum[shift + threadIdx.x] += x_val;
        offset += n_rows;
        __syncthreads();
    }

    __syncthreads();
    atomicAdd(&block_row_sum[(n % n_rows) * gridDim.z + blockIdx.z], row_sum[threadIdx.x]);
    atomicAdd(&block_row_sum[((n + G_BLOCK_SIZE) % n_rows) * gridDim.z + blockIdx.z], row_sum[threadIdx.x + G_BLOCK_SIZE]);
}

template <typename T>
__global__ void reduce_group_sum_step(const int n_block,
        const int n_groups,
        const T *sum_in,
        T *sum_out) {
    
    const int offset_out = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    const int offset_block = blockIdx.x * REDUCE_GROUP_SUM_BLOCK_SIZE + threadIdx.x;
    const int work_size = min(
            CEILDIV(n_block - offset_block, REDUCE_GROUP_SUM_THREAD_SIZE),
            REDUCE_GROUP_SUM_WORK_SIZE
    );
    int offset_in = blockIdx.z * n_block * n_groups + blockIdx.y * n_block + offset_block;
    T sum = 0;
    
    if (threadIdx.x == 0) {
        sum_out[offset_out] = 0;
    }
    __syncthreads();

    
    for (int idx = 0; idx < work_size; idx++) {
        sum += sum_in[offset_in];
        offset_in += REDUCE_GROUP_SUM_THREAD_SIZE;
    }
    sum = _warp_reduce_sum<T>(sum);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&sum_out[offset_out], sum);
    }
}

template <typename T>
__global__ void reduce_axis_sum_step(const int n_block,
        const T *sum_in,
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
    sum = _warp_reduce_sum<T>(sum);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&sum_out[offset_out], sum);
    }
}

template <typename T>
__global__ void group_lasso_regularizer_update_aux(
        const int n_rows,
        const int n_cols,
        T *phi1,
        T *phi2,
        const T *p,
        const T *q,
        const T *row_sum,
        const T *col_sum,
        const T *x_sum,
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
        residual = _warp_reduce_sum<T>(residual);
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
        residual = _warp_reduce_sum<T>(residual);
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
