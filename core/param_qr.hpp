#ifndef PARAM_HPP_
#define PARAM_HPP_

#define G_BLOCK_SIZE 64

#define INIT_X_BLOCK_SIZE_X G_BLOCK_SIZE // block size in x
#define INIT_X_BLOCK_SIZE_Y 64 // block size in y
#define INIT_X_THREAD_SIZE_X G_BLOCK_SIZE // number of threads in x for one block
#define INIT_X_WORK_SIZE INIT_X_BLOCK_SIZE_Y // thread work size

#define UPDATE_X_WORK_SIZE_SLOPE       0.002793
#define UPDATE_X_WORK_SIZE_Y_INTERCEPT 3.480

#define UPDATE_X_BLOCK_SIZE_X G_BLOCK_SIZE
#define UPDATE_X_THREAD_SIZE_X G_BLOCK_SIZE

#define REDUCE_AXIS_SUM_WORK_SIZE 16
#define REDUCE_AXIS_SUM_BLOCK_SIZE_X (G_BLOCK_SIZE * REDUCE_AXIS_SUM_WORK_SIZE)
#define REDUCE_AXIS_SUM_THREAD_SIZE_X G_BLOCK_SIZE

#define UPDATE_AUX_WORK_SIZE 1
#define UPDATE_AUX_BLOCK_SIZE_X (G_BLOCK_SIZE * UPDATE_AUX_WORK_SIZE)

#define FIX_X_WORK_SIZE 4

#endif