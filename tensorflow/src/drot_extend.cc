#include "drot_extend.h"
#include <math.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#define CEILDIV(x, y) ((x+y-1)/y)
// #define MAX(x, y) ((x < y)?y:x)
#define G_BLOCK_SIZE 64
#define UPDATE_X_WORK_SIZE_SLOPE       0.002793
#define UPDATE_X_WORK_SIZE_Y_INTERCEPT 3.480
#define UPDATE_X_BLOCK_SIZE_X G_BLOCK_SIZE

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// register op
REGISTER_OP("QuadraticDrot")
    .Input("cost: float") // cost
    .Input("rho: float") // step size (unscaled)
    .Input("r_weight: float") // quadratic regularizer's weight
    .Input("p: float") // p
    .Input("q: float") // q
    .Input("eps: float") // residual threshold
    .Input("max_iter: int64") // maximal number of iterations
    .Output("x: float") // x
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return tsl::OkStatus(); //Status::OK();
    });

// implementation
template <>
struct QuadraticDrotFuntor<GPUDevice, float> {
    void operator() (const GPUDevice& d,
            const float *c,
            const float *p,
            const float *q,
            const int n_rows,
            const int n_cols,
            const float* step_size,
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
        
        quadratic_regularizer_drot_tf_float32(c, p, q, n_rows, n_cols, step_size, r_weight, max_iters, eps,
                work_size_update_x, x, a, row_sum, row_sum_1, row_sum_2, b, col_sum, col_sum_1,
                col_sum_2, phi1, phi2, aux);
    }
};

// template <>
// struct DrotExtendFuntor<GPUDevice, double> {
//     void operator() (const GPUDevice& d,
//             const double *c,
//             const double *p,
//             const double *q,
//             const int n_rows,
//             const int n_cols,
//             const double* step_size,
//             const double* r_weight,
//             const int64_t* max_iters,
//             const double* eps,
//             const int work_size_update_x,
//             double *x,
//             double *a,
//             double *row_sum,
//             double *row_sum_1,
//             double *row_sum_2,
//             double *b,
//             double *col_sum,
//             double *col_sum_1,
//             double *col_sum_2,
//             double *phi1,
//             double *phi2,
//             double *aux) {
        
//         quadratic_regularizer_drot_tf_double(c, p, q, n_rows, n_cols, step_size, r_weight, max_iters, eps,
//                 work_size_update_x, x, a, row_sum, row_sum_1, row_sum_2, b, col_sum, col_sum_1,
//                 col_sum_2, phi1, phi2, aux);
//     }
// };

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.

template <typename Device>
class QuadraticDrotOp : public OpKernel {
    public:
        explicit QuadraticDrotOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& c = context->input(0);
            const Tensor& rho = context->input(1);
            const Tensor& r_weight = context->input(2);
            const Tensor& p = context->input(3);
            const Tensor& q = context->input(4);
            const Tensor& eps = context->input(5);
            const Tensor& max_iter = context->input(6);

            int n_cols = static_cast<int>(c.shape().dim_size(0));
            int n_rows = static_cast<int>(c.shape().dim_size(1));
            int work_size_update_x = _q_get_work_size_update_x(n_rows, n_cols);

            OP_REQUIRES(context, c.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument("Too many elements in cost matrix"));
            OP_REQUIRES(context, p.NumElements() == n_rows,
                        errors::InvalidArgument("Number of elements in p should match with c.shape[0]"));
            OP_REQUIRES(context, q.NumElements() == n_cols,
                        errors::InvalidArgument("Number of elements in q should match with c.shape[1]"));


            // Create an output tensor
            Tensor* x = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, c.shape(), &x));

            // Create temporary tensors
            const TensorShape &row_shape = TensorShape({n_rows});
            const TensorShape &row_sum_shape = TensorShape({n_rows * CEILDIV(n_cols, work_size_update_x)});
            const TensorShape &col_shape = TensorShape({n_cols});
            const TensorShape &col_sum_shape = TensorShape({n_cols * CEILDIV(n_rows, UPDATE_X_BLOCK_SIZE_X)});
            const TensorShape &aux_shape = TensorShape({5});

            Tensor a, b, row_sum, row_sum_1, row_sum_2, col_sum, col_sum_1, col_sum_2;
            Tensor phi1, phi2, aux;
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, row_shape, &a));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, row_shape, &row_sum));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, row_sum_shape, &row_sum_1));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, row_sum_shape, &row_sum_2));

            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, col_shape, &b));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, col_shape, &col_sum));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, col_sum_shape, &col_sum_1));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, col_sum_shape, &col_sum_2));

            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, row_shape, &phi1));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, col_shape, &phi2));
            
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, aux_shape, &aux));

            QuadraticDrotFuntor<Device, float>()(
                context->eigen_device<Device>(),
                c.flat<float>().data(),
                p.flat<float>().data(),
                q.flat<float>().data(),
                n_rows,
                n_cols,
                rho.scalar<float>().data(),
                r_weight.scalar<float>().data(),
                max_iter.scalar<int64>().data(),
                eps.scalar<float>().data(),
                work_size_update_x,
                x->flat<float>().data(),
                a.flat<float>().data(),
                row_sum.flat<float>().data(),
                row_sum_1.flat<float>().data(),
                row_sum_2.flat<float>().data(),
                b.flat<float>().data(),
                col_sum.flat<float>().data(),
                col_sum_1.flat<float>().data(),
                col_sum_2.flat<float>().data(),
                phi1.flat<float>().data(),
                phi2.flat<float>().data(),
                aux.flat<float>().data()
            );
        }
};

// Register the GPU kernel.
REGISTER_KERNEL_BUILDER(                   \
    Name("QuadraticDrot").Device(DEVICE_GPU), \
    QuadraticDrotOp<GPUDevice>);
