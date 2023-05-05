#include <torch/extension.h>

torch::Tensor quadratic_drot_cuda(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    float rho,
    float r_weight,
    int max_iters,
    float eps);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor quadratic_drot(
    torch::Tensor c,
    torch::Tensor p,
    torch::Tensor q,
    float rho,
    float r_weight,
    int max_iter,
    float eps) {
    
    CHECK_INPUT(c);
    CHECK_INPUT(p);
    CHECK_INPUT(q);
    
    return quadratic_drot_cuda(c, p, q, rho, r_weight, max_iter, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quadratic_drot", &quadratic_drot, "Quadratic Regularized DROT");
}
