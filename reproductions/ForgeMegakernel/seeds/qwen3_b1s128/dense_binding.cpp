#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>
void launch_dense(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& outputs, torch::Tensor scratch);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run", &launch_dense);}
