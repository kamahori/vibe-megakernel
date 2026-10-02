#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "dense_kernel.cu"
void launch_dense(const std::vector<torch::Tensor>& in,const std::vector<torch::Tensor>& out,torch::Tensor scratch) {
  TORCH_CHECK(in.size()==16 && out.size()==4,"invalid tensor count");
  c10::cuda::CUDAGuard guard(in[0].device());
  const B* p[15];
  TORCH_CHECK(in[0].is_cuda() && in[0].scalar_type()==torch::kInt64 && in[0].numel()==1,"invalid token");
  for(int i=1;i<16;++i) {
    TORCH_CHECK(in[i].device()==in[0].device() && in[i].is_contiguous() && in[i].scalar_type()==torch::kBFloat16,"invalid weight/cache");
    p[i-1]=reinterpret_cast<const B*>(in[i].data_ptr());
  }
  dense_step<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(in[0].data_ptr<int64_t>(),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],out[0].data_ptr<float>(),out[1].data_ptr<int64_t>(),reinterpret_cast<B*>(out[2].data_ptr()),reinterpret_cast<B*>(out[3].data_ptr()),scratch.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
