#include "abi.cuh"
extern "C" __global__ void toy(Params p) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += gridDim.x * blockDim.x) {
    Regs r; stage_scale(p, r, i); stage_bias(p, r, i);
  }
}
