// Tap throughput on a resident tile: every CTA (one per SM) runs
// gemv_ksplit<8, BF16, RB, KS> (or gemv_mma<8, KS>, rb = 0) `iters` times over the same smem tile of
// `rows` x `k` BF16 weights, so the time per call is the tap's own cost per
// slot, without HBM or ring waits. Compare tile_bytes / time with the
// ~46-54 KB/us per SM the stream delivers: a tap slower than that makes the
// kernel compute bound (see bench/run_taps.py).
#include <cuda_runtime.h>
#include "loadstream/ls.cuh"

namespace {

constexpr int NW = 8;

template <int RB, int KS>
__global__ void __launch_bounds__(32 * NW, 1) tap_loop(int rows, int k, int iters, float* sink) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int pitch = k * 2;
    uint8_t* tile = smem;
    float* x = reinterpret_cast<float*>(smem + rows * pitch);
    float* part = x + k;
    for (int i = threadIdx.x; i < rows * k; i += blockDim.x)
        reinterpret_cast<__nv_bfloat16*>(tile)[i] = __float2bfloat16(((i * 37) % 17 - 8) * 0.01f);
    for (int i = threadIdx.x; i < k; i += blockDim.x) x[i] = ((i * 13) % 7 - 3) * 0.1f;
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    float* pw = part + (warp % KS) * rows;
    for (int it = 0; it < iters; ++it)
        ls::gemv_ksplit<NW, ls::BF16, RB, KS>(tile, rows, k, pitch, x, warp, [&](int r, float v) { pw[r] += v; });
    __syncthreads();
    if (threadIdx.x == 0) sink[blockIdx.x] = part[0];
}

template <int RB, int KS>
int go(int rows, int k, int iters, float* sink, int grid, cudaStream_t s) {
    const int smem = rows * k * 2 + 4 * k + 4 * NW * rows;
    if (int e = (int)cudaFuncSetAttribute(tap_loop<RB, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return e;
    tap_loop<RB, KS><<<grid, 32 * NW, smem, s>>>(rows, k, iters, sink);
    return (int)cudaGetLastError();
}

template <int KS>
__global__ void __launch_bounds__(32 * NW, 1) tap_mma_loop(int rows, int k, int iters, float* sink) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int pitch = k * 2 + 16;
    uint8_t* tile = smem;
    auto* x = reinterpret_cast<__nv_bfloat16*>(smem + rows * pitch);
    float* part = reinterpret_cast<float*>(x + k);
    for (int i = threadIdx.x; i < rows * pitch / 2; i += blockDim.x)
        reinterpret_cast<__nv_bfloat16*>(tile)[i] = __float2bfloat16(((i * 37) % 17 - 8) * 0.01f);
    for (int i = threadIdx.x; i < k; i += blockDim.x) x[i] = __float2bfloat16(((i * 13) % 7 - 3) * 0.1f);
    for (int i = threadIdx.x; i < NW * rows; i += blockDim.x) part[i] = 0.f;
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    float* pw = part + (warp % KS) * rows;
    for (int it = 0; it < iters; ++it)
        ls::gemv_mma<NW, KS>(tile, rows, k, pitch, x, warp, [&](int r, float v) { pw[r] += v; });
    __syncthreads();
    if (threadIdx.x == 0) sink[blockIdx.x] = part[0];
}

template <int KS>
int go_mma(int rows, int k, int iters, float* sink, int grid, cudaStream_t s) {
    const int smem = rows * (k * 2 + 16) + 2 * k + 4 * NW * rows;
    if (int e = (int)cudaFuncSetAttribute(tap_mma_loop<KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return e;
    tap_mma_loop<KS><<<grid, 32 * NW, smem, s>>>(rows, k, iters, sink);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" int bench_tap(int rb, int ks, int rows, int k, int iters, float* sink, int grid, cudaStream_t s) {
    if (rb == 0) {
        if (ks == 1) return go_mma<1>(rows, k, iters, sink, grid, s);
        if (ks == 2) return go_mma<2>(rows, k, iters, sink, grid, s);
        if (ks == 4) return go_mma<4>(rows, k, iters, sink, grid, s);
        if (ks == 8) return go_mma<8>(rows, k, iters, sink, grid, s);
        return -1;
    }
#define C(R, S) \
    if (rb == R && ks == S) return go<R, S>(rows, k, iters, sink, grid, s);
    C(1, 1) C(1, 2) C(1, 4) C(1, 8) C(2, 1) C(2, 2) C(2, 4) C(2, 8)
    C(4, 1) C(4, 2) C(4, 4) C(4, 8) C(8, 1) C(8, 2) C(8, 4) C(8, 8)
#undef C
    return -1;
}
