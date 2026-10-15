// Phase-1 calibration microbenchmarks for B200 (see bench/run_micro.py).
//  stream_ring<SLOTS,SLOT_BYTES>: each CTA streams a byte band of a large
//      buffer through the WeightRing, consumers only touch one word per slot.
//      -> the streaming ceiling for a given ring depth / slot size.
//  stream_ldg: the same bytes with plain 16-byte ld.global.nc loads.
//  event_pingpong: one-way latency of an ls::signal -> ls::wait_geq hop
//      between CTA 0 and the last CTA.
//  grid_barriers: cost of ls::grid_barrier with every SM participating.
//  mma_rate: dense mma.sync m16n8k16 BF16 throughput.
#include <cuda_runtime.h>
#include "loadstream/ls.cuh"

namespace {

constexpr int NW = 4;

template <int SLOTS, int SLOT_BYTES>
__global__ void __launch_bounds__(32 * (NW + 1), 1) stream_ring(const uint8_t* buf, size_t bytes, float* sink) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto* bars = reinterpret_cast<ls::RingBarriers<SLOTS>*>(smem + SLOTS * SLOT_BYTES);
    ls::WeightRing<SLOTS, SLOT_BYTES>::init(bars, NW);
    __syncthreads();
    const size_t tiles = bytes / SLOT_BYTES;
    const ls::Range r = ls::band((int)tiles, gridDim.x, blockIdx.x);
    if (threadIdx.x < 32) {
        ls::Producer<SLOTS, SLOT_BYTES> prod(smem, bars, ls::policy_evict_first());
        for (int t = r.begin; t < r.end; ++t) prod.tile_contiguous(buf + (size_t)t * SLOT_BYTES, SLOT_BYTES);
    } else {
        ls::Consumer<SLOTS, SLOT_BYTES> cons(smem, bars);
        float acc = 0.f;
        for (int t = r.begin; t < r.end; ++t) {
            const uint8_t* s = cons.wait();
            acc += reinterpret_cast<const float*>(s)[threadIdx.x];
            cons.release();
        }
        if (acc == 12345.f) sink[0] = acc;
    }
}

__global__ void __launch_bounds__(512, 1) stream_ldg(const uint4* buf, size_t vecs, float* sink) {
    const ls::Range r = ls::band((int)(vecs / 512), gridDim.x, blockIdx.x);
    uint32_t acc = 0;
    const uint4* p = buf + (size_t)r.begin * 512 + threadIdx.x;
    const int n = r.size();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint4 v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = ls::ld_stream(p + (size_t)(i + j) * 512);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc ^= v[j].x ^ v[j].w;
    }
    for (; i < n; ++i) acc ^= ls::ld_stream(p + (size_t)i * 512).y;
    if (acc == 0x12345u) sink[0] = (float)acc;
}

__global__ void event_pingpong(uint32_t* ctr, int iters, unsigned long long* out_ns) {
    if (threadIdx.x != 0) return;
    const bool ping = blockIdx.x == 0, pong = blockIdx.x == gridDim.x - 1;
    if (!ping && !pong) return;
    const unsigned long long t0 = ls::globaltimer();
    for (int i = 0; i < iters; ++i) {
        if (ping) {
            ls::signal(&ctr[0], 1);
            ls::wait_geq(&ctr[1], i + 1);
        } else {
            ls::wait_geq(&ctr[0], i + 1);
            ls::signal(&ctr[1], 1);
        }
    }
    if (ping) out_ns[0] = ls::globaltimer() - t0;
}

__global__ void grid_barriers(uint32_t* ctr, int iters, unsigned long long* out_ns) {
    const unsigned long long t0 = ls::globaltimer();
    for (int i = 0; i < iters; ++i) ls::grid_barrier(ctr, 1, iters, i);
    if (blockIdx.x == 0 && threadIdx.x == 0) out_ns[0] = ls::globaltimer() - t0;
}

__global__ void __launch_bounds__(256) mma_rate(int iters, float* sink) {
    ls::MmaAcc acc[4] = {};
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x ^ 7u, 11u};
    uint32_t b[2] = {threadIdx.x * 5u, 13u};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int t = 0; t < 4; ++t) ls::mma_bf16_16816(acc[t], a, b);
    }
    float s = 0.f;
    for (int t = 0; t < 4; ++t) s += acc[t].c[0] + acc[t].c[1] + acc[t].c[2] + acc[t].c[3];
    if (s == 1.2345f) sink[0] = s;
}

template <int SLOTS, int SLOT_BYTES>
int launch_ring(const void* buf, size_t bytes, float* sink, int grid, cudaStream_t stream) {
    const int smem = SLOTS * SLOT_BYTES + (int)sizeof(ls::RingBarriers<SLOTS>);
    cudaFuncSetAttribute(stream_ring<SLOTS, SLOT_BYTES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    stream_ring<SLOTS, SLOT_BYTES><<<grid, 32 * (NW + 1), smem, stream>>>((const uint8_t*)buf, bytes, sink);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" int micro_stream_ring(int slots, int slot_kb, const void* buf, size_t bytes, float* sink, int grid,
                                 cudaStream_t s) {
#define RING_CASE(S, KB) \
    if (slots == S && slot_kb == KB) return launch_ring<S, KB * 1024>(buf, bytes, sink, grid, s);
    RING_CASE(2, 16) RING_CASE(4, 16) RING_CASE(8, 16) RING_CASE(12, 16)
    RING_CASE(2, 32) RING_CASE(4, 32) RING_CASE(6, 32)
    RING_CASE(2, 64) RING_CASE(3, 64)
    RING_CASE(4, 8) RING_CASE(8, 8) RING_CASE(16, 8)
    RING_CASE(8, 4) RING_CASE(16, 4) RING_CASE(32, 4)
#undef RING_CASE
    return -1;
}

extern "C" int micro_stream_ldg(const void* buf, size_t bytes, float* sink, int grid, cudaStream_t s) {
    stream_ldg<<<grid, 512, 0, s>>>((const uint4*)buf, bytes / 16, sink);
    return (int)cudaGetLastError();
}

extern "C" int micro_event_pingpong(uint32_t* ctr, int iters, unsigned long long* out, int grid, cudaStream_t s) {
    void* args[] = {&ctr, &iters, &out};
    return (int)cudaLaunchCooperativeKernel((void*)event_pingpong, dim3(grid), dim3(32), args, 0, s);
}

extern "C" int micro_grid_barriers(uint32_t* ctr, int iters, unsigned long long* out, int grid, int threads,
                                   cudaStream_t s) {
    void* args[] = {&ctr, &iters, &out};
    return (int)cudaLaunchCooperativeKernel((void*)grid_barriers, dim3(grid), dim3(threads), args, 0, s);
}

extern "C" int micro_mma_rate(int iters, float* sink, int grid, int threads, cudaStream_t s) {
    mma_rate<<<grid, threads, 0, s>>>(iters, sink);
    return (int)cudaGetLastError();
}
