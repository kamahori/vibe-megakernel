// GPU unit tests for the ring + taps (driven by tests/run_tests.py).
// Each CTA streams its band of W [rows, k] (any format, row-major) through
// the ring in tiles of `tile_rows` rows using copy_rows with a padded pitch,
// and the consumers compute y = W x (gemv) or Y = X W^T (mma, BF16).
#include <cuda_runtime.h>
#include "loadstream/ls.cuh"

namespace {

constexpr int NW = 4;
constexpr int SLOTS = 3;
constexpr int SLOT_BYTES = 32 * 1024;

template <class Fmt>
__global__ void __launch_bounds__(32 * (NW + 1)) gemv_test(const uint8_t* w, int rows, int k, int row_bytes,
                                                          const float* x, const float* scale, float* y,
                                                          int tile_rows, int pad) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto* bars = reinterpret_cast<ls::RingBarriers<SLOTS>*>(smem + SLOTS * SLOT_BYTES);
    float* xs = reinterpret_cast<float*>(bars + 1);
    float* scratch = xs + k;
    ls::WeightRing<SLOTS, SLOT_BYTES>::init(bars, NW);
    for (int i = threadIdx.x; i < k; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const ls::Range r = ls::band(rows, gridDim.x, blockIdx.x);
    const int pitch = row_bytes + pad;
    if (threadIdx.x < 32) {
        ls::Producer<SLOTS, SLOT_BYTES> prod(smem, bars, ls::policy_evict_first());
        for (int r0 = r.begin; r0 < r.end; r0 += tile_rows) {
            const int n = min(tile_rows, r.end - r0);
            prod.begin(n * row_bytes);
            prod.copy_rows(0, w + (size_t)r0 * row_bytes, n, row_bytes, row_bytes, pitch);
            prod.commit();
        }
    } else {
        ls::Consumer<SLOTS, SLOT_BYTES> cons(smem, bars);
        const int warp = (threadIdx.x >> 5) - 1;
        for (int r0 = r.begin; r0 < r.end; r0 += tile_rows) {
            const int n = min(tile_rows, r.end - r0);
            const uint8_t* t = cons.wait();
            ls::gemv_tile<NW, Fmt>(t, n, k, pitch, xs, scratch, warp, 1,
                                   [&](int i, float v) { y[r0 + i] = v * (scale ? scale[r0 + i] : 1.f); });
            cons.release();
        }
    }
}

// y = W x on tensor cores (gemv_mma), BF16 W streamed with a padded pitch,
// x BF16 in smem; KS K-slices reduced per tile through scratch.
template <int KS>
__global__ void __launch_bounds__(32 * (NW + 1)) gemv_mma_test(const __nv_bfloat16* w, int rows, int k,
                                                              const __nv_bfloat16* x, float* y) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto* bars = reinterpret_cast<ls::RingBarriers<SLOTS>*>(smem + SLOTS * SLOT_BYTES);
    auto* xs = reinterpret_cast<__nv_bfloat16*>(bars + 1);
    float* scratch = reinterpret_cast<float*>(xs + k);
    ls::WeightRing<SLOTS, SLOT_BYTES>::init(bars, NW);
    for (int i = threadIdx.x; i < k; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const ls::Range r = ls::band(rows, gridDim.x, blockIdx.x);
    const int pitch = 2 * k + 16, tile_rows = ls::rows_per_slot(SLOT_BYTES, pitch);
    if (threadIdx.x < 32) {
        ls::Producer<SLOTS, SLOT_BYTES> prod(smem, bars, ls::policy_evict_first());
        for (int r0 = r.begin; r0 < r.end; r0 += tile_rows) {
            const int n = min(tile_rows, r.end - r0);
            prod.begin(n * 2 * k);
            prod.copy_rows(0, w + (size_t)r0 * k, n, 2 * k, 2 * k, pitch);
            prod.commit();
        }
    } else {
        ls::Consumer<SLOTS, SLOT_BYTES> cons(smem, bars);
        const int warp = (threadIdx.x >> 5) - 1, tid = threadIdx.x - 32;
        for (int r0 = r.begin; r0 < r.end; r0 += tile_rows) {
            const int n = min(tile_rows, r.end - r0);
            const uint8_t* t = cons.wait();
            ls::gemv_mma<NW, KS>(t, n, k, pitch, xs, warp, [&](int i, float v) { scratch[(warp % KS) * n + i] = v; });
            cons.release();
            ls::named_sync(1, NW * 32);
            ls::reduce_ksplit<NW, KS>(scratch, n, n, tid, [&](int i, float v) { y[r0 + i] = v; });
            ls::named_sync(1, NW * 32);
        }
    }
}

// Y[t][row] for T = 8*NT tokens. A slot holds a K-chunk (KC columns) of a
// block of up to NW 16-row groups, one group per consumer warp; each warp
// keeps its accumulators across the chunks of its block.
template <int NT>
__global__ void __launch_bounds__(32 * (NW + 1)) mma_test(const __nv_bfloat16* w, int rows, int k,
                                                         const __nv_bfloat16* x, int tokens, float* y, int pad) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto* bars = reinterpret_cast<ls::RingBarriers<SLOTS>*>(smem + SLOTS * SLOT_BYTES);
    uint8_t* xs = reinterpret_cast<uint8_t*>(bars + 1);
    const int x_pitch = k * 2 + pad;
    ls::WeightRing<SLOTS, SLOT_BYTES>::init(bars, NW);
    for (int i = threadIdx.x; i < NT * 8 * k; i += blockDim.x) {
        const int t = i / k, c = i % k;
        reinterpret_cast<__nv_bfloat16*>(xs + t * x_pitch)[c] = t < tokens ? x[t * k + c] : __float2bfloat16(0.f);
    }
    __syncthreads();
    const ls::Range r = ls::band_rows(rows, 16, gridDim.x, blockIdx.x);
    const int row_bytes = k * 2;
    const int kc = min(k, ((SLOT_BYTES / (16 * NW) - pad) / 2) / 16 * 16);  // columns per chunk
    const int block = 16 * NW;
    if (threadIdx.x < 32) {
        ls::Producer<SLOTS, SLOT_BYTES> prod(smem, bars, ls::policy_evict_first());
        for (int r0 = r.begin; r0 < r.end; r0 += block) {
            const int n = min(block, r.end - r0);
            for (int k0 = 0; k0 < k; k0 += kc) {
                const int cols = min(kc, k - k0);
                prod.begin(n * cols * 2);
                prod.copy_rows(0, w + (size_t)r0 * k + k0, n, cols * 2, row_bytes, kc * 2 + pad);
                prod.commit();
            }
        }
    } else {
        ls::Consumer<SLOTS, SLOT_BYTES> cons(smem, bars);
        const int warp = (threadIdx.x >> 5) - 1;
        for (int r0 = r.begin; r0 < r.end; r0 += block) {
            const int n = min(block, r.end - r0);
            const bool mine = warp * 16 < n;
            ls::MmaAcc acc[NT] = {};
            for (int k0 = 0; k0 < k; k0 += kc) {
                const int cols = min(kc, k - k0);
                const uint8_t* t = cons.wait();
                if (mine) ls::mma_rows16<NT>(acc, t + warp * 16 * (kc * 2 + pad), kc * 2 + pad, xs + k0 * 2, x_pitch, cols);
                cons.release();
            }
            if (mine)
                ls::mma_for_each<NT>(acc, [&](int m, int tok, float v) {
                    if (tok < tokens) y[(size_t)tok * rows + r0 + warp * 16 + m] = v;
                });
        }
    }
}

int ring_smem() { return SLOTS * SLOT_BYTES + (int)sizeof(ls::RingBarriers<SLOTS>); }

template <class K>
int prep(K kernel, int smem) {
    return (int)cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}

}  // namespace

extern "C" int test_gemv(int fmt, const void* w, int rows, int k, int row_bytes, const float* x, const float* scale,
                         float* y, int tile_rows, int pad, int grid) {
    const int smem = ring_smem() + 4 * k + 4 * NW * tile_rows;
#define GO(F)                                                                                           \
    {                                                                                                   \
        if (int e = prep(gemv_test<F>, smem)) return e;                                                 \
        gemv_test<F><<<grid, 32 * (NW + 1), smem>>>((const uint8_t*)w, rows, k, row_bytes, x, scale, y, \
                                                    tile_rows, pad);                                     \
        return (int)cudaGetLastError();                                                                 \
    }
    if (fmt == 0) GO(ls::BF16)
    if (fmt == 1) GO(ls::FP8E4M3)
    if (fmt == 2) GO(ls::I8)
    if (fmt == 3) GO(ls::U4Off8)
#undef GO
    return -1;
}

extern "C" int test_gemv_mma(const void* w, int rows, int k, const void* x, float* y, int ks, int grid) {
    const int pitch = 2 * k + 16;
    const int smem = ring_smem() + 2 * k + 4 * NW * (SLOT_BYTES / pitch);
#define GO(KS)                                                                                             if (ks == KS) {                                                                                            if (int e = prep(gemv_mma_test<KS>, smem)) return e;                                                   gemv_mma_test<KS><<<grid, 32 * (NW + 1), smem>>>((const __nv_bfloat16*)w, rows, k,                                                                      (const __nv_bfloat16*)x, y);                          return (int)cudaGetLastError();                                                                    }
    GO(1) GO(2) GO(4)
#undef GO
    return -1;
}

extern "C" int test_mma(const void* w, int rows, int k, const void* x, int tokens, float* y, int pad, int grid) {
    const int nt = (tokens + 7) / 8;
    const int smem = ring_smem() + nt * 8 * (k * 2 + pad);
#define GO(NT)                                                                                              \
    if (nt == NT) {                                                                                         \
        if (int e = prep(mma_test<NT>, smem)) return e;                                                     \
        mma_test<NT><<<grid, 32 * (NW + 1), smem>>>((const __nv_bfloat16*)w, rows, k, (const __nv_bfloat16*)x, \
                                                    tokens, y, pad);                                         \
        return (int)cudaGetLastError();                                                                     \
    }
    GO(1) GO(2) GO(7)
#undef GO
    return -1;
}
