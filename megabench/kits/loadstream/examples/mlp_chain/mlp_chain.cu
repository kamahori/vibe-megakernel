// Worked example: a chain of L residual MLP blocks at batch 1, as one
// persistent cooperative kernel built from loadstream primitives.
//
//   for l in 0..L-1:  n = rms(x) * norm[l]
//                     h = silu(Wg[l] n) * (Wu[l] n)        Wg, Wu: [I, H] bf16
//                     x = x + Wd[l] h                       Wd:     [H, I] bf16
//
// Structure (the load-centric pattern):
//  * warp 0 = producer: streams this CTA's byte-balanced weight bands for
//    every op of every layer, in order, into a ring of SLOTS x SLOT_BYTES.
//    It waits only for free slots, so layer l+1's gate/up weights land in
//    shared memory while layer l's down projection is still finishing
//    elsewhere on the GPU. A slot is one 16-row group x KC columns (a
//    K-chunk), rows landing with a padded pitch (2*KC + 16 bytes) for
//    conflict-free ldmatrix; gate/up groups are 8 gate rows + the same 8
//    up rows, so silu(g)*u pairs land in the same slot.
//  * warps 1..NW = consumers (taps): wait on events for their activation
//    inputs, compute on each slot with tensor-core GEMV (ls::gemv_mma, x in
//    BF16 as at a model's matmul input), publish outputs with events.
//  * no grid barriers: ev_h[l] counts rows of h written, ev_x[l] counts
//    rows of x_{l+1} written; each call adds I resp. H, so the wait target
//    is epoch * I resp. epoch * H. Each CTA signals once per op, after its
//    whole band (the consumer of h needs all of h anyway).
//  * every consumer warp takes 1/NW of each slot's k-steps and keeps its
//    MMA accumulators across the K-chunks of a group; partials for the
//    CTA's whole band go to smem, so each op costs ONE consumer sync and one
//    reduction instead of syncs per slot.
//  * rms(x) is recomputed by every CTA from x (L2-resident) instead of being
//    produced once and broadcast through another event; the taps see
//    x * norm and the scalar 1/rms is applied in the reductions.
#include <cuda_runtime.h>
#include "loadstream/ls.cuh"

namespace {

#ifndef MLP_SLOTS
#define MLP_SLOTS 3
#endif
#ifndef MLP_KC
#define MLP_KC 2048                          // columns per K-chunk
#endif
constexpr int NW = 8;                        // consumer warps
constexpr int THREADS = 32 * (NW + 1);
constexpr int KC = MLP_KC;
constexpr int PITCH = 2 * KC + 16;           // padded smem row pitch, bytes
constexpr int SLOTS = MLP_SLOTS;
constexpr int SLOT_BYTES = 16 * PITCH;       // one 16-row group x KC columns
constexpr int CONSUMER_BAR = 1;              // named barrier id for consumer warps

// Stamp tags (consumer warp 0 lane 0, and producer lane 0).
enum Tag : uint32_t {
    T_START = 1, T_WAIT_X = 2, T_GOT_X = 3, T_NORM = 4, T_GU_TAPS = 5, T_GU_DONE = 6, T_WAIT_H = 7,
    T_GOT_H = 8, T_H_LOADED = 9, T_DN_TAPS = 10, T_DOWN_DONE = 11, T_PROD_LAYER = 12, T_PROD_DONE = 13,
    T_END = 14,
};

struct Params {
    const float* x_in;               // [H]
    const __nv_bfloat16* norm;       // [L, H]
    const __nv_bfloat16* wg;         // [L, I, H]
    const __nv_bfloat16* wu;         // [L, I, H]
    const __nv_bfloat16* wd;         // [L, H, I]
    float* xs;                       // [L, H]   x after layer l
    float* hs;                       // [L, I]
    uint32_t* ev_h;                  // [L]
    uint32_t* ev_x;                  // [L]
    int L, H, I;
    uint32_t epoch;                  // 1-based call index
    ls::Stamps stamps;
};

struct Smem {
    ls::RingBarriers<SLOTS> bars;
    float red[NW];
};

__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

__device__ void producer(const Params& p, uint8_t* ring_data, Smem* sm) {
    ls::Producer<SLOTS, SLOT_BYTES> prod(ring_data, &sm->bars, ls::policy_evict_first());
    const ls::Range gu = ls::band(p.I, gridDim.x, blockIdx.x);
    const ls::Range dn = ls::band(p.H, gridDim.x, blockIdx.x);
    for (int l = 0; l < p.L; ++l) {
        if (prod.lane == 0) ls::stamp(p.stamps, T_PROD_LAYER, l);
        const __nv_bfloat16* wg = p.wg + (size_t)l * p.I * p.H;
        const __nv_bfloat16* wu = p.wu + (size_t)l * p.I * p.H;
        for (int b = gu.begin; b < gu.end; b += 8) {          // 8 gate rows + the same 8 up rows
            const int nb = min(8, gu.end - b);
            for (int c = 0; c < p.H; c += KC) {
                const int cols = min(KC, p.H - c);
                prod.begin(2 * nb * cols * 2);
                prod.copy_rows(0, wg + (size_t)b * p.H + c, nb, cols * 2, p.H * 2, PITCH);
                prod.copy_rows(nb * PITCH, wu + (size_t)b * p.H + c, nb, cols * 2, p.H * 2, PITCH);
                prod.commit();
            }
        }
        const __nv_bfloat16* wd = p.wd + (size_t)l * p.H * p.I;
        for (int b = dn.begin; b < dn.end; b += 16) {
            const int nb = min(16, dn.end - b);
            for (int c = 0; c < p.I; c += KC) {
                const int cols = min(KC, p.I - c);
                prod.begin(nb * cols * 2);
                prod.copy_rows(0, wd + (size_t)b * p.I + c, nb, cols * 2, p.I * 2, PITCH);
                prod.commit();
            }
        }
    }
    if (prod.lane == 0) ls::stamp(p.stamps, T_PROD_DONE, (uint32_t)(prod.bytes_issued >> 10));
}

// Partial sums: NW slices x (2 * gate/up band) floats, reused by down.
__host__ __device__ inline int part_stride(int H, int I, int grid) {
    const int gu = (I + grid - 1) / grid, dn = (H + grid - 1) / grid;
    return 2 * gu > dn ? 2 * gu : dn;
}

// Norm weights per consumer thread, loaded before the x wait (H <= 256 * 16).
constexpr int NORM_REGS = 16;

__device__ void consumer(const Params& p, uint8_t* ring_data, Smem* sm, float* x_s, __nv_bfloat16* n_s,
                         __nv_bfloat16* h_s, float* part) {
    ls::Consumer<SLOTS, SLOT_BYTES> cons(ring_data, &sm->bars);
    const int warp = (threadIdx.x >> 5) - 1, lane = threadIdx.x & 31, tid = threadIdx.x - 32;
    constexpr int T = NW * 32;
    const bool lead = tid == 0;
    const ls::Range gu = ls::band(p.I, gridDim.x, blockIdx.x);
    const ls::Range dn = ls::band(p.H, gridDim.x, blockIdx.x);
    const int stride = part_stride(p.H, p.I, gridDim.x), gu_n = gu.size();
    if (lead) ls::stamp(p.stamps, T_START);

    for (int l = 0; l < p.L; ++l) {
        // ---- n = x_l * norm (1/rms applied in the reductions) ----
        // The norm weights do not depend on x_l: load them before the wait.
        float nw[NORM_REGS];
#pragma unroll
        for (int j = 0; j < NORM_REGS; ++j) {
            const int i = tid + j * T;
            nw[j] = i < p.H ? __bfloat162float(p.norm[(size_t)l * p.H + i]) : 0.f;
        }
        const float* x = l == 0 ? p.x_in : p.xs + (size_t)(l - 1) * p.H;
        if (lead) {
            ls::stamp(p.stamps, T_WAIT_X, l);
            if (l > 0) ls::wait_geq(&p.ev_x[l - 1], p.epoch * p.H, 1000 + l);
            ls::stamp(p.stamps, T_GOT_X, l);
        }
        ls::named_sync(CONSUMER_BAR, T);
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < NORM_REGS; ++j) {
            const int i = tid + j * T;
            if (i < p.H) {
                const float v = ls::ld_cg(x + i);
                x_s[i] = v;
                n_s[i] = __float2bfloat16(v * nw[j]);
                ss += v * v;
            }
        }
        ss = ls::warp_sum(ss);
        if (lane == 0) sm->red[warp] = ss;
        ls::named_sync(CONSUMER_BAR, T);
        if (lead) ls::stamp(p.stamps, T_NORM, l);
        float total = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) total += sm->red[w];
        const float inv = rsqrtf(total / p.H + 1e-6f);

        // ---- gate/up: groups of 8 gate rows + the same 8 up rows ----
        float* pw = part + warp * stride;
        for (int b = gu.begin; b < gu.end; b += 8) {
            const int nb = min(8, gu.end - b);
            ls::GemvAcc acc = {};
            for (int c = 0; c < p.H; c += KC) {
                const uint8_t* tile = cons.wait();
                ls::gemv_mma_acc<NW>(acc, tile, 2 * nb, min(KC, p.H - c), PITCH, n_s + c, warp);
                cons.release();
            }
            ls::gemv_mma_store(acc, 2 * nb, [&](int i, float v) {
                pw[i < nb ? b - gu.begin + i : gu_n + b - gu.begin + i - nb] = v;
            });
        }
        if (lead) ls::stamp(p.stamps, T_GU_TAPS, l);
        ls::named_sync(CONSUMER_BAR, T);
        float* h_out = p.hs + (size_t)l * p.I;
        for (int j = tid; j < gu_n; j += T) {
            float g = 0.f, u = 0.f;
#pragma unroll
            for (int w = 0; w < NW; ++w) g += part[w * stride + j], u += part[w * stride + gu_n + j];
            h_out[gu.begin + j] = silu(g * inv) * (u * inv);
        }
        ls::named_sync(CONSUMER_BAR, T);
        if (lead) {
            ls::signal(&p.ev_h[l], gu_n);
            ls::stamp(p.stamps, T_GU_DONE, l);
        }

        // ---- down: needs all of h_l ----
        if (lead) {
            ls::stamp(p.stamps, T_WAIT_H, l);
            ls::wait_geq(&p.ev_h[l], p.epoch * p.I, 2000 + l);
            ls::stamp(p.stamps, T_GOT_H, l);
        }
        ls::named_sync(CONSUMER_BAR, T);
        for (int i = tid; i < p.I / 4; i += T) {
            const float4 v = ls::ld_cg(reinterpret_cast<const float4*>(h_out) + i);
            reinterpret_cast<__nv_bfloat162*>(h_s)[2 * i] = __floats2bfloat162_rn(v.x, v.y);
            reinterpret_cast<__nv_bfloat162*>(h_s)[2 * i + 1] = __floats2bfloat162_rn(v.z, v.w);
        }
        ls::named_sync(CONSUMER_BAR, T);
        if (lead) ls::stamp(p.stamps, T_H_LOADED, l);
        for (int b = dn.begin; b < dn.end; b += 16) {
            const int nb = min(16, dn.end - b);
            ls::GemvAcc acc = {};
            for (int c = 0; c < p.I; c += KC) {
                const uint8_t* tile = cons.wait();
                ls::gemv_mma_acc<NW>(acc, tile, nb, min(KC, p.I - c), PITCH, h_s + c, warp);
                cons.release();
            }
            ls::gemv_mma_store(acc, nb, [&](int i, float v) { pw[b - dn.begin + i] = v; });
        }
        if (lead) ls::stamp(p.stamps, T_DN_TAPS, l);
        ls::named_sync(CONSUMER_BAR, T);
        float* x_out = p.xs + (size_t)l * p.H;
        ls::reduce_ksplit<NW, NW>(part, stride, dn.size(), tid,
                                         [&](int j, float v) { x_out[dn.begin + j] = x_s[dn.begin + j] + v; });
        ls::named_sync(CONSUMER_BAR, T);
        if (lead) {
            ls::signal(&p.ev_x[l], dn.size());
            ls::stamp(p.stamps, T_DOWN_DONE, l);
        }
    }
    if (lead) ls::stamp(p.stamps, T_END);
}

__global__ void __launch_bounds__(THREADS, 1) mlp_chain_kernel(Params p) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* ring_data = smem_raw;
    Smem* sm = reinterpret_cast<Smem*>(smem_raw + SLOTS * SLOT_BYTES);
    float* x_s = reinterpret_cast<float*>(sm + 1);                       // H fp32 (residual)
    float* part = x_s + p.H;                                             // NW * part_stride fp32
    auto* n_s = reinterpret_cast<__nv_bfloat16*>(part + NW * part_stride(p.H, p.I, gridDim.x));  // H bf16
    auto* h_s = n_s + ((p.H + 7) & ~7);                                  // I bf16
    ls::WeightRing<SLOTS, SLOT_BYTES>::init(&sm->bars, NW);
    __syncthreads();
    if (threadIdx.x < 32)
        producer(p, ring_data, sm);
    else
        consumer(p, ring_data, sm, x_s, n_s, h_s, part);
}

size_t smem_bytes(int H, int I, int grid) {
    return SLOTS * SLOT_BYTES + sizeof(Smem) + sizeof(float) * (H + NW * part_stride(H, I, grid)) +
           2 * (((H + 7) & ~7) + I);
}

}  // namespace

extern "C" int mlp_chain_launch(const float* x_in, const void* norm, const void* wg, const void* wu, const void* wd,
                                float* xs, float* hs, uint32_t* ev_h, uint32_t* ev_x, int L, int H, int I,
                                uint32_t epoch, long long* stamps, int stamp_capacity, int grid,
                                cudaStream_t stream) {
    Params p{x_in, (const __nv_bfloat16*)norm, (const __nv_bfloat16*)wg, (const __nv_bfloat16*)wu,
             (const __nv_bfloat16*)wd, xs, hs, ev_h, ev_x, L, H, I, epoch, {stamps, stamp_capacity}};
    const size_t smem = smem_bytes(H, I, grid);
    static size_t configured = 0;
    if (configured != smem) {
        cudaError_t e = cudaFuncSetAttribute(mlp_chain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return (int)e;
        configured = smem;
    }
    void* args[] = {&p};
    return (int)cudaLaunchCooperativeKernel((void*)mlp_chain_kernel, dim3(grid), dim3(THREADS), args, smem, stream);
}

extern "C" int mlp_chain_smem_bytes(int H, int I, int grid) { return (int)smem_bytes(H, I, grid); }
