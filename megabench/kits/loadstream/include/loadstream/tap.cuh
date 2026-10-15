// Taps: consumers of streamed weight tiles. All are deterministic: a fixed
// assignment of work to warps and a fixed reduction order, so results are
// bit-identical run to run.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "ptx.cuh"

namespace ls {

// ------------------------------------------------------------ reductions
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Argmax with torch.argmax tie-breaking (lowest index wins). NaN is not handled.
__device__ __forceinline__ void warp_argmax(float& v, int& i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, v, o);
        int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (ov > v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
}

// Reduce RB per-lane values (RB a power of two <= 32) across the warp with a
// transpose butterfly: RB/2 + RB/4 + ... + 1 shuffles for the first
// log2(RB) stages, then 5 - log2(RB) plain ones (9 shuffles for RB = 8
// instead of 40). Afterwards lane l holds the sum for row l >> (5 - log2 RB);
// lanes with (l & (32/RB - 1)) == 0 are the writers (see multi_row/multi_writer).
template <int RB>
__device__ __forceinline__ float warp_sum_multi(float (&v)[RB]) {
    static_assert(RB >= 1 && RB <= 32 && (RB & (RB - 1)) == 0, "RB must be a power of two <= 32");
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int n = RB, o = 16; n > 1; n >>= 1, o >>= 1) {
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < n / 2; ++i) {
            const float send = hi ? v[i] : v[i + n / 2];
            const float keep = hi ? v[i + n / 2] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    float s = v[0];
#pragma unroll
    for (int o = 16 / RB; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}
template <int RB>
__device__ __forceinline__ int multi_row(int lane) {
    return RB == 1 ? 0 : lane / (32 / RB);
}
template <int RB>
__device__ __forceinline__ bool multi_writer(int lane) {
    return (lane & (32 / RB - 1)) == 0;
}

// --------------------------------------------- 16-byte weight vector dots
// Each returns sum_j w[j] * x[j] for the elements packed in one uint4.
struct BF16 {  // 8 bf16
    static constexpr int kElems = 8;
    __device__ static float dot(uint4 w, const float* x) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&w);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(h[j]);
            s = fmaf(f.x, x[2 * j], s);
            s = fmaf(f.y, x[2 * j + 1], s);
        }
        return s;
    }
};
struct FP8E4M3 {  // 16 e4m3, scale applied by the caller
    static constexpr int kElems = 16;
    __device__ static float dot(uint4 w, const float* x) {
        const __nv_fp8x2_e4m3* h = reinterpret_cast<const __nv_fp8x2_e4m3*>(&w);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float2 f = static_cast<float2>(h[j]);
            s = fmaf(f.x, x[2 * j], s);
            s = fmaf(f.y, x[2 * j + 1], s);
        }
        return s;
    }
};
struct I8 {  // 16 int8, per-row scale applied by the caller
    static constexpr int kElems = 16;
    __device__ static float dot(uint4 w, const float* x) {
        const int8_t* q = reinterpret_cast<const int8_t*>(&w);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) s = fmaf(static_cast<float>(q[j]), x[j], s);
        return s;
    }
};
struct U4Off8 {  // 32 nibbles: byte b holds col 2b (low) and 2b+1 (high), value = nibble - 8
    static constexpr int kElems = 32;
    __device__ static float dot(uint4 w, const float* x) {
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&w);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint32_t byte = (u[j] >> (8 * b)) & 0xffu;
                s = fmaf(static_cast<float>(static_cast<int>(byte & 15u) - 8), x[8 * j + 2 * b], s);
                s = fmaf(static_cast<float>(static_cast<int>(byte >> 4) - 8), x[8 * j + 2 * b + 1], s);
            }
        }
        return s;
    }
};

// Lane-partial dots of RB rows against the same x segment: row i starts at
// w + off[i] (byte offsets; rows may repeat, e.g. clamped padding rows).
// Each lane takes uint4 vectors v0 + lane, v0 + lane + 32, ... < v1, so
// every x load feeds RB rows and the unrolled loop keeps several smem loads
// in flight. Not reduced across lanes.
template <class Fmt, int RB>
__device__ __forceinline__ void lane_dot_rows(float (&s)[RB], const uint8_t* w, const int (&off)[RB],
                                              const float* x, int v0, int v1) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < RB; ++r) s[r] = 0.f;
#pragma unroll 2
    for (int v = v0 + lane; v < v1; v += 32) {
        uint4 q[RB];
#pragma unroll
        for (int r = 0; r < RB; ++r) q[r] = reinterpret_cast<const uint4*>(w + off[r])[v];
        float xv[Fmt::kElems];
#pragma unroll
        for (int j = 0; j < Fmt::kElems; j += 4) {
            float4 f = *reinterpret_cast<const float4*>(x + v * Fmt::kElems + j);
            xv[j] = f.x, xv[j + 1] = f.y, xv[j + 2] = f.z, xv[j + 3] = f.w;
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) s[r] += Fmt::dot(q[r], xv);
    }
}

// Single-row lane-partial dot (x 16-byte aligned, fp32).
template <class Fmt>
__device__ __forceinline__ float lane_dot(const uint4* w, const float* x, int vecs) {
    float s[1];
    const int off[1] = {0};
    lane_dot_rows<Fmt, 1>(s, reinterpret_cast<const uint8_t*>(w), off, x, 0, vecs);
    return s[0];
}

// RB rows r0 .. r0+RB-1 of a tile, rows past `rows` clamped to the last row
// (computed, not stored); returns this lane's reduced value for row
// r0 + multi_row<RB>(lane), valid where multi_writer<RB>(lane).
template <class Fmt, int RB>
__device__ __forceinline__ float rows_block(const uint8_t* tile, int r0, int rows, int pitch, const float* x,
                                            int v0, int v1) {
    int off[RB];
#pragma unroll
    for (int i = 0; i < RB; ++i) off[i] = min(r0 + i, rows - 1) * pitch;
    float s[RB];
    lane_dot_rows<Fmt, RB>(s, tile, off, x, v0, v1);
    return warp_sum_multi<RB>(s);
}

// ------------------------------------------------------------- GEMV taps
// Building blocks, none of which synchronizes warps:
//
//  gemv_rows    blocks of RB rows strided over warps, full K per warp;
//               emit(r, y). Best when the tile has >= NW*RB rows.
//  gemv_ksplit  every warp takes 1/NW of K for every row of the tile and
//               calls store(r, partial). Use for tiles of few rows (one
//               slot of a band: 2-10 rows). Store partials for a whole band
//               (all its slots) in smem, then do ONE consumer sync and
//               reduce_ksplit -- never a sync per slot.
//  reduce_ksplit fixed-order sum of the NW partials, one row per thread.
//
// RB rows share every x load and one transpose reduction (warp_sum_multi),
// which is what keeps a tap ahead of the stream: with 1-4 vectors per lane
// per row, shuffle latency, not FMA, is the cost. All are deterministic.
// `warp` = 0..NW-1 within the consumer group. x is fp32 in smem, 16-byte
// aligned. emit/store are called by one lane per row.
template <int NW, class Fmt, int RB = 4, class Emit>
__device__ __forceinline__ void gemv_rows(const uint8_t* tile, int rows, int k, int pitch, const float* x, int warp,
                                          Emit&& emit) {
    const int lane = threadIdx.x & 31;
    const int vecs = k / Fmt::kElems;
    for (int r0 = warp * RB; r0 < rows; r0 += NW * RB) {
        const float y = rows_block<Fmt, RB>(tile, r0, rows, pitch, x, 0, vecs);
        const int r = r0 + multi_row<RB>(lane);
        if (multi_writer<RB>(lane) && r < rows) emit(r, y);
    }
}

// KS = number of K slices (default NW). With KS < NW the warps form NW/KS
// row groups: warp w takes slice w % KS of the rows in blocks
// (w / KS) * RB, step (NW / KS) * RB. Pick KS so each lane has >= 1-2
// vectors per row: k / Fmt::kElems / 32 / KS >= 1 (e.g. KS = 4 for BF16
// k = 1024). The partial passed to store(r, v) belongs to slice w % KS;
// reduce over KS slices.
template <int NW, class Fmt, int RB = 8, int KS = NW, class Store>
__device__ __forceinline__ void gemv_ksplit(const uint8_t* tile, int rows, int k, int pitch, const float* x,
                                            int warp, Store&& store) {
    static_assert(NW % KS == 0, "KS must divide NW");
    const int lane = threadIdx.x & 31;
    const int vecs = k / Fmt::kElems;
    const int per = (vecs + KS - 1) / KS, slice = warp % KS;
    const int v0 = min(vecs, slice * per), v1 = min(vecs, v0 + per);
    for (int r0 = (warp / KS) * RB; r0 < rows; r0 += (NW / KS) * RB) {
        const float y = rows_block<Fmt, RB>(tile, r0, rows, pitch, x, v0, v1);
        const int r = r0 + multi_row<RB>(lane);
        if (multi_writer<RB>(lane) && r < rows) store(r, y);
    }
}

// Call after a sync over the NW consumer warps; `tid` = 0..NW*32-1.
// Sums KS slices part[s * part_stride + r] in order; emit(r, sum) once per
// row in [0, rows).
template <int NW, int KS = NW, class Emit>
__device__ __forceinline__ void reduce_ksplit(const float* part, int part_stride, int rows, int tid, Emit&& emit) {
    for (int r = tid; r < rows; r += NW * 32) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < KS; ++w) s += part[w * part_stride + r];
        emit(r, s);
    }
}

// Convenience for one self-contained tile (used by the unit tests): rows
// mode when rows >= 4 * NW, else K-split with a named barrier `bar_id` over
// the NW consumer warps (two syncs per tile -- fine for tests, too slow for
// a hot loop; there, keep gemv_ksplit partials for the band instead).
// `scratch` needs NW * rows floats.
template <int NW, class Fmt, class Emit>
__device__ __forceinline__ void gemv_tile(const uint8_t* tile, int rows, int k, int pitch, const float* x,
                                          float* scratch, int warp, int bar_id, Emit&& emit) {
    if (rows >= 4 * NW) {
        gemv_rows<NW, Fmt>(tile, rows, k, pitch, x, warp, emit);
        return;
    }
    gemv_ksplit<NW, Fmt>(tile, rows, k, pitch, x, warp, [&](int r, float v) { scratch[warp * rows + r] = v; });
    named_sync(bar_id, NW * 32);
    reduce_ksplit<NW>(scratch, rows, rows, warp * 32 + (threadIdx.x & 31), emit);
    named_sync(bar_id, NW * 32);  // scratch is reused by the next tile
}

// -------------------------------------------- small-M tensor-core tap (BF16)
// Y[n][m] += sum_k W[m][k] * X[n][k] for 16 weight rows (m) x 8 tokens (n)
// per mma, via mma.sync.m16n8k16 (A = weight tile, B = activations).
// W tile rows and X rows live in shared memory with pitches in bytes;
// pad pitches to (16 mod 128) bytes to avoid ldmatrix bank conflicts.
struct MmaAcc {
    float c[4];  // c0,c1: row lane/4, cols 2*(lane%4)+{0,1}; c2,c3: row lane/4+8
};

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16_16816(MmaAcc& acc, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(acc.c[0]), "+f"(acc.c[1]), "+f"(acc.c[2]), "+f"(acc.c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One warp: accumulate a 16-row x NT*8-token block over `kn` columns
// (multiple of 16). w = column 0 of the first of the 16 weight rows, x =
// the same column of token 0's row. For a K-chunked tile (the slot holds
// columns [k0, k0+kn) of 16*G rows), pass x + 2*k0 and keep acc across
// slots.
template <int NT>
__device__ __forceinline__ void mma_rows16(MmaAcc (&acc)[NT], const uint8_t* w, int w_pitch, const uint8_t* x,
                                           int x_pitch, int kn) {
    const int lane = threadIdx.x & 31;
    const uint8_t* a_base = w + (lane & 15) * w_pitch + (lane >> 4) * 16;
    const uint8_t* b_base = x + (lane & 7) * x_pitch + ((lane >> 3) & 1) * 16;
    for (int k = 0; k < kn; k += 16) {
        uint32_t a[4];
        ldmatrix_x4(a, a_base + k * 2);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            uint32_t b[2];
            ldmatrix_x2(b, b_base + t * 8 * x_pitch + k * 2);
            mma_bf16_16816(acc[t], a, b);
        }
    }
}

// ------------------------------------------- tensor-core GEMV tap (BF16)
// y[r] = sum_k W[r][k] * x[k] for one token, on tensor cores: each
// mma.m16n8k16 takes a 16 x 16 weight block (ldmatrix from the slot) and x as
// the only nonzero column of B (lanes 0-3 load it from smem), so one
// instruction covers 256 weights. Use it for every BF16 matrix at batch 1;
// the CUDA-core taps barely keep up with the stream (bench/run_taps.py).
// x is BF16 (as at the matmul input of the reference models), accumulation
// FP32.
//
// Slot layout: ONE 16-row group x a K-chunk of kn columns (multiple of 16),
// rows `pitch` bytes apart with pitch = 2*kn + 16 (copy_rows with dst_pitch
// = row_bytes + 16; an unpadded pitch is an 8-way ldmatrix bank conflict).
// Rows >= `rows` are clamped (computed, never stored), so a band of 7 rows
// works, but each slot costs the same MMA work as 16 rows: put as many rows
// in the group as the band allows. Size slots as 16 * (2*KC + 16) bytes,
// e.g. KC = 1024 -> 33,024 B (slots need not be powers of two).
//
// Per warp: zero a GemvAcc, call gemv_mma_acc for each K-chunk slot of the
// group (the warp does the k-steps of `slice` of KS), then gemv_mma_store
// once. With all NW warps as the KS slices of one group (KS = NW), reduce
// the NW partials per row with reduce_ksplit<NW, NW>.
// mma.sync latency on B200 is ~100+ cycles while its issue cost is ~2, so
// a warp needs many independent accumulator chains in flight.
constexpr int kGemvChains = 8;
struct GemvAcc {
    MmaAcc a[kGemvChains];
};

template <int KS>
__device__ __forceinline__ void gemv_mma_acc(GemvAcc& acc, const uint8_t* tile, int rows, int kn, int pitch,
                                             const __nv_bfloat16* x, int slice) {
    const int lane = threadIdx.x & 31;
    const int steps = kn / 16, per = (steps + KS - 1) / KS;
    const int s0 = min(steps, slice * per), s1 = min(steps, s0 + per);
    const uint8_t* a_base = tile + min(lane & 15, rows - 1) * pitch + (lane >> 4) * 16;
    const uint32_t* xw = reinterpret_cast<const uint32_t*>(x) + (lane & 3);  // lanes 0-3: k pairs 2*lane
    int s = s0;
    // ldmatrix/mma are volatile asm (issued in program order), so batch the
    // loads of kGemvChains steps before their mmas: all loads overlap, then
    // all mmas (independent accumulators) overlap.
    constexpr int U = kGemvChains;
    for (; s + U <= s1; s += U) {
        uint32_t a[U][4], b[U][2];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            ldmatrix_x4(a[u], a_base + (s + u) * 32);
            b[u][0] = lane < 4 ? xw[(s + u) * 8] : 0u;
            b[u][1] = lane < 4 ? xw[(s + u) * 8 + 4] : 0u;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) mma_bf16_16816(acc.a[u], a[u], b[u]);
    }
    if (s + U / 2 <= s1) {
        uint32_t a[U / 2][4], b[U / 2][2];
#pragma unroll
        for (int u = 0; u < U / 2; ++u) {
            ldmatrix_x4(a[u], a_base + (s + u) * 32);
            b[u][0] = lane < 4 ? xw[(s + u) * 8] : 0u;
            b[u][1] = lane < 4 ? xw[(s + u) * 8 + 4] : 0u;
        }
#pragma unroll
        for (int u = 0; u < U / 2; ++u) mma_bf16_16816(acc.a[u], a[u], b[u]);
        s += U / 2;
    }
    for (; s < s1; ++s) {
        uint32_t a[4], b[2];
        ldmatrix_x4(a, a_base + s * 32);
        b[0] = lane < 4 ? xw[s * 8] : 0u;
        b[1] = lane < 4 ? xw[s * 8 + 4] : 0u;
        mma_bf16_16816(acc.a[0], a, b);  // static index: keeps acc in registers
    }
}

// store(r, value) for r < rows (row within the group), once per row.
template <class Store>
__device__ __forceinline__ void gemv_mma_store(const GemvAcc& acc, int rows, Store&& store) {
    const int lane = threadIdx.x & 31;
    if ((lane & 3) == 0) {  // token column 0: c0 = row lane/4, c2 = row lane/4 + 8
        const int r = lane >> 2;
        float lo = 0.f, hi = 0.f;
#pragma unroll
        for (int u = 0; u < kGemvChains; ++u) lo += acc.a[u].c[0], hi += acc.a[u].c[2];
        if (r < rows) store(r, lo);
        if (r + 8 < rows) store(r + 8, hi);
    }
}

// Whole-K convenience (unit tests, small k): a tile of `rows` full rows
// (k columns, pitch 2k + 16), 16-row groups over NW/KS row-group lanes, KS
// K-slices; store(r, partial) per row per slice (slice = warp % KS).
template <int NW, int KS = NW, class Store>
__device__ __forceinline__ void gemv_mma(const uint8_t* tile, int rows, int k, int pitch, const __nv_bfloat16* x,
                                         int warp, Store&& store) {
    static_assert(NW % KS == 0, "KS must divide NW");
    for (int g = (warp / KS) * 16; g < rows; g += (NW / KS) * 16) {
        GemvAcc acc = {};
        gemv_mma_acc<KS>(acc, tile + g * pitch, rows - g, k, pitch, x, warp % KS);
        gemv_mma_store(acc, rows - g, [&](int r, float v) { store(g + r, v); });
    }
}

// Visit the accumulator elements: fn(row_in_16, token, value).
template <int NT, class Fn>
__device__ __forceinline__ void mma_for_each(const MmaAcc (&acc)[NT], Fn&& fn) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int n = t * 8 + 2 * (lane & 3);
        fn(lane >> 2, n, acc[t].c[0]);
        fn(lane >> 2, n + 1, acc[t].c[1]);
        fn((lane >> 2) + 8, n, acc[t].c[2]);
        fn((lane >> 2) + 8, n + 1, acc[t].c[3]);
    }
}

}  // namespace ls
