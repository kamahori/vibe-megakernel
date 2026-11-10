// Gemma3-4B W8A16 decode step as one persistent kernel (148 CTAs x 512 threads, software grid barrier).
// Phases per layer: QKV (FFMA, cta_tile16 split-K) | attention (SIMT, 8 CTAs) | o-proj (mma.sync cta_tile16)
//   | gate+up+gelu (FFMA cta_tile16, paired rows) | down (mma.sync cta_tile16).  Then LM head (bf16 warp-per-row) + argmax.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "timeline_probe.cuh"

typedef __nv_bfloat16 bf16;
constexpr int H = 2560, QD = 2048, KD = 1024, I = 10240, V = 262144, L = 34, D = 256, NTH = 512;
constexpr int NWC = 16;  // consumer warps of a producer/consumer ring (arrivals per empty mbarrier)
#ifndef CTX
#define CTX 128
#endif
#ifndef WIN
#define WIN 1024
#endif
constexpr int NPOS = CTX + 1;
constexpr int MAXJ = (NPOS + 15) / 16;
// split-KV attention: 8 q heads x SKS contiguous position splits (CTA b: head b / SKS, split b % SKS)
constexpr int SKS = 4, SKC = CTX / SKS, SKJ = (SKC + 15) / 16;
// visible = [0, CTX]: splits 0..SKS-2 hold SKC cached positions, the last holds SKC cached + the new position (j = SKC)
static_assert(CTX % SKS == 0 && WIN >= CTX + 1, "split-KV layout assumes the full context is visible");
#ifndef GU_D
#define GU_D 4
#endif
constexpr int TL_EVENTS = L * 5 * 2 + 8;

enum { P_TOKEN, P_EMBED, P_LN1, P_LN2, P_LN3, P_LN4, P_QN, P_KN, P_FNORM, P_WQ, P_WK, P_WV, P_WO, P_WG, P_WU, P_WD,
       P_SQ, P_SK, P_SV, P_SO, P_SG, P_SU, P_SD, P_KC, P_VC, P_ROPE, P_QKV, P_ATTN, P_OBUF, P_ACT, P_MLP, P_LOGITS,
       P_NEXT, P_KW, P_VW, P_BCNT, P_BGEN, P_TICKET, P_PV, P_PI, P_DPART, P_TL, P_APART, P_COUNT };
__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldcg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float i8m(uint32_t x, int i) { return __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7650 + i)) - 8388736.f; }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&r); }
__device__ __forceinline__ void i8x4_bf16(uint32_t w, uint32_t& a, uint32_t& b) {
  uint32_t x = w ^ 0x80808080u; a = pack_bf16(i8m(x, 0), i8m(x, 1)); b = pack_bf16(i8m(x, 2), i8m(x, 3)); }
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }

// ===================== shared producer/consumer ring helper (phase-independent) =====================
// The block has PC_NTB = 544 threads: warps 0..15 (threads 0..511, PC_NC) are the compute/consumer warps of every phase,
// warp 16 (PC_NCW) is the ring producer. Phases without a ring run on warps 0..15 only (warp 16 skips them), so every
// intra-phase barrier is the 512-thread named barrier PC_BAR_CONS (pc_csync); only gbar uses bar 0 (all 544 threads).
// A ring is S slots of `stage` bytes at shared address `data`, then 2S mbarriers at `mb`: full[s] (count 1, tx bytes),
// empty[s] (count PC_NCW: one arrive per consumer warp). Chunk g (counted from 0 within the phase) lives in slot g % S,
// use g / S: consumers wait full parity (g / S) & 1, the producer waits empty parity (g / S - 1) & 1 before refilling.
// Per phase: producer lane 0 pc_ring_init -> pc_ring_start (producer arrives, consumers sync, PC_BAR_INIT over 544)
// -> stream -> producer pc_ring_drain_inval (waits every empty release, invalidates the 2S mbarriers).
constexpr int PC_NCW = 16, PC_NC = PC_NCW * 32, PC_NTB = PC_NC + 32;
constexpr int PC_BAR_CONS = 1, PC_BAR_INIT = 2;
// launched block size: PC_NC (512) while no phase uses a producer warp, PC_NTB (544) otherwise
#ifndef MK_NTB
#define MK_NTB PC_NC
#endif
__device__ __forceinline__ void pc_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void pc_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void pc_csync() { pc_bar_sync(PC_BAR_CONS, PC_NC); }
__device__ __forceinline__ void pc_mb_init(uint32_t a, unsigned n) { asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void pc_mb_inval(uint32_t a) { asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void pc_mb_expect_tx(uint32_t a, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void pc_mb_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void pc_mb_wait(uint32_t a, uint32_t parity) {
  asm volatile("{ .reg .pred p; PCW%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1; @!p bra PCW%=; }" :: "r"(a), "r"(parity) : "memory"); }
// global -> shared bulk copy completing on mbarrier mb, L2 evict_first (streamed weights)
__device__ __forceinline__ void pc_bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t mb) {
  asm volatile("{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
               "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], pol; }"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory"); }
template <int S>
struct PcRing {
  uint32_t data, mb, stage;
  __device__ __forceinline__ uint32_t full(unsigned g) const { return mb + 8u * (g % S); }
  __device__ __forceinline__ uint32_t empty(unsigned g) const { return mb + 8u * (S + g % S); }
  __device__ __forceinline__ uint32_t slot(unsigned g) const { return data + stage * (g % S); }
  // producer lane 0, phase start
  __device__ __forceinline__ void init() const {
#pragma unroll
    for (int s = 0; s < S; ++s) { pc_mb_init(mb + 8u * s, 1); pc_mb_init(mb + 8u * (S + s), PC_NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n\tfence.proxy.async.shared::cta;" ::: "memory");
  }
  // init handshake: producer warp arrives (does not wait), consumer warps wait (all PC_NTB threads counted)
  __device__ __forceinline__ void start_producer() const { pc_bar_arrive(PC_BAR_INIT, PC_NTB); }
  __device__ __forceinline__ void start_consumer() const { pc_bar_sync(PC_BAR_INIT, PC_NTB); }
  // producer: wait until slot of chunk g is free, then arm its full barrier (lane 0) with `bytes`
  __device__ __forceinline__ void acquire(unsigned g, unsigned bytes) const {
    if (g >= (unsigned)S) pc_mb_wait(empty(g), ((g / S) - 1u) & 1u);
    if ((threadIdx.x & 31) == 0) pc_mb_expect_tx(full(g), bytes);
    __syncwarp();
  }
  __device__ __forceinline__ void wait_full(unsigned g) const { pc_mb_wait(full(g), (g / S) & 1u); }
  // consumer warp: release chunk g (after the warp's last read of the slot)
  __device__ __forceinline__ void release(unsigned g) const { __syncwarp(); if ((threadIdx.x & 31) == 0) pc_mb_arrive(empty(g)); }
  // producer warp, phase end: wait for the release of the last min(total, S) chunks, then invalidate the barriers
  __device__ __forceinline__ void drain_inval(unsigned total) const {
    for (unsigned g = total > (unsigned)S ? total - S : 0u; g < total; ++g) pc_mb_wait(empty(g), (g / S) & 1u);
    __syncwarp();
    if ((threadIdx.x & 31) < 2 * S) pc_mb_inval(mb + 8u * (threadIdx.x & 31));
  }
};
// =================== end shared producer/consumer ring helper ===================
// ==== shared producer/consumer ring helper (phase-independent; one copy for all phases) ====
// Layout at mb (8-byte aligned smem): full[0..S) (count 1, producer arrive.expect_tx + bulk complete_tx),
// empty[S..2S) (count NWC: one arrive per consumer warp). Chunk g (counted from 0 at phase start, same sequence in
// producer and consumers) uses slot g % S, round g / S: full parity (g/S)&1, empty parity (g/S-1)&1.
// Phase start: one thread calls QkRing<S>::init before its own global loads (the init's release fence waits for the
// thread's outstanding memory ops), then a block barrier precedes any use. With a dedicated producer warp (17-warp
// block) that barrier is qk_handoff_arrive (producer) / qk_handoff_wait (consumers), id 2, 544 threads.
// Phase end: after every consumer's last qk_cwait and a block barrier, one thread per mbarrier calls QkRing<S>::inval
// (the smem is reused by other phases after the grid barrier).
template <int S> struct QkRing {
  static __device__ __forceinline__ uint32_t full(uint32_t mb, unsigned g) { return mb + 8u * (g % S); }
  static __device__ __forceinline__ uint32_t empty(uint32_t mb, unsigned g) { return mb + 8u * (S + g % S); }
  static __device__ __forceinline__ void init(uint32_t mb) {  // one thread
#pragma unroll
    for (int s = 0; s < S; ++s) {
      asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb + 8 * s) : "memory");
      asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mb + 8 * (S + s)), "r"(NWC) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n\tfence.proxy.async.shared::cta;" ::: "memory");
  }
  static __device__ __forceinline__ void inval(uint32_t mb, int i) {  // i in [0, 2S), one thread each
    if (i < 2 * S) asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(mb + 8 * i) : "memory");
  }
};
__device__ __forceinline__ void qk_wait(uint32_t a, uint32_t ph) {
  asm volatile("{ .reg .pred p; PCW: mbarrier.try_wait.parity.shared.b64 p, [%0], %1; @!p bra PCW; }" :: "r"(a), "r"(ph) : "memory"); }
__device__ __forceinline__ void qk_handoff_arrive() { asm volatile("bar.arrive 2, 544;" ::: "memory"); }
__device__ __forceinline__ void qk_handoff_wait() { asm volatile("bar.sync 2, 544;" ::: "memory"); }
// producer side, chunk g: wait until the slot is free (skip on the first round), then arm the full barrier
template <int S> __device__ __forceinline__ void qk_pacquire(uint32_t mb, unsigned g, uint32_t bytes) {
  if (g >= (unsigned)S) qk_wait(QkRing<S>::empty(mb, g), ((g / S) - 1u) & 1u);
  if ((threadIdx.x & 31) == 0) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(QkRing<S>::full(mb, g)), "r"(bytes) : "memory");
  __syncwarp();
}
// one bulk global->smem copy (16-B aligned, bytes % 16 == 0), completes on mbarrier bar, L2 evict_first
__device__ __forceinline__ void qk_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
               "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], pol; }"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
template <int S> __device__ __forceinline__ void qk_cwait(uint32_t mb, unsigned g) { qk_wait(QkRing<S>::full(mb, g), (g / S) & 1u); }
template <int S> __device__ __forceinline__ void qk_crelease(uint32_t mb, unsigned g) {  // whole consumer warp
  __syncwarp();
  if ((threadIdx.x & 31) == 0) asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(QkRing<S>::empty(mb, g)) : "memory");
}
// ==== end shared ring helper ====

// Grid barrier: arrival counter + generation; last arriver resets the counter and publishes generation+1.
__device__ __forceinline__ void gbar(unsigned* cnt, unsigned* gen, unsigned& lg) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    unsigned old;
    asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
    if (old == gridDim.x - 1) {
      asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" :: "l"(cnt) : "memory");
      asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(gen), "r"(lg + 1) : "memory");
    } else {
      unsigned v;
      do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(gen) : "memory"); } while (v == lg);
    }
    lg += 1;
  }
  __syncthreads();
}

__device__ __forceinline__ float block_sum(float v, float* rr) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (lane == 0) rr[warp] = v;
  pc_csync();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) t += rr[i];
  pc_csync();
  return t;
}
// gemma rms over H=2560 elements held 5 per thread (i = tid + 512 j); result rounded to bf16 values.
__device__ __forceinline__ void rms_regs(float (&v)[5], const bf16* w, float* rr) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 5; ++j) s += v[j] * v[j];
  s = block_sum(s, rr);
  const float r = 1.0f / sqrtf(s / (float)H + 1e-6f);
#pragma unroll
  for (int j = 0; j < 5; ++j) { float n = v[j] * r; v[j] = rb(n * (1.0f + bf2f(__ldg(w + threadIdx.x + NTH * j)))); }
}

// rms_regs with the norm weights already in registers (same arithmetic).
__device__ __forceinline__ void rms_pre(float (&v)[5], const float (&w)[5], float* rr) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 5; ++j) s += v[j] * v[j];
  s = block_sum(s, rr);
  const float r = 1.0f / sqrtf(s / (float)H + 1e-6f);
#pragma unroll
  for (int j = 0; j < 5; ++j) { float n = v[j] * r; v[j] = rb(n * (1.0f + w[j])); }
}

// 16-row-tile GEMV, K split over the CTA's 16 warps (lane (g,t) owns rows g and g+8, 16 k per 64-k step).
// NT tiles share the pass (gate+up). Result: fin[n*16+r] = unscaled sum, valid after return.
template <int NT, int K, bool MMA, int U>
__device__ __forceinline__ void gemv16(const int8_t* const (&wrow)[NT], const bf16* xs, float* red, float* fin) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  constexpr int NS = K / 64;
  const int8_t* p0[NT];
  const int8_t* p1[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) { p0[n] = wrow[n] + (size_t)g * K + t * 16; p1[n] = p0[n] + (size_t)8 * K; }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
  for (int st0 = warp; st0 < NS; st0 += 16 * U) {
    uint4 a[U][NT], b[U][NT];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int st = st0 + 16 * u;
      if (st < NS) {
#pragma unroll
        for (int n = 0; n < NT; ++n) { a[u][n] = ldg_nc(p0[n] + st * 64); b[u][n] = ldg_nc(p1[n] + st * 64); }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int st = st0 + 16 * u;
      if (st < NS) {
        if constexpr (MMA) {
          uint32_t bx[8];
          if (g == 0) {
            const uint4* xp = reinterpret_cast<const uint4*>(xs + st * 64 + t * 16);
            uint4 x0 = xp[0], x1 = xp[1];
            bx[0] = x0.x; bx[1] = x0.y; bx[2] = x0.z; bx[3] = x0.w; bx[4] = x1.x; bx[5] = x1.y; bx[6] = x1.z; bx[7] = x1.w;
          } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) bx[i] = 0u;
          }
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            uint32_t pw[8], qw[8];
            const uint32_t* pp = reinterpret_cast<const uint32_t*>(&a[u][n]);
            const uint32_t* qq = reinterpret_cast<const uint32_t*>(&b[u][n]);
#pragma unroll
            for (int i = 0; i < 4; ++i) { i8x4_bf16(pp[i], pw[2 * i], pw[2 * i + 1]); i8x4_bf16(qq[i], qw[2 * i], qw[2 * i + 1]); }
#pragma unroll
            for (int j = 0; j < 4; ++j) mma16816(acc[n], pw[2 * j], qw[2 * j], pw[2 * j + 1], qw[2 * j + 1], bx[2 * j], bx[2 * j + 1]);
          }
        } else {
          const uint4* xp = reinterpret_cast<const uint4*>(xs + st * 64 + t * 16);
          const uint4 xa = xp[0], xb = xp[1];
          const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            const uint32_t* pr = reinterpret_cast<const uint32_t*>(&a[u][n]);
            const uint32_t* qr = reinterpret_cast<const uint32_t*>(&b[u][n]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t aa = pr[i] ^ 0x80808080u, bb = qr[i] ^ 0x80808080u;
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const uint32_t xv = xw[(4 * i + e) >> 1];
                const float xf = (e & 1) ? __uint_as_float(xv & 0xffff0000u) : __uint_as_float(xv << 16);
                acc[n][0] = fmaf(i8m(aa, e), xf, acc[n][0]);
                acc[n][2] = fmaf(i8m(bb, e), xf, acc[n][2]);
              }
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    if constexpr (!MMA) {
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) { acc[n][0] += __shfl_xor_sync(0xffffffffu, acc[n][0], o); acc[n][2] += __shfl_xor_sync(0xffffffffu, acc[n][2], o); }
    }
    if (t == 0) { red[(warp * NT + n) * 16 + g] = acc[n][0]; red[(warp * NT + n) * 16 + g + 8] = acc[n][2]; }
  }
  pc_csync();
  if (tid < 16 * NT) {
    const int n = tid >> 4, r = tid & 15;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) s += red[(w * NT + n) * 16 + r];
    fin[tid] = s;
  }
  pc_csync();
}

__device__ __forceinline__ void st_tag(unsigned long long* p, float v) {
  const unsigned long long x = (1ull << 32) | __float_as_uint(v);
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(x) : "memory");
}
// acc.{x,y} += (bits{w0,w1} - (2^23+128)) * {x0,x1} with packed f32x2 add/fma (sm_100); exact dequant, fp32 accumulate.
__device__ __forceinline__ void gu_fma2(float2& acc, uint32_t w0, uint32_t w1, float x0, float x1) {
  asm("{.reg .b64 a, w, x, nb;\n\t"
      "mov.b64 a, {%0,%1};\n\tmov.b64 w, {%2,%3};\n\tmov.b64 x, {%4,%5};\n\tmov.b64 nb, {%6,%6};\n\t"
      "add.rn.f32x2 w, w, nb;\n\tfma.rn.f32x2 a, w, x, a;\n\tmov.b64 {%0,%1}, a;}"
      : "+f"(acc.x), "+f"(acc.y) : "r"(w0), "r"(w1), "f"(x0), "f"(x1), "f"(-8388736.f));
}
__device__ __forceinline__ uint32_t gu_prmt(uint32_t x, uint32_t k, int e) {
  uint32_t r;
  if (e == 0) asm("prmt.b32 %0, %1, %2, 0x7650;" : "=r"(r) : "r"(x), "r"(k));
  else if (e == 1) asm("prmt.b32 %0, %1, %2, 0x7651;" : "=r"(r) : "r"(x), "r"(k));
  else if (e == 2) asm("prmt.b32 %0, %1, %2, 0x7652;" : "=r"(r) : "r"(x), "r"(k));
  else asm("prmt.b32 %0, %1, %2, 0x7653;" : "=r"(r) : "r"(x), "r"(k));
  return r;
}

// E1 smem ring (live only inside the gateup phase): GU_S x GU_STAGE bytes at GU_RING_OFF, then GU_S mbarriers.
#ifndef GU_S
#define GU_S 2
#endif
// E3 consumer math: acc.{x,y} += (bits{w0,w1} - (2^23+128)) * {x0,x1}, packed f32x2 (exact dequant, fp32 accumulate).
__device__ __forceinline__ void dn_fma2(float2& acc, uint32_t w0, uint32_t w1, float x0, float x1) {
  asm("{.reg .b64 a, w, x, nb;\n\t"
      "mov.b64 a, {%0,%1};\n\tmov.b64 w, {%2,%3};\n\tmov.b64 x, {%4,%5};\n\tmov.b64 nb, {%6,%6};\n\t"
      "add.rn.f32x2 w, w, nb;\n\tfma.rn.f32x2 a, w, x, a;\n\tmov.b64 {%0,%1}, a;}"
      : "+f"(acc.x), "+f"(acc.y) : "r"(w0), "r"(w1), "f"(x0), "f"(x1), "f"(-8388736.f));
}
// byte e of w (already ^0x80) as the float bits of 2^23 + byte
__device__ __forceinline__ uint32_t dn_prmt(uint32_t w, int e) { return __byte_perm(w, 0x4B000000u, 0x7650 + e); }
// E3 down ring (live only inside the down phase; union with the E1 ring): DN_S x DN_STAGE bytes at DN_RING_OFF, then 2 DN_S mbarriers.
constexpr int DN_S = 4, DN_SK = 2560, DN_CH = I / DN_SK, DN_RS = DN_SK + 16, DN_STAGE = 16 * DN_RS, DN_UNITS = (H / 16) * DN_CH;
#ifndef DN_PRE
#define DN_PRE 1
#endif
constexpr int DN_RING_OFF = 30720, DN_RING_END = DN_RING_OFF + DN_S * DN_STAGE + 2 * DN_S * 8;
constexpr int GU_RS = H + 32, GU_STAGE = 16 * GU_RS, GU_RING_OFF = 30720, GU_RING_END = GU_RING_OFF + GU_S * GU_STAGE + GU_S * 8;
__device__ __forceinline__ void gu_wait(uint32_t a, uint32_t ph) {
  asm volatile("{ .reg .pred p; GUW: mbarrier.try_wait.parity.shared.b64 p, [%0], %1; @!p bra GUW; }" :: "r"(a), "r"(ph) : "memory"); }

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) { __half2 r = __floats2half2_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&r); }
// 4 int8 -> 2 f16x2 exactly: (0x6400 | (b ^ 0x80)) = 1024 + 128 + b, minus 1152.
__device__ __forceinline__ void i8x4_h2(uint32_t w, uint32_t& a, uint32_t& b) {
  const uint32_t x = w ^ 0x80808080u;
  uint32_t p, q;
  asm("prmt.b32 %0, %1, 0x64646464, 0x4140;" : "=r"(p) : "r"(x));
  asm("prmt.b32 %0, %1, 0x64646464, 0x4342;" : "=r"(q) : "r"(x));
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(a) : "r"(p), "r"(0x64806480u));
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(b) : "r"(q), "r"(0x64806480u));
}
__device__ __forceinline__ void mma16816h(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }
// QKV ring (live only inside the qkv phase): QK_S stages of 16 rows x QK_RS bytes at QK_RING_OFF, then 2*QK_S mbarriers.
#ifndef QK_S
#define QK_S 2
#endif
constexpr int QK_RS = H + 16, QK_STAGE = 16 * QK_RS, QK_RING_OFF = 30720, QK_RING_END = QK_RING_OFF + QK_S * QK_STAGE + 2 * QK_S * 8;

// ===================== producer/consumer ring helper (phase-independent) =====================
// Per slot one full mbarrier (count 1 + tx bytes) at mb + 8*slot and one empty mbarrier (count = consumer warps) at
// mb + 8*(S + slot). A phase inits them at its start (op_init, one warp; make the init visible to the consumers with a
// CTA/named barrier before their first op_full_wait) and invalidates them at its end (op_inval, after the last wait).
// The chunk counter c runs from 0 within a phase in producer and consumers alike: slot c % S, use c / S; full parity
// (c / S) & 1, empty parity (c / S - 1) & 1 (no empty wait for the first S chunks).
constexpr int OP_RING_OFF = 30720;  // ring region shared (as a union) by the phases; below it: xs, red, fin, rr, ...
__device__ __forceinline__ void op_mbwait(uint32_t a, uint32_t ph) {
  asm volatile("{ .reg .pred p; PCW: mbarrier.try_wait.parity.shared.b64 p, [%0], %1; @!p bra PCW; }" :: "r"(a), "r"(ph) : "memory"); }
template <int S>
__device__ __forceinline__ void op_init(uint32_t mb, int ecnt, int lane) {  // one full warp
  if (lane < S) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb + 8 * lane) : "memory");
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mb + 8 * (S + lane)), "r"(ecnt) : "memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n\tfence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
}
template <int S>
__device__ __forceinline__ void op_inval(uint32_t mb, int tid) {
  if (tid < 2 * S) asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(mb + 8 * tid) : "memory"); }
template <int S>
__device__ __forceinline__ void op_full_wait(uint32_t mb, unsigned c) { op_mbwait(mb + 8 * (c % S), (c / S) & 1u); }
template <int S>
__device__ __forceinline__ void op_empty_wait(uint32_t mb, unsigned c) { if (c >= (unsigned)S) op_mbwait(mb + 8 * (S + c % S), (c / S - 1u) & 1u); }
template <int S>  // producer lane 0: arm slot of chunk c for `bytes` of transactions
__device__ __forceinline__ void op_expect(uint32_t mb, unsigned c, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(mb + 8 * (c % S)), "r"(bytes) : "memory"); }
template <int S>  // one bulk copy into the slot of chunk c (n bytes, 16-aligned), completing on its full mbarrier; L2 evict_first
__device__ __forceinline__ void op_bulk(uint32_t mb, unsigned c, uint32_t dst, const void* src, uint32_t n) {
  asm volatile("{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
               "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], pol; }"
               :: "r"(dst), "l"(src), "r"(n), "r"(mb + 8 * (c % S)) : "memory"); }
template <int S>  // consumer warp: release the slot of chunk c (one arrive per warp)
__device__ __forceinline__ void op_release(uint32_t mb, unsigned c, int lane) {
  __syncwarp();
  if (lane == 0) asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(mb + 8 * (S + c % S)) : "memory"); }
__device__ __forceinline__ void op_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void op_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
// =================== end producer/consumer ring helper ===================

// E6 o-proj ring: 2 stages of whole rows (int8, row stride QD), at OP_RING_OFF.
#ifndef OP_PW
#define OP_PW 0   // producer warp of o-proj: 0 = warp 0 issues all stages then consumes; 16 = dedicated warp (544 threads)
#endif
// stage 0 = rows [0,16) (32 KB, contiguous, row stride QD), stage 1 = rows [16,R) (<= 8 rows); both issued at phase start.
constexpr int OP_S = 2, OP_RMAX = 24, OP_ST1 = 16 * QD, OP_RINGB = OP_RMAX * QD;
constexpr int OP_RING_END = OP_RING_OFF + OP_RINGB + 2 * OP_S * 8;
constexpr int NTHB = OP_PW >= 16 ? NTH + 32 : NTH;  // launch block size
// dot of 16 int8 weights (one uint4) with 16 bf16 x values at xp (two uint4), fp32 FMA in k order.
__device__ __forceinline__ float op_dot16(uint4 w, const bf16* xp, float acc) {
  const uint4 xa = reinterpret_cast<const uint4*>(xp)[0], xb = reinterpret_cast<const uint4*>(xp)[1];
  const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  const uint32_t wr[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t a = wr[i] ^ 0x80808080u;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t xv = xw[(4 * i + e) >> 1];
      const float xf = (e & 1) ? __uint_as_float(xv & 0xffff0000u) : __uint_as_float(xv << 16);
      acc = fmaf(i8m(a, e), xf, acc);
    }
  }
  return acc;
}
struct Ptrs { const void* a[P_COUNT]; float emb_scale; int tl_words; };

__global__ void __launch_bounds__(MK_NTB, 1) mk_kernel(Ptrs P) {
  extern __shared__ __align__(16) uint8_t smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);                       // 10240 bf16 (GEMV x operand)
  float* red = reinterpret_cast<float*>(smem + 20480);            // 16 warps x 2 tiles x 16
  float* fin = red + 512;                                         // 32
  float* rr = fin + 32;                                           // 16
  float* qv = rr + 16;                                            // 256
  float* kv = qv + 256;                                           // 256
  float* nrm = kv + 256;                                          // 512
  float* sc = nrm + 512;                                          // 160
  float* pr = sc + 160;                                           // 160
  float* wbv = pr + 160;                                          // 16
  int* wbi = reinterpret_cast<int*>(wbv + 16);                    // 16
  float* pvs = reinterpret_cast<float*>(wbi + 16);                // 160
  int* pis = reinterpret_cast<int*>(pvs + 160);                   // 160
  int* flag = pis + 160;

  MB_TIMELINE_INIT((uint64_t*)P.a[P_TL], (uint64_t)P.tl_words, 0, 1, threadIdx.x == 0);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t* token = (const int64_t*)P.a[P_TOKEN];
  const bf16* embed = (const bf16*)P.a[P_EMBED];
  const bf16 *ln1 = (const bf16*)P.a[P_LN1], *ln2 = (const bf16*)P.a[P_LN2], *ln3 = (const bf16*)P.a[P_LN3], *ln4 = (const bf16*)P.a[P_LN4];
  const bf16 *qn = (const bf16*)P.a[P_QN], *kn = (const bf16*)P.a[P_KN], *fnorm = (const bf16*)P.a[P_FNORM];
  const int8_t *wq = (const int8_t*)P.a[P_WQ], *wk = (const int8_t*)P.a[P_WK], *wv = (const int8_t*)P.a[P_WV], *wo = (const int8_t*)P.a[P_WO],
              *wg = (const int8_t*)P.a[P_WG], *wu = (const int8_t*)P.a[P_WU], *wd = (const int8_t*)P.a[P_WD];
  const float *sq = (const float*)P.a[P_SQ], *sk = (const float*)P.a[P_SK], *sv = (const float*)P.a[P_SV], *so = (const float*)P.a[P_SO],
              *sg = (const float*)P.a[P_SG], *su = (const float*)P.a[P_SU], *sd = (const float*)P.a[P_SD];
  const bf16 *kc = (const bf16*)P.a[P_KC], *vc = (const bf16*)P.a[P_VC], *ropecs = (const bf16*)P.a[P_ROPE];
  bf16 *qkv = (bf16*)P.a[P_QKV], *attn = (bf16*)P.a[P_ATTN], *obuf = (bf16*)P.a[P_OBUF], *act = (bf16*)P.a[P_ACT], *mlp = (bf16*)P.a[P_MLP];
  float* logits = (float*)P.a[P_LOGITS];
  int64_t* next = (int64_t*)P.a[P_NEXT];
  bf16 *kw = (bf16*)P.a[P_KW], *vw = (bf16*)P.a[P_VW];
  unsigned *bcnt = (unsigned*)P.a[P_BCNT], *bgen = (unsigned*)P.a[P_BGEN], *ticket = (unsigned*)P.a[P_TICKET];
  float* part_v = (float*)P.a[P_PV];
  int* part_i = (int*)P.a[P_PI];
  unsigned long long* dpart = (unsigned long long*)P.a[P_DPART];

  unsigned lg = 0;
  // attn->oproj done counter: monotonic across launches (never reset). Its epoch base is read here, before the first grid
  // barrier; no attention CTA adds to it before that barrier, and every add of the previous launch completed at the kernel
  // boundary. Layer l's handoff completes when the counter reaches base + 32 (l + 1) (wrap-safe signed compare).
  unsigned* acnt = reinterpret_cast<unsigned*>((float*)P.a[P_APART] + 8 * SKS * D + 64);  // own 128-B line
  unsigned abase = 0;
  if (tid == 0) {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(lg) : "l"(bgen) : "memory");
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(abase) : "l"(acnt) : "memory");
  }

  float xr[5];
  const int64_t tok = token[0];
  for (int layer = 0; layer < L; ++layer) {
    // ---------------- QKV ----------------
    MB_TIMELINE_BEGIN(0);
    {
      // QKV (E5+E4) on the bulk ring. Row-balanced: CTA b owns rows [b*4096/G, (b+1)*4096/G) of the stacked [q; k; v]
      // weights (27-28 rows at 148 CTAs), as chunks of <= 16 rows; chunk c goes to ring slot c % QK_S (one 2560-B bulk copy
      // per row, all on the slot's full mbarrier). The first QK_S chunks are issued at phase start (below); a CTA has
      // <= 2 chunks for G >= 128, so no slot is ever waited for. Later chunks (G < 128 only) are issued by warp 0 after
      // the block barrier that ends the reads of their slot.
      // All 16 warps run the ln4-residual + ln1 prologue (mlp and both norm weight vectors loaded together; parent
      // arithmetic and reduction order), then consume chunk by chunk: FFMA, warp w owns k [160w, 160w+160),
      // lane = (row, half), fixed-order reduce over the 16 warps.
      const uint32_t qk_ring = (uint32_t)__cvta_generic_to_shared(smem + QK_RING_OFF), qk_mb = qk_ring + QK_S * QK_STAGE;
      unsigned qk_b, qk_g;
      asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(qk_b)); asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(qk_g));
      constexpr unsigned QK_ROWS = QD + 2 * KD;
      const int qk_r0 = (int)((qk_b * QK_ROWS) / qk_g), qk_nr = (int)(((qk_b + 1) * QK_ROWS) / qk_g) - qk_r0;
      const unsigned qk_n = (unsigned)(qk_nr + 15) >> 4;
      auto qk_row = [&](int r, const float*& s) -> const int8_t* {  // stacked row r -> weight row, scale
        if (r < QD) { s = sq + (size_t)layer * QD + r; return wq + ((size_t)layer * QD + r) * H; }
        if (r < QD + KD) { s = sk + (size_t)layer * KD + r - QD; return wk + ((size_t)layer * KD + r - QD) * H; }
        s = sv + (size_t)layer * KD + r - QD - KD; return wv + ((size_t)layer * KD + r - QD - KD) * H;
      };
      auto qk_issue = [&](unsigned c) {  // warp 0 only
        if (c < qk_n) {
          const int nrc = min(16, qk_nr - 16 * (int)c);
          if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(QkRing<QK_S>::full(qk_mb, c)), "r"(nrc * H) : "memory");
          __syncwarp();
          if (lane < nrc) {
            const float* s;
            const int8_t* w = qk_row(qk_r0 + 16 * (int)c + lane, s);
            qk_bulk(qk_ring + (c % QK_S) * QK_STAGE + lane * QK_RS, w, H, QkRing<QK_S>::full(qk_mb, c));
          }
        }
      };
      // ring start, before any load of this thread (the init's release fence would otherwise wait for them): thread 0
      // inits the barriers and arms the first QK_S chunks; after one block barrier (behind the prologue loads), lane 0
      // of warp w issues row w of each chunk (2 bulk copies per warp, not a serial per-row loop in one warp)
      if (tid == 0) {
        QkRing<QK_S>::init(qk_mb);
#pragma unroll
        for (unsigned c = 0; c < (unsigned)QK_S; ++c)
          if (c < qk_n) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(QkRing<QK_S>::full(qk_mb, c)), "r"(min(16, qk_nr - 16 * (int)c) * H) : "memory");
      }
      // prologue loads, issued together
      float m[5], w4[5], w1[5], qsc[2] = {0.f, 0.f};
      if (layer == 0) {
#pragma unroll
        for (int j = 0; j < 5; ++j) m[j] = bf2f(embed[(size_t)tok * H + tid + NTH * j]);
      } else {
#pragma unroll
        for (int j = 0; j < 5; ++j) { m[j] = bf2f(__ldcg(mlp + tid + NTH * j)); w4[j] = bf2f(__ldg(ln4 + (size_t)(layer - 1) * H + tid + NTH * j)); }
      }
#pragma unroll
      for (int j = 0; j < 5; ++j) w1[j] = bf2f(__ldg(ln1 + (size_t)layer * H + tid + NTH * j));
      if (tid < 16) {
#pragma unroll
        for (int c = 0; c < 2; ++c) if (16 * c + tid < qk_nr) { const float* s; qk_row(qk_r0 + 16 * c + tid, s); qsc[c] = __ldg(s); }
      }
      __syncthreads();  // ring barriers initialised and armed
      if (lane == 0) {
#pragma unroll
        for (unsigned c = 0; c < (unsigned)QK_S; ++c)
          if (c < qk_n && warp < qk_nr - 16 * (int)c) {
            const float* s;
            const int8_t* w = qk_row(qk_r0 + 16 * (int)c + warp, s);
            qk_bulk(qk_ring + (c % QK_S) * QK_STAGE + warp * QK_RS, w, H, QkRing<QK_S>::full(qk_mb, c));
          }
      }
      // gemma rms (rms_pre arithmetic; block sum = warp xor tree, then warps 0..15 in order); the two sums use
      // separate scratch (rr, rr + 16), so no barrier is needed after each read
      auto qk_rms = [&](float (&v)[5], const float (&w)[5], float* scr) {
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 5; ++j) ss += v[j] * v[j];
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) scr[warp] = ss;
        __syncthreads();
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) t += scr[i];
        const float r = 1.0f / sqrtf(t / (float)H + 1e-6f);
#pragma unroll
        for (int j = 0; j < 5; ++j) { float n = v[j] * r; v[j] = rb(n * (1.0f + w[j])); }
      };
      if (layer == 0) {
#pragma unroll
        for (int j = 0; j < 5; ++j) xr[j] = rb(m[j] * P.emb_scale);
      } else {
        qk_rms(m, w4, rr);
#pragma unroll
        for (int j = 0; j < 5; ++j) xr[j] = rb(xr[j] + m[j]);
      }
#pragma unroll
      for (int j = 0; j < 5; ++j) m[j] = xr[j];
      qk_rms(m, w1, rr + 16);
#pragma unroll
      for (int j = 0; j < 5; ++j) xs[tid + NTH * j] = __float2bfloat16_rn(m[j]);
      __syncthreads();  // xs complete
      const int row = lane >> 1, half = lane & 1;
      constexpr int LB = H / 32;  // 80 k per thread per chunk
      const bf16* xb = xs + warp * (H / 16) + half * LB;
#pragma unroll 1
      for (unsigned c = 0; c < qk_n; ++c) {
        qk_cwait<QK_S>(qk_mb, c);
        const uint8_t* wb = smem + QK_RING_OFF + (c % QK_S) * QK_STAGE + row * QK_RS + warp * (H / 16) + half * LB;
        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for (int v = 0; v < LB / 16; ++v) {
          const uint4 wv4 = *reinterpret_cast<const uint4*>(wb + v * 16);
          const uint4* xp = reinterpret_cast<const uint4*>(xb + v * 16);
          const uint4 xa = xp[0], xc = xp[1];
          const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
          const uint32_t pr[4] = {wv4.x, wv4.y, wv4.z, wv4.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t a = pr[i] ^ 0x80808080u;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint32_t xv = xw[(4 * i + e) >> 1];
              const float xf = (e & 1) ? __uint_as_float(xv & 0xffff0000u) : __uint_as_float(xv << 16);
              if (e & 1) acc1 = fmaf(i8m(a, e), xf, acc1); else acc0 = fmaf(i8m(a, e), xf, acc0);
            }
          }
        }
        float acc = acc0 + acc1;
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        float* rd = red + (c & 1u) * 256;  // double-buffered: one block barrier per chunk
        if (half == 0) rd[warp * 16 + row] = acc;
        __syncthreads();
        if (tid < min(16, qk_nr - 16 * (int)c)) {
          float sc = c == 0 ? qsc[0] : qsc[1];
          if (c >= 2) { const float* s; qk_row(qk_r0 + 16 * (int)c + tid, s); sc = __ldg(s); }
          float sum = 0.f;
#pragma unroll
          for (int w2 = 0; w2 < 16; ++w2) sum += rd[w2 * 16 + tid];
          qkv[qk_r0 + 16 * (int)c + tid] = __float2bfloat16_rn(sum * sc);
        }
        if (warp == 0) qk_issue(c + QK_S);  // slot c % QK_S is free: every warp passed the barrier after reading it
      }
      QkRing<QK_S>::inval(qk_mb, tid);  // the last chunk's barrier put every warp past its last ring wait
    }
    gbar(bcnt, bgen, lg);
    MB_TIMELINE_END(0);
    // ---------------- attention ----------------
    MB_TIMELINE_BEGIN(1);
    // Split-KV: CTA b < 8*SKS takes q head h = b / SKS and visible positions [q0, q1) (split sp = b % SKS, contiguous).
    // Each split CTA redoes the q/k norm + rope of its head (cheap), computes bf16 scores, a local max m and
    // unnormalised o = sum_j exp(s_j - m) v_j and z = sum_j exp(s_j - m), published as fp32 partials.
    // After the phase barrier every o-proj CTA merges the SKS partials of each head in fixed split order (no atomics)
    // into its xs operand (bf16), so the combine is in the oproj window and attn adds no barrier.
    if (warp < PC_NCW) {  // warp 16 (ring producer) idles in this phase
      // volatile reads keep the layer-invariant split/lane math inside the phase (out of the other phases' live set)
      unsigned bx, tx;
      asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bx));
      asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tx));
      const int tid = (int)tx, lane = tid & 31, warp = tid >> 5;
      float* apo = (float*)P.a[P_APART];  // [8*SKS][256] unnormalised split outputs, then [8*SKS][2] (max, sum of exp)
      float* apmz = apo + 8 * SKS * D;
      const int h = bx / SKS, sp = bx % SKS, kvh = h >> 1;
      if (bx < 8 * SKS) {
        static_assert(L <= 64 && L < 35, "global-layer mask");
        const bool local = ((0x20820820ull >> layer) & 1ull) == 0ull;  // global layers: (layer + 1) % 6 == 0
        const int q0 = sp * SKC;
        const bool hasnew = sp == SKS - 1, newwarp = hasnew && warp == (SKC & 15);  // new position: j = SKC
        const int nl = SKC + (hasnew ? 1 : 0);
        // prefetch this split's cached K/V rows; the new position's V comes from this layer's qkv, its K is computed below
        uint4 kk[SKJ], vv[SKJ], vn = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int jj = 0; jj < SKJ; ++jj) {
          const int j = warp + 16 * jj;
          kk[jj] = vv[jj] = make_uint4(0u, 0u, 0u, 0u);  // defined on every path: nothing carried across layers
          if (j < SKC) {
            const size_t off = (((size_t)layer * CTX + q0 + j) * 4 + kvh) * D + lane * 8;
            kk[jj] = ldg_nc(kc + off); vv[jj] = ldg_nc(vc + off);
          }
        }
        if (newwarp) vn = ldcg16(qkv + QD + KD + kvh * D + lane * 8);
        // q/k norm + rope
        const int d = tid & 255; const bool isk = tid >= 256;
        float val = bf2f(__ldcg(qkv + (isk ? (QD + kvh * D) : (h * D)) + d));
        float s2 = val * val;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
        if (lane == 0) rr[warp] = s2;
        pc_csync();
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) tot += rr[(isk ? 8 : 0) + i];
        const float r = 1.0f / sqrtf(tot / (float)D + 1e-6f);
        const bf16* nw = (isk ? kn : qn) + (size_t)layer * D;
        nrm[tid] = rb((val * r) * (1.0f + bf2f(__ldg(nw + d))));
        pc_csync();
        {
          const bf16* cs = ropecs + (local ? 0 : 2 * D);
          const float cosv = bf2f(cs[d]), sinv = bf2f(cs[D + d]);
          const float partner = nrm[(isk ? 256 : 0) + ((d + 128) & 255)];
          const float rot = d < 128 ? -partner : partner;
          const float res = rb(rb(nrm[tid] * cosv) + rb(rot * sinv));
          const bool wr = (h & 1) == 0 && sp == 0;
          if (isk) { kv[d] = res; if (wr) kw[(size_t)layer * KD + kvh * D + d] = __float2bfloat16_rn(res); }
          else { qv[d] = res; if (wr) vw[(size_t)layer * KD + kvh * D + d] = __ldcg(qkv + QD + KD + kvh * D + d); }
        }
        pc_csync();
        // scores of this split
        float qr[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) qr[e] = qv[lane * 8 + e];
#pragma unroll
        for (int jj = 0; jj < SKJ; ++jj) {
          const int j = warp + 16 * jj;
          if (j < SKC) {
            float dot = 0.f;
            const uint32_t* w4 = reinterpret_cast<const uint32_t*>(&kk[jj]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              dot = fmaf(qr[2 * e], __uint_as_float(w4[e] << 16), dot);
              dot = fmaf(qr[2 * e + 1], __uint_as_float(w4[e] & 0xffff0000u), dot);
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) sc[j] = rb(dot) * 0.0625f;
          }
        }
        if (newwarp) {
          float dot = 0.f;
#pragma unroll
          for (int e = 0; e < 8; ++e) dot = fmaf(qr[e], kv[lane * 8 + e], dot);
#pragma unroll
          for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
          if (lane == 0) sc[SKC] = rb(dot) * 0.0625f;
        }
        pc_csync();
        // local max, computed by every warp (no extra block sync)
        float mx = -INFINITY;
        for (int j = lane; j < nl; j += 32) mx = fmaxf(mx, sc[j]);
#pragma unroll
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (warp == 15) {
          float z = 0.f;
          for (int j = lane; j < nl; j += 32) z += expf(sc[j] - mx);
#pragma unroll
          for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
          if (lane == 0) { __stcg(apmz + bx * 2, mx); __stcg(apmz + bx * 2 + 1, z); }
        }
        float oa[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) oa[e] = 0.f;
#pragma unroll
        for (int jj = 0; jj <= SKJ; ++jj) {
          const int j = jj < SKJ ? warp + 16 * jj : SKC;
          if (jj < SKJ ? j < SKC : newwarp) {
            const uint32_t* w4 = reinterpret_cast<const uint32_t*>(jj < SKJ ? &vv[jj] : &vn);
            const float pj = expf(sc[j] - mx);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              oa[2 * e] = fmaf(pj, __uint_as_float(w4[e] << 16), oa[2 * e]);
              oa[2 * e + 1] = fmaf(pj, __uint_as_float(w4[e] & 0xffff0000u), oa[2 * e + 1]);
            }
          }
        }
        float4* opart = reinterpret_cast<float4*>(smem);  // aliases xs (dead here), [16][256]
        opart[warp * 64 + lane * 2] = make_float4(oa[0], oa[1], oa[2], oa[3]);
        opart[warp * 64 + lane * 2 + 1] = make_float4(oa[4], oa[5], oa[6], oa[7]);
        pc_csync();
        if (tid < 256) {
          const float* op = reinterpret_cast<const float*>(smem);
          float s = 0.f;
#pragma unroll
          for (int w = 0; w < 16; ++w) s += op[w * 256 + tid];
          __stcg(apo + bx * D + tid, s);
        }
      }
    }
    // attn->oproj handoff (no grid barrier): the 8*SKS attention CTAs release-add the done counter after their partials
    // are stored (the block barrier orders every thread's stores before thread 0's cumulative release); every CTA's
    // thread 0 acquire-spins to the layer's target, then a block barrier hands the acquire to the whole CTA.
    __syncthreads();
    if (tid == 0) {
      if (blockIdx.x < 8 * SKS) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(acnt) : "memory");
      const unsigned tgt = abase + (unsigned)(8 * SKS) * (unsigned)(layer + 1);
      unsigned v;
      do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(acnt) : "memory"); } while ((int)(v - tgt) < 0);
    }
    __syncthreads();
    MB_TIMELINE_END(1);
    // ---------------- o-proj ----------------
    MB_TIMELINE_BEGIN(2);
    {
      // Row-balanced split: CTA c owns rows [floor(c*H/grid), floor((c+1)*H/grid)) (16..24 rows) over the full K, no fixup.
      // Producer/consumer ring, 2 stages of whole contiguous rows: stage 0 = rows 0..15 (32 KB, 4 bulk copies of 8 KB),
      // stage 1 = rows 16..R-1 (one bulk copy); the producer warp (OP_PW) inits the mbarriers and issues both right after
      // the barrier, while the 16 consumer warps run the split-KV combine into xs. Compute (FFMA): warp w owns row w of
      // stage 0 over all k (lane l: k = 512 v + 16 l), and half (w & 1) of row 16 + (w >> 1) of stage 1. Per row: k-ordered
      // FMA per lane, fixed shuffle tree, halves added in order (deterministic).
      const int r0 = (int)((blockIdx.x * (unsigned)H) / gridDim.x), r1 = (int)(((blockIdx.x + 1) * (unsigned)H) / gridDim.x);
      const int R = r1 - r0, R2 = R - 16;  // host asserts 107 <= grid <= 160: 16 <= R <= 24
      const uint32_t op_ring = (uint32_t)__cvta_generic_to_shared(smem + OP_RING_OFF), op_mb = op_ring + OP_RINGB;
      if (warp == OP_PW) {
        op_init<OP_S>(op_mb, 16, lane);
        if constexpr (OP_PW >= 16) op_bar_arrive(1, NTHB);  // init visible to the consumers at their first named barrier
        const int8_t* src = wo + ((size_t)layer * H + r0) * QD;
        if (lane == 0) { op_expect<OP_S>(op_mb, 0u, (uint32_t)OP_ST1); op_expect<OP_S>(op_mb, 1u, (uint32_t)(R2 * QD)); }
        __syncwarp();
        if (lane < 4) op_bulk<OP_S>(op_mb, 0u, op_ring + lane * (OP_ST1 / 4), src + lane * (OP_ST1 / 4), OP_ST1 / 4);
        else if (lane == 4 && R2 > 0) op_bulk<OP_S>(op_mb, 1u, op_ring + OP_ST1, src + OP_ST1, (uint32_t)(R2 * QD));
      }
      if (warp < 16) {
        const float osc = tid < R ? __ldg(so + (size_t)layer * H + r0 + tid) : 0.f;
        {  // split-KV combine (fixed split order): thread -> head tid / 64, dims 4 * (tid % 64) .. +3
          static_assert(SKS == 4 && NTH * 4 == QD, "combine mapping");
          unsigned tx;  // volatile read keeps the layer-invariant combine addresses out of the loop-wide live set
          asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tx));
          const float* apo = (const float*)P.a[P_APART];
          const float* apmz = apo + 8 * SKS * D;
          const int ch = tx >> 6, d4 = (tx & 63) * 4;
          const float4 mz0 = __ldcg(reinterpret_cast<const float4*>(apmz + ch * 8));
          const float4 mz1 = __ldcg(reinterpret_cast<const float4*>(apmz + ch * 8 + 4));
          float4 o4[SKS];
#pragma unroll
          for (int s = 0; s < SKS; ++s) o4[s] = __ldcg(reinterpret_cast<const float4*>(apo + (ch * SKS + s) * D + d4));
          const float M = fmaxf(fmaxf(mz0.x, mz0.z), fmaxf(mz1.x, mz1.z));
          const float f[SKS] = {expf(mz0.x - M), expf(mz0.z - M), expf(mz1.x - M), expf(mz1.z - M)};
          const float zz[SKS] = {mz0.y, mz0.w, mz1.y, mz1.w};
          float Z = 0.f; float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (int s = 0; s < SKS; ++s) {
            Z = fmaf(f[s], zz[s], Z);
            A.x = fmaf(f[s], o4[s].x, A.x); A.y = fmaf(f[s], o4[s].y, A.y); A.z = fmaf(f[s], o4[s].z, A.z); A.w = fmaf(f[s], o4[s].w, A.w);
          }
          const float iz = 1.0f / Z;
          reinterpret_cast<uint2*>(xs)[tx] = make_uint2(pack_bf16(A.x * iz, A.y * iz), pack_bf16(A.z * iz, A.w * iz));
        }
        op_bar_sync(1, NTHB);  // xs complete (+ mbarrier init visible when the producer is a separate warp)
        const uint8_t* ring = smem + OP_RING_OFF;
        float a0 = 0.f, a1 = 0.f;
        {
          op_full_wait<OP_S>(op_mb, 0u);
          uint4 w[4];
#pragma unroll
          for (int v = 0; v < 4; ++v) w[v] = *reinterpret_cast<const uint4*>(ring + warp * QD + v * 512 + lane * 16);
#pragma unroll
          for (int v = 0; v < 4; ++v) a0 = op_dot16(w[v], xs + v * 512 + lane * 16, a0);
          op_release<OP_S>(op_mb, 0u, lane);
        }
        const int er = warp >> 1, kh = (warp & 1) * (QD / 2);
        if (er < R2) {
          op_full_wait<OP_S>(op_mb, 1u);
          uint4 w[2];
#pragma unroll
          for (int v = 0; v < 2; ++v) w[v] = *reinterpret_cast<const uint4*>(ring + OP_ST1 + er * QD + kh + v * 512 + lane * 16);
#pragma unroll
          for (int v = 0; v < 2; ++v) a1 = op_dot16(w[v], xs + kh + v * 512 + lane * 16, a1);
        }
        op_release<OP_S>(op_mb, 1u, lane);
#pragma unroll
        for (int o = 16; o; o >>= 1) { a0 += __shfl_xor_sync(0xffffffffu, a0, o); a1 += __shfl_xor_sync(0xffffffffu, a1, o); }
        if (lane == 0) { red[warp] = a0; red[16 + warp] = a1; }
        op_bar_sync(1, NTH);
        if (tid < R) {
          const float sm = tid < 16 ? red[tid] : red[16 + 2 * (tid - 16)] + red[17 + 2 * (tid - 16)];
          obuf[r0 + tid] = __float2bfloat16_rn(sm * osc);
        }
        op_inval<OP_S>(op_mb, tid);  // every wait on these mbarriers is done (all consumers passed the named barrier)
      }
    }
    gbar(bcnt, bgen, lg);
    MB_TIMELINE_END(2);
    // ---------------- gate + up + gelu ----------------
    MB_TIMELINE_BEGIN(3);
    if (warp < PC_NCW) {  // warp 16 (ring producer) idles in this phase
      // E1 ring: GU_S stages x 40 KB (8 gate rows + 8 up rows of 2560 B each), cp.async.bulk onto one mbarrier per slot.
      // Stage t of this CTA holds rows [rw0 + 8t, rw0 + 8t + 8) of the CTA's row-balanced range; the ring index runs
      // continuously over the range (no drain per tile). The first GU_S stages are issued before the norm prologue.
      unsigned gu_c, gu_n;
      asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(gu_c)); asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(gu_n));
      const int gu_rw0 = (int)((gu_c * (unsigned)I) / gu_n), gu_rw1 = (int)(((gu_c + 1) * (unsigned)I) / gu_n);
      const int gu_ns = (gu_rw1 - gu_rw0 + 7) >> 3;
      const uint32_t gu_ring = (uint32_t)__cvta_generic_to_shared(smem + GU_RING_OFF), gu_mb = gu_ring + GU_S * GU_STAGE;
      float* gu_rs = reinterpret_cast<float*>(smem + 10240);  // [row_local<70][gate w0..15, up w0..15] (8960 B)
      const int8_t* gu_g = wg + (size_t)layer * I * H;
      const int8_t* gu_u = wu + (size_t)layer * I * H;
      auto gu_issue = [&](int t) {  // warp 0 only
        if (t < gu_ns) {
          const int slot = t - (t / GU_S) * GU_S, nr = min(8, gu_rw1 - gu_rw0 - 8 * t);
          if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(gu_mb + 8 * slot), "r"(2 * nr * H) : "memory");
          __syncwarp();
          if ((lane & 7) < nr && lane < 16) {
            const int8_t* src = (lane < 8 ? gu_g : gu_u) + (size_t)(gu_rw0 + 8 * t + (lane & 7)) * H;
            asm volatile("{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
                         "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], pol; }"
                         :: "r"(gu_ring + slot * GU_STAGE + lane * GU_RS), "l"(src), "r"(H), "r"(gu_mb + 8 * slot) : "memory");
          }
        }
      };
      if (warp == 0) {
        if (lane < GU_S) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(gu_mb + 8 * lane) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;\n\tfence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
#pragma unroll 1
        for (int t = 0; t < GU_S; ++t) gu_issue(t);
      }
      // prologue loads issued together: obuf, both norm weight vectors and this thread's epilogue scales
      float m[5], w2[5], w3[5], gsc = 0.f, usc = 0.f;
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        m[j] = bf2f(__ldcg(obuf + tid + NTH * j));
        w2[j] = bf2f(__ldg(ln2 + (size_t)layer * H + tid + NTH * j));
        w3[j] = bf2f(__ldg(ln3 + (size_t)layer * H + tid + NTH * j));
      }
      if (tid < gu_rw1 - gu_rw0) { gsc = __ldg(sg + (size_t)layer * I + gu_rw0 + tid); usc = __ldg(su + (size_t)layer * I + gu_rw0 + tid); }
      rms_pre(m, w2, rr);
#pragma unroll
      for (int j = 0; j < 5; ++j) xr[j] = rb(xr[j] + m[j]);
#pragma unroll
      for (int j = 0; j < 5; ++j) m[j] = xr[j];
      rms_pre(m, w3, rr);
      // x (bf16-valued fp32) linear in the xs region; lanes g == 0 of each warp keep their mma B fragments as f16x2
      // (exact: bf16 values in the f16 normal range) for the warp's fixed k range [160 warp, 160 warp + 160).
      float* gu_xf = reinterpret_cast<float*>(smem);
#pragma unroll
      for (int j = 0; j < 5; ++j) gu_xf[tid + NTH * j] = m[j];
      pc_csync();
      const int g = lane >> 2, t4 = lane & 3;
      uint32_t gxh[5][4];
#pragma unroll
      for (int s5 = 0; s5 < 5; ++s5) {
        if (g == 0) {
          const float4* xp = reinterpret_cast<const float4*>(gu_xf + warp * 160 + s5 * 32 + t4 * 8);
          const float4 x0 = xp[0], x1 = xp[1];
          gxh[s5][0] = pack_h2(x0.x, x0.y); gxh[s5][1] = pack_h2(x0.z, x0.w); gxh[s5][2] = pack_h2(x1.x, x1.y); gxh[s5][3] = pack_h2(x1.z, x1.w);
        } else {
          gxh[s5][0] = gxh[s5][1] = gxh[s5][2] = gxh[s5][3] = 0u;
        }
      }
#pragma unroll 1
      for (int t = 0; t < gu_ns; ++t) {
        const int slot = t - (t / GU_S) * GU_S;
        gu_wait(gu_mb + 8 * slot, (unsigned)((t / GU_S) & 1));
        const int nr = min(8, gu_rw1 - gu_rw0 - 8 * t);
        // stage rows: g = gate row 8t+g, g+8 = up row 8t+g; this thread's 8 k per 32-k step at warp*160 + 32 s + 8 t4
        const uint8_t* ra = smem + GU_RING_OFF + slot * GU_STAGE + g * GU_RS + warp * 160 + t4 * 8;
        uint2 wa[5], wb[5];
#pragma unroll
        for (int s5 = 0; s5 < 5; ++s5) {
          wa[s5] = *reinterpret_cast<const uint2*>(ra + s5 * 32);
          wb[s5] = *reinterpret_cast<const uint2*>(ra + 8 * GU_RS + s5 * 32);
        }
        __syncthreads();  // slot copied to registers by all warps: refill it before the math (early refill)
        if (warp == 0) gu_issue(t + GU_S);
        float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s5 = 0; s5 < 5; ++s5) {
          uint32_t pa[4], pb[4];
          i8x4_h2(wa[s5].x, pa[0], pa[1]); i8x4_h2(wa[s5].y, pa[2], pa[3]);
          i8x4_h2(wb[s5].x, pb[0], pb[1]); i8x4_h2(wb[s5].y, pb[2], pb[3]);
          float (&c)[4] = (s5 & 1) ? c1 : c0;
          mma16816h(c, pa[0], pb[0], pa[1], pb[1], gxh[s5][0], gxh[s5][1]);
          mma16816h(c, pa[2], pb[2], pa[3], pb[3], gxh[s5][2], gxh[s5][3]);
        }
        if (t4 == 0 && g < nr) {
          float* d = gu_rs + (8 * t + g) * 32 + warp;
          d[0] = c0[0] + c1[0]; d[16] = c0[2] + c1[2];
        }
      }
      pc_csync();
      if (tid < gu_rw1 - gu_rw0) {
        const int row = gu_rw0 + tid;
        const float* q = gu_rs + tid * 32;
        float gsum = 0.f, usum = 0.f;
#pragma unroll
        for (int w = 0; w < 16; ++w) { gsum += q[w]; usum += q[16 + w]; }
        const float gt = rb(gsum * gsc);
        const float upv = rb(usum * usc);
        const float inner = 0.7978845608028654f * (gt + 0.044715f * (gt * gt * gt));
        const float ge = rb(0.5f * gt * (1.0f + tanhf(inner)));
        act[row] = __float2bfloat16_rn(ge * upv);
      }
      if (tid < GU_S) asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(gu_mb + 8 * tid) : "memory");
    }
    gbar(bcnt, bgen, lg);
    MB_TIMELINE_END(3);
    // ---------------- down ----------------
    MB_TIMELINE_BEGIN(4);
    {
      // E3 stream-K over (16-row tile, DN_SK-k chunk) units on the producer/consumer ring (DN_S stages of 16 rows x DN_SK).
      // CTA c owns units [c*DN_UNITS/G, (c+1)*DN_UNITS/G) in tile-major order (4-5 units, so a tile spans at most 2 CTAs).
      // Its first segment may be the last chunks of a tile begun by c-1: it publishes the 16 fp32 row partials (tagged
      // u64 in dpart[row], one store each). Its last segment may be the first chunks of a tile finished by c+1: it waits
      // for those tags, resets them, and adds partial (own + neighbour) in fixed order. No float atomics.
      int G, cta;
      asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(G));
      asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(cta));
      if (G > DN_UNITS / DN_CH) __trap();  // every CTA must own >= DN_CH units
      const int u0 = (int)((long long)cta * DN_UNITS / G), total = (int)((long long)(cta + 1) * DN_UNITS / G) - u0;
      const PcRing<DN_S> rg{(uint32_t)__cvta_generic_to_shared(smem + DN_RING_OFF),
                            (uint32_t)__cvta_generic_to_shared(smem + DN_RING_OFF + DN_S * DN_STAGE), (uint32_t)DN_STAGE};
      const int8_t* wl = wd + (size_t)layer * H * I;
      auto issue = [&](int c) {  // warp 0: the ring's producer role (no producer warp; 512 threads)
        const int u = u0 + c, tile = u / DN_CH, ch = u % DN_CH;
        rg.acquire((unsigned)c, 16u * DN_SK);
        if (lane < 16) pc_bulk_g2s(rg.slot((unsigned)c) + lane * DN_RS, wl + (size_t)(tile * 16 + lane) * I + ch * DN_SK, DN_SK, rg.full((unsigned)c));
      };
      if (warp < PC_NCW) {  // 16 compute warps (a launched warp 16 idles here)
        // epilogue operands issued at phase start (their latency is off the tail): row scales of the CTA's (<= 3) tiles,
        // loaded before the x staging and parked in fin[3][16] (free in this phase) after it
        const int t0 = u0 / DN_CH, tl = (u0 + total - 1) / DN_CH;
        float sc0 = 0.f, sc1 = 0.f, sc2 = 0.f;
        if (tid < 16) {
          const float* sdl = sd + (size_t)layer * H + tid;
          sc0 = __ldg(sdl + t0 * 16);
          if (t0 + 1 <= tl) sc1 = __ldg(sdl + (t0 + 1) * 16);
          if (t0 + 2 <= tl) sc2 = __ldg(sdl + (t0 + 2) * 16);
        }
        if (tl - t0 > 2) __trap();  // at most 3 tiles per CTA (G >= 107)
        const unsigned xsa = (unsigned)__cvta_generic_to_shared(xs);
        for (int i = tid; i < I / 8; i += PC_NC)  // act -> xs (L2-coherent cp.async), requested ahead of the weights
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(xsa + i * 16), "l"(act + i * 8) : "memory");
        if (warp == 0) {  // ring init + chunks 0..DN_PRE-1 in order (row copies hidden under the x staging)
          if (lane == 0) rg.init();
          __syncwarp();
#pragma unroll 1
          for (int c = 0; c < min(total, DN_PRE); ++c) issue(c);
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        if (tid < 16) { fin[tid] = sc0; fin[16 + tid] = sc1; fin[32 + tid] = sc2; }  // fin[32..47] reaches rr: dead here
        pc_csync();  // ring initialised, xs complete, scales parked
        if (warp == 0) {  // chunks DN_PRE .. DN_S-1 in order, issued while every warp waits for chunk 0 (UBLKCP issue ~25-40 ns/op)
#pragma unroll 1
          for (int c = DN_PRE; c < min(total, DN_S); ++c) issue(c);
        }
        // thread = (k quarter q, row pair r / r+8): 40 k of the warp's 160-k slice of each chunk, both rows, packed f32x2 math
        const int q = lane >> 3, r = lane & 7;
        const int koff = warp * (DN_SK / 16) + q * (DN_SK / 64);
        float2 aA = make_float2(0.f, 0.f), aB = make_float2(0.f, 0.f);
        int seg = 0;
#pragma unroll 1
        for (int c = 0; c < total; ++c) {
          const int u = u0 + c, ch = u % DN_CH;
          // neighbour partial of a head tile: fetched (L2-coherent cp.async, 16 B = the tag pair holding this row) into
          // qv[4 tid .. 4 tid + 3] (attention scratch, dead here) before the last chunk's math; read after the reduce
          if (c == total - 1 && ch != DN_CH - 1 && tid < 16)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(qv + 4 * tid)),
                         "l"(dpart + (u / DN_CH) * 16 + (tid & ~1)) : "memory");
          rg.wait_full((unsigned)c);
          const uint8_t* wa = smem + DN_RING_OFF + (c % DN_S) * DN_STAGE + r * DN_RS + koff;
          const bf16* xb = xs + ch * DN_SK + koff;
#pragma unroll
          for (int j = 0; j < DN_SK / 64 / 8; ++j) {
            const uint2 A = *reinterpret_cast<const uint2*>(wa + j * 8);
            const uint2 B = *reinterpret_cast<const uint2*>(wa + 8 * DN_RS + j * 8);
            const uint4 X = *reinterpret_cast<const uint4*>(xb + j * 8);
            const uint32_t aw[2] = {A.x ^ 0x80808080u, A.y ^ 0x80808080u}, bw[2] = {B.x ^ 0x80808080u, B.y ^ 0x80808080u};
            const uint32_t xw[4] = {X.x, X.y, X.z, X.w};
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                const uint32_t xv = xw[2 * i + h];
                const float x0 = __uint_as_float(xv << 16), x1 = __uint_as_float(xv & 0xffff0000u);
                dn_fma2(aA, dn_prmt(aw[i], 2 * h), dn_prmt(aw[i], 2 * h + 1), x0, x1);
                dn_fma2(aB, dn_prmt(bw[i], 2 * h), dn_prmt(bw[i], 2 * h + 1), x0, x1);
              }
            }
          }
          if (total > DN_S) {  // refill (chunk c + DN_S into chunk c's slot once every warp released it)
            rg.release((unsigned)c);
            if (warp == 0 && c + DN_S < total) issue(c + DN_S);
          }
          if (ch == DN_CH - 1 || c == total - 1) {  // end of a tile segment: reduce the 16 warps (red double-buffered by segment)
            float sA = aA.x + aA.y, sB = aB.x + aB.y;
            aA = aB = make_float2(0.f, 0.f);
            sA += __shfl_xor_sync(0xffffffffu, sA, 8); sB += __shfl_xor_sync(0xffffffffu, sB, 8);
            sA += __shfl_xor_sync(0xffffffffu, sA, 16); sB += __shfl_xor_sync(0xffffffffu, sB, 16);
            float* rd = red + (seg & 1) * 256;
            if (q == 0) { rd[warp * 16 + r] = sA; rd[warp * 16 + r + 8] = sB; }
            pc_csync();
            if (c == total - 1 && tid < 2 * DN_S) pc_mb_inval(rg.mb + 8u * tid);  // every slot consumed by all warps
            if (tid < 16) {
              float sm = 0.f;
#pragma unroll
              for (int w = 0; w < 16; ++w) sm += rd[w * 16 + tid];
              const int orow = (u / DN_CH) * 16 + tid;
              const float scl = fin[seg * 16 + tid];  // segment s is tile t0 + s
              const bool startmid = seg == 0 && (u0 % DN_CH) != 0;  // tile begun by CTA c-1
              if (ch == DN_CH - 1 && !startmid) {
                mlp[orow] = __float2bfloat16_rn(sm * scl);
              } else if (ch == DN_CH - 1) {
                st_tag(dpart + orow, sm);
              } else {  // first chunks (last segment); CTA c+1 publishes the rest
                asm volatile("cp.async.wait_all;" ::: "memory");
                unsigned long long tag = reinterpret_cast<const unsigned long long*>(qv + 4 * tid)[tid & 1];
                while ((tag >> 32) == 0ull) asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(tag) : "l"(dpart + orow) : "memory");
                asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(dpart + orow), "l"(0ull) : "memory");
                mlp[orow] = __float2bfloat16_rn((sm + __uint_as_float((unsigned)tag)) * scl);
              }
            }
            ++seg;
          }
        }
      }
    }
    gbar(bcnt, bgen, lg);
    MB_TIMELINE_END(4);
  }
  // ---------------- LM head ----------------
  MB_TIMELINE_BEGIN(5);
  if (warp < PC_NCW) {  // warp 16 (ring producer) idles in the LM head
  float* lmx = reinterpret_cast<float*>(smem);  // 2560 floats
  {
    float m[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) m[j] = bf2f(__ldcg(mlp + tid + NTH * j));
    rms_regs(m, ln4 + (size_t)(L - 1) * H, rr);
#pragma unroll
    for (int j = 0; j < 5; ++j) xr[j] = rb(xr[j] + m[j]);
#pragma unroll
    for (int j = 0; j < 5; ++j) m[j] = xr[j];
    rms_regs(m, fnorm, rr);
#pragma unroll
    for (int j = 0; j < 5; ++j) lmx[tid + NTH * j] = m[j];
    pc_csync();
  }
  float best = -INFINITY; int besti = 0x7fffffff;
  {
    const int gw = blockIdx.x * 16 + warp, tw = gridDim.x * 16;
    for (int row = gw; row < V; row += tw) {
      const bf16* wr = embed + (size_t)row * H;
      uint4 v[10];
#pragma unroll
      for (int u = 0; u < 10; ++u) v[u] = ldg_nc(wr + lane * 8 + u * 256);
      float acc = 0.f;
#pragma unroll
      for (int u = 0; u < 10; ++u) {
        const float4* xp = reinterpret_cast<const float4*>(lmx + lane * 8 + u * 256);
        const float4 x0 = xp[0], x1 = xp[1];
        acc = fmaf(__uint_as_float(v[u].x << 16), x0.x, acc); acc = fmaf(__uint_as_float(v[u].x & 0xffff0000u), x0.y, acc);
        acc = fmaf(__uint_as_float(v[u].y << 16), x0.z, acc); acc = fmaf(__uint_as_float(v[u].y & 0xffff0000u), x0.w, acc);
        acc = fmaf(__uint_as_float(v[u].z << 16), x1.x, acc); acc = fmaf(__uint_as_float(v[u].z & 0xffff0000u), x1.y, acc);
        acc = fmaf(__uint_as_float(v[u].w << 16), x1.z, acc); acc = fmaf(__uint_as_float(v[u].w & 0xffff0000u), x1.w, acc);
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      if (lane == 0) {
        const float lv = rb(acc);
        logits[row] = lv;
        if (lv > best) { best = lv; besti = row; }
      }
    }
  }
  if (lane == 0) { wbv[warp] = best; wbi[warp] = besti; }
  pc_csync();
  if (tid == 0) {
    float bv = wbv[0]; int bi = wbi[0];
    for (int w = 1; w < 16; ++w) if (wbv[w] > bv || (wbv[w] == bv && wbi[w] < bi)) { bv = wbv[w]; bi = wbi[w]; }
    part_v[blockIdx.x] = bv; part_i[blockIdx.x] = bi;
    __threadfence();
    const unsigned old = atomicAdd(ticket, 1u);
    *flag = (old == gridDim.x - 1) ? 1 : 0;
  }
  pc_csync();
  if (*flag) {
    __threadfence();
    if (tid < gridDim.x) { pvs[tid] = __ldcg(part_v + tid); pis[tid] = __ldcg(part_i + tid); }
    pc_csync();
    if (tid == 0) {
      float bv = pvs[0]; int bi = pis[0];
      for (int c = 1; c < (int)gridDim.x; ++c) if (pvs[c] > bv || (pvs[c] == bv && pis[c] < bi)) { bv = pvs[c]; bi = pis[c]; }
      next[0] = (int64_t)bi;
      *ticket = 0;
    }
  }
  }
  MB_TIMELINE_END(5);
}

constexpr size_t SMEM_BASE = 20480 + (512 + 32 + 16 + 256 + 256 + 512 + 160 + 160 + 16 + 16 + 160 + 160) * 4 + 16;
static_assert(NTH == PC_NC, "NTH is the compute width (warps 0..15)");
static_assert(SMEM_BASE <= GU_RING_OFF, "E1 ring overlaps the static smem layout");
static_assert(SMEM_BASE <= DN_RING_OFF && SMEM_BASE <= QK_RING_OFF && SMEM_BASE <= OP_RING_OFF, "ring overlaps the static smem layout");
constexpr size_t mx2(size_t a, size_t b) { return a > b ? a : b; }
constexpr size_t SMEM_BYTES = mx2(mx2(mx2(SMEM_BASE, (size_t)GU_RING_END), mx2((size_t)DN_RING_END, (size_t)QK_RING_END)), (size_t)OP_RING_END);

extern "C" int mk_init() {
  return (int)cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
}
extern "C" int mk_launch(const void* const* ptrs, float emb_scale, int tl_words, int grid, void* stream) {
  Ptrs p;
  for (int i = 0; i < P_COUNT; ++i) p.a[i] = ptrs[i];
  p.emb_scale = emb_scale; p.tl_words = tl_words;
  mk_kernel<<<grid, MK_NTB, SMEM_BYTES, (cudaStream_t)stream>>>(p);
  return (int)cudaGetLastError();
}
extern "C" int mk_pcount() { return P_COUNT; }
