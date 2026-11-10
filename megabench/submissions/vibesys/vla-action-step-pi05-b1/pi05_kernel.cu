// pi0.5 action-expert megakernel (single persistent launch).
// Phases: hoisted t-only FP32 pass, then 10 steps x 18 layers of
//   row-norm -> QKV(tcgen05) -> rope -> fused flash attention item (QK, chunk softmax, PV; mma.sync)
//   -> weighted chunk merge -> o_proj(tcgen05)
//   -> row-norm -> gate/up(tcgen05) -> gelu*up -> down(tcgen05)
// BF16 GEMMs: tcgen05 swap-AB M128 N64 K16, TMEM fp32 accumulator, cp.async 4-stage ring,
// stream-K across CTAs, partials reduced with red.global.add.f32 into zeroed fp32 workspaces.
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "timeline_probe.cuh"
typedef __nv_bfloat16 bf16;

#define KC 128
#define NP 64
#define STAGES 4
#define QP 264   // padded bf16 row stride (elements) for 256-wide smem tiles
#define PPD 72   // padded row stride for P tile
#define SSP 65   // padded fp32 row stride for the score tile
#define NCHUNK 16
#define HID 1024
#define NTOK 50
#define NLAYER 18
#define NSTEP 10
#define NPREFIX 968
#define NKEY 1018
#define NMOD 37

struct Args {
  const float *action_in_w, *action_in_b, *time_in_w, *time_in_b, *time_out_w, *time_out_b, *action_out_w, *action_out_b;
  const bf16 *wq, *wk, *wv, *wo, *wg, *wu, *wd;
  const float *attn_w, *attn_b, *ffn_w, *ffn_b, *final_w, *final_b;
  const float* noise;
  const bf16 *pk, *pv;
  const bool* mask;
  float* out;
  float *temb, *hid1, *cond, *mod, *Yqkv, *Yo, *Ygu, *Yd;
  float* Opart;    // [NCHUNK][NTOK][2048] fp32 unnormalized chunk partials of P@V, rewritten every layer
  const float* inv;
  bf16 *h, *X1, *Xo, *X2;
  float2* St;      // [8][NTOK][NCHUNK] chunk stats (m_c, l_c), rewritten every layer
  unsigned* bar;
  unsigned epoch0;
  float2* cs;      // [NTOK][128] (bf16r(cos), bf16r(sin)) of (nvalid+t)*inv[j], built per call
  bf16 *Qr, *Kr;   // roped Q [NTOK][2048], roped action-K [NTOK][256], written by the rope phase
#ifdef MEGABENCH_TIMELINE
  uint64_t* tl;
  unsigned long long tlw;
#endif
};

#define NBAR (4 + NSTEP * (NLAYER * 10 + 2))

__device__ __forceinline__ void grid_bar(unsigned* ctr, unsigned& epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    epoch += gridDim.x;
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(ctr) : "memory");
    unsigned v;
    do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ctr) : "memory"); } while ((int)(v - epoch) < 0);
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  __syncthreads();
}

// ---- timeline probes (compile switch MEGABENCH_TIMELINE; production build carries none)
// phase event k: BEGIN after the previous barrier releases, END when the whole CTA arrives at
// the closing barrier (after its first __syncthreads); event TLB+k: closing barrier wait.
enum { EV_TEMB, EV_TIME_IN, EV_TIME_OUT, EV_MOD, EV_NORM_ATTN, EV_QKV, EV_ROPE, EV_SCORES, EV_PV, EV_CONV,
       EV_OPROJ, EV_NORM_FFN, EV_GATEUP, EV_GELU, EV_DOWN, EV_FINAL_NORM, EV_ACTION_OUT, EV_N };
#define TLB 32
#ifdef MEGABENCH_TIMELINE
// lean writer in the timeline_probe.cuh buffer layout (one writer per CTA, thread 0): only a
// 32-bit record counter stays live across the layer loop, so the probe build does not spill.
__device__ __forceinline__ void tl_rec(uint64_t* buf, unsigned long long words, unsigned& k, int ev, unsigned type) {
  if (threadIdx.x != 0) return;
  uint32_t now, sm;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(now));
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const unsigned long long idx = 1ull + blockIdx.x + (unsigned long long)k * gridDim.x;
  if (idx < words)
    buf[idx] = ((uint64_t)now << 32) | ((sm & 0xFFu) << 24) | ((blockIdx.x & 0xFFFu) << 12) | (((unsigned)ev & 0x3FFu) << 2) | type;
  ++k;
}
__device__ __forceinline__ void grid_bar_tl(unsigned* ctr, unsigned& epoch, const Args& A, unsigned& k, int ev) {
  __syncthreads();
  if (threadIdx.x == 0) {
    tl_rec(A.tl, A.tlw, k, ev, 1u); tl_rec(A.tl, A.tlw, k, TLB + ev, 0u);
    epoch += gridDim.x;
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(ctr) : "memory");
    unsigned v;
    do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ctr) : "memory"); } while ((int)(v - epoch) < 0);
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  __syncthreads();
  tl_rec(A.tl, A.tlw, k, TLB + ev, 1u);
}
#define GBAR(ev) grid_bar_tl(A.bar, epoch, A, tlk, (ev))
#define TLBEGIN(ev) tl_rec(A.tl, A.tlw, tlk, (ev), 0u)
#else
#define GBAR(ev) grid_bar(A.bar, epoch)
#define TLBEGIN(ev) do {} while (0)
#endif

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------ tcgen05 helpers
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr) {
  uint64_t d = 0; d |= (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)1 << 16; d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 46; d |= (uint64_t)2 << 61; return d;
}
__device__ __forceinline__ void mma_bf16_ss(uint32_t dt, uint64_t ad, uint64_t bd, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" :: "r"(dt), "l"(ad), "l"(bd), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void mbar_wait_p(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" :: "r"(mbar), "r"(phase));
}
__device__ __forceinline__ void tc_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); __syncthreads(); asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void cp_async16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void commit_to(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar)); }
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

struct Tc { uint8_t* base; uint64_t* mb; uint32_t tmem; long long J0; int dph; };
struct GemmDesc { const bf16* w[3]; int nt[3]; int K, N; const bf16* X; float* Y; };

// GFIX > 0: stream-K over exactly GFIX CTAs; CTAs >= GFIX do no work (tc state untouched)
template <int GFIX = 0>
__device__ void gemm_phase(Tc& tc, const GemmDesc& d) {
  constexpr int S = STAGES;
  constexpr int A_BYTES = 128 * KC * 2, B_BYTES = NP * KC * 2, STAGE = A_BYTES + B_BYTES;
  if (GFIX > 0 && (int)blockIdx.x >= GFIX) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = blockIdx.x, G = GFIX > 0 ? GFIX : (int)gridDim.x;
  const int K = d.K, N = d.N, KI = K / KC, ntiles = N / 128;
  const long long T = (long long)ntiles * KI;
  const long long g0 = (long long)b * T / G, g1 = (long long)(b + 1) * T / G;
  const int n = (int)(g1 - g0);
  uint8_t* base = tc.base; uint64_t* mb = tc.mb; const uint32_t tmem = tc.tmem;
  const uint32_t idesc = 1u << 4 | 1u << 7 | 1u << 10 | (uint32_t)(NP >> 3) << 17 | (uint32_t)(128 >> 4) << 24;
  const long long J0 = tc.J0; int dph = tc.dph;
  auto wptr = [&](int tile) -> const bf16* {
    int s = 0, lt = tile;
    if (lt >= d.nt[0]) { lt -= d.nt[0]; s = 1; if (lt >= d.nt[1]) { lt -= d.nt[1]; s = 2; } }
    return d.w[s] + (size_t)lt * 128 * K;
  };
  auto stage_a = [&](long long J) { return base + (size_t)(J % S) * STAGE; };
  auto stage_b = [&](long long J) { return base + (size_t)(J % S) * STAGE + A_BYTES; };
  auto issue = [&](int j) {
    if (j < n) {
      const int g = (int)(g0 + j), tile = g / KI, kc = g % KI; const long long J = J0 + j;
      const bf16* Wt = wptr(tile) + (size_t)kc * KC;
      const uint32_t sa = su32(stage_a(J)), sb = su32(stage_b(J));
#pragma unroll
      for (int q = 0; q < 8; ++q) { const int i = tid + 256 * q, row = i >> 4, c16 = i & 15, atom = c16 >> 3, c = c16 & 7;
        cp_async16(sa + atom * 16384 + row * 128 + ((c ^ (row & 7)) << 4), Wt + (size_t)row * K + c16 * 8); }
      for (int i = tid; i < NP * 16; i += 256) { const int row = i >> 4, c16 = i & 15, atom = c16 >> 3, c = c16 & 7;
        cp_async16(sb + atom * (NP * 128) + row * 128 + ((c ^ (row & 7)) << 4), d.X + (size_t)row * K + (size_t)kc * KC + c16 * 8); }
    }
    cp_commit();
  };
  for (int j = 0; j < S - 1; ++j) issue(j);
  int seg_k0 = 0;
  for (int j = 0; j < n; ++j) {
    const int g = (int)(g0 + j), tile = g / KI, kc = g % KI; const long long J = J0 + j;
    asm volatile("cp.async.wait_group %0;" ::"n"(S - 2) : "memory");
    asm volatile("fence.proxy.async.shared::cta;");
    tc_sync();
    const bool seg_start = (j == 0) || kc == 0, seg_end = (kc == KI - 1) || (j == n - 1);
    if (seg_start) seg_k0 = kc;
    (void)seg_k0;
    if (tid == 0) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint64_t ad = make_sdesc(su32(stage_a(J)) + (kk >> 2) * 16384 + (kk & 3) * 32);
        const uint64_t bd = make_sdesc(su32(stage_b(J)) + (kk >> 2) * (NP * 128) + (kk & 3) * 32);
        mma_bf16_ss(tmem, ad, bd, idesc, (seg_start && kk == 0) ? 0u : 1u);
      }
      commit_to(su32(mb + J % S));
      if (seg_end) commit_to(su32(mb + S));
    }
    if (j + S - 1 < n && j >= 1) mbar_wait_p(su32(mb + (J - 1) % S), (uint32_t)(((J - 1) / S) & 1));
    issue(j + S - 1);
    if (seg_end) {
      mbar_wait_p(su32(mb + S), dph); dph ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (warp < 4) {
        const int row = tile * 128 + warp * 32 + lane;
#pragma unroll
        for (int c0 = 0; c0 < NP / 8; ++c0) {
          // GFIX: the two CTAs of a tile finish together; odd CTAs walk the token columns in
          // reverse so the pair's red.adds do not hit the same addresses at the same time
          const int c = (GFIX > 0 && (b & 1)) ? NP / 8 - 1 - c0 : c0;
          uint32_t r[8];
          const uint32_t ta = tmem + ((uint32_t)(warp * 32) << 16) + c * 8;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                       : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(ta));
          asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
          for (int e = 0; e < 8; ++e) { const int t = c * 8 + e;
            if (t < NTOK) atomicAdd(d.Y + (size_t)t * N + row, __uint_as_float(r[e])); }
        }
      }
      tc_sync();
    }
  }
  tc.J0 = J0 + n; tc.dph = dph;
  // drain outstanding (empty) cp.async groups before smem is reused by other phases
  asm volatile("cp.async.wait_all;" ::: "memory");
}

// ------------------------------------------------------------------ prologue (t-only FP32 work)
__device__ void p_temb(const Args& A) {
  const int gt = blockIdx.x * blockDim.x + threadIdx.x, nt = gridDim.x * blockDim.x;
  const double step = 1.0 / 511.0;
  for (int i = gt; i < NSTEP * 512; i += nt) {
    const int s = i / 512, k = i % 512;
    const double frac = (k < 256) ? (double)k * step : 1.0 - (double)(511 - k) * step;
    const double period = 4e-3 * pow(4.0 / 4e-3, frac);
    const float tf = (float)(1.0 + (double)s * (-1.0 / 10));
    const double ph = (double)tf * (1.0 / period * 2.0 * 3.141592653589793);
    A.temb[s * 1024 + k] = (float)sin(ph);
    A.temb[s * 1024 + 512 + k] = (float)cos(ph);
  }
  // initialize output with the noise (the Euler state lives in `out`)
  for (int i = gt; i < NTOK * 32; i += nt) A.out[i] = A.noise[i];
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

// out[s][j] = act(sum_i W[j][i]*in[s][i] + b[j]) for 10 steps, one warp per row j
__device__ void p_linear10(const float* W, const float* bias, const float* in, float* outp, float* sm) {
  float4* s4 = (float4*)sm;
  for (int i = threadIdx.x; i < NSTEP * 256; i += blockDim.x) s4[i] = ((const float4*)in)[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), nw = gridDim.x * (blockDim.x >> 5);
  for (int j = gw; j < HID; j += nw) {
    float4 w[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) w[q] = ((const float4*)(W + (size_t)j * HID))[lane + 32 * q];
    for (int s = 0; s < NSTEP; ++s) {
      float acc = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) { const float4 x = s4[s * 256 + lane + 32 * q];
        acc += w[q].x * x.x; acc += w[q].y * x.y; acc += w[q].z * x.z; acc += w[q].w * x.w; }
#pragma unroll
      for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      if (lane == 0) outp[s * HID + j] = silu(acc + bias[j]);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float4 ldnc4(const float* p) { float4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ void mma_tf32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// mod[s][g][j] = sum_k W_g[j][k] * cond[s][k] + b_g[j]: swap-AB mma.sync m16n8k8 tf32 (fp32 bits as tf32 operands).
// One warp item = 16 consecutive W rows (MMA M) x K=1024; the 10 cond vectors are N (two n8 tiles, tokens 10..15 = 0).
// Lane (q,t) loads rows q and q+8 at physical k = c*32 + t*8 + [0..7] (2 float4 each, ld.global.nc), MOD_CH k-chunks
// in flight; the same k permutation is used for the cond (B) fragment from smem, so no shuffles are needed.
// Group boundaries (3072 rows) are multiples of 16, so an item never straddles two weight tensors.
#define MOD_CH 8
#define MOD_XS 1028   // padded smem row stride (floats) of the staged cond rows: conflict-free float4 fragment reads
__device__ void p_mod(const Args& A, float* sm) {
  for (int i = threadIdx.x; i < NSTEP * 256; i += blockDim.x) {
    const int s = i >> 8, c = i & 255;
    *(float4*)(sm + s * MOD_XS + c * 4) = ((const float4*)A.cond)[i];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, q = lane >> 2, t = lane & 3;
  const int gw = (threadIdx.x >> 5) * gridDim.x + blockIdx.x, nw = gridDim.x * (blockDim.x >> 5);
  const int items = NMOD * 3072 / 16;
  const float* x0 = sm + q * MOD_XS + t * 8;            // tokens q (0..7)
  const float* x1 = sm + (q + 8) * MOD_XS + t * 8;      // tokens q+8, valid for q < 2 only
  const bool x1v = q < 2;
  for (int it = gw; it < items; it += nw) {
    const int r0 = it * 16, g = r0 / 3072, j0 = r0 % 3072;
    const float *W, *bias;
    if (g == 36) { W = A.final_w + (size_t)j0 * HID; bias = A.final_b + j0; }
    else { const int l = g >> 1;
      if (g & 1) { W = A.ffn_w + ((size_t)l * 3072 + j0) * HID; bias = A.ffn_b + (size_t)l * 3072 + j0; }
      else { W = A.attn_w + ((size_t)l * 3072 + j0) * HID; bias = A.attn_b + (size_t)l * 3072 + j0; } }
    const float* w0 = W + (size_t)q * HID + t * 8;
    const float* w1 = w0 + (size_t)8 * HID;
    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c0 = 0; c0 < HID / 32; c0 += MOD_CH) {
      float4 Wr[MOD_CH][4];
#pragma unroll
      for (int c = 0; c < MOD_CH; ++c) { const int o = (c0 + c) * 32;
        Wr[c][0] = ldnc4(w0 + o); Wr[c][1] = ldnc4(w0 + o + 4); Wr[c][2] = ldnc4(w1 + o); Wr[c][3] = ldnc4(w1 + o + 4); }
#pragma unroll
      for (int c = 0; c < MOD_CH; ++c) {
        const int o = (c0 + c) * 32;
        const float4 xa0 = *(const float4*)(x0 + o), xa1 = *(const float4*)(x0 + o + 4);
        float4 xb0 = make_float4(0.f, 0.f, 0.f, 0.f), xb1 = xb0;
        if (x1v) { xb0 = *(const float4*)(x1 + o); xb1 = *(const float4*)(x1 + o + 4); }
        const float wa[8] = {Wr[c][0].x, Wr[c][0].y, Wr[c][0].z, Wr[c][0].w, Wr[c][1].x, Wr[c][1].y, Wr[c][1].z, Wr[c][1].w};
        const float wb[8] = {Wr[c][2].x, Wr[c][2].y, Wr[c][2].z, Wr[c][2].w, Wr[c][3].x, Wr[c][3].y, Wr[c][3].z, Wr[c][3].w};
        const float xa[8] = {xa0.x, xa0.y, xa0.z, xa0.w, xa1.x, xa1.y, xa1.z, xa1.w};
        const float xb[8] = {xb0.x, xb0.y, xb0.z, xb0.w, xb1.x, xb1.y, xb1.z, xb1.w};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const uint32_t a0 = __float_as_uint(wa[2 * u]), a1 = __float_as_uint(wb[2 * u]);
          const uint32_t a2 = __float_as_uint(wa[2 * u + 1]), a3 = __float_as_uint(wb[2 * u + 1]);
          mma_tf32(d0, a0, a1, a2, a3, __float_as_uint(xa[2 * u]), __float_as_uint(xa[2 * u + 1]));
          mma_tf32(d1, a0, a1, a2, a3, __float_as_uint(xb[2 * u]), __float_as_uint(xb[2 * u + 1]));
        }
      }
    }
    // C fragment: d[0..1] = W row q, tokens 2t, 2t+1; d[2..3] = W row q+8 (d1: tokens 8+2t+e, valid for t == 0)
    const float b0 = bias[q], b1 = bias[q + 8];
    float* m = A.mod + (size_t)g * 3072 + j0 + q;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int s0 = 2 * t + e;
      m[(size_t)s0 * NMOD * 3072] = d0[e] + b0; m[(size_t)s0 * NMOD * 3072 + 8] = d0[2 + e] + b1;
      if (t == 0) { const int s1 = 8 + e;
        m[(size_t)s1 * NMOD * 3072] = d1[e] + b0; m[(size_t)s1 * NMOD * 3072 + 8] = d1[2 + e] + b1; }
    }
  }
}

// ------------------------------------------------------------------ per-row norm phases
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += red[i];
  return s;
}

// mode 0: h = bf16(action_in(x)); mode 1/2: h = bf16(h + bf16(bf16(Y)*gate)); then ada-rms with mod[nidx] -> X1
__device__ void row_norm(const Args& A, int step, int t, int mode, int gidx, int nidx, const float* Y, float* red) {
  const int tid = threadIdx.x;
  float hv[4];
  if (mode == 0) {
    float xr[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) xr[j] = A.out[t * 32 + j];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + 256 * i; float acc = A.action_in_b[c];
      const float4* w = (const float4*)(A.action_in_w + (size_t)c * 32);
#pragma unroll
      for (int q = 0; q < 8; ++q) { const float4 v = w[q];
        acc += xr[4 * q] * v.x; acc += xr[4 * q + 1] * v.y; acc += xr[4 * q + 2] * v.z; acc += xr[4 * q + 3] * v.w; }
      hv[i] = bf16r(acc);
    }
  } else {
    const float* gm = A.mod + ((size_t)step * NMOD + gidx) * 3072 + 2048;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = tid + 256 * i;
      const float hold = __bfloat162float(A.h[t * HID + c]);
      const float y = bf16r(Y[(size_t)t * HID + c]);
      const float g = bf16r(gm[c]);
      hv[i] = bf16r(hold + bf16r(y * g));
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { A.h[t * HID + tid + 256 * i] = __float2bfloat16(hv[i]); ss += hv[i] * hv[i]; }
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / 1024.f + 1e-6f);
  const float* md = A.mod + ((size_t)step * NMOD + nidx) * 3072;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = tid + 256 * i;
    const float nrm = hv[i] * r;
    A.X1[t * HID + c] = __float2bfloat16(nrm * (1.f + md[c]) + md[1024 + c]);
  }
}

__device__ void zero_buf(float* p, size_t n) {
  const size_t gt = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  for (size_t i = gt; i < n / 4; i += nt) ((float4*)p)[i] = make_float4(0, 0, 0, 0);
}

// ------------------------------------------------------------------ attention
__device__ __forceinline__ void rope_pair(const Args& A, int pos, int j, float x1, float x2, float& o1, float& o2) {
  const float ph = (float)pos * A.inv[j];
  const float c = bf16r(cosf(ph)), s = bf16r(sinf(ph));
  o1 = bf16r(bf16r(x1 * c) + bf16r(-x2 * s));
  o2 = bf16r(bf16r(x2 * c) + bf16r(x1 * s));
}

// prologue: cos/sin table, the same values rope_pair computes (runs once per call, runtime nvalid)
__device__ void p_rope_table(const Args& A, int nvalid) {
  const int gt = blockIdx.x * blockDim.x + threadIdx.x, nt = gridDim.x * blockDim.x;
  for (int i = gt; i < NTOK * 128; i += nt) {
    const int t = i >> 7, j = i & 127;
    const float ph = (float)(nvalid + t) * A.inv[j];
    A.cs[i] = make_float2(bf16r(cosf(ph)), bf16r(sinf(ph)));
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(a)) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}

// rope phase: Q (8 heads) and action-K of all 50 tokens roped once per layer from the table.
// unit = (token, 4 consecutive pair indices) of one head (288 units per token: 8x32 Q + 32 K);
// warps are spread over CTAs so all 148 CTAs take part.
__device__ void p_rope(const Args& A, int nvalid) {
  const int lane = threadIdx.x & 31, gw = (threadIdx.x >> 5) * gridDim.x + blockIdx.x, nw = gridDim.x * (blockDim.x >> 5);
  for (int u = gw * 32 + lane; u < NTOK * 288; u += nw * 32) {
    const int t = u / 288, r = u - t * 288;
    const bool isq = r < 256;
    const int j0 = (r & 31) * 4, col = isq ? (r >> 5) * 256 : 2048;
    const float4 a = *(const float4*)(A.Yqkv + (size_t)t * 2560 + col + j0);
    const float4 b = *(const float4*)(A.Yqkv + (size_t)t * 2560 + col + j0 + 128);
    const float4 c01 = *(const float4*)(A.cs + t * 128 + j0), c23 = *(const float4*)(A.cs + t * 128 + j0 + 2);
    const float x1[4] = {bf16r(a.x), bf16r(a.y), bf16r(a.z), bf16r(a.w)};
    const float x2[4] = {bf16r(b.x), bf16r(b.y), bf16r(b.z), bf16r(b.w)};
    const float cc[4] = {c01.x, c01.z, c23.x, c23.z}, ss[4] = {c01.y, c01.w, c23.y, c23.w};
    float o1[4], o2[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      o1[e] = bf16r(bf16r(x1[e] * cc[e]) + bf16r(-x2[e] * ss[e]));
      o2[e] = bf16r(bf16r(x2[e] * cc[e]) + bf16r(x1[e] * ss[e]));
    }
#ifdef ROPE_DEBUG
    for (int e = 0; e < 4; ++e) {
      float r1, r2; rope_pair(A, nvalid + t, j0 + e, x1[e], x2[e], r1, r2);
      if (__float_as_uint(r1) != __float_as_uint(o1[e]) || __float_as_uint(r2) != __float_as_uint(o2[e])) atomicAdd(A.bar + 2, 1u);
    }
#endif
    (void)nvalid;
    bf16* dst = isq ? A.Qr + (size_t)t * 2048 + col + j0 : A.Kr + (size_t)t * 256 + j0;
    *(uint2*)dst = make_uint2(pack_bf2(o1[0], o1[1]), pack_bf2(o1[2], o1[3]));
    *(uint2*)(dst + 128) = make_uint2(pack_bf2(o2[0], o2[1]), pack_bf2(o2[2], o2[3]));
  }
}

// fused flash item (h = b/16, chunk c = b%16): 64 keys x 50 tokens of head h.
// S = bf16r(Q K^T) / 16 (masked keys -inf); per row m_c = max, P = bf16(exp(S - m_c)), l_c = sum exp(S - m_c);
// Opart[c][t][h*256+d] = P @ V (plain stores), St[h][t][c] = (m_c, l_c). A chunk row with no valid key gives
// m_c = -inf, l_c = 0 and a zero partial. All global Q/K/V loads are issued before the smem stores.
__device__ void attn_fused(const Args& A, int layer, bf16* sm) {
  const int item = blockIdx.x; if (item >= 128) return;
  const int h = item >> 4, c = item & 15;
  bf16* Qs = sm; bf16* Ks = Qs + 64 * QP; bf16* Vs = Ks + 64 * QP;
  float* Ss = (float*)(Vs + 64 * QP); bf16* Ps = (bf16*)(Ss + 64 * SSP); float* Ms = (float*)(Ps + 64 * PPD); float* Vf = Ms + 64;  // Vf: fp32 action-V stage [NTOK][256]
  const int tid = threadIdx.x;
  {
    uint4 rq[8], rk[8], rv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = tid + 256 * j, r = i >> 5, q = i & 31, kk = c * 64 + r;
      rq[j] = (r < NTOK) ? *(const uint4*)(A.Qr + (size_t)r * 2048 + h * 256 + q * 8) : make_uint4(0, 0, 0, 0);
      if (kk < NPREFIX) {
        rk[j] = *(const uint4*)(A.pk + ((size_t)layer * NPREFIX + kk) * 256 + q * 8);
        rv[j] = *(const uint4*)(A.pv + ((size_t)layer * NPREFIX + kk) * 256 + q * 8);
      } else if (kk < NKEY) {
        // action V is fp32 in Yqkv: cp.async into the fp32 stage (no wait here), converted below by this thread
        const int t = kk - NPREFIX;
        rk[j] = *(const uint4*)(A.Kr + (size_t)t * 256 + q * 8);
        const float* y = A.Yqkv + (size_t)t * 2560 + 2304 + q * 8;
        cp_async16(su32(Vf + t * 256 + q * 8), y); cp_async16(su32(Vf + t * 256 + q * 8 + 4), y + 4);
      } else { rk[j] = make_uint4(0, 0, 0, 0); rv[j] = make_uint4(0, 0, 0, 0); }
    }
    if (tid < 64) { const int kk = c * 64 + tid; Ms[tid] = ((kk < NPREFIX) ? A.mask[kk] : (kk < NKEY)) ? 0.f : -INFINITY; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = tid + 256 * j, r = i >> 5, q = i & 31;
      const int kk = c * 64 + r;
      *(uint4*)(Qs + r * QP + q * 8) = rq[j]; *(uint4*)(Ks + r * QP + q * 8) = rk[j];
      if (kk < NPREFIX || kk >= NKEY) *(uint4*)(Vs + r * QP + q * 8) = rv[j];
    }
    if (c * 64 + 63 >= NPREFIX) {  // chunk with action keys (CTA-uniform)
      cp_commit();
      asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = tid + 256 * j, r = i >> 5, q = i & 31, kk = c * 64 + r;
        if (kk >= NPREFIX && kk < NKEY) {
          const float* f = Vf + (kk - NPREFIX) * 256 + q * 8;
          const float4 va = *(const float4*)f, vb = *(const float4*)(f + 4);
          *(uint4*)(Vs + r * QP + q * 8) = make_uint4(pack_bf2(va.x, va.y), pack_bf2(va.z, va.w), pack_bf2(vb.x, vb.y), pack_bf2(vb.z, vb.w));
        }
      }
    }
  }
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int mt = warp & 3, nh = warp >> 2;
  float* redm = Ss; float* redl = Ss + 128;  // [key half nh][64 rows] partial row max / sum
  {
    float acc[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
    const uint32_t qaddr = su32(Qs + (mt * 16 + (lane & 15)) * QP + (lane >> 4) * 8);
    const uint32_t kaddr = su32(Ks + (nh * 32 + (lane & 7) + ((lane >> 4) << 3)) * QP + ((lane >> 3) & 1) * 8);
#pragma unroll 4
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7;
      ldsm_x4(qaddr + ks * 32, a0, a1, a2, a3);
      ldsm_x4(kaddr + ks * 32, b0, b1, b2, b3);
      ldsm_x4(kaddr + 16 * QP * 2 + ks * 32, b4, b5, b6, b7);
      mma16816(acc[0], a0, a1, a2, a3, b0, b1);
      mma16816(acc[1], a0, a1, a2, a3, b2, b3);
      mma16816(acc[2], a0, a1, a2, a3, b4, b5);
      mma16816(acc[3], a0, a1, a2, a3, b6, b7);
    }
    // chunk softmax on the accumulators: row ra = mt*16+g (e = 0,1), rb = ra+8 (e = 2,3), key i = (nh*4+n)*8+2tq+(e&1);
    // per-row max / sum over this warp's 32 keys by a 2-level shuffle over tq, the two key halves (nh) combined in smem
    const int ra = mt * 16 + g, rb = ra + 8;
    float ma = -INFINITY, mb = -INFINITY;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[n][e] = bf16r(acc[n][e]) * 0.0625f + Ms[(nh * 4 + n) * 8 + 2 * tq + (e & 1)];
        if (e < 2) ma = fmaxf(ma, acc[n][e]); else mb = fmaxf(mb, acc[n][e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) { ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o)); mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o)); }
    if (tq == 0) { redm[nh * 64 + ra] = ma; redm[nh * 64 + rb] = mb; }
    __syncthreads();
    ma = fmaxf(redm[ra], redm[64 + ra]); mb = fmaxf(redm[rb], redm[64 + rb]);
    float la = 0.f, lb = 0.f;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int i = (nh * 4 + n) * 8 + 2 * tq;
      const float e0 = (ma == -INFINITY) ? 0.f : expf(acc[n][0] - ma), e1 = (ma == -INFINITY) ? 0.f : expf(acc[n][1] - ma);
      const float e2 = (mb == -INFINITY) ? 0.f : expf(acc[n][2] - mb), e3 = (mb == -INFINITY) ? 0.f : expf(acc[n][3] - mb);
      la += e0 + e1; lb += e2 + e3;
      *(uint32_t*)(Ps + ra * PPD + i) = pack_bf2(e0, e1);
      *(uint32_t*)(Ps + rb * PPD + i) = pack_bf2(e2, e3);
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) { la += __shfl_xor_sync(0xffffffffu, la, o); lb += __shfl_xor_sync(0xffffffffu, lb, o); }
    if (tq == 0) { redl[nh * 64 + ra] = la; redl[nh * 64 + rb] = lb; }
  }
  __syncthreads();
  if (nh == 0 && tq == 0) {  // chunk stats (m_c, l_c) of rows ra, rb
    const int ra = mt * 16 + g, rb = ra + 8;
    const float ma = fmaxf(redm[ra], redm[64 + ra]), mb = fmaxf(redm[rb], redm[64 + rb]);
    if (ra < NTOK) A.St[((size_t)h * NTOK + ra) * NCHUNK + c] = make_float2(ma, redl[ra] + redl[64 + ra]);
    if (rb < NTOK) A.St[((size_t)h * NTOK + rb) * NCHUNK + c] = make_float2(mb, redl[rb] + redl[64 + rb]);
  }
  float acc[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
  const uint32_t paddr = su32(Ps + (mt * 16 + (lane & 15)) * PPD + (lane >> 4) * 8);
  const uint32_t vaddr = su32(Vs + ((lane & 7) + ((lane >> 3) & 1) * 8) * QP + nh * 128 + (lane >> 4) * 8);
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    uint32_t a0, a1, a2, a3;
    ldsm_x4(paddr + ks * 32, a0, a1, a2, a3);
#pragma unroll
    for (int n = 0; n < 16; n += 2) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4_t(vaddr + (ks * 16 * QP + n * 8) * 2, b0, b1, b2, b3);
      mma16816(acc[n], a0, a1, a2, a3, b0, b1);
      mma16816(acc[n + 1], a0, a1, a2, a3, b2, b3);
    }
  }
#pragma unroll
  for (int n = 0; n < 16; ++n) {
#pragma unroll
    for (int e2 = 0; e2 < 2; ++e2) {
      const int t = mt * 16 + g + (e2 ? 8 : 0), d = (nh * 16 + n) * 8 + 2 * tq;
      if (t < NTOK) *(float2*)(A.Opart + ((size_t)c * NTOK + t) * 2048 + h * 256 + d) = make_float2(acc[n][2 * e2], acc[n][2 * e2 + 1]);
    }
  }
}

// ------------------------------------------------------------------ elementwise phases
// weighted chunk merge, one warp per (t, h, 128-column half): M = max_c m_c,
// w_c = exp(m_c - M) / sum_c l_c exp(m_c - M) (chunks with l_c = 0 get w_c = 0), Xo = bf16(sum_c w_c Opart_c); zero Yd
__device__ void p_conv(const Args& A) {  // warps interleaved over CTAs: the 800 units spread over all SMs
  const int lane = threadIdx.x & 31, gw = (threadIdx.x >> 5) * gridDim.x + blockIdx.x, nw = gridDim.x * 8;
  for (int u = gw; u < NTOK * 16; u += nw) {
    const int t = u >> 4, h = (u >> 1) & 7, half = u & 1;
    const float4* src = (const float4*)(A.Opart + (size_t)t * 2048 + h * 256) + half * 32 + lane;
    float4 v[NCHUNK];
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c) v[c] = src[(size_t)c * NTOK * 512];
    const float2 p = lane < NCHUNK ? A.St[((size_t)h * NTOK + t) * NCHUNK + lane] : make_float2(-INFINITY, 0.f);
    float M = p.x;
#pragma unroll
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float e = p.y == 0.f ? 0.f : expf(p.x - M);
    float L = e * p.y;
#pragma unroll
    for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
    const float w = e / L;
    float4 s0 = make_float4(0, 0, 0, 0);
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      const float wc = __shfl_sync(0xffffffffu, w, c);
      s0.x += wc * v[c].x; s0.y += wc * v[c].y; s0.z += wc * v[c].z; s0.w += wc * v[c].w;
    }
    *(uint2*)(A.Xo + (size_t)t * 2048 + h * 256 + half * 128 + lane * 4) = make_uint2(pack_bf2(s0.x, s0.y), pack_bf2(s0.z, s0.w));
  }
  zero_buf(A.Yd, NTOK * 1024);
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__device__ void p_gelu(const Args& A) {
  const size_t gt = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  for (size_t i = gt; i < NTOK * 4096; i += nt) {
    const size_t t = i >> 12, c = i & 4095;
    const float g = bf16r(A.Ygu[t * 8192 + c]), u = bf16r(A.Ygu[t * 8192 + 4096 + c]);
    A.X2[i] = __float2bfloat16(bf16r(gelu_tanh(g)) * u);
  }
}

__device__ void p_action_out(const Args& A) {
  const int lane = threadIdx.x & 31, gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), nw = gridDim.x * (blockDim.x >> 5);
  const float dt = -0.1f;
  for (int p = gw; p < NTOK * 32; p += nw) {
    const int t = p >> 5, j = p & 31;
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int i = (lane + 32 * q) * 4;
      const float4 w = *(const float4*)(A.action_out_w + (size_t)j * HID + i);
      const uint2 xu = *(const uint2*)(A.X1 + t * HID + i);
      const bf16* xb = (const bf16*)&xu;
      acc += w.x * __bfloat162float(xb[0]); acc += w.y * __bfloat162float(xb[1]);
      acc += w.z * __bfloat162float(xb[2]); acc += w.w * __bfloat162float(xb[3]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) { const float v = acc + A.action_out_b[j]; A.out[t * 32 + j] = __fadd_rn(A.out[t * 32 + j], __fmul_rn(dt, v)); }
  }
}

// ------------------------------------------------------------------ main kernel
__global__ void __launch_bounds__(256, 1) pi05_kernel(Args A) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint64_t* mb = (uint64_t*)(base + (size_t)STAGES * (128 * KC * 2 + NP * KC * 2));
  uint32_t* tslot = (uint32_t*)(mb + STAGES + 1);
  __shared__ float red[8];
  const int tid = threadIdx.x, warp = tid >> 5;
  if (tid == 0) { for (int s = 0; s <= STAGES; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(mb + s))); asm volatile("fence.mbarrier_init.release.cluster;"); }
  if (warp == 0) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" :: "r"(su32(tslot))); asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
  tc_sync();
  Tc tc; tc.base = base; tc.mb = mb; tc.tmem = *tslot; tc.J0 = 0; tc.dph = 0;
  unsigned epoch = A.epoch0;
#ifdef MEGABENCH_TIMELINE
  unsigned tlk = 0;
  if (blockIdx.x == 0 && tid == 0) A.tl[0] = (uint64_t)gridDim.x | ((uint64_t)1 << 32);
#endif
  // number of valid prefix tokens
  int cnt = 0;
  for (int i = tid; i < NPREFIX; i += 256) cnt += A.mask[i] ? 1 : 0;
  const int nvalid = (int)block_sum((float)cnt, red);
  bf16* smb = (bf16*)base;
  float* smf = (float*)base;

  // ---- hoisted t-only work
  TLBEGIN(EV_TEMB);
  p_temb(A);
  p_rope_table(A, nvalid);
  GBAR(EV_TEMB);
  TLBEGIN(EV_TIME_IN);
  p_linear10(A.time_in_w, A.time_in_b, A.temb, A.hid1, smf);
  GBAR(EV_TIME_IN);
  TLBEGIN(EV_TIME_OUT);
  p_linear10(A.time_out_w, A.time_out_b, A.hid1, A.cond, smf);
  GBAR(EV_TIME_OUT);
  TLBEGIN(EV_MOD);
  p_mod(A, smf);
  GBAR(EV_MOD);

  for (int step = 0; step < NSTEP; ++step) {
    for (int l = 0; l < NLAYER; ++l) {
      // ---- row norm (attn)
      TLBEGIN(EV_NORM_ATTN);
      if (blockIdx.x < NTOK) {
        if (l == 0) row_norm(A, step, blockIdx.x, 0, 0, 0, nullptr, red);
        else row_norm(A, step, blockIdx.x, 2, 2 * (l - 1) + 1, 2 * l, A.Yd, red);
      }
      zero_buf(A.Yqkv, NTOK * 2560); zero_buf(A.Yo, NTOK * 1024); zero_buf(A.Ygu, NTOK * 8192);
      GBAR(EV_NORM_ATTN);
      TLBEGIN(EV_QKV);
      { GemmDesc d; d.w[0] = A.wq + (size_t)l * 2048 * 1024; d.nt[0] = 16; d.w[1] = A.wk + (size_t)l * 256 * 1024; d.nt[1] = 2;
        d.w[2] = A.wv + (size_t)l * 256 * 1024; d.nt[2] = 2; d.K = 1024; d.N = 2560; d.X = A.X1; d.Y = A.Yqkv; gemm_phase(tc, d); }
      GBAR(EV_QKV);
      TLBEGIN(EV_ROPE);
      p_rope(A, nvalid);
      GBAR(EV_ROPE);
      TLBEGIN(EV_SCORES);
      attn_fused(A, l, smb);
      GBAR(EV_SCORES);
      TLBEGIN(EV_CONV);
      p_conv(A);
      GBAR(EV_CONV);
      TLBEGIN(EV_OPROJ);
      { GemmDesc d; d.w[0] = A.wo + (size_t)l * 1024 * 2048; d.nt[0] = 8; d.w[1] = d.w[2] = d.w[0]; d.nt[1] = d.nt[2] = 0;
        d.K = 2048; d.N = 1024; d.X = A.Xo; d.Y = A.Yo; gemm_phase(tc, d); }
      GBAR(EV_OPROJ);
      TLBEGIN(EV_NORM_FFN);
      if (blockIdx.x < NTOK) row_norm(A, step, blockIdx.x, 1, 2 * l, 2 * l + 1, A.Yo, red);
      GBAR(EV_NORM_FFN);
      TLBEGIN(EV_GATEUP);
      { GemmDesc d; d.w[0] = A.wg + (size_t)l * 4096 * 1024; d.nt[0] = 32; d.w[1] = A.wu + (size_t)l * 4096 * 1024; d.nt[1] = 32;
        d.w[2] = d.w[1]; d.K = 1024; d.N = 8192; d.nt[2] = 0; d.X = A.X1; d.Y = A.Ygu; gemm_phase<128>(tc, d); }  // 512 iters = 4 x 128 CTAs
      GBAR(EV_GATEUP);
      TLBEGIN(EV_GELU);
      p_gelu(A);
      GBAR(EV_GELU);
      TLBEGIN(EV_DOWN);
      { GemmDesc d; d.w[0] = A.wd + (size_t)l * 1024 * 4096; d.nt[0] = 8; d.w[1] = d.w[2] = d.w[0]; d.nt[1] = d.nt[2] = 0;
        d.K = 4096; d.N = 1024; d.X = A.X2; d.Y = A.Yd; gemm_phase(tc, d); }
      GBAR(EV_DOWN);
    }
    TLBEGIN(EV_FINAL_NORM);
    if (blockIdx.x < NTOK) row_norm(A, step, blockIdx.x, 2, 2 * 17 + 1, 36, A.Yd, red);
    GBAR(EV_FINAL_NORM);
    TLBEGIN(EV_ACTION_OUT);
    p_action_out(A);
    GBAR(EV_ACTION_OUT);
  }
  if (blockIdx.x == 0 && tid == 0) A.bar[1] = epoch;
  tc_sync();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(tc.tmem));
}

extern "C" int pi05_nbar() { return NBAR; }

// pointer table order must match submission.py
extern "C" int pi05_launch(void** p, int grid, unsigned epoch0, void* stream) {
  Args A;
  int i = 0;
  A.action_in_w = (const float*)p[i++]; A.action_in_b = (const float*)p[i++];
  A.time_in_w = (const float*)p[i++]; A.time_in_b = (const float*)p[i++];
  A.time_out_w = (const float*)p[i++]; A.time_out_b = (const float*)p[i++];
  A.action_out_w = (const float*)p[i++]; A.action_out_b = (const float*)p[i++];
  A.wq = (const bf16*)p[i++]; A.wk = (const bf16*)p[i++]; A.wv = (const bf16*)p[i++]; A.wo = (const bf16*)p[i++];
  A.wg = (const bf16*)p[i++]; A.wu = (const bf16*)p[i++]; A.wd = (const bf16*)p[i++];
  A.attn_w = (const float*)p[i++]; A.attn_b = (const float*)p[i++];
  A.ffn_w = (const float*)p[i++]; A.ffn_b = (const float*)p[i++];
  A.final_w = (const float*)p[i++]; A.final_b = (const float*)p[i++];
  A.noise = (const float*)p[i++];
  A.pk = (const bf16*)p[i++]; A.pv = (const bf16*)p[i++]; A.mask = (const bool*)p[i++];
  A.out = (float*)p[i++];
  A.temb = (float*)p[i++]; A.hid1 = (float*)p[i++]; A.cond = (float*)p[i++]; A.mod = (float*)p[i++];
  A.Yqkv = (float*)p[i++]; A.Yo = (float*)p[i++]; A.Ygu = (float*)p[i++]; A.Yd = (float*)p[i++]; A.Opart = (float*)p[i++];
  A.inv = (const float*)p[i++];
  A.h = (bf16*)p[i++]; A.X1 = (bf16*)p[i++]; A.Xo = (bf16*)p[i++]; A.X2 = (bf16*)p[i++]; A.St = (float2*)p[i++];
  A.bar = (unsigned*)p[i++];
  A.epoch0 = epoch0;
  A.cs = (float2*)p[i++]; A.Qr = (bf16*)p[i++]; A.Kr = (bf16*)p[i++];
#ifdef MEGABENCH_TIMELINE
  A.tl = (uint64_t*)p[i++]; A.tlw = (unsigned long long)(uintptr_t)p[i++];
#endif
  const int smem = 1024 + STAGES * (128 * KC * 2 + NP * KC * 2) + 256;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(pi05_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  void* args[] = {&A};
  return (int)cudaLaunchCooperativeKernel((void*)pi05_kernel, dim3(grid), dim3(256), args, smem, (cudaStream_t)stream);
}
