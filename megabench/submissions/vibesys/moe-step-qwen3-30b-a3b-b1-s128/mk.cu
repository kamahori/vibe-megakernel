// Persistent single-launch Qwen3-30B-A3B decode step (batch 1, ctx 128). sm_100a.
// Round-1 baseline: all phases SIMT FFMA warp-per-row (see progress/microbench/build/round-0001.md).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "timeline_probe.cuh"

typedef __nv_bfloat16 bf16;
constexpr int NB = 148, NT = 512, NW = NT / 16 * 1;   // NW unused
constexpr int WPB = NT / 32;                          // warps per block = 16
constexpr int H = 2048, QH = 32, KVH = 4, D = 128, NE = 128, TK = 8, I = 768, V = 151936, L = 48, CTX = 128;
constexpr int QKVN = 4096 + 512 + 512;

struct P {
  const bf16 *ln1, *ln2, *fnorm, *wq, *wk, *wv, *wo, *qn, *kn, *wrt, *wg, *wu, *wd, *embed, *lm, *kc, *vc;
  const long long* token;
  const float *cosb, *sinb;
  bf16 *kw, *vw; float* logits; long long* next;
  unsigned char* scratch;   // zero-initialised, persistent
  uint64_t* tl; uint64_t tl_words; int nblocks;
};
// scratch layout
constexpr size_t O_BAR = 0, O_QKV = 64, O_ATT = O_QKV + QKVN * 2, O_X = O_ATT + 4096 * 2, O_HB = O_X + H * 2,
                 O_RL = O_HB + TK * I * 2, O_PV = O_RL + NE * 4, O_PI = O_PV + 256 * 4;
// fp32 stream-K red.add targets (tcgen05 GEMV phases); each zeroed in a phase before its producer
constexpr size_t O_QKVF = O_PI + 256 * 4 + 64, O_YO = O_QKVF + QKVN * 4, O_GU = O_YO + H * 4, O_END = O_GU + TK * 2 * I * 4;
// E2 down: per-expert outputs y_e[TK][H] as bf16 rb(full K=768 fp32 sum) (one CTA per tile, plain stores; no zeroing),
// and the second residual buffer: X_l lives in
// xbuf[l & 1] (O_X for even l, O_X2 for odd l), written by its row owner in layer l's ln1 prologue (E2|E3E6 fixup).
constexpr size_t O_YE = O_END, O_X2 = O_YE + TK * H * 2, O_END2 = O_X2 + H * 2;

__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ldb(const bf16* p) { return __bfloat162float(__ldcg(p)); }
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ uint4 ldnc(const bf16* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void fma8(float& a, const uint4& w, const float* xs) {
  float4 x0 = *reinterpret_cast<const float4*>(xs), x1 = *reinterpret_cast<const float4*>(xs + 4);
  const uint32_t u[4] = {w.x, w.y, w.z, w.w};
  float f[8];
#pragma unroll
  for (int i = 0; i < 4; i++) { f[2 * i] = __uint_as_float(u[i] << 16); f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u); }
  a = fmaf(f[0], x0.x, a); a = fmaf(f[1], x0.y, a); a = fmaf(f[2], x0.z, a); a = fmaf(f[3], x0.w, a);
  a = fmaf(f[4], x1.x, a); a = fmaf(f[5], x1.y, a); a = fmaf(f[6], x1.z, a); a = fmaf(f[7], x1.w, a);
}
template <int K>
__device__ __forceinline__ float dot_row(const bf16* __restrict__ w, const float* xs, int lane) {
  constexpr int N = K / 256;
  uint4 v[N];
#pragma unroll
  for (int i = 0; i < N; i++) v[i] = ldnc(w + (i * 32 + lane) * 8);
  float a = 0.f;
#pragma unroll
  for (int i = 0; i < N; i++) fma8(a, v[i], xs + (i * 32 + lane) * 8);
  return wsum(a);
}

__device__ __forceinline__ void gsync(unsigned long long* ctr, int nb) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned long long old = atomicAdd(ctr, 1ull);
    unsigned long long target = (old / nb + 1) * nb;
    unsigned long long cur;
    do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(ctr) : "memory"); } while (cur < target);
    __threadfence();
  }
  __syncthreads();
}

// Grid barrier / E9 arrival counter on a monotone u64 (never reset; each use adds exactly NB): release on
// arrival, acquire on the spin. Same arrive/wait shape as gsync without its two full fences (MEMBAR.SC + CCTL.IVALL).
__device__ __forceinline__ void arrive_wait(unsigned long long* ctr, int nb) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long old, cur;
    asm volatile("atom.add.release.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(ctr) : "memory");
    const unsigned long long target = (old / nb + 1) * nb;
    do { asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(ctr) : "memory"); } while (cur < target);
  }
  __syncthreads();
}

__device__ float bsum(float v, float* red) {   // block sum, all threads get result
  v = wsum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < WPB; i++) s += red[i];
  return s;
}
// xn = bf16(bf16(x*rsqrt(mean(x^2)+eps)) * w) as floats in smem
__device__ void rms_smem(const bf16* x, const bf16* w, float* out, float* red) {
  float v[4], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) { v[i] = ldb(x + threadIdx.x + i * NT); ss += v[i] * v[i]; }
  ss = bsum(ss, red);
  float r = rsqrtf(ss * (1.f / H) + 1e-6f);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int d = threadIdx.x + i * NT;
    out[d] = rb(rb(v[i] * r) * __bfloat162float(w[d]));
  }
  __syncthreads();
}
// E2|E3E6 fixup, vector layout d = 4*threadIdx.x + k: pre[k] = rb(X_{l-1}[d] + rb(yo[d])) (loaded at E2 start, before
// the barrier); y_e = bf16 rb(down outputs). E2's per-expert rounding contract: acc = fmaf(rb(a_s[d]), prob[s], acc) in
// s order; x = bf16(pre + rb(acc)). X_l goes to xs (smem float[H]; caller syncs and reads it back in its own layout, so
// the rms sum order is unchanged); the chunk owner (threadIdx.x % NB == blockIdx.x) publishes X_l to xw (if non-null).
__device__ __forceinline__ void fixup_x(const float pre[4], const bf16* ye, const float* prob, float* xs, bf16* xw) {
  const int d0 = threadIdx.x * 4;
  uint2 e[TK];
#pragma unroll
  for (int s = 0; s < TK; s++) e[s] = __ldcg(reinterpret_cast<const uint2*>(ye + s * H + d0));
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < TK; s++) {
    const float ps = prob[s];
    acc[0] = fmaf(__uint_as_float(e[s].x << 16), ps, acc[0]); acc[1] = fmaf(__uint_as_float(e[s].x & 0xffff0000u), ps, acc[1]);
    acc[2] = fmaf(__uint_as_float(e[s].y << 16), ps, acc[2]); acc[3] = fmaf(__uint_as_float(e[s].y & 0xffff0000u), ps, acc[3]);
  }
  float x[4];
#pragma unroll
  for (int k = 0; k < 4; k++) x[k] = rb(pre[k] + rb(acc[k]));
  *reinterpret_cast<float4*>(xs + d0) = make_float4(x[0], x[1], x[2], x[3]);
  if (xw && threadIdx.x % NB == blockIdx.x) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(x[0], x[1]), hi = __floats2bfloat162_rn(x[2], x[3]);
    uint2 u; u.x = *reinterpret_cast<uint32_t*>(&lo); u.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(xw + d0) = u;
  }
}
// pre[k] = rb(X[d] + rb(yo[d])), d = 4*threadIdx.x + k (the residual of E4, final once E9 has read it)
__device__ __forceinline__ void fixup_pre(float pre[4], const bf16* x, const float* yo) {
  const int d0 = threadIdx.x * 4;
  const uint2 xv = __ldcg(reinterpret_cast<const uint2*>(x + d0));
  const float4 y = __ldcg(reinterpret_cast<const float4*>(yo + d0));
  pre[0] = rb(__uint_as_float(xv.x << 16) + rb(y.x)); pre[1] = rb(__uint_as_float(xv.x & 0xffff0000u) + rb(y.y));
  pre[2] = rb(__uint_as_float(xv.y << 16) + rb(y.z)); pre[3] = rb(__uint_as_float(xv.y & 0xffff0000u) + rb(y.w));
}


// ================= tcgen05 swap-AB GEMV engine (E3E6 QKV, E4 wo, E2 down) =================
// W tile 128 rows x 128 k (SWIZZLE_128B K-major, cp.async ring of RS stages) = MMA A (M128);
// x = MMA B (N16) read straight from a contiguous bf16 copy in smem (SWIZZLE_NONE, LBO=16, SBO=128):
// only column 0 of the accumulator is used, rows 1..15 of B are don't-care.
// Stream-K over 148 CTAs of (tiles x K/128) units, red.add.f32 of each segment into fp32 scratch.
#ifndef XPF
#define XPF 1   // 1: issue the next GEMV phase's first RS-1 stages before the preceding barrier
#endif
#ifndef RS
#define RS 3
#endif
constexpr int KC = 128, STAGE_B = 128 * KC * 2;
// XB region: x of QKV (4 KB) / wo (8 KB) / hb of down (12 KB) + 512 B tail of the don't-care B rows; also the 16 KB
// attention PV reduce buffer and lm_head's x/partials (both time-disjoint from the GEMV phases' B operand).
constexpr size_t XB_SZ = 16384 + 512;
constexpr size_t XB_OFF = (size_t)RS * STAGE_B, MB_OFF = XB_OFF + XB_SZ, DSMEM = 1024 + MB_OFF + 8 * (RS + 2);
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ uint64_t desc_a(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | (uint64_t)1 << 16 | (uint64_t)(1024 >> 4) << 32 | (uint64_t)1 << 46 | (uint64_t)2 << 61;
}
__device__ __forceinline__ uint64_t desc_b(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | (uint64_t)(16 >> 4) << 16 | (uint64_t)(128 >> 4) << 32 | (uint64_t)1 << 46;
}
constexpr uint32_t IDESC = 1u << 4 | 1u << 7 | 1u << 10 | (uint32_t)(16 >> 3) << 17 | (uint32_t)(128 >> 4) << 24;
__device__ __forceinline__ void mma_ss(uint32_t dt, uint64_t ad, uint64_t bd, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(dt), "l"(ad), "l"(bd), "r"(IDESC), "r"(acc));
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" :: "r"(mbar), "r"(ph) : "memory");
}
__device__ __forceinline__ void tc_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}
enum { G_QKV, G_WO, G_GU, G_DOWN };
struct Ring {
  uint32_t a0, xb, mb, tmem;   // smem addresses, tmem base
  int J0; uint32_t dph, seg;
};
struct RingSt { uint32_t tmem; int J0; uint32_t dph, seg; };   // persistent ring state, in smem
extern __shared__ uint8_t dsm_raw[];
__device__ __forceinline__ uint32_t dsm_base() { return (su32(dsm_raw) + 1023u) & ~1023u; }
__device__ __forceinline__ Ring ring_load(const RingSt* st) {
  Ring R; const uint32_t b = dsm_base();
  R.a0 = b; R.xb = b + (uint32_t)XB_OFF; R.mb = b + (uint32_t)MB_OFF;
  R.tmem = st->tmem; R.J0 = st->J0; R.dph = st->dph; R.seg = st->seg;
  return R;
}
__device__ __forceinline__ bf16* xb_smem() {
  return reinterpret_cast<bf16*>(((uintptr_t)dsm_raw + 1023) & ~(uintptr_t)1023) + XB_OFF / 2;
}
struct GPh {   // one GEMV phase as seen by this CTA
  int kind, K, KI, g0, n; const bf16 *w0, *w1, *w2; const int* sel; float* out;
};
__device__ __forceinline__ GPh gph(int kind, const bf16* w0, const bf16* w1, const bf16* w2, const int* sel, float* out) {
  GPh g; g.kind = kind; g.w0 = w0; g.w1 = w1; g.w2 = w2; g.sel = sel; g.out = out;
  g.K = kind == G_WO ? 4096 : kind == G_DOWN ? I : H; g.KI = g.K / KC;
  const int tiles = kind == G_QKV ? QKVN / 128 : kind == G_WO ? H / 128 : kind == G_DOWN ? TK * H / 128 : TK * 2 * I / 128;
  const long long T = (long long)tiles * g.KI;
  g.g0 = (int)((long long)blockIdx.x * T / NB); g.n = (int)((long long)(blockIdx.x + 1) * T / NB) - g.g0;
  if (kind == G_DOWN) {   // whole tiles: CTA b < 128 owns tile b (all 6 k-iters; the busiest stream-K CTA also has 6)
    g.g0 = blockIdx.x < tiles ? (int)blockIdx.x * g.KI : 0; g.n = blockIdx.x < tiles ? g.KI : 0;
  }
  return g;
}
__device__ __forceinline__ const bf16* tile_ptr(const GPh& g, int tile) {
  const int r0 = tile * 128;
  if (g.kind == G_QKV)
    return r0 < 4096 ? g.w0 + (size_t)r0 * H : r0 < 4608 ? g.w1 + (size_t)(r0 - 4096) * H : g.w2 + (size_t)(r0 - 4608) * H;
  if (g.kind == G_WO) return g.w0 + (size_t)r0 * 4096;
  if (g.kind == G_DOWN) return g.w0 + ((size_t)g.sel[tile >> 4] * H + (tile & 15) * 128) * I;   // tile = s*16 + row block
  const int s = tile / 12, w2 = (tile % 12) / 6, rr = (tile % 6) * 128;
  return (w2 ? g.w1 : g.w0) + ((size_t)g.sel[s] * I + rr) * H;
}
__device__ __forceinline__ void g_issue(const GPh& g, const Ring& R, int j) {
  if (j < g.n) {
    const int u = g.g0 + j, tile = u / g.KI, kc = u - tile * g.KI, s = (R.J0 + j) % RS;
    const bf16* Wt = tile_ptr(g, tile) + (size_t)kc * KC;
    const uint32_t st = R.a0 + s * STAGE_B;
#pragma unroll
    for (int q = 0; q < 2048 / NT; ++q) {
      const int i = threadIdx.x + NT * q, row = i >> 4, c16 = i & 15, atom = c16 >> 3, c = c16 & 7;
      cp16(st + atom * 16384 + row * 128 + ((c ^ (row & 7)) << 4), Wt + (size_t)row * g.K + c16 * 8);
    }
  }
  cp_commit();
}
// Out of line, scalar arguments only: keeps the engine's registers out of the other phases' allocation.
__device__ __noinline__ void g_prefetch(int kind, const bf16* w0, const bf16* w1, const bf16* w2, const int* sel, float* out,
                                        const RingSt* st) {
  const GPh g = gph(kind, w0, w1, w2, sel, out);
  const Ring R = ring_load(st);
  for (int j = 0; j < RS - 1; ++j) g_issue(g, R, j);
}
// Runs the phase; the first RS-1 units must already be issued (g_prefetch). x (bf16, g.K) must be in R.xb.
__device__ __noinline__ void g_run(int kind, const bf16* w0, const bf16* w1, const bf16* w2, const int* sel, float* out,
                                   RingSt* st) {
  const GPh g = gph(kind, w0, w1, w2, sel, out);
  Ring R = ring_load(st);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // x written with generic stores
  for (int j = 0; j < g.n; ++j) {
    const int J = R.J0 + j;
    const int u = g.g0 + j, tile = u / g.KI, kc = u - tile * g.KI, s = J % RS;
    cp_wait<RS - 2>();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    tc_sync();
    const bool seg_start = (j == 0) || kc == 0, seg_end = (kc == g.KI - 1) || (j == g.n - 1);
    const uint32_t dt = R.tmem + (R.seg & 1) * 16;
    const uint32_t boff = g.kind == G_DOWN ? (uint32_t)(tile >> 4) * (I * 2) : 0u;   // down: B = hb segment of expert slot s
    if (threadIdx.x == 0) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint64_t ad = desc_a(R.a0 + s * STAGE_B + (kk >> 2) * 16384 + (kk & 3) * 32);
        const uint64_t bd = desc_b(R.xb + boff + kc * 256 + kk * 32);
        mma_ss(dt, ad, bd, (seg_start && kk == 0) ? 0u : 1u);
      }
      tc_commit(R.mb + 8 * s);
      if (seg_end) tc_commit(R.mb + 8 * RS);
    }
    if (j + RS - 1 < g.n && J >= 1) mbar_wait(R.mb + 8 * (uint32_t)((J - 1) % RS), (uint32_t)(((J - 1) / RS) & 1));
    g_issue(g, R, j + RS - 1);
    if (seg_end) {
      if (warp < 4) {
        mbar_wait(R.mb + 8 * RS, R.dph);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        uint32_t r;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r) : "r"(dt + ((uint32_t)(warp * 32) << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        const float v = __uint_as_float(r);   // row warp*32+lane; pack 4 rows per lane for red.v4
        const float a0 = __shfl_sync(0xffffffffu, v, (lane * 4) & 31), a1 = __shfl_sync(0xffffffffu, v, (lane * 4 + 1) & 31),
                    a2 = __shfl_sync(0xffffffffu, v, (lane * 4 + 2) & 31), a3 = __shfl_sync(0xffffffffu, v, (lane * 4 + 3) & 31);
        if (g.kind == G_DOWN)   // full-K sum of one expert row: store rb() as bf16 (the E2 rounding point)
          reinterpret_cast<bf16*>(g.out)[tile * 128 + warp * 32 + lane] = __float2bfloat16_rn(v);
        else if (lane < 8)
          asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(g.out + tile * 128 + warp * 32 + lane * 4),
                       "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
      }
      R.dph ^= 1; R.seg++;
    }
  }
  if (threadIdx.x == 0) { st->J0 = R.J0 + g.n; st->dph = R.dph; st->seg = R.seg; }
  tc_sync();
}
__device__ __forceinline__ float qkvv(const float* q, int i) { return rb(__ldcg(q + i)); }

enum { EV_QKV, EV_ATT, EV_WO, EV_ROUT, EV_GU, EV_DOWN, EV_LM, EV_BAR };

__global__ void __launch_bounds__(NT, 1) mk_kernel(P p) {
  __shared__ __align__(16) float xn[H];
  __shared__ __align__(16) float qs[D], ks[D];
  __shared__ float sc[132], red[WPB];
  __shared__ int sel[TK]; __shared__ float prob[TK];
  __shared__ float pbv[WPB]; __shared__ int pbi[WPB];
  MB_TIMELINE_INIT(p.tl, p.tl_words, 0, 1, threadIdx.x == 0);
  unsigned long long* bar = reinterpret_cast<unsigned long long*>(p.scratch + O_BAR);
  bf16* qkv = reinterpret_cast<bf16*>(p.scratch + O_QKV);
  bf16* att = reinterpret_cast<bf16*>(p.scratch + O_ATT);
  bf16* xb = reinterpret_cast<bf16*>(p.scratch + O_X);
  bf16* xb2 = reinterpret_cast<bf16*>(p.scratch + O_X2);
  bf16* ye = reinterpret_cast<bf16*>(p.scratch + O_YE);
  bf16* hb = reinterpret_cast<bf16*>(p.scratch + O_HB);
  float* rl = reinterpret_cast<float*>(p.scratch + O_RL);
  float* pv = reinterpret_cast<float*>(p.scratch + O_PV);
  int* pi = reinterpret_cast<int*>(p.scratch + O_PI);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = blockIdx.x * WPB + warp, NGW = NB * WPB;
  const bf16* x0 = p.embed + (size_t)(*p.token) * H;
  __shared__ RingSt rst;
  if (threadIdx.x == 0) {
    const uint32_t mb = dsm_base() + (uint32_t)MB_OFF;
    for (int i = 0; i <= RS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb + 8 * i));
    asm volatile("fence.mbarrier_init.release.cluster;");
    rst.J0 = 0; rst.dph = 0; rst.seg = 0;
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(su32(&rst.tmem)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_sync();
  if (XPF) g_prefetch(G_QKV, p.wq + (size_t)(0) * 4096 * H, p.wk + (size_t)(0) * 512 * H, p.wv + (size_t)(0) * 512 * H, sel, reinterpret_cast<float*>(p.scratch + O_QKVF), &rst);

  float pre[4] = {0.f, 0.f, 0.f, 0.f};   // E2|E3E6 fixup: rb(X_l + rb(yo)) of the current layer, set at E2 start
#define BARRIER() do { MB_TIMELINE_BEGIN(EV_BAR); gsync(bar, NB); MB_TIMELINE_END(EV_BAR); } while (0)

  for (int l = 0; l < L; l++) {
    // ---- ln1 + QKV ----
    MB_TIMELINE_BEGIN(EV_QKV);
    {
      bf16* xbs = xb_smem();
      if (!XPF) g_prefetch(G_QKV, p.wq + (size_t)(l) * 4096 * H, p.wk + (size_t)(l) * 512 * H, p.wv + (size_t)(l) * 512 * H, sel, reinterpret_cast<float*>(p.scratch + O_QKVF), &rst);
      // X_l -> xbuf[l & 1]: layer 0 = embed row; else the E2|E3E6 fixup of layer l-1
      bf16* xw = (l & 1) ? xb2 : xb;
      float v[4], ss = 0.f;
      if (l == 0) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int d = threadIdx.x + i * NT;
          v[i] = ldb(x0 + d);
          if (d % NB == (int)blockIdx.x) xw[d] = x0[d];
        }
      } else {
        fixup_x(pre, ye, prob, xn, xw);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; i++) v[i] = xn[threadIdx.x + i * NT];
      }
#pragma unroll
      for (int i = 0; i < 4; i++) ss += v[i] * v[i];
      ss = bsum(ss, red);
      float r = rsqrtf(ss * (1.f / H) + 1e-6f);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int d = threadIdx.x + i * NT;
        xbs[d] = __float2bfloat16_rn(rb(v[i] * r) * __bfloat162float(p.ln1[(size_t)l * H + d]));
      }
      __syncthreads();
      g_run(G_QKV, p.wq + (size_t)(l) * 4096 * H, p.wk + (size_t)(l) * 512 * H, p.wv + (size_t)(l) * 512 * H, sel, reinterpret_cast<float*>(p.scratch + O_QKVF), &rst);
      // zero the fp32 target of wo (yo; last read at the previous E2 start and E9, next written by this layer's wo)
      if (blockIdx.x >= NB - 4) reinterpret_cast<float*>(p.scratch + O_YO)[(blockIdx.x - (NB - 4)) * NT + threadIdx.x] = 0.f;
      if (XPF) g_prefetch(G_WO, p.wo + (size_t)(l) * H * 4096, nullptr, nullptr, sel, reinterpret_cast<float*>(p.scratch + O_YO), &rst);
    }
    MB_TIMELINE_END(EV_QKV);
    BARRIER();
    // ---- attention (CTA per q head) ----
    MB_TIMELINE_BEGIN(EV_ATT);
    if (blockIdx.x < QH) {
      // D1b: one CTA per q head, parallel inner loop (r4_att arm B). Rounding points as before.
      const int h = blockIdx.x, g = h >> 3;
      const float* qkvf = reinterpret_cast<const float*>(p.scratch + O_QKVF);
      if (warp < 2) {
        const bool isq = warp == 0;
        const int src = isq ? h * D : 4096 + g * D;
        const bf16* nw = (isq ? p.qn : p.kn) + (size_t)l * D;
        float v[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) { v[i] = qkvv(qkvf, src + lane + 32 * i); ss += v[i] * v[i]; }
        ss = wsum(ss);
        float r = rsqrtf(ss * (1.f / D) + 1e-6f);
#pragma unroll
        for (int i = 0; i < 4; i++) v[i] = rb(rb(v[i] * r) * __bfloat162float(nw[lane + 32 * i]));
        float* o = isq ? qs : ks;
#pragma unroll
        for (int i = 0; i < 2; i++) {
          int dl = lane + 32 * i, dh = dl + 64;
          float cl = rb(p.cosb[dl]), sl = rb(p.sinb[dl]), ch = rb(p.cosb[dh]), sh = rb(p.sinb[dh]);
          float lo = v[i], hi = v[i + 2];
          o[dl] = rb(rb(lo * cl) + rb(-hi * sl));
          o[dh] = rb(rb(hi * ch) + rb(lo * sh));
        }
      }
      __syncthreads();
      // scores: half-warp per position (2 per warp), 16 B of K per lane, all K loads issued before any FMA.
      const int hw = lane >> 4, hl = lane & 15;
      const bf16* kl = p.kc + (size_t)l * CTX * KVH * D + g * D;
      uint4 u[4];
#pragma unroll
      for (int j = 0; j < 4; j++) u[j] = __ldg(reinterpret_cast<const uint4*>(kl + (size_t)(j * 32 + warp * 2 + hw) * KVH * D) + hl);
      const float4 q0 = *reinterpret_cast<const float4*>(qs + hl * 8), q1 = *reinterpret_cast<const float4*>(qs + hl * 8 + 4);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float a = __uint_as_float(u[j].x << 16) * q0.x + __uint_as_float(u[j].x & 0xffff0000u) * q0.y +
                  __uint_as_float(u[j].y << 16) * q0.z + __uint_as_float(u[j].y & 0xffff0000u) * q0.w +
                  __uint_as_float(u[j].z << 16) * q1.x + __uint_as_float(u[j].z & 0xffff0000u) * q1.y +
                  __uint_as_float(u[j].w << 16) * q1.z + __uint_as_float(u[j].w & 0xffff0000u) * q1.w;
#pragma unroll
        for (int o = 8; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (hl == 0) sc[j * 32 + warp * 2 + hw] = rb(rb(a) * 0.08838834764831845f);
      }
      if (warp == 0) {   // position CTX: the current token's k (ks)
        const float4 k0 = *reinterpret_cast<const float4*>(ks + hl * 8), k1 = *reinterpret_cast<const float4*>(ks + hl * 8 + 4);
        float a = hw ? 0.f : k0.x * q0.x + k0.y * q0.y + k0.z * q0.z + k0.w * q0.w + k1.x * q1.x + k1.y * q1.y + k1.z * q1.z + k1.w * q1.w;
#pragma unroll
        for (int o = 8; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) sc[CTX] = rb(rb(a) * 0.08838834764831845f);
      }
      // PV operands: issued before the score barrier so their latency overlaps the softmax.
      const int c = threadIdx.x & 15, pg = threadIdx.x >> 4;
      const bf16* vl = p.vc + (size_t)l * CTX * KVH * D + g * D + c * 8;
      uint4 vv[4];
#pragma unroll
      for (int j = 0; j < 4; j++) vv[j] = __ldg(reinterpret_cast<const uint4*>(vl + (size_t)(pg + 32 * j) * KVH * D));
      __syncthreads();
      // softmax: every warp computes m and s redundantly (no extra barrier)
      float m = -INFINITY;
      for (int i = lane; i < CTX + 1; i += 32) m = fmaxf(m, sc[i]);
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      float s = 0.f;
      for (int i = lane; i < CTX + 1; i += 32) s += expf(sc[i] - m);
      s = wsum(s);
      // PV: thread = (8-dim chunk c, position group pg), independent accumulators
      float acc[8];
#pragma unroll
      for (int k = 0; k < 8; k++) acc[k] = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float pr = rb(expf(sc[pg + 32 * j] - m) / s);
        acc[0] = fmaf(pr, __uint_as_float(vv[j].x << 16), acc[0]); acc[1] = fmaf(pr, __uint_as_float(vv[j].x & 0xffff0000u), acc[1]);
        acc[2] = fmaf(pr, __uint_as_float(vv[j].y << 16), acc[2]); acc[3] = fmaf(pr, __uint_as_float(vv[j].y & 0xffff0000u), acc[3]);
        acc[4] = fmaf(pr, __uint_as_float(vv[j].z << 16), acc[4]); acc[5] = fmaf(pr, __uint_as_float(vv[j].z & 0xffff0000u), acc[5]);
        acc[6] = fmaf(pr, __uint_as_float(vv[j].w << 16), acc[6]); acc[7] = fmaf(pr, __uint_as_float(vv[j].w & 0xffff0000u), acc[7]);
      }
      if (pg == 0) {
        const float pr = rb(expf(sc[CTX] - m) / s);
#pragma unroll
        for (int k = 0; k < 8; k++) acc[k] = fmaf(pr, qkvv(qkvf, 4608 + g * D + c * 8 + k), acc[k]);
      }
      float* ored32 = reinterpret_cast<float*>(xb_smem());   // [32][D] in the XB region (idle during attention)
#pragma unroll
      for (int k = 0; k < 8; k += 4)
        *reinterpret_cast<float4*>(ored32 + pg * D + c * 8 + k) = make_float4(acc[k], acc[k + 1], acc[k + 2], acc[k + 3]);
      __syncthreads();
      if (threadIdx.x < D) {
        const int d = threadIdx.x;
        float o = 0.f;
#pragma unroll 8
        for (int j = 0; j < 32; j++) o += ored32[j * D + d];
        att[h * D + d] = __float2bfloat16_rn(o);
        if ((h & 7) == 0) {
          p.kw[((size_t)l * KVH + g) * D + d] = __float2bfloat16_rn(ks[d]);
          p.vw[((size_t)l * KVH + g) * D + d] = __float2bfloat16_rn(qkvv(qkvf, 4608 + g * D + d));
        }
      }
    }
    MB_TIMELINE_END(EV_ATT);
    BARRIER();
    // ---- wo + residual (in place on xb; layer 0 reads embed row) ----
    MB_TIMELINE_BEGIN(EV_WO);
    {
      // zero the fp32 qkv target for the next layer (read by attention before the last barrier)
      if (!XPF) g_prefetch(G_WO, p.wo + (size_t)(l) * H * 4096, nullptr, nullptr, sel, reinterpret_cast<float*>(p.scratch + O_YO), &rst);
      float* qkvf = reinterpret_cast<float*>(p.scratch + O_QKVF);
      bf16* xbs = xb_smem();
      for (int i = blockIdx.x * NT + threadIdx.x; i < QKVN; i += NB * NT) qkvf[i] = 0.f;
      reinterpret_cast<uint4*>(xbs)[threadIdx.x] = __ldcg(reinterpret_cast<const uint4*>(att) + threadIdx.x);
      __syncthreads();
      g_run(G_WO, p.wo + (size_t)(l) * H * 4096, nullptr, nullptr, sel, reinterpret_cast<float*>(p.scratch + O_YO), &rst);
    }
    MB_TIMELINE_END(EV_WO);
    MB_TIMELINE_BEGIN(EV_BAR); arrive_wait(bar, NB); MB_TIMELINE_END(EV_BAR);
    // ---- ln2 + router logits ----
    MB_TIMELINE_BEGIN(EV_ROUT);
    // router logits split-K ks=8: item = e*8 + ks (256-wide K slice), spread warp-major over all CTAs
    // (warps 0..6); the weight slice is loaded before ln2 so its latency overlaps the residual/ln2 pass.
    const int ritem = warp * NB + blockIdx.x;
    uint4 rw = make_uint4(0u, 0u, 0u, 0u);
    if (ritem < NE * 8) rw = ldnc(p.wrt + ((size_t)l * NE + (ritem >> 3)) * H + (ritem & 7) * 256 + lane * 8);
    {   // residual of E4 (wo) applied here with the reference rounding: x = bf16(x + bf16(wo))
      const float* yo = reinterpret_cast<const float*>(p.scratch + O_YO);
      float v[4], ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int d = threadIdx.x + i * NT;
        v[i] = rb(ldb(((l & 1) ? xb2 : xb) + d) + rb(__ldcg(yo + d))); ss += v[i] * v[i];
      }
      ss = bsum(ss, red);
      float r = rsqrtf(ss * (1.f / H) + 1e-6f);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int d = threadIdx.x + i * NT;
        xn[d] = rb(rb(v[i] * r) * __bfloat162float(p.ln2[(size_t)l * H + d]));
      }
      __syncthreads();
    }
    // split-K partials summed by red.add.u64 of int64 fixed point (partial * 2^40, exact for |partial| >= 2^-17,
    // range 2^23): integer addition is associative, so the logits are bit-identical whatever the arrival order.
    // rl2[l & 1] (u64[NE] in the unused bf16 qkv scratch slot) is zeroed one layer ahead: rl2[(l+1)&1] was last
    // read by layer l-1's top-8, which every CTA finished before layer l-1's gate_up barrier.
    unsigned long long* rp = reinterpret_cast<unsigned long long*>(p.scratch + O_QKV) + (l & 1) * NE;
    if (blockIdx.x == 0 && threadIdx.x < NE) reinterpret_cast<unsigned long long*>(p.scratch + O_QKV)[((l + 1) & 1) * NE + threadIdx.x] = 0ull;
    if (ritem < NE * 8) {
      float a = 0.f;
      fma8(a, rw, xn + (ritem & 7) * 256 + lane * 8);
      a = wsum(a);
      if (lane == 0)
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"(rp + (ritem >> 3)), "l"((unsigned long long)__float2ll_rn(a * 1099511627776.f)) : "memory");
    }
    MB_TIMELINE_END(EV_ROUT);
    MB_TIMELINE_BEGIN(EV_BAR);
    arrive_wait(reinterpret_cast<unsigned long long*>(p.scratch + O_BAR + 16), NB);
    MB_TIMELINE_END(EV_BAR);
    MB_TIMELINE_BEGIN(EV_ROUT);
    if (warp == 0) {   // top-8 + softmax, redundant in every CTA
      // keys: order-preserving u32 of v (+0.0f folds -0 into +0, as the == compare of the float version does);
      // max by redux.sync, ties to the lowest index (lowest lane, then lowest i); a taken key becomes 0.
      uint32_t k[4];
      const ulonglong2 q0 = __ldcg(reinterpret_cast<const ulonglong2*>(rp) + lane * 2), q1 = __ldcg(reinterpret_cast<const ulonglong2*>(rp) + lane * 2 + 1);
      const float lgv[4] = {(float)((double)(long long)q0.x * 9.094947017729282e-13), (float)((double)(long long)q0.y * 9.094947017729282e-13),
                            (float)((double)(long long)q1.x * 9.094947017729282e-13), (float)((double)(long long)q1.y * 9.094947017729282e-13)};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint32_t b = __float_as_uint(lgv[i] + 0.0f);
        k[i] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
      }
      float tv[TK]; int ti[TK];
#pragma unroll
      for (int s = 0; s < TK; s++) {
        const uint32_t m = max(max(k[0], k[1]), max(k[2], k[3]));
        const uint32_t M = __reduce_max_sync(0xffffffffu, m);
        const int w = __ffs(__ballot_sync(0xffffffffu, m == M)) - 1;
        const int i = k[0] == M ? 0 : k[1] == M ? 1 : k[2] == M ? 2 : 3;   // meaningful in lane w only
#pragma unroll
        for (int j = 0; j < 4; j++) if (lane == w && j == i) k[j] = 0u;
        ti[s] = __shfl_sync(0xffffffffu, lane * 4 + i, w);
        tv[s] = __uint_as_float((M & 0x80000000u) ? (M & 0x7fffffffu) : ~M);
      }
      float sum = 0.f, e8[TK];
#pragma unroll
      for (int s = 0; s < TK; s++) { e8[s] = expf(tv[s] - tv[0]); sum += e8[s]; }
#pragma unroll
      for (int s = 0; s < TK; s++) if (lane == s) { sel[s] = ti[s]; prob[s] = e8[s] / sum; }
    }
    __syncthreads();
    MB_TIMELINE_END(EV_ROUT);
    // ---- expert gate/up ----
    MB_TIMELINE_BEGIN(EV_GU);
    for (int it = gw; it < TK * I; it += NGW) {
      int s = it / I, j = it - s * I;
      size_t off = (((size_t)l * NE + sel[s]) * I + j) * H;
      float g = dot_row<H>(p.wg + off, xn, lane);
      float u = dot_row<H>(p.wu + off, xn, lane);
      if (lane == 0) {
        g = rb(g); u = rb(u);
        float sg = rb(g / (1.f + expf(-g)));
        hb[it] = __float2bfloat16_rn(sg * u);
      }
    }
    if (XPF) g_prefetch(G_DOWN, p.wd + (size_t)l * NE * H * I, nullptr, nullptr, sel, reinterpret_cast<float*>(ye), &rst);   // E1|E2: down ring fill
    MB_TIMELINE_END(EV_GU);
    BARRIER();
    // ---- expert down + weighted sum + residual ----
    MB_TIMELINE_BEGIN(EV_DOWN);
    {   // tcgen05 swap-AB per-expert whole tiles: CTA b<128 owns tile b (8 slots x 16 row blocks), full K=768 sum stored once into y_e
      if (!XPF) g_prefetch(G_DOWN, p.wd + (size_t)l * NE * H * I, nullptr, nullptr, sel, reinterpret_cast<float*>(ye), &rst);
      // B operand: only this CTA's hb segment (tile b -> slot b >> 4), at its offset in the XB region
      uint4* xbs = reinterpret_cast<uint4*>(xb_smem());
      const int sseg = (int)blockIdx.x >> 4;
      if (blockIdx.x < TK * H / 128 && threadIdx.x < I / 8)
        xbs[sseg * (I / 8) + threadIdx.x] = __ldcg(reinterpret_cast<const uint4*>(hb) + sseg * (I / 8) + threadIdx.x);
      fixup_pre(pre, (l & 1) ? xb2 : xb, reinterpret_cast<const float*>(p.scratch + O_YO));
      __syncthreads();
      g_run(G_DOWN, p.wd + (size_t)l * NE * H * I, nullptr, nullptr, sel, reinterpret_cast<float*>(ye), &rst);
      if (XPF && l + 1 < L)   // E2|E3E6: next layer's QKV ring fill streams during the barrier and the ln1 fixup
        g_prefetch(G_QKV, p.wq + (size_t)(l + 1) * 4096 * H, p.wk + (size_t)(l + 1) * 512 * H, p.wv + (size_t)(l + 1) * 512 * H, sel,
                   reinterpret_cast<float*>(p.scratch + O_QKVF), &rst);
    }
    MB_TIMELINE_END(EV_DOWN);
    BARRIER();
  }
  // ---- final norm + lm_head + argmax ----
  MB_TIMELINE_BEGIN(EV_LM);
  {   // final norm on X_L = fixup of layer L-1 (every CTA, no publish)
    float v[4], ss = 0.f;
    fixup_x(pre, ye, prob, xn, nullptr);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = xn[threadIdx.x + i * NT]; ss += v[i] * v[i]; }
    ss = bsum(ss, red);
    const float r = rsqrtf(ss * (1.f / H) + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int d = threadIdx.x + i * NT;
      xn[d] = rb(rb(v[i] * r) * __bfloat162float(p.fnorm[d]));
    }
    __syncthreads();
  }
  // mma.sync m16n8k16 swap-AB, rowblock16 split-K ks=4: the 4 warps of a quad take the 4 K quarters of one
  // 16-row block; partials meet in smem (fixed order, deterministic), the bf16-rounded logit is stored directly.
  // Blocks are handed out dynamically (per-SM HBM throughput is uneven): a persistent u64 ticket counter at
  // scratch+8 advances by exactly NRB + NGQ per call (every quad draws one failing ticket), so ticket % C is
  // the block index of this call.
  float bv = -INFINITY; int bi = 0x7fffffff;
  {
    constexpr int KP = H / 4, NRB = V / 16, NGQ = NB * WPB / 4;
    constexpr unsigned long long C = NRB + NGQ;
    unsigned long long* tick = reinterpret_cast<unsigned long long*>(p.scratch + O_BAR + 8);
    float* hs = reinterpret_cast<float*>(xb_smem());   // XB region (ring idle after the last down)
    uint32_t* xp = reinterpret_cast<uint32_t*>(hs);   // x as packed bf16 pairs (exact: xn is bf16-valued)
    float* part = hs + H / 2;                         // [4 quads][2 slots][4 kp][16 rows]
    int* tks = reinterpret_cast<int*>(part + 512);    // [4 quads][2 slots] block index
    const int g = lane >> 2, t = lane & 3, quad = warp >> 2, kp = warp & 3;
    const bool leader = kp == 0 && lane == 0;
    if (leader) tks[quad * 2] = (int)(atomicAdd(tick, 1ull) % C);
    for (int i = threadIdx.x; i < H / 2; i += NT) {
      __nv_bfloat162 h2 = __floats2bfloat162_rn(xn[2 * i], xn[2 * i + 1]);
      xp[i] = *reinterpret_cast<uint32_t*>(&h2);
    }
    __syncthreads();
    const uint4* xv = reinterpret_cast<const uint4*>(xp) + (kp * KP + t * 8) / 8;
    float* qpart = part + quad * 128;
    int slot = 0;
    while (true) {
      const int rbk = tks[quad * 2 + slot];
      if (rbk >= NRB) break;   // quad-uniform
      if (leader) tks[quad * 2 + (slot ^ 1)] = (int)(atomicAdd(tick, 1ull) % C);   // next block, overlapped
      const bf16* W0 = p.lm + (size_t)(rbk * 16 + g) * H + kp * KP + t * 8;
      const bf16* W1 = W0 + 8 * H;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int kb = 0; kb < KP; kb += 32) {
        uint4 u0 = ldnc(W0 + kb), u1 = ldnc(W1 + kb);
        uint4 bx = xv[kb / 8];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(u0.x), "r"(u1.x), "r"(u0.y), "r"(u1.y), "r"(bx.x), "r"(bx.y));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(u0.z), "r"(u1.z), "r"(u0.w), "r"(u1.w), "r"(bx.z), "r"(bx.w));
      }
      float* sp = qpart + slot * 64;
      if (t == 0) { sp[kp * 16 + g] = c[0]; sp[kp * 16 + g + 8] = c[2]; }
      asm volatile("bar.sync %0, 128;" :: "r"(1 + quad) : "memory");
      if (kp == 0) {   // whole warp: lanes 16-31 duplicate lanes 0-15 (same value and index)
        const int j = lane & 15, r = rbk * 16 + j;
        float a = rb(((sp[j] + sp[16 + j]) + sp[32 + j]) + sp[48 + j]);
        if (lane < 16) p.logits[r] = a;
        if (a > bv || (a == bv && r < bi)) { bv = a; bi = r; }
      }
      slot ^= 1;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
  }
  if (lane == 0) { pbv[warp] = bv; pbi[warp] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float b = pbv[0]; int bj = pbi[0];
    for (int i = 1; i < WPB; i++) if (pbv[i] > b || (pbv[i] == b && pbi[i] < bj)) { b = pbv[i]; bj = pbi[i]; }
    pv[blockIdx.x] = b; pi[blockIdx.x] = bj;
  }
  MB_TIMELINE_END(EV_LM);
  BARRIER();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float b = __ldcg(pv); int bj = __ldcg(pi);
    for (int i = 1; i < NB; i++) { float v = __ldcg(pv + i); int j = __ldcg(pi + i); if (v > b || (v == b && j < bj)) { b = v; bj = j; } }
    *p.next = bj;
  }
  tc_sync();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(rst.tmem));
}

extern "C" int mk_launch(unsigned long long* a, void* stream) {
  P p;
  const void** f = reinterpret_cast<const void**>(&p);
  int n = 17;  // pointer members in order up to vc
  for (int i = 0; i < n; i++) f[i] = (const void*)a[i];
  p.token = (const long long*)a[17]; p.cosb = (const float*)a[18]; p.sinb = (const float*)a[19];
  p.kw = (bf16*)a[20]; p.vw = (bf16*)a[21]; p.logits = (float*)a[22]; p.next = (long long*)a[23];
  p.scratch = (unsigned char*)a[24]; p.tl = (uint64_t*)a[25]; p.tl_words = a[26]; p.nblocks = NB;
  void* args[] = {&p};
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute((const void*)mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM); attr = true; }
  cudaError_t e = cudaLaunchCooperativeKernel((const void*)mk_kernel, dim3(NB), dim3(NT), args, DSMEM, (cudaStream_t)stream);
  return (int)e;
}
