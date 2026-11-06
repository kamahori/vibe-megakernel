// Small-token GEMM engines on sm_100a: Y[M][N] = X[M][K] * W[N][K]^T, M = tokens (1..64), W large.
// Two non-tcgen05 engines, each a standalone kernel with a small tile/launch autotune per point:
//   ffma     SIMT GEMV-style. Each warp owns R rows of W; lane loads 16 B of W per row per step
//            (coalesced 512 B / warp / row), X chunks are staged in smem with a cp.async double
//            buffer, fp32 FFMA with split partial sums, warp reduce-scatter, optional split-K
//            across the warps of a block (KG warps share rows, reduce through smem).
//   mma_sync swap-AB mma.sync.m16n8k16 (bf16 in, fp32 acc). W rows are the 16-row A operand, tokens
//            are the N=8 operand (padded to NT*8). W goes global -> registers with 16 B loads
//            (no smem hop); the K order inside an MMA is permuted so a lane's 16 B vector feeds
//            its own A fragments and X is read with LDS.128 (padded rows, conflict free).
//            KG warps split K, accumulators reduce through smem.
// Weight dtypes: bf16, and int8 with an fp32 per-output-channel scale. int8 -> bf16 uses the
// PRMT + 2^23 FADD magic path from dequant.cu (exact for |v| <= 127); the scale is applied to the
// fp32 accumulator in the epilogue (so dequant costs no multiply).
// L2 cold: W copies rotate over >= 576 MB, kernels run back to back inside one CUDA graph.
// L2 warm: the same W every launch. us = graph time / launches (includes the inter-kernel gap).
// Reference: fp32 kernel (exact bf16 products, fp32 accumulate). Tolerance per element:
//   |y - ref| <= 2^-8 |ref| + 1e-4 rms(ref)   (bf16 output rounding is <= 2^-9 relative; the rest
//   is fp32 accumulation-order slack). Candidates that fail are dropped, never timed.
// Usage: ./gemm_small [shape_idx]   (JSON lines on stdout; progress on stderr)
#include "pb_common.cuh"
#include <cstring>
#include <functional>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
enum { WBF16 = 0, WI8 = 1 };

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(smem)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ float lo_f(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_f(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float i8f(uint32_t x, int i) {  // x = w ^ 0x80808080 (biased bytes)
  return __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7650 + i)) - 8388736.f;  // 2^23 + 128
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}
// 4 packed int8 -> two bf16x2 (exact)
__device__ __forceinline__ void i8x4_bf16(uint32_t w, uint32_t& a, uint32_t& b) {
  uint32_t x = w ^ 0x80808080u;
  a = pack_bf16(i8f(x, 0), i8f(x, 1));
  b = pack_bf16(i8f(x, 2), i8f(x, 3));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------- FFMA engine
// Template: WT dtype, M tokens, R rows/warp, KG warps splitting K, U 16B-steps per chunk.
template <int WT, int M, int R, int KG, int U>
__global__ void __launch_bounds__(256) k_ffma(const void* Wv, const float* scale, const bf16* X, bf16* Y, int N, int K) {
  constexpr int EPL = WT == WBF16 ? 8 : 16, ESZ = WT == WBF16 ? 2 : 1;
  constexpr int RG = 8 / KG, BR = RG * R, STEP = 32 * EPL, KCH = KG * U * STEP, NG = KCH / 8;
  constexpr int Q = R * M;
  constexpr int P = Q >= 32 ? 1 : (Q >= 16 ? 2 : 4);  // partial sums per accumulator (FFMA chain ILP)
  extern __shared__ __align__(16) uint8_t sm[];
  bf16* xs = reinterpret_cast<bf16*>(sm);                       // [2][M][KCH]
  float* red = reinterpret_cast<float*>(sm + 2 * M * KCH * 2);  // [KG][M][BR]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, kg = warp % KG, rg = warp / KG;
  const int row0 = blockIdx.x * BR + rg * R;
  const uint8_t* W = reinterpret_cast<const uint8_t*>(Wv);
  const int nch = K / KCH;

  auto issue_x = [&](int c, int buf) {
    for (int i = tid; i < M * NG; i += 256) {
      int m = i / NG, gi = i % NG;
      int pos = EPL == 16 ? ((gi & ~63) | ((gi & 1) << 5) | ((gi >> 1) & 31)) : gi;  // lane-conflict-free LDS
      cp_async16(xs + ((size_t)buf * M + m) * KCH + pos * 8, X + (size_t)m * K + (size_t)c * KCH + gi * 8);
    }
    cp_commit();
  };
  uint4 wc[R][U], wn[R][U];
  auto ldw = [&](int c, uint4 (&d)[R][U]) {
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r)
        d[r][u] = ldg_stream(W + ((size_t)(row0 + r) * K + (size_t)c * KCH + (size_t)(kg * U + u) * STEP + lane * EPL) * ESZ);
  };

  float acc[R][M][P];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int p = 0; p < P; ++p) acc[r][m][p] = 0.f;

  issue_x(0, 0);
  ldw(0, wc);
  for (int c = 0; c < nch; ++c) {
    cp_wait0();
    __syncthreads();
    if (c + 1 < nch) { issue_x(c + 1, (c + 1) & 1); ldw(c + 1, wn); }
    const bf16* xb = xs + (size_t)(c & 1) * M * KCH + (size_t)(kg * U) * STEP;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float wf[R][EPL];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&wc[r][u]);
        if constexpr (WT == WBF16) {
#pragma unroll
          for (int j = 0; j < 4; ++j) { wf[r][2 * j] = lo_f(w[j]); wf[r][2 * j + 1] = hi_f(w[j]); }
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t x = w[j] ^ 0x80808080u;
#pragma unroll
            for (int i = 0; i < 4; ++i) wf[r][4 * j + i] = i8f(x, i);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[EPL];
#pragma unroll
        for (int h = 0; h < EPL / 8; ++h) {
          uint4 v = *reinterpret_cast<const uint4*>(xb + (size_t)m * KCH + u * STEP + (h * 32 + lane) * 8);
          const uint32_t* x = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
          for (int j = 0; j < 4; ++j) { xf[h * 8 + 2 * j] = lo_f(x[j]); xf[h * 8 + 2 * j + 1] = hi_f(x[j]); }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int e = 0; e < EPL; ++e) acc[r][m][e % P] = fmaf(wf[r][e], xf[e], acc[r][m][e % P]);
      }
    }
    if (c + 1 < nch) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int u = 0; u < U; ++u) wc[r][u] = wn[r][u];
    }
  }
  // ---- reduce over lanes
  float v[Q];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float s = 0.f;
#pragma unroll
      for (int p = 0; p < P; ++p) s += acc[r][m][p];
      v[r * M + m] = s;
    }
  __syncthreads();  // xs is dead; red aliases nothing but keep ordering simple
  if constexpr (Q < 32) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1) v[q] += __shfl_xor_sync(0xffffffffu, v[q], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < Q; ++q) red[kg * (BR * M) + (q % M) * BR + rg * R + q / M] = v[q];
    }
  } else {
    static_assert(Q % 32 == 0, "reduce-scatter needs groups of 32");
#pragma unroll
    for (int g = 0; g < Q / 32; ++g) {
      float a[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) a[i] = v[g * 32 + i];
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1) {
        const bool up = lane & o;
#pragma unroll
        for (int i = 0; i < o; ++i) {
          float send = up ? a[i] : a[i + o], keep = up ? a[i + o] : a[i];
          a[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
      }
      int q = g * 32 + lane;
      red[kg * (BR * M) + (q % M) * BR + rg * R + q / M] = a[0];
    }
  }
  __syncthreads();
  for (int t = tid; t < BR * M; t += 256) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KG; ++k) s += red[k * (BR * M) + t];
    int m = t / BR, rl = t % BR, row = blockIdx.x * BR + rl;
    if (WT == WI8) s *= scale[row];
    Y[(size_t)m * N + row] = __float2bfloat16_rn(s);
  }
}

// ---------------------------------------------------------------- mma.sync swap-AB engine
// NT n8 tiles (tokens padded to NT*8), RT 16-row tiles per warp, KG warps splitting K, U steps/chunk.
template <int WT, int NT, int RT, int KG, int U>
__global__ void __launch_bounds__(256) k_mma(const void* Wv, const float* scale, const bf16* X, bf16* Y, int M, int N, int K) {
  constexpr int V = WT == WBF16 ? 8 : 16, ESZ = WT == WBF16 ? 2 : 1, NJ = V / 4, NW = V / 2;
  constexpr int RG = 8 / KG, BROWS = RG * RT * 16, STEP = 4 * V, KCH = KG * U * STEP, NG = KCH / 8;
  constexpr int PAD = WT == WBF16 ? 64 : 16, ROWB = KCH * 2 + PAD, XBUF = NT * 8 * ROWB;
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, kg = warp % KG, rg = warp / KG;
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* W = reinterpret_cast<const uint8_t*>(Wv);
  const int rowb = blockIdx.x * BROWS + rg * RT * 16;
  const int nch = K / KCH;

  // zero the padded token rows of both buffers (never written by cp.async)
  for (int i = tid; i < 2 * (NT * 8 - M) * (ROWB / 16); i += 256) {
    int per = (NT * 8 - M) * (ROWB / 16), b = i / per, rem = i % per;
    *reinterpret_cast<uint4*>(sm + (size_t)b * XBUF + (size_t)M * ROWB + rem * 16) = make_uint4(0, 0, 0, 0);
  }
  auto issue_x = [&](int c, int buf) {
    for (int i = tid; i < M * NG; i += 256) {
      int m = i / NG, gi = i % NG;
      cp_async16(sm + (size_t)buf * XBUF + (size_t)m * ROWB + gi * 16, X + (size_t)m * K + (size_t)c * KCH + gi * 8);
    }
    cp_commit();
  };
  uint4 wc[RT][U][2], wn[RT][U][2];
  auto ldw = [&](int c, uint4 (&d)[RT][U][2]) {
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int rt = 0; rt < RT; ++rt) {
        size_t k = (size_t)c * KCH + (size_t)(kg * U + u) * STEP + t * V;
        d[rt][u][0] = ldg_stream(W + ((size_t)(rowb + rt * 16 + g) * K + k) * ESZ);
        d[rt][u][1] = ldg_stream(W + ((size_t)(rowb + rt * 16 + g + 8) * K + k) * ESZ);
      }
  };
  float acc[RT][NT][4];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[rt][nt][r] = 0.f;

  issue_x(0, 0);
  ldw(0, wc);
  for (int c = 0; c < nch; ++c) {
    cp_wait0();
    __syncthreads();
    if (c + 1 < nch) { issue_x(c + 1, (c + 1) & 1); ldw(c + 1, wn); }
    const uint8_t* xb = sm + (size_t)(c & 1) * XBUF + (size_t)((kg * U) * STEP) * 2 + t * V * 2;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint32_t pw[RT][NW], qw[RT][NW];
#pragma unroll
      for (int rt = 0; rt < RT; ++rt) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(&wc[rt][u][0]);
        const uint32_t* q = reinterpret_cast<const uint32_t*>(&wc[rt][u][1]);
        if constexpr (WT == WBF16) {
#pragma unroll
          for (int i = 0; i < 4; ++i) { pw[rt][i] = p[i]; qw[rt][i] = q[i]; }
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            i8x4_bf16(p[i], pw[rt][2 * i], pw[rt][2 * i + 1]);
            i8x4_bf16(q[i], qw[rt][2 * i], qw[rt][2 * i + 1]);
          }
        }
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint32_t bx[NW];
        const uint8_t* xp = xb + (size_t)(nt * 8 + g) * ROWB + u * STEP * 2;
#pragma unroll
        for (int h = 0; h < V / 8; ++h) *reinterpret_cast<uint4*>(&bx[4 * h]) = *reinterpret_cast<const uint4*>(xp + 16 * h);
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
          for (int j = 0; j < NJ; ++j)
            mma_bf16(acc[rt][nt], pw[rt][2 * j], qw[rt][2 * j], pw[rt][2 * j + 1], qw[rt][2 * j + 1], bx[2 * j], bx[2 * j + 1]);
      }
    }
    if (c + 1 < nch) {
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int u = 0; u < U; ++u) { wc[rt][u][0] = wn[rt][u][0]; wc[rt][u][1] = wn[rt][u][1]; }
    }
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(sm);  // [KG][RG][RT][NT][4][32]
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        red[((((kg * RG + rg) * RT + rt) * NT + nt) * 4 + r) * 32 + lane] = acc[rt][nt][r];
  __syncthreads();
  constexpr int TOT = RG * RT * NT * 4 * 32;
  for (int idx = tid; idx < TOT; idx += 256) {
    int ln = idx & 31, rest = idx >> 5, r = rest & 3;
    rest >>= 2;
    int nt = rest % NT; rest /= NT;
    int rt = rest % RT, rgi = rest / RT;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KG; ++k) s += red[((((k * RG + rgi) * RT + rt) * NT + nt) * 4 + r) * 32 + ln];
    int token = nt * 8 + 2 * (ln & 3) + (r & 1);
    int row = blockIdx.x * BROWS + rgi * RT * 16 + rt * 16 + (ln >> 2) + 8 * (r >> 1);
    if (token < M) {
      if (WT == WI8) s *= scale[row];
      Y[(size_t)token * N + row] = __float2bfloat16_rn(s);
    }
  }
}

// ---------------------------------------------------------------- init / reference / clock
__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed) {
  uint32_t h = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32) * 40503u ^ seed;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return h;
}
__global__ void k_init_w(void* W, size_t n, int wt) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = hash32(i, 17);
    if (wt == WBF16) reinterpret_cast<bf16*>(W)[i] = __float2bfloat16_rn(0.05f * ((h >> 8) * (2.f / 16777216.f) - 1.f));
    else reinterpret_cast<int8_t*>(W)[i] = (int8_t)((int)(h % 255u) - 127);
  }
}
__global__ void k_init_x(bf16* X, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    X[i] = __float2bfloat16_rn((hash32(i, 91) >> 8) * (2.f / 16777216.f) - 1.f);
}
__global__ void k_init_scale(float* s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) s[i] = 0.002f * (1.f + (i % 8) / 8.f);
}
template <int WT>
__global__ void __launch_bounds__(256) k_ref(const void* Wv, const float* scale, const bf16* X, float* ref, int N, int K) {
  constexpr int MR = 64;
  float a[MR];
#pragma unroll
  for (int m = 0; m < MR; ++m) a[m] = 0.f;
  const int n = blockIdx.x;
  for (int k = threadIdx.x; k < K; k += 256) {
    float w = WT == WBF16 ? __bfloat162float(reinterpret_cast<const bf16*>(Wv)[(size_t)n * K + k])
                          : (float)reinterpret_cast<const int8_t*>(Wv)[(size_t)n * K + k];
#pragma unroll
    for (int m = 0; m < MR; ++m) a[m] = fmaf(w, __bfloat162float(X[(size_t)m * K + k]), a[m]);
  }
  __shared__ float s[8][MR];
#pragma unroll
  for (int m = 0; m < MR; ++m) {
    float v = a[m];
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5][m] = v;
  }
  __syncthreads();
  if (threadIdx.x < MR) {
    float v = 0.f;
    for (int w = 0; w < 8; ++w) v += s[w][threadIdx.x];
    if (WT == WI8) v *= scale[n];
    ref[(size_t)threadIdx.x * N + n] = v;
  }
}
__global__ void k_spin(Stamp* st, int iters) {
  float x = threadIdx.x * 1e-3f, y = 1.f;
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) { x = fmaf(x, 0.999f, y); y = fmaf(y, 0.9999f, x * 1e-6f); }
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (threadIdx.x == 0) st[blockIdx.x] = {t1 - t0, g1 - g0};
  if (x == 123.f) st[0].cyc = (long long)y;
}
static double measure_sm_clock() {
  int n = num_sms(); Stamp* d; CK(cudaMalloc(&d, n * sizeof(Stamp)));
  k_spin<<<n, 1024>>>(d, 1000); CK(cudaDeviceSynchronize());
  k_spin<<<n, 1024>>>(d, 200000); CK(cudaDeviceSynchronize());
  std::vector<Stamp> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(Stamp), cudaMemcpyDeviceToHost)); cudaFree(d);
  return clock_mhz(h);
}

// ---------------------------------------------------------------- candidates
struct Ctx { const float* scale; const bf16* X; bf16* Y; int M, N, K; };
struct Cand {
  std::string cfg, sched; int rows, tile_n, grid, threads = 256, regs = 0, local = 0; size_t smem;
  std::function<void(cudaStream_t, const void*)> launch;
};
static const size_t SMEM_CAP = 160 * 1024;

template <int WT, int M, int R, int KG, int U>
static void reg_ffma(std::vector<Cand>& v, const Ctx& cx) {
  constexpr int EPL = WT == WBF16 ? 8 : 16, RG = 8 / KG, BR = RG * R, KCH = KG * U * 32 * EPL, Q = R * M;
  if constexpr (Q > 128 || (Q >= 32 && Q % 32 != 0)) return;
  else {
    if (cx.K % KCH || cx.N % BR) return;
    size_t smem = 2 * (size_t)M * KCH * 2 + (size_t)KG * BR * M * 4;
    if (smem > SMEM_CAP) return;
    auto fn = k_ffma<WT, M, R, KG, U>;
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
    Cand c; c.cfg = "ffma R=" + std::to_string(R) + " KG=" + std::to_string(KG) + " U=" + std::to_string(U) +
                    " kchunk=" + std::to_string(KCH) + " 8warps x " + std::to_string(R) + "rows";
    c.sched = "warp_rows" + std::string(KG > 1 ? "+splitK_warps" + std::to_string(KG) : "") + " cp.async_x_dbuf";
    c.rows = BR; c.tile_n = M; c.grid = cx.N / BR; c.smem = smem; c.regs = a.numRegs; c.local = (int)a.localSizeBytes;
    c.launch = [=](cudaStream_t s, const void* W) { fn<<<cx.N / BR, 256, smem, s>>>(W, cx.scale, cx.X, cx.Y, cx.N, cx.K); };
    v.push_back(c);
  }
}
template <int WT, int M>
static void reg_ffma_all(std::vector<Cand>& v, const Ctx& cx) {
  reg_ffma<WT, M, 1, 1, 4>(v, cx); reg_ffma<WT, M, 1, 1, 2>(v, cx); reg_ffma<WT, M, 1, 1, 1>(v, cx);
  reg_ffma<WT, M, 2, 1, 2>(v, cx); reg_ffma<WT, M, 2, 1, 1>(v, cx); reg_ffma<WT, M, 4, 1, 1>(v, cx);
  reg_ffma<WT, M, 1, 2, 2>(v, cx); reg_ffma<WT, M, 1, 4, 1>(v, cx);
  reg_ffma<WT, M, 2, 1, 4>(v, cx); reg_ffma<WT, M, 1, 1, 8>(v, cx); reg_ffma<WT, M, 4, 1, 2>(v, cx);
}
template <int WT, int NT, int RT, int KG, int U>
static void reg_mma(std::vector<Cand>& v, const Ctx& cx) {
  constexpr int V = WT == WBF16 ? 8 : 16, RG = 8 / KG, BROWS = RG * RT * 16, KCH = KG * U * 4 * V;
  constexpr int PAD = WT == WBF16 ? 64 : 16, ROWB = KCH * 2 + PAD;
  if constexpr (RT * NT > 16) return;
  else {
    if (cx.K % KCH || cx.N % BROWS) return;
    size_t xs = 2 * (size_t)NT * 8 * ROWB, rd = (size_t)KG * RG * RT * NT * 4 * 32 * 4, smem = xs > rd ? xs : rd;
    if (smem > SMEM_CAP) return;
    auto fn = k_mma<WT, NT, RT, KG, U>;
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
    Cand c; c.cfg = "mma.m16n8k16 swapAB RT=" + std::to_string(RT) + " NT=" + std::to_string(NT) + " KG=" + std::to_string(KG) +
                    " U=" + std::to_string(U) + " kchunk=" + std::to_string(KCH) + " blockrows=" + std::to_string(BROWS);
    c.sched = "swap_ab W_regs_direct" + std::string(KG > 1 ? "+splitK_warps" + std::to_string(KG) : "") + " cp.async_x_dbuf";
    c.rows = BROWS; c.tile_n = NT * 8; c.grid = cx.N / BROWS; c.smem = smem; c.regs = a.numRegs; c.local = (int)a.localSizeBytes;
    c.launch = [=](cudaStream_t s, const void* W) { fn<<<cx.N / BROWS, 256, smem, s>>>(W, cx.scale, cx.X, cx.Y, cx.M, cx.N, cx.K); };
    v.push_back(c);
  }
}
template <int WT, int NT>
static void reg_mma_all(std::vector<Cand>& v, const Ctx& cx) {
  reg_mma<WT, NT, 1, 8, 4>(v, cx); reg_mma<WT, NT, 1, 8, 2>(v, cx); reg_mma<WT, NT, 1, 8, 1>(v, cx);
  reg_mma<WT, NT, 2, 8, 2>(v, cx); reg_mma<WT, NT, 2, 4, 2>(v, cx); reg_mma<WT, NT, 2, 4, 1>(v, cx);
  reg_mma<WT, NT, 1, 4, 2>(v, cx); reg_mma<WT, NT, 4, 4, 1>(v, cx); reg_mma<WT, NT, 4, 2, 1>(v, cx);
  reg_mma<WT, NT, 2, 8, 4>(v, cx); reg_mma<WT, NT, 4, 8, 2>(v, cx); reg_mma<WT, NT, 1, 8, 8>(v, cx); reg_mma<WT, NT, 2, 4, 4>(v, cx);
}
template <int WT>
static std::vector<Cand> candidates(bool mma, const Ctx& cx) {
  std::vector<Cand> v;
  if (mma) {
    switch ((cx.M + 7) / 8) {
      case 1: reg_mma_all<WT, 1>(v, cx); break; case 2: reg_mma_all<WT, 2>(v, cx); break;
      case 4: reg_mma_all<WT, 4>(v, cx); break; default: reg_mma_all<WT, 8>(v, cx); break;
    }
  } else {
    switch (cx.M) {
      case 1: reg_ffma_all<WT, 1>(v, cx); break; case 2: reg_ffma_all<WT, 2>(v, cx); break;
      case 4: reg_ffma_all<WT, 4>(v, cx); break; case 8: reg_ffma_all<WT, 8>(v, cx); break;
      case 16: reg_ffma_all<WT, 16>(v, cx); break; case 32: reg_ffma_all<WT, 32>(v, cx); break;
      default: reg_ffma_all<WT, 64>(v, cx); break;
    }
  }
  return v;
}

// ---------------------------------------------------------------- host driver
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

static bool verify(const std::vector<uint16_t>& y, const std::vector<float>& ref, int M, int N, double& max_err) {
  double ss = 0; size_t n = (size_t)M * N;
  for (size_t i = 0; i < n; ++i) ss += (double)ref[i] * ref[i];
  double rms = std::sqrt(ss / n); max_err = 0; bool ok = true;
  for (size_t i = 0; i < n; ++i) {
    double e = std::fabs((double)bf2f(y[i]) - ref[i]);
    if (!(e <= 1.0 / 256 * std::fabs(ref[i]) + 1e-4 * rms)) ok = false;
    if (!(e <= max_err)) max_err = std::isnan(e) ? INFINITY : e;
  }
  return ok;
}

static double time_cand(const Cand& c, const std::vector<void*>& copies, int L, std::vector<double>& reps) {
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < L; ++i) c.launch(s, copies[i % copies.size()]);
  CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  reps.clear();
  for (int i = 0; i < 9; ++i) {
    CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); reps.push_back(ms * 1e3 / L);
  }
  CK(cudaGetLastError());
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
  return median(reps);
}

#ifndef GEMM_SMALL_LIB   // gemm_small_tc.cu includes this file for its harness and supplies its own main
int main(int argc, char** argv) {
  const int NS = 4, shapes[NS][2] = {{4096, 4096}, {8192, 8192}, {28672, 4096}, {4096, 14336}};
  const int Ms[] = {1, 2, 4, 8, 16, 32, 64};
  int only = argc > 1 ? atoi(argv[1]) : -1;
  const double mhz = measure_sm_clock();
  fprintf(stderr, "sm clock (spin) %.0f MHz\n", mhz);
  for (int si = 0; si < NS; ++si) {
    if (only >= 0 && si != only) continue;
    const int N = shapes[si][0], K = shapes[si][1];
    for (int wt = 0; wt < 2; ++wt) {
      const size_t wbytes = (size_t)N * K * (wt == WBF16 ? 2 : 1);
      const int ncopy = (int)((576ull << 20) / wbytes + 1);
      std::vector<void*> copies(ncopy);
      for (auto& p : copies) CK(cudaMalloc(&p, wbytes));
      k_init_w<<<2048, 256>>>(copies[0], wbytes / (wt == WBF16 ? 2 : 1), wt);
      for (int i = 1; i < ncopy; ++i) CK(cudaMemcpy(copies[i], copies[0], wbytes, cudaMemcpyDeviceToDevice));
      bf16* X; float *scale, *refd; bf16* Y;
      CK(cudaMalloc(&X, (size_t)64 * K * 2)); CK(cudaMalloc(&scale, N * 4)); CK(cudaMalloc(&refd, (size_t)64 * N * 4));
      CK(cudaMalloc(&Y, (size_t)64 * N * 2));
      k_init_x<<<256, 256>>>(X, (size_t)64 * K); k_init_scale<<<(N + 255) / 256, 256>>>(scale, N);
      if (wt == WBF16) k_ref<WBF16><<<N, 256>>>(copies[0], scale, X, refd, N, K);
      else k_ref<WI8><<<N, 256>>>(copies[0], scale, X, refd, N, K);
      CK(cudaDeviceSynchronize());
      std::vector<float> ref((size_t)64 * N); CK(cudaMemcpy(ref.data(), refd, ref.size() * 4, cudaMemcpyDeviceToHost));
      const char* wname = wt == WBF16 ? "bf16" : "int8";
      for (int eng = 0; eng < 2; ++eng) {
        const char* ename = eng ? "mma_sync" : "ffma";
        for (int M : Ms) {
          Ctx cx{scale, X, Y, M, N, K};
          std::vector<Cand> cs = wt == WBF16 ? candidates<WBF16>(eng, cx) : candidates<WI8>(eng, cx);
          std::vector<Cand> ok; std::vector<double> errs;
          std::vector<uint16_t> y((size_t)M * N);
          for (auto& c : cs) {
            CK(cudaMemset(Y, 0xFF, (size_t)M * N * 2));
            c.launch(0, copies[0]); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(y.data(), Y, y.size() * 2, cudaMemcpyDeviceToHost));
            double e; bool pass = verify(y, ref, M, N, e);
            if (!pass) fprintf(stderr, "DROP %s %s M=%d %dx%d: %s err=%g\n", ename, wname, M, N, K, c.cfg.c_str(), e);
            else { ok.push_back(c); errs.push_back(e); }
          }
          for (int l2 = 0; l2 < 2; ++l2) {  // 0 cold, 1 warm
            int best = -1; double bt = 1e30; std::vector<double> breps, reps;
            std::vector<void*> cp = l2 ? std::vector<void*>{copies[0]} : copies;
            int L = l2 ? 16 : std::max(ncopy, 12);
            for (size_t i = 0; i < ok.size(); ++i) {
              double t = time_cand(ok[i], cp, L, reps);
              if (t < bt) { bt = t; best = (int)i; breps = reps; }
            }
            if (best < 0) { fprintf(stderr, "NO valid candidate %s %s M=%d %dx%d\n", ename, wname, M, N, K); continue; }
            const Cand& c = ok[best];
            double gbps = (double)wbytes / (bt * 1e-6) / 1e9, tflops = 2.0 * M * N * K / (bt * 1e-6) / 1e12;
            bool sane = l2 == 1 || gbps < 8000;  // cold run above HBM peak means a caching bug
            if (!sane) fprintf(stderr, "SUSPECT cold run above HBM peak: %s %s M=%d %dx%d %.0f GB/s\n", ename, wname, M, N, K, gbps);
            Json j;
            j.s("family", "gemm").s("primitive", "gemm.primbench").s("p.engine", ename).s("p.w_dtype", wname).s("p.x_dtype", "bf16")
                .i("p.M", M).i("p.N", N).i("p.K", K).s("p.config", c.cfg).i("p.tile_m", c.rows).i("p.tile_n", c.tile_n)
                .s("p.sched", c.sched)
                .f("m.us", bt).f("m.tflops", tflops).f("m.hbm_gbps", gbps).f("m.max_abs_err", errs[best])
                .i("m.regs", c.regs).i("m.local_bytes", c.local)
                .s("c.l2", l2 ? "warm" : "cold").f("c.sm_clock_mhz", mhz).i("c.ctas", c.grid).i("c.threads", c.threads)
                .i("c.smem", (long long)c.smem)
                .t("verified", sane).i("n_reps", (long long)breps.size()).f("cv", cv(breps))
                .s("s.bench", "primbench/gemm_small.cu").s("s.timer", "cudaEvent_graph")
                .s("note", std::string("Y[M][N] = X[M][K] W[N][K]^T, M = tokens; best of an autotuned tile list (all verified). ") +
                               (wt == WI8 ? "int8 W with per-row fp32 scale applied in the fp32 epilogue; magic-number int8->bf16 in registers. " : "") +
                               "us = CUDA-graph average over back-to-back launches (includes the inter-kernel gap, ~1-2 us). " +
                               (l2 ? "L2 warm: same W every launch (W <= ~100 MB can be L2 resident, then hbm_gbps is not HBM)."
                                   : "L2 cold: W copies rotate over >= 576 MB."));
            j.print();
          }
        }
      }
      for (auto p : copies) cudaFree(p);
      cudaFree(X); cudaFree(scale); cudaFree(refd); cudaFree(Y);
      fprintf(stderr, "done %dx%d %s\n", N, K, wname);
    }
  }
  return 0;
}
#endif
