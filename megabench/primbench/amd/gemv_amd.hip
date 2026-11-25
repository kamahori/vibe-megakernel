// Batch-1 bf16 GEMV y[N] = W[N,K] x[K] (fp32 accumulate) on CDNA, the dominant phase of a
// decode megakernel. Engines:
//   valu_r1 / valu_r4 : one wave per 1 or 4 rows, 16 B loads, fp32 FMA, DPP wave reduction
//   mfma16            : one wave per 16-row tile, v_mfma_f32_16x16x16_bf16 with x broadcast to
//                       every column (15/16 of the MFMA is wasted; no cross-lane reduction)
//   mfma4             : one wave per 4-row tile, v_mfma_f32_4x4x4_16b_bf16 with the 16 blocks
//                       splitting K (block b takes k-slice b); one 16-lane reduction at the end
// One launch of NB persistent blocks (1 per CU) runs the GEMV on `copies` copies of W in turn
// (more bytes than the Infinity Cache, so weights come from HBM). Reported:
//   us_single : one GEMV per launch on a cold copy, timed in-kernel (max end - min start)
//   gbps      : streaming rate over all copies in one launch
// Every engine's y is checked against a CPU fp32 reference.
// M tokens (M up to 128, `--msweep` runs only the token sweep): valu engines keep MC = min(M, 16)
// accumulators per row and loop over token chunks, reloading the row (L2 hits) per chunk; mfma16
// keeps ceil(M/16) 16x16 accumulator tiles and reuses each weight fragment for all of them.
// mfma_lds_sk (--msweep only) is the GEMM-style engine: each block stages x[0..M)[k-chunk of 128]
// in LDS once and its waves (16 rows each) read B fragments from there; K is split S ways so that
// (N / (16 * waves)) * S >= #blocks, with fp32 atomics combining the S partial sums.
// mfma_lds_db (--msweep only): the same scheme with the LDS x chunk double-buffered (the next
// chunk's x and weights are loaded into registers while the current chunk is computed, one
// __syncthreads per chunk). At M = 128 two 128-k chunks exceed 64 KB of LDS, so it uses 64-k
// chunks there (p.k_chunk). A 32x32x8 variant (32 rows x 32 tokens per wave) was measured on
// MI210 and MI300X and was 1.3-2.3x slower than 16x16 everywhere, so it is not kept.
// The other engines read x from global (L1/L2), not staged in LDS, so at large M the x traffic per wave
// (M x 16 B per lane per k-step) can rival the weight traffic.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx942 -o gemv_amd gemv_amd.hip
#include "pb_hip.h"
#include <cstring>

typedef __attribute__((ext_vector_type(4))) short s4;
typedef __attribute__((ext_vector_type(4))) float f4;

__device__ __forceinline__ float bf(unsigned short h) { return __uint_as_float((unsigned)h << 16); }
__device__ __forceinline__ float lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }

#define DPP(v, ctrl) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xf, 0xf, false))
__device__ __forceinline__ float row16_sum(float v) {
  v += DPP(v, 0xb1); v += DPP(v, 0x4e); v += DPP(v, 0x141); v += DPP(v, 0x140); return v;
}
__device__ __forceinline__ float wave_sum(float v) {
  v = row16_sum(v);
  return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0)) +
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16)) +
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32)) +
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
}

__device__ __forceinline__ float dot8(uint4 w, uint4 x) {
  float a = lo(w.x) * lo(x.x);
  a = fmaf(hi(w.x), hi(x.x), a); a = fmaf(lo(w.y), lo(x.y), a); a = fmaf(hi(w.y), hi(x.y), a);
  a = fmaf(lo(w.z), lo(x.z), a); a = fmaf(hi(w.z), hi(x.z), a); a = fmaf(lo(w.w), lo(x.w), a);
  a = fmaf(hi(w.w), hi(x.w), a);
  return a;
}

enum Eng { E_VALU1, E_VALU4, E_MFMA16, E_MFMA4, NENG };
// LDS engines, outside the M = 1 engine loop
static const int E_MLDS = NENG, E_MLDS_DB = NENG + 1;
static const char* eng_name[] = {"valu_r1", "valu_r4", "mfma16", "mfma4", "mfma_lds_sk", "mfma_lds_db"};
static const int eng_rows[] = {1, 4, 16, 4, 16, 16};

// One GEMV on W (N x K bf16 row-major), units of `rows` rows dealt to waves round-robin.
// M tokens: x is [M][K], y is [M][N]. valu engines reuse each weight load for all M tokens;
// mfma16 puts token m in column m (useful fraction M/16).
template <int E, int M = 1>
__device__ __forceinline__ void gemv(const unsigned short* __restrict__ W, const unsigned short* __restrict__ x,
                                     float* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 63;
  const int wave = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int nwaves = gridDim.x * (blockDim.x >> 6);
  constexpr int R = E == E_VALU1 ? 1 : E == E_VALU4 ? 4 : E == E_MFMA16 ? 16 : 4;
  const int units = N / R;
  for (int u = wave; u < units; u += nwaves) {
    const int r0 = u * R;
    if (E == E_VALU1 || E == E_VALU4) {
      constexpr int MC = M < 16 ? M : 16;  // token chunk held in registers
      for (int m0 = 0; m0 < M; m0 += MC) {
        float acc[R][MC];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int m = 0; m < MC; ++m) acc[r][m] = 0;
        for (int k = lane * 8; k < K; k += 512) {
          uint4 wv[R];
#pragma unroll
          for (int r = 0; r < R; ++r) wv[r] = *(const uint4*)(W + (size_t)(r0 + r) * K + k);
#pragma unroll
          for (int m = 0; m < MC; ++m) {
            uint4 xv = *(const uint4*)(x + (size_t)(m0 + m) * K + k);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][m] += dot8(wv[r], xv);
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int m = 0; m < MC; ++m) {
            float s = wave_sum(acc[r][m]);
            if (lane == 0) y[(size_t)(m0 + m) * N + r0 + r] = s;
          }
      }
    } else if (E == E_MFMA16) {
      // lane l: row r0 + l%16, k-group l/16 (8 contiguous k per group per 32-k step)
      // T = ceil(M/16) token tiles; tile t holds tokens t*16 .. t*16+15 in its columns
      constexpr int T = (M + 15) / 16;
      const int row = r0 + (lane & 15), g = lane >> 4;
      f4 c[T];
#pragma unroll
      for (int t = 0; t < T; ++t) c[t] = f4{0, 0, 0, 0};
      for (int k = 0; k < K; k += 32) {
        uint4 wv = *(const uint4*)(W + (size_t)row * K + k + g * 8);
        s4 a0 = {(short)wv.x, (short)(wv.x >> 16), (short)wv.y, (short)(wv.y >> 16)};
        s4 a1 = {(short)wv.z, (short)(wv.z >> 16), (short)wv.w, (short)(wv.w >> 16)};
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const int col = M == 1 ? 0 : t * 16 + (lane & 15);  // token in this column (M == 1: x broadcast)
          uint4 xv = col < M ? *(const uint4*)(x + (size_t)col * K + k + g * 8) : make_uint4(0, 0, 0, 0);
          s4 b0 = {(short)xv.x, (short)(xv.x >> 16), (short)xv.y, (short)(xv.y >> 16)};
          s4 b1 = {(short)xv.z, (short)(xv.z >> 16), (short)xv.w, (short)(xv.w >> 16)};
          c[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a0, b0, c[t], 0, 0, 0);
          c[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a1, b1, c[t], 0, 0, 0);
        }
      }
      // D: lane l holds column l%16, rows (l/16)*4 + i; with M == 1 every column is the same
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const int m = M == 1 ? 0 : t * 16 + (lane & 15);
        if ((M == 1 && (lane & 15) == 0) || (M > 1 && m < M)) {
#pragma unroll
          for (int i = 0; i < 4; ++i) y[(size_t)m * N + r0 + g * 4 + i] = c[t][i];
        }
      }
    } else {  // E_MFMA4: 16 blocks of 4x4x4; block b = lane/4 owns k-slice b of each 64-k step
      // A (4x4 per block): lane l -> row l%4, k 4 values; B: lane l -> column l%4, same k
      const int row = r0 + (lane & 3), blk = lane >> 2;
      f4 c = {0, 0, 0, 0};
      for (int k = 0; k < K; k += 128) {
        uint4 wv = *(const uint4*)(W + (size_t)row * K + k + blk * 8);
        uint4 xv = *(const uint4*)(x + k + blk * 8);
        s4 a0 = {(short)wv.x, (short)(wv.x >> 16), (short)wv.y, (short)(wv.y >> 16)};
        s4 a1 = {(short)wv.z, (short)(wv.z >> 16), (short)wv.w, (short)(wv.w >> 16)};
        s4 b0 = {(short)xv.x, (short)(xv.x >> 16), (short)xv.y, (short)(xv.y >> 16)};
        s4 b1 = {(short)xv.z, (short)(xv.z >> 16), (short)xv.w, (short)(xv.w >> 16)};
        c = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a0, b0, c, 0, 0, 0);
        c = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a1, b1, c, 0, 0, 0);
      }
      // D per block: lane l holds column l%4, rows 0..3 of block l/4; sum blocks (lanes l%4 == 0 per block)
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float s = row16_sum(c[i]);  // sums the 16 lanes of a row of 16 = 4 blocks x 4 columns
        s = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 0)) +
            __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 16)) +
            __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 32)) +
            __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 48));
        v[i] = s * 0.25f;  // every column holds the same sum
      }
      if (lane == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) y[r0 + i] = v[i];
      }
    }
  }
}

template <int E, int M = 1>
__global__ void k_gemv(const unsigned short* W, const unsigned short* x, float* y, int N, int K,
                       int first, int copies, unsigned long long* stamps) {
  unsigned long long t0 = wall_clock64();
  for (int c = first; c < first + copies; ++c) gemv<E, M>(W + (size_t)c * N * K, x, y, N, K);
  __syncthreads();
  if (threadIdx.x == 0) { stamps[2 * blockIdx.x] = t0; stamps[2 * blockIdx.x + 1] = wall_clock64(); }
}

// Same kernel capped at 512 threads, so the compiler may use 256 VGPRs (the default 1024-thread
// bound caps it at 128 and the larger M variants spill). Used by the --msweep rows.
template <int E, int M>
__global__ __launch_bounds__(512) void k_gemv512(const unsigned short* W, const unsigned short* x, float* y, int N,
                                                 int K, int first, int copies, unsigned long long* stamps) {
  unsigned long long t0 = wall_clock64();
  for (int c = first; c < first + copies; ++c) gemv<E, M>(W + (size_t)c * N * K, x, y, N, K);
  __syncthreads();
  if (threadIdx.x == 0) { stamps[2 * blockIdx.x] = t0; stamps[2 * blockIdx.x + 1] = wall_clock64(); }
}

// ---- mfma_lds_sk: x staged in LDS per 128-k chunk, split-K across blocks ----
constexpr int LKC = 128;  // k per LDS chunk
// k per LDS chunk for the gemm_lds2 engines: 128, or 64 when NBUF chunks of 128 would not fit in 64 KB
__host__ __device__ constexpr int lds_kch(int M, int TM, bool DB) {
  return ((M + TM - 1) / TM * TM) * (128 + 8) * 2 * (DB ? 2 : 1) <= 65536 ? 128 : 64;
}
__host__ __device__ inline int lds_split(int N, int K, int nw, int nblocks, int rows = 16, int kch = LKC) {
  const int groups = N / (rows * nw), nk = K / kch;
  for (int d = 1; d <= nk; ++d)
    if (nk % d == 0 && groups * d >= nblocks) return d;
  return nk;
}
template <int M>
__device__ __forceinline__ void gemm_lds(const unsigned short* __restrict__ W, const unsigned short* __restrict__ x,
                                         float* __restrict__ y, int N, int K) {
  constexpr int MP = (M + 15) / 16 * 16, LS = LKC + 8, T = MP / 16;  // LS: +16 B row pad
  __shared__ __attribute__((aligned(16))) unsigned short xs[MP * LS];
  const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6, nw = blockDim.x >> 6, g = lane >> 4;
  const int S = lds_split(N, K, nw, gridDim.x), groups = N / (16 * nw), kper = K / LKC / S;
  for (int u = blockIdx.x; u < groups * S; u += gridDim.x) {
    const int grp = u / S, sl = u % S;
    const int tile = grp * 16 * nw + wv * 16, row = tile + (lane & 15);
    f4 c[T];
#pragma unroll
    for (int t = 0; t < T; ++t) c[t] = f4{0, 0, 0, 0};
    for (int kc = sl * kper; kc < (sl + 1) * kper; ++kc) {
      const int k0 = kc * LKC;
      uint4 w[LKC / 32];  // this wave's weight fragments, issued before the LDS round trip
#pragma unroll
      for (int j = 0; j < LKC / 32; ++j) w[j] = *(const uint4*)(W + (size_t)row * K + k0 + j * 32 + g * 8);
      __syncthreads();
      for (int i = threadIdx.x; i < MP * (LKC / 8); i += blockDim.x) {
        const int m = i / (LKC / 8), q = i % (LKC / 8);
        *(uint4*)(xs + m * LS + q * 8) = m < M ? *(const uint4*)(x + (size_t)m * K + k0 + q * 8) : make_uint4(0, 0, 0, 0);
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < LKC / 32; ++j) {
        s4 a0 = {(short)w[j].x, (short)(w[j].x >> 16), (short)w[j].y, (short)(w[j].y >> 16)};
        s4 a1 = {(short)w[j].z, (short)(w[j].z >> 16), (short)w[j].w, (short)(w[j].w >> 16)};
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const uint4 xv = *(const uint4*)(xs + (t * 16 + (lane & 15)) * LS + j * 32 + g * 8);
          s4 b0 = {(short)xv.x, (short)(xv.x >> 16), (short)xv.y, (short)(xv.y >> 16)};
          s4 b1 = {(short)xv.z, (short)(xv.z >> 16), (short)xv.w, (short)(xv.w >> 16)};
          c[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a0, b0, c[t], 0, 0, 0);
          c[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a1, b1, c[t], 0, 0, 0);
        }
      }
    }
    // D: lane l holds token column t*16 + l%16, rows g*4 + i of the tile
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const int m = t * 16 + (lane & 15);
      if (m < M) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float* p = y + (size_t)m * N + tile + g * 4 + i;
          if (S == 1) *p = c[t][i]; else unsafeAtomicAdd(p, c[t][i]);
        }
      }
    }
  }
}
// 16x16x16 MFMA, 16 rows per wave; DB = double-buffered LDS
template <int M, bool DB>
__device__ __forceinline__ void gemm_lds2(const unsigned short* __restrict__ W, const unsigned short* __restrict__ x,
                                          float* __restrict__ y, int N, int K) {
  constexpr int TM = 16, KC = lds_kch(M, TM, DB);
  constexpr int MP = (M + TM - 1) / TM * TM, LS = KC + 8, T = MP / TM, NBUF = DB ? 2 : 1;
  constexpr int KSTEP = 32;                  // k covered by one 16 B load per lane across the wave
  constexpr int NJ = KC / KSTEP;            // weight uint4 per lane per chunk
  constexpr int XV = MP * (KC / 8);         // uint4 in one x chunk
  constexpr int XR = (XV + 255) / 256;       // per-thread x registers (blockDim >= 256)
  __shared__ __attribute__((aligned(16))) unsigned short xs[NBUF][MP * LS];
  const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6, nw = blockDim.x >> 6;
  const int rr = lane % TM, g = lane / TM;
  const int S = lds_split(N, K, nw, gridDim.x, TM, KC), groups = N / (TM * nw), kper = K / KC / S;
  uint4 w[NJ], xr[XR];
  auto loadw = [&](int row, int kc, uint4* dst) {
#pragma unroll
    for (int j = 0; j < NJ; ++j) dst[j] = *(const uint4*)(W + (size_t)row * K + kc * KC + j * KSTEP + g * 8);
  };
  auto loadx = [&](int kc) {
#pragma unroll
    for (int i = 0; i < XR; ++i) {
      const int idx = threadIdx.x + i * blockDim.x, m = idx / (KC / 8), q = idx % (KC / 8);
      if (idx < XV) xr[i] = m < M ? *(const uint4*)(x + (size_t)m * K + kc * KC + q * 8) : make_uint4(0, 0, 0, 0);
    }
  };
  auto storex = [&](int b) {
#pragma unroll
    for (int i = 0; i < XR; ++i) {
      const int idx = threadIdx.x + i * blockDim.x, m = idx / (KC / 8), q = idx % (KC / 8);
      if (idx < XV) *(uint4*)(&xs[b][m * LS + q * 8]) = xr[i];
    }
  };
  for (int u = blockIdx.x; u < groups * S; u += gridDim.x) {
    const int grp = u / S, sl = u % S, kb = sl * kper, ke = kb + kper;
    const int tile = grp * TM * nw + wv * TM, row = tile + rr;
    f4 c[T];
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[t][i] = 0;
    auto compute = [&](int b) {
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        s4 a0 = {(short)w[j].x, (short)(w[j].x >> 16), (short)w[j].y, (short)(w[j].y >> 16)};
        s4 a1 = {(short)w[j].z, (short)(w[j].z >> 16), (short)w[j].w, (short)(w[j].w >> 16)};
#pragma unroll
        for (int t = 0; t < T; ++t) {
          const uint4 xv = *(const uint4*)(&xs[b][(t * TM + rr) * LS + j * KSTEP + g * 8]);
          s4 b0 = {(short)xv.x, (short)(xv.x >> 16), (short)xv.y, (short)(xv.y >> 16)};
          s4 b1 = {(short)xv.z, (short)(xv.z >> 16), (short)xv.w, (short)(xv.w >> 16)};
          c[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a0, b0, c[t], 0, 0, 0);
          c[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a1, b1, c[t], 0, 0, 0);
        }
      }
    };
    if constexpr (DB) {
      loadw(row, kb, w);
      loadx(kb);
      storex(0);
      __syncthreads();
      for (int kc = kb; kc < ke; ++kc) {
        const int b = (kc - kb) & 1;
        uint4 wn[NJ];
        const bool more = kc + 1 < ke;
        if (more) { loadw(row, kc + 1, wn); loadx(kc + 1); }  // in flight during compute(b)
        compute(b);
        if (more) storex(b ^ 1);  // buffer b^1 was last read before the previous __syncthreads
        __syncthreads();
        if (more)
#pragma unroll
          for (int j = 0; j < NJ; ++j) w[j] = wn[j];
      }
    } else {
      for (int kc = kb; kc < ke; ++kc) {
        loadw(row, kc, w);
        __syncthreads();
        loadx(kc);
        storex(0);
        __syncthreads();
        compute(0);
      }
    }
    // D: lane holds token column t*16 + l%16, rows g*4 + i of the tile
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const int m = t * TM + rr;
      if (m < M) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float* p = y + (size_t)m * N + tile + g * 4 + i;
          if (S == 1) *p = c[t][i]; else unsafeAtomicAdd(p, c[t][i]);
        }
      }
    }
  }
}
template <int M, bool DB>
__global__ __launch_bounds__(512) void k_gemm_lds2(const unsigned short* W, const unsigned short* x, float* y, int N,
                                                   int K, int first, int copies, unsigned long long* stamps) {
  unsigned long long t0 = wall_clock64();
  for (int c = first; c < first + copies; ++c) gemm_lds2<M, DB>(W + (size_t)c * N * K, x, y, N, K);
  __syncthreads();
  if (threadIdx.x == 0) { stamps[2 * blockIdx.x] = t0; stamps[2 * blockIdx.x + 1] = wall_clock64(); }
}

template <int M>
__global__ __launch_bounds__(512) void k_gemm_lds(const unsigned short* W, const unsigned short* x, float* y, int N,
                                                  int K, int first, int copies, unsigned long long* stamps) {
  unsigned long long t0 = wall_clock64();
  for (int c = first; c < first + copies; ++c) gemm_lds<M>(W + (size_t)c * N * K, x, y, N, K);
  __syncthreads();
  if (threadIdx.x == 0) { stamps[2 * blockIdx.x] = t0; stamps[2 * blockIdx.x + 1] = wall_clock64(); }
}

typedef void (*KFn)(const unsigned short*, const unsigned short*, float*, int, int, int, int, unsigned long long*);
static KFn kfns[] = {k_gemv<0>, k_gemv<1>, k_gemv<2>, k_gemv<3>};
// small-M (tokens) variants: {engine, M, kernel}
struct MK { int e, m; KFn k; };
static const MK mkfns[] = {{E_VALU1, 4, k_gemv<0, 4>}, {E_VALU1, 8, k_gemv<0, 8>}, {E_VALU1, 16, k_gemv<0, 16>},
                           {E_VALU4, 4, k_gemv<1, 4>},
                           {E_MFMA16, 4, k_gemv<2, 4>}, {E_MFMA16, 8, k_gemv<2, 8>}, {E_MFMA16, 16, k_gemv<2, 16>}};
// large-M sweep (--msweep, together with the rows above and M = 1)
static const MK mkbig[] = {
    {E_VALU1, 1, k_gemv512<0, 1>}, {E_VALU1, 16, k_gemv512<0, 16>}, {E_VALU1, 32, k_gemv512<0, 32>},
    {E_VALU1, 64, k_gemv512<0, 64>}, {E_VALU1, 128, k_gemv512<0, 128>},
    {E_VALU4, 16, k_gemv512<1, 16>}, {E_VALU4, 32, k_gemv512<1, 32>}, {E_VALU4, 64, k_gemv512<1, 64>},
    {E_VALU4, 128, k_gemv512<1, 128>},
    {E_MFMA16, 16, k_gemv512<2, 16>}, {E_MFMA16, 32, k_gemv512<2, 32>}, {E_MFMA16, 64, k_gemv512<2, 64>},
    {E_MFMA16, 128, k_gemv512<2, 128>},
    {E_MLDS, 1, k_gemm_lds<1>}, {E_MLDS, 4, k_gemm_lds<4>}, {E_MLDS, 8, k_gemm_lds<8>}, {E_MLDS, 16, k_gemm_lds<16>},
    {E_MLDS, 32, k_gemm_lds<32>}, {E_MLDS, 64, k_gemm_lds<64>}, {E_MLDS, 128, k_gemm_lds<128>},
    {E_MLDS_DB, 1, k_gemm_lds2<1, true>}, {E_MLDS_DB, 4, k_gemm_lds2<4, true>},
    {E_MLDS_DB, 8, k_gemm_lds2<8, true>}, {E_MLDS_DB, 16, k_gemm_lds2<16, true>},
    {E_MLDS_DB, 32, k_gemm_lds2<32, true>}, {E_MLDS_DB, 64, k_gemm_lds2<64, true>},
    {E_MLDS_DB, 128, k_gemm_lds2<128, true>},};
static const int MMAX = 128;

static float bf2f(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
  const bool msweep = argc > 1 && !strcmp(argv[1], "--msweep");
  const int ncu = num_cus();
  const double mhz = wall_mhz();
  // decode shapes: Qwen3-0.6B qkv / o / gate+up / down, Qwen3-30B-A3B expert gate+up / down,
  // Llama-3.1-8B qkv, and a vocabulary head
  const int shapes[][2] = {{4096, 1024}, {1024, 2048}, {6144, 1024}, {1024, 3072},
                           {1536, 2048}, {2048, 768}, {6144, 4096}, {151936, 1024}};
  const size_t budget = (size_t)1 << 30;  // bytes of W copies per shape (4x the Infinity Cache)
  unsigned short *dW, *dx; float* dy; unsigned long long* st;
  CK(hipMalloc(&dW, budget + ((size_t)151936 * 1024 * 2))); CK(hipMalloc(&dx, (size_t)MMAX * 4096 * 2));
  CK(hipMalloc(&dy, (size_t)MMAX * 151936 * 4)); CK(hipMalloc(&st, 2 * 2 * ncu * 8));
  for (auto& sh : shapes) {
    const int N = sh[0], K = sh[1];
    const bool small_m = (N == 4096 && K == 1024) || (N == 6144 && K == 4096) || (N == 1536 && K == 2048) ||
                         (msweep && N == 2048 && K == 768);
    if (msweep && !small_m) continue;
    const int mref = small_m ? (msweep ? MMAX : 16) : 1;
    const size_t wbytes = (size_t)N * K * 2;
    const int copies = (int)std::max<size_t>(2, budget / wbytes);
    std::vector<unsigned short> hW((size_t)N * K), hx((size_t)MMAX * K);
    uint32_t s = 12345u + N + K;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; float f = ((s >> 9) & 0x3fff) / 16384.f - 0.5f; unsigned u; memcpy(&u, &f, 4); return (unsigned short)(u >> 16); };
    for (auto& v : hW) v = rnd();
    for (auto& v : hx) v = rnd();
    for (int c = 0; c < copies; ++c) CK(hipMemcpy(dW + (size_t)c * N * K, hW.data(), wbytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dx, hx.data(), (size_t)MMAX * K * 2, hipMemcpyHostToDevice));
    std::vector<double> ref((size_t)mref * N);
    for (int m = 0; m < mref; ++m)
      for (int n = 0; n < N; ++n) {
        double a = 0;
        for (int k = 0; k < K; ++k) a += (double)bf2f(hW[(size_t)n * K + k]) * bf2f(hx[(size_t)m * K + k]);
        ref[(size_t)m * N + n] = a;
      }
    auto run = [&](int e, int M, KFn kf, int threads, int maxthr = 1024) {
      const int nb = ncu;
      const bool lds = e >= E_MLDS;
      const int kch = e == E_MLDS ? LKC : lds_kch(M, eng_rows[e], true);
      const int split = lds ? lds_split(N, K, threads / 64, nb, eng_rows[e], kch) : 1;
      if (lds && (N % (eng_rows[e] * (threads / 64)) || K % kch)) return;
      CK(hipMemset(dy, 0, (size_t)M * N * 4));
      hipLaunchKernelGGL(kf, dim3(nb), dim3(threads), 0, 0, dW, dx, dy, N, K, 0, 1, st);
      CK(hipDeviceSynchronize());
      std::vector<float> y((size_t)M * N); CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
      double maxerr = 0, scale = 0;
      for (size_t i = 0; i < y.size(); ++i) { maxerr = std::max(maxerr, std::fabs(y[i] - ref[i])); scale = std::max(scale, std::fabs(ref[i])); }
      const bool ok = maxerr <= 1e-3 * std::max(1.0, scale);
      std::vector<double> single;
      for (int r = 0; r < 15; ++r) {
        hipLaunchKernelGGL(kf, dim3(nb), dim3(threads), 0, 0, dW, dx, dy, N, K, r % copies, 1, st);
        CK(hipDeviceSynchronize());
        std::vector<unsigned long long> h(2 * nb); CK(hipMemcpy(h.data(), st, 2 * nb * 8, hipMemcpyDeviceToHost));
        unsigned long long a = ~0ull, b = 0;
        for (int i = 0; i < nb; ++i) { a = std::min(a, h[2 * i]); b = std::max(b, h[2 * i + 1]); }
        if (r >= 3) single.push_back((b - a) / mhz);
      }
      auto ms = time_ms([&] {
        hipLaunchKernelGGL(kf, dim3(nb), dim3(threads), 0, 0, dW, dx, dy, N, K, 0, copies, st);
      }, 5, 1);
      const double gbps = (double)wbytes * copies / (median(ms) * 1e-3) / 1e9;
      Json j; j.s("family", "gemv").s("primitive", std::string("gemv.bf16.") + eng_name[e]).dev("gemv_amd.hip");
      j.i("p.N", N).i("p.K", K).i("p.M", M).i("p.blocks", nb).i("p.threads", threads).i("p.rows_per_wave_unit", eng_rows[e]).i("p.launch_bounds", maxthr).i("p.split_k", split).i("p.k_chunk", lds ? kch : 0)
       .f("m.us_single", median(single)).f("m.gbps_single", wbytes / (median(single) * 1e-6) / 1e9)
       .f("m.gbps_stream", gbps).f("m.tflops_single", 2.0 * M * N * K / (median(single) * 1e-6) / 1e12).f("m.max_abs_err", maxerr).s("c.l2", "cold").i("c.copies", copies)
       .t("verified", ok).i("n_reps", (long long)single.size()).f("cv", cv(single))
       .s("s.timer", "wall_clock64 span (single), hipEvent (stream)")
       .s("note", "M tokens share each weight load; us_single: one GEMV per launch, max block end - min block "
          "start (no launch overhead); gbps_stream: back-to-back GEMVs over cold copies in one launch").print();
    };
    for (int e = 0; e < NENG; ++e)
      for (int threads : {256, 512, 1024})
        if (!msweep || threads != 1024) run(e, 1, kfns[e], threads);
    if (small_m)
      for (const MK& mk : mkfns)
        for (int threads : {256, 512}) run(mk.e, mk.m, mk.k, threads);
    if (msweep)
      for (const MK& mk : mkbig)
        for (int threads : {256, 512}) run(mk.e, mk.m, mk.k, threads, 512);
  }
  return 0;
}
