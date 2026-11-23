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
static const char* eng_name[] = {"valu_r1", "valu_r4", "mfma16", "mfma4"};
static const int eng_rows[] = {1, 4, 16, 4};

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
      float acc[R][M];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0;
      for (int k = lane * 8; k < K; k += 512) {
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r] = *(const uint4*)(W + (size_t)(r0 + r) * K + k);
#pragma unroll
        for (int m = 0; m < M; ++m) {
          uint4 xv = *(const uint4*)(x + (size_t)m * K + k);
#pragma unroll
          for (int r = 0; r < R; ++r) acc[r][m] += dot8(wv[r], xv);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) { float s = wave_sum(acc[r][m]); if (lane == 0) y[(size_t)m * N + r0 + r] = s; }
    } else if (E == E_MFMA16) {
      // lane l: row r0 + l%16, k-group l/16 (8 contiguous k per group per 32-k step)
      const int row = r0 + (lane & 15), g = lane >> 4;
      f4 c = {0, 0, 0, 0};
      for (int k = 0; k < K; k += 32) {
        uint4 wv = *(const uint4*)(W + (size_t)row * K + k + g * 8);
        const int col = M == 1 ? 0 : (lane & 15);  // token in this column (M == 1: x broadcast)
        uint4 xv = col < M ? *(const uint4*)(x + (size_t)col * K + k + g * 8) : make_uint4(0, 0, 0, 0);
        s4 a0 = {(short)wv.x, (short)(wv.x >> 16), (short)wv.y, (short)(wv.y >> 16)};
        s4 a1 = {(short)wv.z, (short)(wv.z >> 16), (short)wv.w, (short)(wv.w >> 16)};
        s4 b0 = {(short)xv.x, (short)(xv.x >> 16), (short)xv.y, (short)(xv.y >> 16)};
        s4 b1 = {(short)xv.z, (short)(xv.z >> 16), (short)xv.w, (short)(xv.w >> 16)};
        c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a0, b0, c, 0, 0, 0);
        c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a1, b1, c, 0, 0, 0);
      }
      // D: lane l holds column l%16, rows (l/16)*4 + i; every column is the same
      if ((M == 1 && (lane & 15) == 0) || (M > 1 && (lane & 15) < M)) {
        const int m = M == 1 ? 0 : (lane & 15);
#pragma unroll
        for (int i = 0; i < 4; ++i) y[(size_t)m * N + r0 + g * 4 + i] = c[i];
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

typedef void (*KFn)(const unsigned short*, const unsigned short*, float*, int, int, int, int, unsigned long long*);
static KFn kfns[] = {k_gemv<0>, k_gemv<1>, k_gemv<2>, k_gemv<3>};
// small-M (tokens) variants: {engine, M, kernel}
struct MK { int e, m; KFn k; };
static const MK mkfns[] = {{E_VALU1, 4, k_gemv<0, 4>}, {E_VALU1, 8, k_gemv<0, 8>}, {E_VALU1, 16, k_gemv<0, 16>},
                           {E_VALU4, 4, k_gemv<1, 4>},
                           {E_MFMA16, 4, k_gemv<2, 4>}, {E_MFMA16, 8, k_gemv<2, 8>}, {E_MFMA16, 16, k_gemv<2, 16>}};

static float bf2f(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main() {
  const int ncu = num_cus();
  const double mhz = wall_mhz();
  // decode shapes: Qwen3-0.6B qkv / o / gate+up / down, Qwen3-30B-A3B expert gate+up / down,
  // Llama-3.1-8B qkv, and a vocabulary head
  const int shapes[][2] = {{4096, 1024}, {1024, 2048}, {6144, 1024}, {1024, 3072},
                           {1536, 2048}, {2048, 768}, {6144, 4096}, {151936, 1024}};
  const size_t budget = (size_t)1 << 30;  // bytes of W copies per shape (4x the Infinity Cache)
  unsigned short *dW, *dx; float* dy; unsigned long long* st;
  CK(hipMalloc(&dW, budget + ((size_t)151936 * 1024 * 2))); CK(hipMalloc(&dx, (size_t)16 * 4096 * 2));
  CK(hipMalloc(&dy, (size_t)16 * 151936 * 4)); CK(hipMalloc(&st, 2 * 2 * ncu * 8));
  for (auto& sh : shapes) {
    const int N = sh[0], K = sh[1];
    const size_t wbytes = (size_t)N * K * 2;
    const int copies = (int)std::max<size_t>(2, budget / wbytes);
    std::vector<unsigned short> hW((size_t)N * K), hx((size_t)16 * K);
    uint32_t s = 12345u + N + K;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; float f = ((s >> 9) & 0x3fff) / 16384.f - 0.5f; unsigned u; memcpy(&u, &f, 4); return (unsigned short)(u >> 16); };
    for (auto& v : hW) v = rnd();
    for (auto& v : hx) v = rnd();
    for (int c = 0; c < copies; ++c) CK(hipMemcpy(dW + (size_t)c * N * K, hW.data(), wbytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dx, hx.data(), (size_t)16 * K * 2, hipMemcpyHostToDevice));
    std::vector<double> ref((size_t)16 * N);
    const bool small_m = (N == 4096 && K == 1024) || (N == 6144 && K == 4096) || (N == 1536 && K == 2048);
    for (int m = 0; m < (small_m ? 16 : 1); ++m)
      for (int n = 0; n < N; ++n) {
        double a = 0;
        for (int k = 0; k < K; ++k) a += (double)bf2f(hW[(size_t)n * K + k]) * bf2f(hx[(size_t)m * K + k]);
        ref[(size_t)m * N + n] = a;
      }
    auto run = [&](int e, int M, KFn kf, int threads) {
      const int nb = ncu;
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
      j.i("p.N", N).i("p.K", K).i("p.M", M).i("p.blocks", nb).i("p.threads", threads).i("p.rows_per_wave_unit", eng_rows[e])
       .f("m.us_single", median(single)).f("m.gbps_single", wbytes / (median(single) * 1e-6) / 1e9)
       .f("m.gbps_stream", gbps).f("m.max_abs_err", maxerr).s("c.l2", "cold").i("c.copies", copies)
       .t("verified", ok).i("n_reps", (long long)single.size()).f("cv", cv(single))
       .s("s.timer", "wall_clock64 span (single), hipEvent (stream)")
       .s("note", "M tokens share each weight load; us_single: one GEMV per launch, max block end - min block "
          "start (no launch overhead); gbps_stream: back-to-back GEMVs over cold copies in one launch").print();
    };
    for (int e = 0; e < NENG; ++e)
      for (int threads : {256, 512, 1024}) run(e, 1, kfns[e], threads);
    if (small_m)
      for (const MK& mk : mkfns)
        for (int threads : {256, 512}) run(mk.e, mk.m, mk.k, threads);
  }
  return 0;
}
