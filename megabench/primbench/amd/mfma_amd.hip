// MFMA instruction latency and issue rate, and wave64 reduction cost, on CDNA.
//   mfma : each instruction shape, 1 dependent chain (latency) and 4 independent chains
//          (issue rate) per wave, 4 waves per CU (one per SIMD), all CUs; all-ones operands so
//          every accumulator must equal iters*K (checked).
//   reduce : 64-lane fp32 sum by __shfl_xor (ds_bpermute) vs DPP + readlane, cycles per sum.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx942 -o mfma_amd mfma_amd.hip
#include "pb_hip.h"
#include <cstring>

typedef __attribute__((ext_vector_type(2))) int i2;
typedef __attribute__((ext_vector_type(4))) short s4;
typedef __attribute__((ext_vector_type(4))) _Float16 h4;
typedef __attribute__((ext_vector_type(4))) float f4;
typedef __attribute__((ext_vector_type(16))) float f16v;
typedef __attribute__((ext_vector_type(4))) int i4;
typedef __attribute__((ext_vector_type(16))) int i16v;

// One struct per instruction: operand/accumulator types, shape, an all-ones operand, the op.
// Every struct exists in every compile pass; on an arch without the instruction the op is a
// no-op and main() never launches it.
#define MFMA_DEF(NAME, STR, AT, CT, M_, N_, K_, B_, ONE, GUARD, CALL)                       \
  struct NAME {                                                                            \
    typedef AT A; typedef CT C;                                                            \
    static constexpr int M = M_, N = N_, K = K_, B = B_;                                   \
    static constexpr const char* name = STR;                                               \
    __device__ static A one() { return ONE; }                                              \
    __device__ static C op(A a, A b, C c) {                                                \
      if constexpr (GUARD) { return CALL; } else { (void)a; (void)b; return c; }           \
    }                                                                                      \
  };

#if defined(__gfx942__) || defined(__gfx950__)
#define HAS_FP8 1
#else
#define HAS_FP8 0
#endif
#if defined(__gfx950__)
#define HAS_950 1
#define FP8_ONE 0x3838383838383838LL  // OCP e4m3 1.0
typedef __attribute__((ext_vector_type(8))) __bf16 bf8v;
#define BF8ONE (bf8v{(__bf16)1.f, (__bf16)1.f, (__bf16)1.f, (__bf16)1.f, (__bf16)1.f, (__bf16)1.f, (__bf16)1.f, (__bf16)1.f})
#define CALL950(f) f(a, b, c, 0, 0, 0)
#else
#define HAS_950 0
#define FP8_ONE 0x4040404040404040LL  // e4m3fnuz 1.0 (gfx942)
typedef __attribute__((ext_vector_type(8))) short bf8v;
#define BF8ONE (bf8v{})
#define CALL950(f) c
#endif
#if HAS_FP8
#define CALLFP8(f) f(a, b, c, 0, 0, 0)
#else
#define CALLFP8(f) c
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#define CALLALL(f) f(a, b, c, 0, 0, 0)
#else
#define CALLALL(f) c
#undef CALLFP8
#define CALLFP8(f) c
#undef CALL950
#define CALL950(f) c
#endif

MFMA_DEF(Bf16_32x32x8, "v_mfma_f32_32x32x8_bf16", s4, f16v, 32, 32, 8, 1, (s4{0x3F80, 0x3F80, 0x3F80, 0x3F80}), true,
         CALLALL(__builtin_amdgcn_mfma_f32_32x32x8bf16_1k))
MFMA_DEF(Bf16_16x16x16, "v_mfma_f32_16x16x16_bf16", s4, f4, 16, 16, 16, 1, (s4{0x3F80, 0x3F80, 0x3F80, 0x3F80}), true,
         CALLALL(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k))
MFMA_DEF(Bf16_4x4x4, "v_mfma_f32_4x4x4_16b_bf16", s4, f4, 4, 4, 4, 16, (s4{0x3F80, 0x3F80, 0x3F80, 0x3F80}), true,
         CALLALL(__builtin_amdgcn_mfma_f32_4x4x4bf16_1k))
MFMA_DEF(F16_32x32x8, "v_mfma_f32_32x32x8_f16", h4, f16v, 32, 32, 8, 1, (h4{1, 1, 1, 1}), true,
         CALLALL(__builtin_amdgcn_mfma_f32_32x32x8f16))
MFMA_DEF(F16_16x16x16, "v_mfma_f32_16x16x16_f16", h4, f4, 16, 16, 16, 1, (h4{1, 1, 1, 1}), true,
         CALLALL(__builtin_amdgcn_mfma_f32_16x16x16f16))
MFMA_DEF(Fp8_16x16x32, "v_mfma_f32_16x16x32_fp8_fp8", long, f4, 16, 16, 32, 1, (long)FP8_ONE, HAS_FP8,
         CALLFP8(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8))
MFMA_DEF(Fp8_32x32x16, "v_mfma_f32_32x32x16_fp8_fp8", long, f16v, 32, 32, 16, 1, (long)FP8_ONE, HAS_FP8,
         CALLFP8(__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8))
MFMA_DEF(I8_16x16x32, "v_mfma_i32_16x16x32_i8", long, i4, 16, 16, 32, 1, (long)0x0101010101010101LL, HAS_FP8,
         CALLFP8(__builtin_amdgcn_mfma_i32_16x16x32_i8))
MFMA_DEF(Bf16_16x16x32, "v_mfma_f32_16x16x32_bf16", bf8v, f4, 16, 16, 32, 1, BF8ONE, HAS_950,
         CALL950(__builtin_amdgcn_mfma_f32_16x16x32_bf16))
MFMA_DEF(Bf16_32x32x16, "v_mfma_f32_32x32x16_bf16", bf8v, f16v, 32, 32, 16, 1, BF8ONE, HAS_950,
         CALL950(__builtin_amdgcn_mfma_f32_32x32x16_bf16))

template <class I, int CH>
__global__ void k_mfma(int iters, float* out, long long* cyc) {
  typename I::A a = I::one();
  typename I::C c[CH];
#pragma unroll
  for (int ch = 0; ch < CH; ++ch) c[ch] = typename I::C{};
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int ch = 0; ch < CH; ++ch) c[ch] = I::op(a, a, c[ch]);
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int ch = 0; ch < CH; ++ch) s += (float)c[ch][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s / CH;
}

template <class I, int CH>
static void mfma_point() {
  const int ncu = num_cus(), threads = 256, nb = ncu, iters = 4096;
  float* out; long long* cyc;
  CK(hipMalloc(&out, (size_t)nb * threads * 4)); CK(hipMalloc(&cyc, nb * 8));
  auto ms = time_ms([&] { hipLaunchKernelGGL((k_mfma<I, CH>), dim3(nb), dim3(threads), 0, 0, iters, out, cyc); }, 5, 1);
  std::vector<float> o((size_t)nb * threads); CK(hipMemcpy(o.data(), out, o.size() * 4, hipMemcpyDeviceToHost));
  std::vector<long long> cy(nb); CK(hipMemcpy(cy.data(), cyc, nb * 8, hipMemcpyDeviceToHost));
  const float expect = (float)iters * I::K;
  bool ok = true; for (float v : o) ok &= (v == expect);
  std::vector<double> c; for (auto v : cy) c.push_back((double)v / iters / CH);
  const double t = median(ms) * 1e-3;
  const double flops = 2.0 * I::M * I::N * I::K * I::B * iters * CH * (threads / 64) * nb;
  const double clk_mhz = (double)median(std::vector<double>(cy.begin(), cy.end())) / t / 1e6;
  Json j; j.s("family", "mfma").s("primitive", I::name).dev("mfma_amd.hip");
  j.i("p.M", I::M).i("p.N", I::N).i("p.K", I::K).i("p.blocks_per_instr", I::B).i("p.chains", CH)
   .i("p.waves_per_cu", threads / 64)
   .f(CH == 1 ? "m.latency_cyc" : "m.issue_cyc", median(c)).f("m.tflops_device", flops / t / 1e12)
   .f("c.clock_mhz_est", clk_mhz).t("verified", ok).i("n_reps", (long long)ms.size()).f("cv", cv(ms))
   .s("s.timer", "clock64 (s_memtime) per wave; hipEvent for tflops")
   .s("note", CH == 1 ? "dependent chain: cycles per instruction = issue-to-result latency of one wave"
                      : "4 independent accumulators per wave, one wave per SIMD: cycles per instruction per wave").print();
  CK(hipFree(out)); CK(hipFree(cyc));
}

// ---------------------------------------------------------------- reductions
__device__ __forceinline__ float dpp_f(float v, int ctrl);
#define DPP(v, ctrl) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xf, 0xf, false))
__device__ __forceinline__ float wave_sum_dpp(float v) {
  v += DPP(v, 0xb1);   // quad_perm [1,0,3,2]
  v += DPP(v, 0x4e);   // quad_perm [2,3,0,1]
  v += DPP(v, 0x141);  // row_half_mirror
  v += DPP(v, 0x140);  // row_mirror: every lane of a row of 16 holds the row sum
  return __builtin_amdgcn_readlane(__float_as_int(v), 0) == 0 && false ? 0.f :
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0)) +
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16)) +
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32)) +
         __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
}
__device__ __forceinline__ float wave_sum_shfl(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

template <int KIND>
__global__ void k_reduce(int iters, float* out, long long* cyc) {
  float v = (float)(threadIdx.x & 63);
  float acc = 0;
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
    float s = KIND ? wave_sum_dpp(v) : wave_sum_shfl(v);
    acc += s;
    v = v + s * 0.0f;  // keep the chain dependent
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc / iters;
}

static void reduce_points() {
  const int nb = num_cus(), threads = 256, iters = 4096;
  float* out; long long* cyc;
  CK(hipMalloc(&out, (size_t)nb * threads * 4)); CK(hipMalloc(&cyc, nb * 8));
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<double> reps;
    for (int r = 0; r < 5; ++r) {
      if (kind) hipLaunchKernelGGL(k_reduce<1>, dim3(nb), dim3(threads), 0, 0, iters, out, cyc);
      else hipLaunchKernelGGL(k_reduce<0>, dim3(nb), dim3(threads), 0, 0, iters, out, cyc);
      CK(hipDeviceSynchronize());
      std::vector<long long> cy(nb); CK(hipMemcpy(cy.data(), cyc, nb * 8, hipMemcpyDeviceToHost));
      std::vector<double> c; for (auto v : cy) c.push_back((double)v / iters);
      reps.push_back(median(c));
    }
    std::vector<float> o((size_t)nb * threads); CK(hipMemcpy(o.data(), out, o.size() * 4, hipMemcpyDeviceToHost));
    bool ok = true; for (float v : o) ok &= (v == 2016.f);  // sum 0..63
    Json j; j.s("family", "simt").s("primitive", kind ? "wave_sum.dpp_readlane" : "wave_sum.shfl_xor").dev("mfma_amd.hip");
    j.i("p.lanes", 64).i("p.waves_per_cu", threads / 64).f("m.cyc", median(reps)).t("verified", ok)
     .i("n_reps", (long long)reps.size()).f("cv", cv(reps)).s("s.timer", "clock64")
     .s("note", kind ? "quad_perm x2, row_half_mirror, row_mirror DPP adds, then 4 v_readlane: every lane gets the sum"
                     : "6 rounds of __shfl_xor (ds_bpermute_b32): every lane gets the sum").print();
  }
  CK(hipFree(out)); CK(hipFree(cyc));
}

int main(int argc, char** argv) {
  const std::string a = arch_name();
  const bool all = argc < 2;
  auto want = [&](const char* m) { if (all) return true; for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], m)) return true; return false; };
  if (want("mfma")) {
    mfma_point<Bf16_32x32x8, 1>(); mfma_point<Bf16_32x32x8, 4>();
    mfma_point<Bf16_16x16x16, 1>(); mfma_point<Bf16_16x16x16, 4>();
    mfma_point<Bf16_4x4x4, 1>(); mfma_point<Bf16_4x4x4, 4>();
    mfma_point<F16_32x32x8, 4>(); mfma_point<F16_16x16x16, 4>();
    if (a == "gfx942" || a == "gfx950") {
      mfma_point<Fp8_16x16x32, 1>(); mfma_point<Fp8_16x16x32, 4>();
      mfma_point<Fp8_32x32x16, 1>(); mfma_point<Fp8_32x32x16, 4>();
      mfma_point<I8_16x16x32, 4>();
    }
    if (a == "gfx950") {
      mfma_point<Bf16_16x16x32, 1>(); mfma_point<Bf16_16x16x32, 4>();
      mfma_point<Bf16_32x32x16, 1>(); mfma_point<Bf16_32x32x16, 4>();
    }
  }
  if (want("reduce")) reduce_points();
  return 0;
}
