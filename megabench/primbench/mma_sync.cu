// Warp-level mma.sync throughput and latency on sm_100a, per shape and dtype.
// One CTA per SM with W warps; each warp runs ILP independent accumulator chains.
// Cycles per MMA come from the slope between two loop lengths (fixed cost removed).
// Verification: all-ones operands, so every accumulator must equal K * iters.
// fp8 (e4m3/e5m2) mma.sync has no native SASS on sm_100a: it lowers to F2FP to f16, two
// HMMA.16816 and FADDs, and fails the all-ones check; it runs only when named explicitly.
// Usage: ./mma_sync [shape_filter]   (JSON lines on stdout)
#include "pb_common.cuh"
#include <cstring>
#include <cuda_fp16.h>

enum Shape { BF16_F32, F16_F32, F16_F16, TF32_F32, S8_S32, E4M3_F32, E5M2_F32, NSHAPE };
struct ShapeInfo { const char* name; const char* shape; const char* dtype; const char* acc; int M, N, K; uint32_t one; };
static const ShapeInfo SH[NSHAPE] = {
  {"bf16_f32", "m16n8k16", "bf16", "f32", 16, 8, 16, 0x3F803F80u},
  {"f16_f32", "m16n8k16", "f16", "f32", 16, 8, 16, 0x3C003C00u},
  {"f16_f16", "m16n8k16", "f16", "f16", 16, 8, 16, 0x3C003C00u},
  {"tf32_f32", "m16n8k8", "tf32", "f32", 16, 8, 8, 0x3F800000u},
  {"s8_s32", "m16n8k32", "s8", "s32", 16, 8, 32, 0x01010101u},
  {"e4m3_f32", "m16n8k32", "e4m3", "f32", 16, 8, 32, 0x38383838u},
  {"e5m2_f32", "m16n8k32", "e5m2", "f32", 16, 8, 32, 0x3C3C3C3Cu},
};

template <int S>
__device__ __forceinline__ void mma(uint32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  if constexpr (S == BF16_F32)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if constexpr (S == F16_F32)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if constexpr (S == F16_F16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if constexpr (S == TF32_F32)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if constexpr (S == S8_S32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if constexpr (S == E4M3_F32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int S, int ILP>
__global__ void __launch_bounds__(1024, 1) k_mma(uint32_t one, int iters, uint32_t* out, Stamp* st) {
  uint32_t a[4] = {one, one, one, one}, b[2] = {one, one};
  uint32_t c[ILP][4] = {};
  __syncthreads();
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) {
    // Opaque operands each iteration: stops the compiler hoisting loop-invariant work
    // (fp8 mma.sync lowers to cvt + f16 HMMA on sm_100a, and the cvt must be paid per MMA).
    asm volatile("" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(b[0]), "+r"(b[1]));
#pragma unroll
    for (int j = 0; j < ILP; ++j) mma<S>(c[j], a, b);
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (threadIdx.x == 0) st[blockIdx.x] = {t1 - t0, g1 - g0};
  if (blockIdx.x == 0 && threadIdx.x == 0) for (int r = 0; r < 4; ++r) out[r] = c[0][r];
  // Keep every chain live (ptxas drops MMAs whose results are unused); never true at run time.
  if (iters < 0) for (int j = 0; j < ILP; ++j) for (int r = 0; r < 4; ++r) out[4 + j * 4 + r] = c[j][r];
}

typedef void (*Fn)(uint32_t, int, uint32_t*, Stamp*);
template <int S> static Fn pick(int ilp) {
  switch (ilp) { case 1: return k_mma<S, 1>; case 2: return k_mma<S, 2>; case 4: return k_mma<S, 4>; default: return k_mma<S, 8>; }
}
static Fn pick(int s, int ilp) {
  switch (s) {
    case BF16_F32: return pick<BF16_F32>(ilp); case F16_F32: return pick<F16_F32>(ilp);
    case F16_F16: return pick<F16_F16>(ilp); case TF32_F32: return pick<TF32_F32>(ilp);
    case S8_S32: return pick<S8_S32>(ilp); case E4M3_F32: return pick<E4M3_F32>(ilp);
    default: return pick<E5M2_F32>(ilp);
  }
}

static bool check(int s, const uint32_t* o, int iters) {
  double want = (double)SH[s].K * iters;
  if (s == F16_F16) { __half_raw h; h.x = o[0] & 0xFFFF; return __half2float(__half(h)) == (float)want; }
  if (s == S8_S32) return (int)o[0] == (int)want;
  float f; memcpy(&f, &o[0], 4); return f == (float)want;
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  int nsm = num_sms();
  uint32_t* out; Stamp* st;
  CK(cudaMalloc(&out, 16 * 9)); CK(cudaMalloc(&st, sizeof(Stamp) * nsm));
  const int I1 = 256, I2 = 2304, VERIFY_IT = 8;
  for (int s = 0; s < NSHAPE; ++s) {
    if (only ? !strstr(SH[s].name, only) : (s == E4M3_F32 || s == E5M2_F32)) continue;  // fp8 is emulated: opt-in only
    for (int ctas : {1, nsm})
    for (int W : {1, 2, 4, 8, 16, 32})
    for (int ilp : {1, 2, 4, 8}) {
      if (ctas > 1 && !(ilp == 4 && (W == 4 || W == 8 || W == 16))) continue;  // all-SM: a few saturating points
      Fn k = pick(s, ilp);
      uint32_t hout[4];
      k<<<1, 32 * W>>>(SH[s].one, VERIFY_IT, out, st);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hout, out, 16, cudaMemcpyDeviceToHost));
      bool ok = check(s, hout, VERIFY_IT);
      std::vector<double> per, mhz;
      for (int rep = 0; rep < 5; ++rep) {
        double c[2];
        int its[2] = {I1, I2};
        for (int x = 0; x < 2; ++x) {
          k<<<ctas, 32 * W>>>(SH[s].one, its[x], out, st);
          CK(cudaDeviceSynchronize());
          std::vector<Stamp> h(ctas);
          CK(cudaMemcpy(h.data(), st, sizeof(Stamp) * ctas, cudaMemcpyDeviceToHost));
          std::vector<double> cy; for (auto& v : h) cy.push_back((double)v.cyc);
          c[x] = median(cy);
          if (x == 1) mhz.push_back(clock_mhz(h));
        }
        per.push_back((c[1] - c[0]) / (I2 - I1));  // cycles per loop iteration (W*ILP MMAs per SM)
      }
      double it = median(per), cyc_sm = it / (W * ilp);
      const ShapeInfo& I = SH[s];
      Json j; j.s("family", "mma_sync").s("primitive", "mma.sync")
        .s("p.shape", I.shape).s("p.dtype", I.dtype).s("p.acc", I.acc)
        .i("p.M", I.M).i("p.N", I.N).i("p.K", I.K).i("p.warps", W).i("p.ilp", ilp)
        .f("m.cyc_per_instr_sm", cyc_sm).f("m.fma_per_clk_sm", (double)I.M * I.N * I.K / cyc_sm)
        .i("c.ctas", ctas).f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", (long long)per.size()).f("cv", cv(per));
      if (W == 1 && ilp == 1) j.f("m.latency_cyc", it);
      if (ctas > 1) j.f("m.allsm_tflops", (double)I.M * I.N * I.K * 2 / cyc_sm * ctas * median(mhz) * 1e6 / 1e12);
      j.s("note", "cyc_per_instr_sm = SM cycles per mma.sync with W warps x ILP chains in flight; "
                  "latency_cyc (W=1, ILP=1) = dependent issue-to-issue time");
      j.print();
    }
  }
  return 0;
}
