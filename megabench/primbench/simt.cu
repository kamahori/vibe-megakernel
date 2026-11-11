// SIMT arithmetic throughput per SM on sm_100a: FFMA, HFMA2 (f16x2), BFMA2 (bf16x2), IMAD, DP4A.
// One CTA per SM with W warps; each thread runs ILP independent dependency chains.
// Cycles per warp instruction come from the slope between two loop lengths.
// Verification: chains start at 0 and add 1*1 per step, so each lane must hold iters.
// Usage: ./simt [op_filter]   (JSON lines on stdout)
#include "pb_common.cuh"
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

enum Op { FFMA, HFMA2, BFMA2, IMAD, DP4A, NOP };
struct OpInfo { const char* name; const char* sass; int flop_per_lane; };  // FMA = 2 flops; dp4a = 4 MACs
static const OpInfo OPS[NOP] = {
  {"ffma", "FFMA", 2}, {"hfma2", "HFMA2", 4}, {"bfma2", "HFMA2.BF16", 4}, {"imad", "IMAD", 2}, {"dp4a", "IDP.4A", 8},
};

template <int OP>
__device__ __forceinline__ void step(uint32_t& c, uint32_t a, uint32_t b) {
  if constexpr (OP == FFMA) c = __float_as_uint(fmaf(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)));
  else if constexpr (OP == HFMA2) { __half2 r = __hfma2(*(__half2*)&a, *(__half2*)&b, *(__half2*)&c); c = *(uint32_t*)&r; }
  else if constexpr (OP == BFMA2) { __nv_bfloat162 r = __hfma2(*(__nv_bfloat162*)&a, *(__nv_bfloat162*)&b, *(__nv_bfloat162*)&c); c = *(uint32_t*)&r; }
  else if constexpr (OP == IMAD) c = a * b + c;
  else c = (uint32_t)__dp4a((int)a, (int)b, (int)c);
}

template <int OP, int ILP>
__global__ void __launch_bounds__(1024, 1) k_simt(uint32_t a, uint32_t b, int iters, uint32_t* out, Stamp* st) {
  uint32_t c[ILP];
#pragma unroll
  for (int j = 0; j < ILP; ++j) { c[j] = j ? j : 0u; asm volatile("" : "+r"(c[j])); }  // distinct: no CSE across chains
  __syncthreads();
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) {
    asm volatile("" : "+r"(a), "+r"(b));
#pragma unroll
    for (int j = 0; j < ILP; ++j) step<OP>(c[j], a, b);
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (threadIdx.x == 0) st[blockIdx.x] = {t1 - t0, g1 - g0};
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = c[0];
  if (iters < 0) for (int j = 0; j < ILP; ++j) out[1 + j] = c[j];  // keep every chain live
}

typedef void (*Fn)(uint32_t, uint32_t, int, uint32_t*, Stamp*);
template <int OP> static Fn pick(int ilp) {
  switch (ilp) { case 1: return k_simt<OP, 1>; case 2: return k_simt<OP, 2>; case 4: return k_simt<OP, 4>; default: return k_simt<OP, 8>; }
}
static Fn pick(int op, int ilp) {
  switch (op) { case FFMA: return pick<FFMA>(ilp); case HFMA2: return pick<HFMA2>(ilp);
    case BFMA2: return pick<BFMA2>(ilp); case IMAD: return pick<IMAD>(ilp); default: return pick<DP4A>(ilp); }
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  int nsm = num_sms();
  uint32_t* out; Stamp* st;
  CK(cudaMalloc(&out, 64)); CK(cudaMalloc(&st, sizeof(Stamp) * nsm));
  // Operands that make each step add exactly 1 (per lane / per packed half).
  const uint32_t ones[NOP] = {0x3F800000u, 0x3C003C00u, 0x3F803F80u, 1u, 0x00000001u};
  const int I1 = 512, I2 = 4608, VERIFY_IT = 64;
  for (int op = 0; op < NOP; ++op) {
    if (only && !strstr(OPS[op].name, only)) continue;
    for (int ctas : {1, nsm})
    for (int W : {1, 4, 8, 16, 32})
    for (int ilp : {1, 2, 4, 8}) {
      if (ctas > 1 && !(ilp == 4 && W >= 8)) continue;
      Fn k = pick(op, ilp);
      k<<<1, 32 * W>>>(ones[op], ones[op], VERIFY_IT, out, st);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      uint32_t h; CK(cudaMemcpy(&h, out, 4, cudaMemcpyDeviceToHost));
      bool ok;
      if (op == FFMA) { float f; memcpy(&f, &h, 4); ok = f == VERIFY_IT; }
      else if (op == HFMA2) { __half_raw r; r.x = h & 0xFFFF; ok = __half2float(__half(r)) == VERIFY_IT; }
      else if (op == BFMA2) { __nv_bfloat16_raw r; r.x = h & 0xFFFF; ok = __bfloat162float(__nv_bfloat16(r)) == VERIFY_IT; }
      else ok = h == (uint32_t)VERIFY_IT;
      std::vector<double> per, mhz;
      for (int rep = 0; rep < 5; ++rep) {
        double c[2]; int its[2] = {I1, I2};
        for (int x = 0; x < 2; ++x) {
          k<<<ctas, 32 * W>>>(ones[op], ones[op], its[x], out, st);
          CK(cudaDeviceSynchronize());
          std::vector<Stamp> hs(ctas);
          CK(cudaMemcpy(hs.data(), st, sizeof(Stamp) * ctas, cudaMemcpyDeviceToHost));
          std::vector<double> cy; for (auto& v : hs) cy.push_back((double)v.cyc);
          c[x] = median(cy);
          if (x == 1) mhz.push_back(clock_mhz(hs));
        }
        per.push_back((c[1] - c[0]) / (I2 - I1));
      }
      double it = median(per), cyc_warp_instr = it / (W * ilp);
      Json j; j.s("family", "simt").s("primitive", std::string("simt.") + OPS[op].name)
        .s("p.op", OPS[op].name).s("p.sass", OPS[op].sass).i("p.warps", W).i("p.ilp", ilp)
        .f("m.cyc_per_warp_instr_sm", cyc_warp_instr).f("m.warp_instr_per_clk_sm", 1.0 / cyc_warp_instr)
        .f("m.flop_per_clk_sm", 32.0 * OPS[op].flop_per_lane / cyc_warp_instr)
        .i("c.ctas", ctas).f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", (long long)per.size()).f("cv", cv(per));
      if (W == 1 && ilp == 1) j.f("m.latency_cyc", it);
      j.s("note", "flop counts an FMA as 2 per element (f16x2/bf16x2: 2 elements; dp4a: 4 MACs). "
                  "SIMT and mma.sync share the warp schedulers' issue slots; tcgen05 math does not");
      j.print();
    }
  }
  return 0;
}
