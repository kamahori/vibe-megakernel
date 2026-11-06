// Weight dequantization throughput per SM on sm_100a, from registers (no memory traffic).
// Each variant turns one packed 32-bit word into 16-bit MMA operands (bf16 or f16, packed x2)
// and applies a scale. Compare elements/clk/SM with the per-SM HBM byte rate to see whether
// dequant can hide behind streaming, and with mma.sync / SIMT rows for issue-slot pressure.
// Verification: every variant is checked element by element against a CPU reference.
// Usage: ./dequant   (JSON lines on stdout)
#include "pb_common.cuh"
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

enum V { I8_I2F, I8_MAGIC, I4_MAGIC, E4M3_CVT, E2M1_CVT, NV };
struct VInfo { const char* name; const char* src; const char* dst; int elems; const char* how; };
static const VInfo VI[NV] = {
  {"i8_i2f", "s8", "bf16", 4, "byte extract + I2F per element, F2FP pack to bf16x2, HMUL2 by scale"},
  {"i8_magic_f32", "s8", "bf16", 4, "PRMT into 2^23 float + FADD bias per element, F2FP pack to bf16x2, HMUL2 by scale"},
  {"i4_magic_bf16", "s4", "bf16", 8, "LOP3 nibble pairs into bf16 128+u, HFMA2 (v-136)*s per pair"},
  {"e4m3_cvt", "e4m3", "f16", 4, "cvt.rn.f16x2.e4m3x2 per pair, HMUL2 by scale"},
  {"e2m1_cvt", "e2m1", "f16", 8, "cvt.rn.f16x2.e2m1x2 per pair, HMUL2 by scale"},
};

template <int VAR>
__device__ __forceinline__ void dq(uint32_t w, uint32_t s2, uint32_t* o) {
  if constexpr (VAR == I8_I2F) {
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = (float)(int8_t)(w >> (8 * i));
    __nv_bfloat162 a = __floats2bfloat162_rn(f[0], f[1]), b = __floats2bfloat162_rn(f[2], f[3]);
    __nv_bfloat162 s = *(__nv_bfloat162*)&s2;
    a = __hmul2(a, s); b = __hmul2(b, s);
    o[0] = *(uint32_t*)&a; o[1] = *(uint32_t*)&b;
  } else if constexpr (VAR == I8_MAGIC) {
    uint32_t x = w ^ 0x80808080u;  // signed -> biased unsigned
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7650 + i) ) - 8388736.f;  // 2^23 + 128
    __nv_bfloat162 a = __floats2bfloat162_rn(f[0], f[1]), b = __floats2bfloat162_rn(f[2], f[3]);
    __nv_bfloat162 s = *(__nv_bfloat162*)&s2;
    a = __hmul2(a, s); b = __hmul2(b, s);
    o[0] = *(uint32_t*)&a; o[1] = *(uint32_t*)&b;
  } else if constexpr (VAR == I4_MAGIC) {
    // Nibble i of w (signed, two's complement) -> bf16. Bias by XOR 0x8 so u = v + 8 in [0, 16).
    uint32_t x = w ^ 0x88888888u;
    __nv_bfloat162 s = *(__nv_bfloat162*)&s2;
    const uint32_t bias = 0x43084308u;  // bf16 (136, 136)
    __nv_bfloat162 nb = __hmul2(*(const __nv_bfloat162*)&bias, s);
    nb = __hneg2(nb);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // pair i holds nibbles i and i+4 (interleaved order, as weight-packing kernels arrange it)
      uint32_t v = (((x >> (4 * i)) & 0xFu) | (((x >> (4 * i + 16)) & 0xFu) << 16)) | 0x43004300u;
      __nv_bfloat162 r = __hfma2(*(__nv_bfloat162*)&v, s, nb);
      o[i] = *(uint32_t*)&r;
    }
  } else if constexpr (VAR == E4M3_CVT) {
    __half2 s = *(__half2*)&s2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t h; asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(w >> (16 * i))));
      __half2 r = __hmul2(*(__half2*)&h, s); o[i] = *(uint32_t*)&r;
    }
  } else {
    __half2 s = *(__half2*)&s2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t h;
      asm volatile("{\n.reg .b8 b;\nmov.b32 {b, _, _, _}, %1;\ncvt.rn.f16x2.e2m1x2 %0, b;\n}" : "=r"(h) : "r"(w >> (8 * i)));
      __half2 r = __hmul2(*(__half2*)&h, s); o[i] = *(uint32_t*)&r;
    }
  }
}

template <int VAR>
__global__ void k_verify(const uint32_t* w, uint32_t s2, uint32_t* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dq<VAR>(w[i], s2, o + 4 * i);
}

template <int VAR, int ILP>
__global__ void __launch_bounds__(1024, 1) k_rate(uint32_t w0, uint32_t s2, int iters, uint32_t* out, Stamp* st) {
  uint32_t w[ILP], acc = 0;
#pragma unroll
  for (int j = 0; j < ILP; ++j) { w[j] = w0 + j * 0x01010101u; asm volatile("" : "+r"(w[j])); }
  __syncthreads();
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      w[j] += 0x01030507u;  // a real new word every iteration (empty asm does not stop ptxas hoisting)
      uint32_t o[4] = {};
      dq<VAR>(w[j], s2, o);
      acc ^= o[0] ^ o[1] ^ o[2] ^ o[3];
    }
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (threadIdx.x == 0) st[blockIdx.x] = {t1 - t0, g1 - g0};
  if (iters < 0) out[threadIdx.x] = acc;
}

// ---------------------------------------------------------------- host reference
static float bf(uint16_t h) { __nv_bfloat16_raw r; r.x = h; return __bfloat162float(__nv_bfloat16(r)); }
static float hf(uint16_t h) { __half_raw r; r.x = h; return __half2float(__half(r)); }
static float e2m1(int v) { static const float t[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6}; return (v & 8 ? -1 : 1) * t[v & 7]; }

static bool check(int var, const std::vector<uint32_t>& w, const std::vector<uint32_t>& o, float s) {
  for (size_t i = 0; i < w.size(); ++i) {
    uint32_t x = w[i]; const uint32_t* r = &o[4 * i];
    std::vector<float> want, got;
    switch (var) {
      case I8_I2F: case I8_MAGIC:
        for (int k = 0; k < 4; ++k) want.push_back((float)(int8_t)(x >> (8 * k)) * s);
        for (int k = 0; k < 2; ++k) { got.push_back(bf(r[k] & 0xFFFF)); got.push_back(bf(r[k] >> 16)); }
        break;
      case I4_MAGIC:
        for (int k = 0; k < 4; ++k) {
          int lo = (int)((x >> (4 * k)) & 0xF), hi = (int)((x >> (4 * k + 16)) & 0xF);
          want.push_back((float)(lo >= 8 ? lo - 16 : lo) * s); want.push_back((float)(hi >= 8 ? hi - 16 : hi) * s);
          got.push_back(bf(r[k] & 0xFFFF)); got.push_back(bf(r[k] >> 16));
        }
        break;
      case E4M3_CVT:
        for (int k = 0; k < 4; ++k) { __nv_fp8_e4m3 e; uint8_t b = x >> (8 * k); memcpy(&e, &b, 1); want.push_back((float)e * s); }
        for (int k = 0; k < 2; ++k) { got.push_back(hf(r[k] & 0xFFFF)); got.push_back(hf(r[k] >> 16)); }
        break;
      default:
        for (int k = 0; k < 8; ++k) want.push_back(e2m1((x >> (4 * k)) & 0xF) * s);
        for (int k = 0; k < 4; ++k) { got.push_back(hf(r[k] & 0xFFFF)); got.push_back(hf(r[k] >> 16)); }
    }
    for (size_t k = 0; k < want.size(); ++k) {
      float tol = std::fabs(want[k]) * (var <= I4_MAGIC ? 1.0f / 128 : 1.0f / 1024) + 1e-6f;  // bf16 / f16 rounding
      if (!(std::fabs(got[k] - want[k]) <= tol)) return false;
    }
  }
  return true;
}

typedef void (*Rate)(uint32_t, uint32_t, int, uint32_t*, Stamp*);
typedef void (*Ver)(const uint32_t*, uint32_t, uint32_t*, int);
template <int VAR> static Rate rate(int ilp) { return ilp == 1 ? k_rate<VAR, 1> : ilp == 2 ? k_rate<VAR, 2> : k_rate<VAR, 4>; }
static Rate rate(int v, int ilp) {
  switch (v) { case I8_I2F: return rate<I8_I2F>(ilp); case I8_MAGIC: return rate<I8_MAGIC>(ilp);
    case I4_MAGIC: return rate<I4_MAGIC>(ilp); case E4M3_CVT: return rate<E4M3_CVT>(ilp); default: return rate<E2M1_CVT>(ilp); }
}
static Ver ver(int v) {
  switch (v) { case I8_I2F: return k_verify<I8_I2F>; case I8_MAGIC: return k_verify<I8_MAGIC>;
    case I4_MAGIC: return k_verify<I4_MAGIC>; case E4M3_CVT: return k_verify<E4M3_CVT>; default: return k_verify<E2M1_CVT>; }
}

int main() {
  int nsm = num_sms();
  const int NW = 4096;
  std::vector<uint32_t> hw(NW);
  srand(7);
  for (auto& x : hw) x = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
  for (auto& x : hw)  // avoid e4m3 NaN encodings (S.1111.111) in every byte
    for (int k = 0; k < 4; ++k) if (((x >> (8 * k)) & 0x7F) == 0x7F) x ^= 1u << (8 * k);
  uint32_t *dw, *dout; Stamp* st;
  CK(cudaMalloc(&dw, NW * 4)); CK(cudaMalloc(&dout, NW * 16)); CK(cudaMalloc(&st, sizeof(Stamp) * nsm));
  CK(cudaMemcpy(dw, hw.data(), NW * 4, cudaMemcpyHostToDevice));
  const float scale = 0.5f;
  const uint32_t s_bf = 0x3F003F00u, s_h = 0x38003800u;  // 0.5 in bf16x2 / f16x2
  const int I1 = 256, I2 = 2304;
  for (int v = 0; v < NV; ++v) {
    uint32_t s2 = v <= I4_MAGIC ? s_bf : s_h;
    ver(v)<<<NW / 256, 256>>>(dw, s2, dout, NW);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<uint32_t> ho(NW * 4);
    CK(cudaMemcpy(ho.data(), dout, NW * 16, cudaMemcpyDeviceToHost));
    bool ok = check(v, hw, ho, scale);
    for (int ctas : {1, nsm})
    for (int W : {4, 8, 16, 32})
    for (int ilp : {1, 2, 4}) {
      if (ctas > 1 && !(W == 16 && ilp == 2)) continue;
      Rate k = rate(v, ilp);
      std::vector<double> per, mhz;
      for (int rep = 0; rep < 5; ++rep) {
        double c[2]; int its[2] = {I1, I2};
        for (int x = 0; x < 2; ++x) {
          k<<<ctas, 32 * W>>>(hw[0], s2, its[x], dout, st);
          CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          std::vector<Stamp> h(ctas);
          CK(cudaMemcpy(h.data(), st, sizeof(Stamp) * ctas, cudaMemcpyDeviceToHost));
          std::vector<double> cy; for (auto& s : h) cy.push_back((double)s.cyc);
          c[x] = median(cy);
          if (x == 1) mhz.push_back(clock_mhz(h));
        }
        per.push_back((c[1] - c[0]) / (I2 - I1));
      }
      double it = median(per);
      double elems = 32.0 * W * ilp * VI[v].elems / it;
      double src_bits = !strcmp(VI[v].src, "s4") || !strcmp(VI[v].src, "e2m1") ? 4 : 8;
      Json j; j.s("family", "dequant").s("primitive", std::string("dequant.") + VI[v].name)
        .s("p.src", VI[v].src).s("p.dst", VI[v].dst).s("p.method", VI[v].how).i("p.warps", W).i("p.ilp", ilp)
        .f("m.elems_per_clk_sm", elems).f("m.src_bytes_per_clk_sm", elems * src_bits / 8)
        .i("c.ctas", ctas).f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", (long long)per.size()).f("cv", cv(per))
        .s("note", "register-to-register only (no loads, no smem stores; one IADD per word makes each word new); src_bytes_per_clk_sm is the packed "
                   "weight rate this conversion can absorb, to compare with the per-SM HBM share. "
                   "These instructions use the same issue slots as mma.sync and SIMT math");
      j.print();
    }
  }
  return 0;
}
