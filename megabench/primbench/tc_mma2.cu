// tcgen05.mma extensions on sm_100a (companion of tc_mma.cu, same metric / param naming):
//  (A2) cta_group::2 (cluster of 2 CTAs, rank 0 issues): M = 128 and 256, N = 16..256 step 16, kind::f16 (bf16) and kind::f8f6f4 (e4m3).
//       Each CTA holds M/2 rows of A and N/2 rows of B (SW128 K-major) in its own smem; D lives in each CTA's TMEM
//       (M = 256: CTA r lanes 0..127 = rows 128r..; M = 128: 64 rows per CTA). TMEM alloc/dealloc cta_group::2, commit multicast to both CTAs.
//  (A3) operand-layout variants for cta_group::1 bf16 M = 128: swizzle none/32/64/128 B, K-major vs MN-major for A and B,
//       N in {16, 64, 256}, plus issuers 1/2/4 (lane 0 of warps 0..3 issue independent MMA chains into separate accumulators).
// Smem images are built on the host from the canonical layouts (PTX ISA shared memory matrix descriptor) and copied into smem.
// Every point is checked against a CPU reference (small integers, exact). Rows whose layout/descriptor combination does not
// verify are still printed with verified=false.
// Throughput = slope between two loop lengths; latency = one MMA from issue to commit arrival.
// Usage: ./tc_mma2 [cg2|layout]   (no args: both), JSON lines on stdout.
#include "pb_common.cuh"
#include <cstring>
#include <map>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

constexpr int A_REG = 16384, B_REG = 32768;                 // per-CTA smem regions (bytes)
constexpr int SMEM_BYTES = 1024 + A_REG + B_REG + 1024;

struct KP {
  int N, iters, issuers, a_len, b_len;
  uint32_t a_off[4], b_off[4];
  uint32_t a_lbo, a_sbo, b_lbo, b_sbo, a_swz, b_swz, idesc;
  const uint8_t *A, *B;      // images: [rank][A_REG] / [rank][B_REG]
  uint32_t* D;               // verify only: [issuer][rank][128 lanes][256] fp32/int bits
  long long* t;              // per block: [issuer][t0, t1(issue end), t2(commit arrival)]
};

__device__ __forceinline__ uint64_t mk_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | (uint64_t)(lbo >> 4) << 16 | (uint64_t)(sbo >> 4) << 32 | (uint64_t)1 << 46 | (uint64_t)swz << 61;
}

#define MMAS(CGS, K) "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::" CGS ".kind::" K " [%0], %1, %2, %3, p;\n}\n"
template <int CG, int KIND>
__device__ __forceinline__ void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  if constexpr (CG == 1 && KIND == 0) asm volatile(MMAS("1", "f16") :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
  else if constexpr (CG == 1) asm volatile(MMAS("1", "f8f6f4") :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
  else if constexpr (KIND == 0) asm volatile(MMAS("2", "f16") :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
  else asm volatile(MMAS("2", "f8f6f4") :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" :: "r"(mbar), "r"(phase) : "memory");
}
__device__ __forceinline__ void tc_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory"); }

template <int CG, int KIND>
__global__ void __launch_bounds__(128, 1) bench(KP p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t *sA = base, *sB = base + A_REG;
  uint64_t* mbar = (uint64_t*)(sB + B_REG);
  uint32_t* tslot = (uint32_t*)(mbar + 8);
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int rank = CG == 2 ? (blockIdx.x & 1) : 0;
  if (warp == 0) {
    if (CG == 2) {
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(tslot)) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    } else {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(tslot)) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  if (tid == 0) {
    for (int w = 0; w < 4; ++w) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(mbar + w)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < p.a_len / 16; i += 128) ((uint4*)sA)[i] = ((const uint4*)(p.A + (size_t)rank * A_REG))[i];
  for (int i = tid; i < p.b_len / 16; i += 128) ((uint4*)sB)[i] = ((const uint4*)(p.B + (size_t)rank * B_REG))[i];
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (CG == 2) cluster_sync_all();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t tmem = *(volatile uint32_t*)tslot;

  const bool issuer = lane == 0 && warp < p.issuers && (CG == 1 || rank == 0);
  __syncthreads();
  if (issuer) {
    uint32_t idesc = p.idesc, d0 = tmem + (uint32_t)(warp * p.N);
    uint64_t ad[4], bd[4];
    for (int j = 0; j < 4; ++j) {
      ad[j] = mk_desc(smem_u32(sA) + p.a_off[j], p.a_lbo, p.a_sbo, p.a_swz);
      bd[j] = mk_desc(smem_u32(sB) + p.b_off[j], p.b_lbo, p.b_sbo, p.b_swz);
    }
    long long t0 = clock64();
    if (p.iters == 1) {
      mma<CG, KIND>(d0, ad[0], bd[0], idesc, 0u);
    } else {
      mma<CG, KIND>(d0, ad[0], bd[0], idesc, 0u);
      mma<CG, KIND>(d0, ad[1], bd[1], idesc, 1u);
      mma<CG, KIND>(d0, ad[2], bd[2], idesc, 1u);
      mma<CG, KIND>(d0, ad[3], bd[3], idesc, 1u);
      for (int i = 4; i < p.iters; i += 4) {
        mma<CG, KIND>(d0, ad[0], bd[0], idesc, 1u);
        mma<CG, KIND>(d0, ad[1], bd[1], idesc, 1u);
        mma<CG, KIND>(d0, ad[2], bd[2], idesc, 1u);
        mma<CG, KIND>(d0, ad[3], bd[3], idesc, 1u);
      }
    }
    long long t1 = clock64();
    if (CG == 2) asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                              :: "r"(smem_u32(mbar)), "h"((unsigned short)3) : "memory");
    else asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(mbar + warp)) : "memory");
    mbar_wait(smem_u32(mbar + (CG == 2 ? 0 : warp)), 0);
    long long t2 = clock64();
    long long* o = p.t + blockIdx.x * 16 + warp * 3;
    o[0] = t0; o[1] = t1; o[2] = t2;
  } else if (CG == 2 && rank == 1 && tid == 0) {
    mbar_wait(smem_u32(mbar), 0);
  }
  tc_sync();
  if (p.D) {
    for (int w = 0; w < p.issuers; ++w)
      for (int c = 0; c < p.N; c += 8) {
        uint32_t r[8]; uint32_t ta = tmem + ((uint32_t)(warp * 32) << 16) + w * p.N + c;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(ta));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        for (int j = 0; j < 8; ++j) p.D[(((size_t)(w * 2 + rank) * 128) + warp * 32 + lane) * 256 + c + j] = r[j];
      }
  }
  tc_sync();
  if (CG == 2) cluster_sync_all();
  if (warp == 0) {
    if (CG == 2) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
    else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
  }
}

// ---------------------------------------------------------------- host: operand images
struct Img { std::vector<uint8_t> b; uint32_t lbo, sbo, swz, off[4]; };

// X: R rows (mn) x K elements (k), row-major, eb bytes per element, K = 128 / eb.
// W = swizzle width in bytes (0 = none, 32, 64, 128); major 0 = K-major, 1 = MN-major.
static Img build_img(const uint8_t* X, int R, int eb, int W, int major) {
  Img im; int K = 128 / eb, kstep = 32 / eb;
  im.b.assign(R <= 128 ? A_REG : B_REG, 0);
  im.swz = W == 0 ? 0 : W == 32 ? 6 : W == 64 ? 4 : 2;
  uint32_t m = W == 32 ? 1 : W == 64 ? 3 : 7;
  uint32_t lbo = 16, sbo = 128; uint32_t slab_b = 0;
  int epc = 16 / eb;  // elements per 16 B core-matrix row
  if (major == 0) {
    if (W) { slab_b = (R * W + 1023) / 1024 * 1024; sbo = 8 * W; lbo = 16; }
    else { sbo = 128; lbo = R / 8 * 128; }
  } else {
    if (W) { sbo = 8 * W; lbo = (K / 8) * sbo; }
    else { sbo = 128; lbo = (uint32_t)((R + epc - 1) / epc) * 128; }
  }
  size_t mx = 0;
  for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) {
    uint32_t raw;
    if (major == 0) {
      if (W) { uint32_t kb = k * eb; raw = (kb / W) * slab_b + (r / 8) * (8 * W) + (r % 8) * W + kb % W; }
      else raw = (k / epc) * lbo + (r / 8) * 128 + (r % 8) * 16 + (k % epc) * eb;
    } else {
      if (W) { int per = W / eb; raw = (r / per) * lbo + (k / 8) * sbo + (k % 8) * W + (r % per) * eb; }
      else raw = (r / epc) * sbo + (k / 8) * lbo + (k % 8) * 16 + (r % epc) * eb;
    }
    if (W) raw ^= ((raw >> 7) & m) << 4;
    if (raw + eb > im.b.size()) { fprintf(stderr, "image overflow R=%d W=%d major=%d\n", R, W, major); exit(1); }
    memcpy(&im.b[raw], X + ((size_t)r * K + k) * eb, eb);
    mx = std::max(mx, (size_t)raw + eb);
  }
  for (int j = 0; j < 4; ++j) {
    if (major == 0) im.off[j] = W ? (32 * j) / W * slab_b + (32 * j) % W : 2 * j * lbo;
    else im.off[j] = W ? j * (kstep / 8) * sbo : j * (kstep / 8) * lbo;
  }
  im.lbo = lbo; im.sbo = sbo;
  (void)mx;
  return im;
}

static void enc(int eb, uint8_t* p, int i, float v) {
  if (eb == 2) { __nv_bfloat16 x(v); memcpy(p + 2 * i, &x, 2); }
  else { __nv_fp8_e4m3 x(v); memcpy(p + i, &x, 1); }
}
static float dec(int eb, const uint8_t* p, int i) {
  if (eb == 2) { __nv_bfloat16_raw r; memcpy(&r.x, p + 2 * i, 2); return __bfloat162float(__nv_bfloat16(r)); }
  __nv_fp8_e4m3 x; memcpy(&x, p + i, 1); return (float)x;
}

struct Cfg { int cg, kind, M, N, issuers, wa, ma, wb, mb; };   // kind 0 = bf16 (f16 kind), 1 = e4m3 (f8f6f4)
struct Dev { uint8_t *A, *B; uint32_t* D; long long* t; };

typedef void (*Fn)(KP);
static Fn pick(int cg, int kind) {
  if (cg == 1) return kind ? bench<1, 1> : bench<1, 0>;
  return kind ? bench<2, 1> : bench<2, 0>;
}

static KP make_kp(const Cfg& c, int iters, Dev& d, int eb, const Img& ia, const Img& ib, bool dump) {
  KP p{}; p.N = c.N; p.iters = iters; p.issuers = c.issuers; p.a_len = A_REG; p.b_len = B_REG;
  for (int j = 0; j < 4; ++j) { p.a_off[j] = ia.off[j]; p.b_off[j] = ib.off[j]; }
  p.a_lbo = ia.lbo; p.a_sbo = ia.sbo; p.b_lbo = ib.lbo; p.b_sbo = ib.sbo; p.a_swz = ia.swz; p.b_swz = ib.swz;
  uint32_t ab = c.kind == 0 ? 1u : 0u;   // bf16 = 1 (f16 kind), e4m3 = 0 (f8f6f4)
  p.idesc = 1u << 4 | ab << 7 | ab << 10 | (uint32_t)(c.ma ? 1 : 0) << 15 | (uint32_t)(c.mb ? 1 : 0) << 16
          | (uint32_t)(c.N >> 3) << 17 | (uint32_t)(c.M >> 4) << 24;
  p.A = d.A; p.B = d.B; p.D = dump ? d.D : nullptr; p.t = d.t;
  (void)eb;
  return p;
}

static void launch(const Cfg& c, const KP& p, int ctas) {
  Fn k = pick(c.cg, c.kind);
  if (c.cg == 2) {
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(ctas); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, (void (*)(KP))k, p));
  } else k<<<ctas, 128, SMEM_BYTES>>>(p);
  CK(cudaGetLastError());
}

// Median over CTAs of elapsed cycles (loop+commit wait) and of issue cycles; CG=2 reads rank-0 CTAs only.
static double run(const Cfg& c, Dev& d, int eb, const Img& ia, const Img& ib, int iters, int ctas,
                  double* wall_us = nullptr, double* issue = nullptr, std::vector<double>* reps = nullptr) {
  KP p = make_kp(c, iters, d, eb, ia, ib, false);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  launch(c, p, ctas); CK(cudaDeviceSynchronize());
  std::vector<double> loops, issues; float best = 1e9;
  for (int rep = 0; rep < 7; ++rep) {
    cudaEventRecord(e0); launch(c, p, ctas); cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
    std::vector<long long> t(ctas * 16);
    CK(cudaMemcpy(t.data(), d.t, t.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> l, s;
    for (int b = 0; b < ctas; ++b) {
      if (c.cg == 2 && (b & 1)) continue;
      long long st = t[b * 16], en = t[b * 16 + 2], is = t[b * 16 + 1];
      for (int w = 1; w < c.issuers; ++w) { st = std::min(st, t[b * 16 + w * 3]); en = std::max(en, t[b * 16 + w * 3 + 2]); is = std::max(is, t[b * 16 + w * 3 + 1]); }
      l.push_back((double)(en - st)); s.push_back((double)(is - st));
    }
    loops.push_back(median(l)); issues.push_back(median(s));
  }
  if (wall_us) *wall_us = best * 1e3;
  if (issue) *issue = median(issues);
  if (reps) *reps = loops;
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return median(loops);
}

static bool verify(const Cfg& c, Dev& d, int eb, const Img& ia, const Img& ib, const std::vector<uint8_t>& hA, const std::vector<uint8_t>& hB) {
  KP p = make_kp(c, 4, d, eb, ia, ib, true);
  const size_t dn = 4 * 2 * 128 * 256;
  CK(cudaMemset(d.D, 0, dn * 4));
  launch(c, p, c.cg);
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> hD(dn);
  CK(cudaMemcpy(hD.data(), d.D, dn * 4, cudaMemcpyDeviceToHost));
  int K = 128 / eb;
  std::map<std::vector<float>, int> rows;   // reference row -> row index
  for (int m = 0; m < c.M; ++m) {
    std::vector<float> r(c.N);
    for (int n = 0; n < c.N; ++n) { double s = 0;
      for (int k = 0; k < K; ++k) s += (double)dec(eb, hA.data(), m * K + k) * dec(eb, hB.data(), n * K + k);
      r[n] = (float)s; }
    rows[r] = m;
  }
  if ((int)rows.size() != c.M) return false;
  for (int w = 0; w < c.issuers; ++w) {
    std::vector<int> hit(c.M, 0);
    auto at = [&](int rk, int l, int n) { uint32_t u = hD[(((size_t)(w * 2 + rk) * 128) + l) * 256 + n]; float f; memcpy(&f, &u, 4); return f; };
    if (c.cg == 2 && c.M == 128) {
      // cta_group::2 M=128 data path (observed): CTA r holds rows 64r..64r+63; lane l (< 64) holds columns [0, N/2) of its row
      // and lane l + 64 holds columns [N/2, N), both at TMEM columns 0..N/2-1.
      int H = c.N / 2;
      for (int rk = 0; rk < 2; ++rk) for (int l = 0; l < 64; ++l) {
        std::vector<float> r(c.N);
        for (int n = 0; n < H; ++n) { r[n] = at(rk, l, n); r[H + n] = at(rk, l + 64, n); }
        auto it = rows.find(r);
        if (it != rows.end()) hit[it->second]++;
      }
    } else {
      for (int rk = 0; rk < c.cg; ++rk) for (int l = 0; l < 128; ++l) {
        std::vector<float> r(c.N);
        for (int n = 0; n < c.N; ++n) r[n] = at(rk, l, n);
        auto it = rows.find(r);
        if (it != rows.end()) hit[it->second]++;
      }
    }
    for (int m = 0; m < c.M; ++m) if (hit[m] != 1) return false;
  }
  return true;
}

static void make_data(int eb, int rows, int seed, std::vector<uint8_t>& h) {
  int K = 128 / eb; h.assign((size_t)rows * K * eb, 0); srand(seed);
  for (int i = 0; i < rows * K; ++i) enc(eb, h.data(), i, (float)(rand() % 7 - 3));
}

static void measure_point(const Cfg& c, Dev& d, int nsm, const char* layout_str, bool cg2_sweep, int ssw, int sama, int samb) {
  int eb = c.kind == 0 ? 2 : 1, K = 128 / eb, kstep = 32 / eb;
  std::vector<uint8_t> hA, hB;
  make_data(eb, c.M, 11 + c.M + c.kind, hA); make_data(eb, c.N, 77 + c.N + c.kind, hB);
  int RA = c.M / c.cg, RB = c.N / c.cg;
  Img ia, ib, ia1, ib1;   // images of rank 0 (used for descriptors) and the concatenated device blobs
  std::vector<uint8_t> blobA(2 * A_REG, 0), blobB(2 * B_REG, 0);
  for (int rk = 0; rk < c.cg; ++rk) {
    Img a = build_img(hA.data() + (size_t)rk * RA * K * eb, RA, eb, c.wa, c.ma);
    Img b = build_img(hB.data() + (size_t)rk * RB * K * eb, RB, eb, c.wb, c.mb);
    memcpy(&blobA[rk * A_REG], a.b.data(), A_REG); memcpy(&blobB[rk * B_REG], b.b.data(), B_REG);
    if (rk == 0) { ia = a; ib = b; }
  }
  CK(cudaMemcpy(d.A, blobA.data(), blobA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.B, blobB.data(), blobB.size(), cudaMemcpyHostToDevice));
  const int I1 = 256, I2 = 4352;
  bool ok = verify(c, d, eb, ia, ib, hA, hB);
  std::vector<double> r1, r2;
  double c1 = run(c, d, eb, ia, ib, I1, c.cg, nullptr, nullptr, &r1);
  double iss, c2 = run(c, d, eb, ia, ib, I2, c.cg, nullptr, &iss, &r2);
  std::vector<double> slopes; for (size_t i = 0; i < r1.size(); ++i) slopes.push_back((r2[i] - r1[i]) / (I2 - I1) / c.issuers);
  double cpm = (c2 - c1) / (I2 - I1) / c.issuers;
  double lat = run(c, d, eb, ia, ib, 1, c.cg);
  int grid = nsm / c.cg * c.cg;
  if (c.cg == 2) {   // clusters of 2 must be co-resident
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(grid); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
    int ncl = 0; CK(cudaOccupancyMaxActiveClusters(&ncl, (void (*)(KP))pick(2, c.kind), &cfg));
    grid = std::min(grid, 2 * ncl);
  }
  double wall, a1 = run(c, d, eb, ia, ib, I1, grid), a2 = run(c, d, eb, ia, ib, I2, grid, &wall);
  double acpm = (a2 - a1) / (I2 - I1) / c.issuers;
  double flop = 2.0 * c.M * c.N * kstep;
  double streams = (double)grid / c.cg * c.issuers;
  Json j; j.s("family", "tc_mma").s("primitive", "tcgen05.mma")
    .s("p.kind", c.kind ? "f8f6f4" : "f16").s("p.dtype", c.kind ? "e4m3" : "bf16").i("p.cta_group", c.cg)
    .i("p.M", c.M).i("p.N", c.N).i("p.K", kstep).s("p.a_src", "smem").i("p.naccum", 1)
    .s("p.layout", layout_str).i("p.issuers", c.issuers);
  if (!cg2_sweep) j.i("p.swizzle_a", c.wa).i("p.swizzle_b", c.wb).s("p.a_major", c.ma ? "MN" : "K").s("p.b_major", c.mb ? "MN" : "K");
  (void)ssw; (void)sama; (void)samb;
  j.f("m.cyc_per_mma", cpm).f("m.latency_cyc", lat).f("m.issue_cyc_per_mma", iss / I2 / c.issuers)
   .f("m.fma_per_clk_sm", (double)c.M * c.N * kstep / cpm / c.cg).f("m.allsm_cyc_per_mma", acpm)
   .f("m.allsm_tflops", flop * I2 * streams / (wall * 1e-6) / 1e12)
   .i("c.ctas", c.cg).i("c.allsm_ctas", grid).t("verified", ok).i("n_reps", (long long)slopes.size()).f("cv", cv(slopes))
   .s("note", c.cg == 2
      ? "cta_group::2 cluster of 2, rank 0 issues back-to-back MMAs; M/N are the pair totals (A rows M/2 and B rows N/2 per CTA, SW128 K-major); "
        "cyc_per_mma per cluster pair, fma_per_clk_sm per SM (pair rate / 2); latency = one MMA issue to multicast commit arrival; allsm_* with clusters on every SM"
      : "cta_group::1 M=128 layout / issuer variant; lane 0 of warps 0..issuers-1 each issue a chain into its own accumulator; cyc_per_mma is per SM "
        "aggregated over issuers (elapsed / (iters*issuers)); latency = one MMA per issuer from issue to commit arrival; allsm_* with one CTA on every SM")
   .print();
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  int nsm = num_sms();
  for (int cg = 1; cg <= 2; ++cg) for (int kd = 0; kd < 2; ++kd)
    CK(cudaFuncSetAttribute((const void*)pick(cg, kd), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
  Dev d;
  CK(cudaMalloc(&d.A, 2 * A_REG)); CK(cudaMalloc(&d.B, 2 * B_REG));
  CK(cudaMalloc(&d.D, 4 * 2 * 128 * 256 * 4)); CK(cudaMalloc(&d.t, 2048 * 16 * 8));
  if (!only || !strcmp(only, "cg2")) {
    for (int kd = 0; kd < 2; ++kd) for (int M : {128, 256}) for (int N = 16; N <= 256; N += 16) {
      Cfg c{2, kd, M, N, 1, 128, 0, 128, 0};
      measure_point(c, d, nsm, "K-major SW128", true, 0, 0, 0);
    }
  }
  if (!only || !strcmp(only, "layout")) {
    const int Ws[4] = {0, 32, 64, 128};
    for (int N : {16, 64, 256}) for (int w : Ws) for (int ma = 0; ma < 2; ++ma) for (int mb = 0; mb < 2; ++mb) {
      Cfg c{1, 0, 128, N, 1, w, ma, w, mb};
      char s[64]; snprintf(s, sizeof s, "A:%s B:%s SW%d", ma ? "MN" : "K", mb ? "MN" : "K", w);
      measure_point(c, d, nsm, s, false, w, ma, mb);
    }
    for (int N : {16, 64, 256}) for (int is : {2, 4}) {
      if (is * N > 512) continue;   // accumulators of all issuers must fit in the 512 TMEM columns
      Cfg c{1, 0, 128, N, is, 128, 0, 128, 0};
      measure_point(c, d, nsm, "A:K B:K SW128", false, 128, 0, 0);
    }
  }
  return 0;
}
