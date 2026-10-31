// tcgen05.mma instruction-level costs on sm_100a (cta_group::1), ported from tcgen05_rule/mma_bench.cu
// and extended to every 8/16/32-bit input kind: kind::f16 (bf16, fp16), kind::tf32, kind::f8f6f4
// (e4m3, e5m2), kind::i8 (s8). Operands in smem (K-major, SWIZZLE_128B) or A in TMEM; D in TMEM.
// Throughput = slope between two loop lengths; latency = one MMA from issue to commit arrival.
// Every shape is checked against a CPU reference (small integers, exact in every kind).
// Usage: ./tc_mma [dtype_filter]     sweep M x N x a_src x naccum, JSON lines
//        ./tc_mma overhead           alloc/dealloc/relinquish/fence/commit costs
#include "pb_common.cuh"
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// SMEM: [A: 128 rows x 128 B][B: 256 rows x 128 B][mbar x4][tmem slot]. 128 B = 4 K-steps of 32 B.
constexpr int A_BYTES = 128 * 128, B_BYTES = 256 * 128;
constexpr int SMEM_BYTES = A_BYTES + B_BYTES + 1024 + 64;

enum Dt { BF16, F16, TF32, E4M3, E5M2, S8, NDT };
struct DtInfo { const char* name; const char* kind; int bytes; int kstep; };
static const DtInfo DT[NDT] = {
  {"bf16", "f16", 2, 16}, {"f16", "f16", 2, 16}, {"tf32", "tf32", 4, 8},
  {"e4m3", "f8f6f4", 1, 32}, {"e5m2", "f8f6f4", 1, 32}, {"s8", "i8", 1, 32},
};

struct Params {
  int M, N, iters, a_tmem, naccum;
  const uint8_t *A, *B; float* D; long long* cycles;
};

__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);  // start address
  d |= (uint64_t)1 << 16;                    // LBO (unused for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;          // SBO: 8-row group stride
  d |= (uint64_t)1 << 46;                    // SM100 descriptor version
  d |= (uint64_t)2 << 61;                    // SWIZZLE_128B
  return d;
}

// Instruction descriptor: D format [4,6), A type [7,10), B type [10,13), N>>3 [17,23), M>>4 [24,29).
template <int DTY>
__device__ __forceinline__ uint32_t make_idesc(int M, int N) {
  uint32_t dfmt = DTY == S8 ? 2u : 1u;                                   // S32 or F32
  uint32_t ab = DTY == BF16 ? 1u : DTY == F16 ? 0u : DTY == TF32 ? 2u    // f16 kind: f16 0, bf16 1; tf32 2
              : DTY == E4M3 ? 0u : DTY == E5M2 ? 1u : 1u;                // f8f6f4: e4m3 0, e5m2 1; i8: signed 1
  return dfmt << 4 | ab << 7 | ab << 10 | (uint32_t)(N >> 3) << 17 | (uint32_t)(M >> 4) << 24;
}

#define MMA_SS(K) "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::" K " [%0], %1, %2, %3, p;\n}\n"
#define MMA_TS(K) "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::" K " [%0], [%1], %2, %3, p;\n}\n"

template <int DTY, int ATMEM>
__device__ __forceinline__ void mma(uint32_t dt, uint64_t adesc, uint32_t at, uint64_t bdesc, uint32_t idesc, uint32_t acc) {
#define ISSUE(K) \
  if constexpr (ATMEM) asm volatile(MMA_TS(K) :: "r"(dt), "r"(at), "l"(bdesc), "r"(idesc), "r"(acc)); \
  else asm volatile(MMA_SS(K) :: "r"(dt), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(acc));
  if constexpr (DTY == BF16 || DTY == F16) { ISSUE("f16") }
  else if constexpr (DTY == TF32) { ISSUE("tf32") }
  else if constexpr (DTY == S8) { ISSUE("i8") }
  else { ISSUE("f8f6f4") }
#undef ISSUE
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n"
               :: "r"(mbar), "r"(phase));
}
__device__ __forceinline__ void tc_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
}

template <int DTY, int ATMEM>
__global__ void __launch_bounds__(128, 1) bench(Params p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t *sA = base, *sB = base + A_BYTES;
  uint64_t* mbar = (uint64_t*)(sB + B_BYTES);
  uint32_t* tslot = (uint32_t*)(mbar + 4);
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

  // 128B swizzle: 16 B chunk c of row r goes to chunk c ^ (r % 8).
  for (int i = tid; i < 128 * 8; i += 128) { int r = i / 8, c = i % 8;
    *(uint4*)(sA + r * 128 + ((c ^ (r & 7)) * 16)) = *(const uint4*)(p.A + r * 128 + c * 16); }
  for (int i = tid; i < 256 * 8; i += 128) { int r = i / 8, c = i % 8;
    *(uint4*)(sB + r * 128 + ((c ^ (r & 7)) * 16)) = *(const uint4*)(p.B + r * 128 + c * 16); }
  if (tid == 0) {
    for (int w = 0; w < 4; ++w) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(mbar + w)));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("fence.proxy.async.shared::cta;");
  tc_sync();
  uint32_t tmem = *tslot, a_t = tmem + 256;
  if (ATMEM) {  // A rows into TMEM lanes, 32 columns of 4 B = the same 128 B of K per row
    uint32_t v[32]; int r = warp * 32 + lane;
    for (int j = 0; j < 32; ++j) v[j] = ((const uint32_t*)(p.A + r * 128))[j];
    uint32_t ta = a_t + ((uint32_t)(warp * 32) << 16);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {"
      "%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
      :: "r"(ta), "r"(v[0]),"r"(v[1]),"r"(v[2]),"r"(v[3]),"r"(v[4]),"r"(v[5]),"r"(v[6]),"r"(v[7]),
         "r"(v[8]),"r"(v[9]),"r"(v[10]),"r"(v[11]),"r"(v[12]),"r"(v[13]),"r"(v[14]),"r"(v[15]),
         "r"(v[16]),"r"(v[17]),"r"(v[18]),"r"(v[19]),"r"(v[20]),"r"(v[21]),"r"(v[22]),"r"(v[23]),
         "r"(v[24]),"r"(v[25]),"r"(v[26]),"r"(v[27]),"r"(v[28]),"r"(v[29]),"r"(v[30]),"r"(v[31]));
    asm volatile("tcgen05.wait::st.sync.aligned;");
    tc_sync();
  }
  if (tid == 0) {
    uint32_t idesc = make_idesc<DTY>(p.M, p.N);
    uint64_t ad[4], bd[4]; uint32_t at[4];
    for (int j = 0; j < 4; ++j) { ad[j] = make_sdesc(smem_u32(sA) + 32 * j); bd[j] = make_sdesc(smem_u32(sB) + 32 * j); at[j] = a_t + 8 * j; }
    uint32_t d0 = tmem, d1 = d0 + (uint32_t)(p.N <= 128 && p.naccum == 2 ? 128 : 0);
    long long t0 = clock64();
    if (p.iters == 1) {
      mma<DTY, ATMEM>(d0, ad[0], at[0], bd[0], idesc, 0u);
    } else {
      mma<DTY, ATMEM>(d0, ad[0], at[0], bd[0], idesc, 0u);
      mma<DTY, ATMEM>(d1, ad[1], at[1], bd[1], idesc, d1 == d0 ? 1u : 0u);
      mma<DTY, ATMEM>(d0, ad[2], at[2], bd[2], idesc, 1u);
      mma<DTY, ATMEM>(d1, ad[3], at[3], bd[3], idesc, 1u);
      for (int i = 4; i < p.iters; i += 4) {
        mma<DTY, ATMEM>(d0, ad[0], at[0], bd[0], idesc, 1u);
        mma<DTY, ATMEM>(d1, ad[1], at[1], bd[1], idesc, 1u);
        mma<DTY, ATMEM>(d0, ad[2], at[2], bd[2], idesc, 1u);
        mma<DTY, ATMEM>(d1, ad[3], at[3], bd[3], idesc, 1u);
      }
    }
    long long t1 = clock64();
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(mbar)));
    mbar_wait(smem_u32(mbar), 0);
    long long t2 = clock64();
    p.cycles[blockIdx.x * 2 + 0] = t2 - t0;
    p.cycles[blockIdx.x * 2 + 1] = t1 - t0;
  }
  tc_sync();
  if (blockIdx.x == 0 && p.D) {  // dump D (128 lanes x N columns) for verification
    for (int c = 0; c < p.N; c += 8) {
      uint32_t r[8]; uint32_t ta = tmem + ((uint32_t)(warp * 32) << 16) + c;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(ta));
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      for (int j = 0; j < 8; ++j) ((uint32_t*)p.D)[(warp * 32 + lane) * 256 + c + j] = r[j];
    }
  }
  tc_sync();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem));
}

// Fixed costs of the tcgen05 path, one CTA, thread 0 stamps. Rows: [alloc, relinquish, dealloc, fence+sync, commit->wait].
__global__ void __launch_bounds__(128, 1) overhead(int ncols, long long* out) {
  __shared__ uint32_t tslot; __shared__ __align__(8) uint64_t mbar;
  int warp = threadIdx.x / 32;
  if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar)));
                          asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  long long t0 = clock64(), t1 = 0, t2 = 0;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&tslot)), "r"(ncols));
    t1 = clock64();
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    t2 = clock64();
  }
  tc_sync();
  long long t3 = clock64();
  tc_sync();
  long long t4 = clock64();
  long long t5 = 0, t6 = 0;
  if (threadIdx.x == 0) {  // commit with no MMA in flight: the bare commit -> mbarrier arrival round trip
    t5 = clock64();
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(&mbar)));
    mbar_wait(smem_u32(&mbar), 0);
    t6 = clock64();
  }
  __syncthreads();
  long long t7 = clock64();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tslot), "r"(ncols));
  long long t8 = clock64();
  if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = t2 - t1; out[2] = t8 - t7; out[3] = t4 - t3; out[4] = t6 - t5; }
}

// ---------------------------------------------------------------- host
static float decode(int dt, const uint8_t* p, int i) {
  switch (dt) {
    case BF16: { __nv_bfloat16_raw r; memcpy(&r.x, p + 2 * i, 2); return __bfloat162float(__nv_bfloat16(r)); }
    case F16: { __half_raw r; memcpy(&r.x, p + 2 * i, 2); return __half2float(__half(r)); }
    case TF32: { float f; memcpy(&f, p + 4 * i, 4); return f; }
    case E4M3: { __nv_fp8_e4m3 x; memcpy(&x, p + i, 1); return (float)x; }
    case E5M2: { __nv_fp8_e5m2 x; memcpy(&x, p + i, 1); return (float)x; }
    default: return (float)(int8_t)p[i];
  }
}
static void encode(int dt, uint8_t* p, int i, float v) {
  switch (dt) {
    case BF16: { __nv_bfloat16 x(v); memcpy(p + 2 * i, &x, 2); break; }
    case F16: { __half x(v); memcpy(p + 2 * i, &x, 2); break; }
    case TF32: memcpy(p + 4 * i, &v, 4); break;
    case E4M3: { __nv_fp8_e4m3 x(v); memcpy(p + i, &x, 1); break; }
    case E5M2: { __nv_fp8_e5m2 x(v); memcpy(p + i, &x, 1); break; }
    default: p[i] = (uint8_t)(int8_t)v;
  }
}

typedef void (*KernelFn)(Params);
template <int D> static KernelFn pick_dt(int at) { return at ? bench<D, 1> : bench<D, 0>; }
static KernelFn pick(int dt, int at) {
  switch (dt) { case BF16: return pick_dt<BF16>(at); case F16: return pick_dt<F16>(at); case TF32: return pick_dt<TF32>(at);
    case E4M3: return pick_dt<E4M3>(at); case E5M2: return pick_dt<E5M2>(at); default: return pick_dt<S8>(at); }
}

struct Dev { uint8_t *A, *B; float* D; long long* cyc; };

static double run(Dev& d, int dt, int M, int N, int iters, int at, int na, int ctas,
                  double* wall_us = nullptr, double* issue_cyc = nullptr, std::vector<double>* reps = nullptr) {
  Params p{M, N, iters, at, na, d.A, d.B, nullptr, d.cyc};
  KernelFn k = pick(dt, at);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<ctas, 128, SMEM_BYTES>>>(p);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<double> loops, issues; float best_ms = 1e9;
  for (int rep = 0; rep < 7; ++rep) {
    cudaEventRecord(e0); k<<<ctas, 128, SMEM_BYTES>>>(p); cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1); best_ms = std::min(best_ms, ms);
    std::vector<long long> c(ctas * 2);
    CK(cudaMemcpy(c.data(), d.cyc, ctas * 2 * sizeof(long long), cudaMemcpyDeviceToHost));
    std::vector<double> l, s; for (int i = 0; i < ctas; ++i) { l.push_back(c[2 * i]); s.push_back(c[2 * i + 1]); }
    loops.push_back(median(l)); issues.push_back(median(s));
  }
  if (wall_us) *wall_us = best_ms * 1e3;
  if (issue_cyc) *issue_cyc = median(issues);
  if (reps) *reps = loops;
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return median(loops);
}

static bool verify(Dev& d, int dt, int M, int N, int at, const std::vector<uint8_t>& hA, const std::vector<uint8_t>& hB) {
  Params p{M, N, 4, at, 1, d.A, d.B, d.D, d.cyc};
  CK(cudaMemset(d.D, 0, 128 * 256 * 4));
  pick(dt, at)<<<1, 128, SMEM_BYTES>>>(p);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<uint32_t> hD(128 * 256);
  CK(cudaMemcpy(hD.data(), d.D, hD.size() * 4, cudaMemcpyDeviceToHost));
  int K = 128 / DT[dt].bytes;
  auto got = [&](int l, int n) { uint32_t u = hD[l * 256 + n]; float f; memcpy(&f, &u, 4); return dt == S8 ? (float)(int32_t)u : f; };
  auto ref = [&](int m, int n) { double s = 0;
    for (int k = 0; k < K; ++k) s += (double)decode(dt, hA.data(), m * K + k) * decode(dt, hB.data(), n * K + k);
    return (float)s; };
  // Map each reference row to the TMEM lane holding it (the lane layout differs for M=64).
  int matched = 0; std::vector<int> used(128, 0);
  for (int m = 0; m < (at ? 128 : M) && matched < M; ++m)
    for (int l = 0; l < 128; ++l) {
      if (used[l]) continue;
      bool ok = true;
      for (int n = 0; n < N && ok; ++n) ok = got(l, n) == ref(m, n);
      if (ok) { used[l] = 1; ++matched; break; }
    }
  return matched == M;
}

int main(int argc, char** argv) {
  int nsm = num_sms();
  if (argc > 1 && !strcmp(argv[1], "overhead")) {
    long long* out; CK(cudaMalloc(&out, 5 * sizeof(long long)));
    const char* names[5] = {"tcgen05.alloc", "tcgen05.relinquish_alloc_permit", "tcgen05.dealloc",
                            "tcgen05.fence_syncthreads", "tcgen05.commit_mbarrier_wait"};
    for (int ncols : {32, 64, 128, 256, 512}) {
      std::vector<double> v[5];
      for (int rep = 0; rep < 9; ++rep) {
        overhead<<<1, 128>>>(ncols, out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long h[5]; CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
        if (rep) for (int i = 0; i < 5; ++i) v[i].push_back((double)h[i]);  // rep 0 is warm-up
      }
      for (int i = 0; i < 5; ++i) {
        if (ncols != 32 && (i == 3 || i == 4)) continue;  // independent of the allocation size
        Json j; j.s("family", "tc_overhead").s("primitive", names[i]).i("p.cta_group", 1).i("p.threads", 128);
        if (i < 3) j.i("p.ncols", ncols);
        j.f("m.cyc", median(v[i])).i("c.ctas", 1).i("n_reps", (long long)v[i].size()).f("cv", cv(v[i]))
         .s("note", "clock64 around one call by warp 0 (alloc/relinquish/dealloc) or thread 0 (commit with no "
                    "MMA in flight, then mbarrier wait); fence_syncthreads = fence::before + __syncthreads + fence::after "
                    "with 4 warps").print();
      }
    }
    return 0;
  }
  const char* only = argc > 1 ? argv[1] : nullptr;
  for (int dt = 0; dt < NDT; ++dt)
    for (int at = 0; at < 2; ++at) CK(cudaFuncSetAttribute(pick(dt, at), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
  Dev d;
  CK(cudaMalloc(&d.A, A_BYTES)); CK(cudaMalloc(&d.B, B_BYTES));
  CK(cudaMalloc(&d.D, 128 * 256 * 4)); CK(cudaMalloc(&d.cyc, 2 * 1024 * sizeof(long long)));
  const int I1 = 256, I2 = 4352;  // slope between two loop lengths removes the fixed cost
  for (int dt = 0; dt < NDT; ++dt) {
    if (only && strcmp(only, DT[dt].name)) continue;
    // Small integers: exact in every input type and in fp32/s32 sums. Distinct rows keep the lane map unique.
    std::vector<uint8_t> hA(A_BYTES), hB(B_BYTES);
    int per_row = 128 / DT[dt].bytes;
    srand(1 + dt);
    for (int i = 0; i < 128 * per_row; ++i) encode(dt, hA.data(), i, (float)(rand() % 7 - 3));
    for (int i = 0; i < 256 * per_row; ++i) encode(dt, hB.data(), i, (float)(rand() % 7 - 3));
    CK(cudaMemcpy(d.A, hA.data(), A_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.B, hB.data(), B_BYTES, cudaMemcpyHostToDevice));
    int K = DT[dt].kstep;
    for (int at = 0; at < 2; ++at)
    for (int M : {64, 128})
    for (int N = 8; N <= 256; N += 8) {
      if (M == 128 && N % 16) continue;
      if (dt == S8 && N > 32 && N % 16) continue;  // kind::i8: N steps of 8 only up to 32, then 16
      for (int na : {1, 2}) {
        if (na == 2 && N > 128) continue;
        bool ok = verify(d, dt, M, N, at, hA, hB);
        std::vector<double> r1, r2;
        double c1 = run(d, dt, M, N, I1, at, na, 1, nullptr, nullptr, &r1);
        double iss, c2 = run(d, dt, M, N, I2, at, na, 1, nullptr, &iss, &r2);
        double cpm = (c2 - c1) / (I2 - I1);
        std::vector<double> slopes; for (size_t i = 0; i < r1.size(); ++i) slopes.push_back((r2[i] - r1[i]) / (I2 - I1));
        double lat = run(d, dt, M, N, 1, at, na, 1);
        double wall, a1 = run(d, dt, M, N, I1, at, na, nsm), a2 = run(d, dt, M, N, I2, at, na, nsm, &wall);
        double acpm = (a2 - a1) / (I2 - I1);
        double flop = 2.0 * M * N * K;
        Json j; j.s("family", "tc_mma").s("primitive", "tcgen05.mma")
          .s("p.kind", DT[dt].kind).s("p.dtype", DT[dt].name).i("p.cta_group", 1)
          .i("p.M", M).i("p.N", N).i("p.K", K).s("p.a_src", at ? "tmem" : "smem").i("p.naccum", na)
          .s("p.layout", "K-major SW128").i("p.issuers", 1)
          .f("m.cyc_per_mma", cpm).f("m.latency_cyc", lat).f("m.issue_cyc_per_mma", iss / I2)
          .f("m.fma_per_clk_sm", (double)M * N * K / cpm).f("m.allsm_cyc_per_mma", acpm)
          .f("m.allsm_tflops", flop * I2 * nsm / (wall * 1e-6) / 1e12)
          .i("c.ctas", 1).t("verified", ok).i("n_reps", (long long)slopes.size()).f("cv", cv(slopes))
          .s("note", "back-to-back MMAs from one thread; cyc_per_mma is steady-state per SM (1 CTA), "
                     "allsm_* with one CTA on every SM; latency = one MMA issue to commit arrival");
        j.print();
      }
    }
  }
  return 0;
}
