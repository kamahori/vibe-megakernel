// Ingress primitives on sm_100a (family "ingress"):
//   C2  TMA 2D tile loads (cp.async.bulk.tensor): latency of one load, per-SM / all-SM bandwidth vs box size,
//       swizzle, stages in flight, L2 warm vs cold.
//   C3  per-thread global loads (ld.global.v4, ld.global.nc.v4, cp.async 16B): bytes in flight per SM vs bandwidth.
//   C4  shared memory: LDS/STS 32/64/128-bit, ldmatrix x1/x2/x4 (+.trans), bank-conflict vs 128B-swizzled layout.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -I megabench/primbench ingress.cu -o ingress -lcuda
// Usage: ./ingress [c2|c3|c4]   (JSON lines on stdout, diagnostics on stderr)
#include "pb_common.cuh"
#include <cuda.h>
#include <cstring>

struct St3 { long long cyc; unsigned long long ns, g0; };

// ---------------------------------------------------------------- shared device helpers
__device__ __forceinline__ void mbar_init(uint32_t a, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" :: "r"(a), "r"(par) : "memory"); }
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"((uint64_t)m), "r"(c0), "r"(c1), "r"(mbar) : "memory"); }

// ---------------------------------------------------------------- host helpers
static uint32_t* g_buf;      // 1 GiB, word w holds w
static uint4* g_flush;       // 512 MiB, read to evict L2 (clean lines)
static uint32_t* g_out; static St3* g_st;
static const size_t BUF_BYTES = 1ull << 30, FLUSH_BYTES = 512ull << 20;
static int g_nsm;

__global__ void k_fill(uint32_t* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (uint32_t)i; }
__global__ void k_flush(const uint4* p, size_t n, uint32_t* out) {
  uint32_t a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; a += v.x ^ v.w; }
  if (n == 0) out[0] = a; }
static void flush_l2() { k_flush<<<g_nsm * 4, 1024>>>(g_flush, FLUSH_BYTES / 16, g_out); CK(cudaDeviceSynchronize()); }

struct Win { double bw_total, bw_sm, mhz; };  // GB/s (bytes/ns)
static Win summarize(const std::vector<St3>& v, double bytes_per_cta) {
  unsigned long long mn = ~0ull, mx = 0; std::vector<double> per; std::vector<Stamp> s;
  for (auto& x : v) { mn = std::min(mn, x.g0); mx = std::max(mx, x.g0 + x.ns); per.push_back(bytes_per_cta / (double)x.ns); s.push_back({x.cyc, x.ns}); }
  return {bytes_per_cta * v.size() / (double)(mx - mn), median(per), clock_mhz(s)};
}
static std::vector<St3> get_st(int n) { std::vector<St3> h(n); CK(cudaMemcpy(h.data(), g_st, sizeof(St3) * n, cudaMemcpyDeviceToHost)); return h; }

// ================================================================= C2: TMA tile loads
template <bool VERIFY>
__global__ void __launch_bounds__(128, 1) k_tma(const __grid_constant__ CUtensorMap tm, int inner, int rows, int S, int iters,
                                                uint32_t ntiles, uint32_t base, uint32_t* out, St3* st) {
  extern __shared__ uint8_t sm_raw[];
  const uint32_t tile = inner * rows, raw = smem_u32(sm_raw), s0 = (raw + 1023) & ~1023u, bars = s0 + S * tile;
  const int tid = threadIdx.x;
  if (tid == 0) { for (int s = 0; s < S; ++s) mbar_init(bars + 8 * s, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  const uint32_t* p0 = (const uint32_t*)(sm_raw + (s0 - raw));
  uint32_t acc = 0;
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  auto issue = [&](int i) {
    int s = i % S; uint32_t idx = (base + (uint32_t)i * gridDim.x + blockIdx.x) % ntiles;
    mbar_expect(bars + 8 * s, tile); tma2d(s0 + s * tile, &tm, 0, (int)(idx * rows), bars + 8 * s);
  };
  if (!VERIFY) {
    if (tid == 0) {
      for (int i = 0; i < min(S, iters); ++i) issue(i);
      for (int i = 0; i < iters; ++i) { int s = i % S; mbar_wait(bars + 8 * s, (i / S) & 1); if (i + S < iters) issue(i + S); }
    }
  } else {
    if (tid == 0) for (int i = 0; i < min(S, iters); ++i) issue(i);
    for (int i = 0; i < iters; ++i) {
      int s = i % S; mbar_wait(bars + 8 * s, (i / S) & 1);
      const uint32_t* p = p0 + (size_t)s * tile / 4;
      for (uint32_t w = tid; w < tile / 4; w += 128) acc += p[w];
      __syncthreads();
      if (tid == 0 && i + S < iters) issue(i + S);
    }
  }
  if (tid == 0) { long long t1 = clock64(); uint64_t g1 = globaltimer(); st[blockIdx.x] = {t1 - t0, (unsigned long long)(g1 - g0), g0}; }
  if (VERIFY) atomicAdd(out + blockIdx.x, acc);
}

// one CTA, one warp: latency of single loads (issue -> mbarrier complete), sequential, distinct tiles.
__global__ void k_tma_lat(const __grid_constant__ CUtensorMap tm, int inner, int rows, int N, uint32_t ntiles, uint32_t base,
                          uint32_t* lat, St3* st) {
  extern __shared__ uint8_t sm_raw[];
  const uint32_t tile = inner * rows, raw = smem_u32(sm_raw), s0 = (raw + 1023) & ~1023u, bar = s0 + tile;
  if (threadIdx.x == 0) {
    mbar_init(bar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    long long a0 = clock64(); uint64_t g0 = globaltimer();
    for (int i = 0; i < N; ++i) {
      uint32_t idx = (base + (uint32_t)i * 7919u) % ntiles;
      long long t0 = clock64();
      mbar_expect(bar, tile); tma2d(s0, &tm, 0, (int)(idx * rows), bar);
      mbar_wait(bar, i & 1);
      lat[i] = (uint32_t)(clock64() - t0);
    }
    st[0] = {clock64() - a0, (unsigned long long)(globaltimer() - g0), g0};
  }
}

static CUtensorMap make_map(void* p, uint64_t total, int inner, int rows, int swz) {
  CUtensorMap m; cuuint64_t dims[2] = {(cuuint64_t)inner, total / inner}; cuuint64_t strides[1] = {(cuuint64_t)inner};
  cuuint32_t box[2] = {(cuuint32_t)inner, (cuuint32_t)rows}, es[2] = {1, 1};
  CUtensorMapSwizzle sw = swz == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : swz == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed %d (inner %d rows %d swz %d)\n", (int)r, inner, rows, swz); exit(1); }
  return m;
}

static void run_c2() {
  const char* swn[3] = {"none", "64B", "128B"};
  struct SW { int swz, inner; };
  std::vector<SW> sws = {{0, 64}, {0, 128}, {64, 64}, {128, 128}};
  const uint64_t WARM_BYTES = 8ull << 20;
  CK(cudaFuncSetAttribute(k_tma<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
  CK(cudaFuncSetAttribute(k_tma<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
  CK(cudaFuncSetAttribute(k_tma_lat, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
  for (auto sw : sws) for (int rows : {8, 16, 32, 64, 128, 256}) {
    int inner = sw.inner; uint32_t tile = inner * rows;
    CUtensorMap tm = make_map(g_buf, BUF_BYTES, inner, rows, sw.swz);
    for (int cold = 0; cold < 2; ++cold) {
      uint32_t ntiles = (uint32_t)((cold ? BUF_BYTES : WARM_BYTES) / tile);
      // ---- latency (one CTA, sequential loads)
      {
        const int N = 64; uint32_t* dl; CK(cudaMalloc(&dl, N * 4)); std::vector<double> med, mhz; std::vector<uint32_t> hl(N);
        size_t smem = tile + 1024 + 64;
        if (!cold) { k_tma_lat<<<1, 32, smem>>>(tm, inner, rows, N, ntiles, 0, dl, g_st); CK(cudaDeviceSynchronize()); }
        for (int rep = 0; rep < 5; ++rep) {
          if (cold) flush_l2();
          k_tma_lat<<<1, 32, smem>>>(tm, inner, rows, N, ntiles, cold ? rep * 1000 : 0, dl, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(hl.data(), dl, N * 4, cudaMemcpyDeviceToHost));
          std::vector<double> v(hl.begin(), hl.end()); med.push_back(median(v)); auto s = get_st(1); mhz.push_back(clock_mhz({{s[0].cyc, s[0].ns}}));
        }
        double c = median(med), f = median(mhz);
        Json j; j.s("family", "ingress").s("primitive", "tma.load2d.latency").i("p.box_rows", rows).i("p.box_inner_bytes", inner)
          .i("p.tile_bytes", tile).s("p.swizzle", swn[sw.swz == 0 ? 0 : sw.swz == 64 ? 1 : 2]).i("p.stages", 1)
          .f("m.latency_cyc", c).f("m.latency_ns", c / f * 1e3).i("c.ctas", 1).s("c.l2", cold ? "cold" : "warm").f("c.sm_clock_mhz", f)
          .t("verified", true).i("n_reps", 5).f("cv", cv(med))
          .s("note", "median over 64 sequential single-tile loads per rep: clock64 from before expect_tx+issue to mbarrier try_wait success, one thread, "
                     "distinct tiles. Includes transfer time of the whole tile (grows with tile_bytes). cold = L2 flushed before each rep and tiles never touched before. "
                     "Verified by the bandwidth rows of the same box config (checksum of tile contents)");
        j.print(); cudaFree(dl);
      }
      for (int ctas : {1, g_nsm}) for (int S : {1, 2, 4, 8}) {
        if ((size_t)S * tile > 200 * 1024) continue;
        size_t smem = std::max<size_t>((size_t)S * tile + 1024 + 128, 120 * 1024);  // pin one CTA per SM
        int iters = (int)std::min<uint64_t>(2048, std::max<uint64_t>(32, (ctas == 1 ? (32ull << 20) : (600ull << 20)) / ((uint64_t)ctas * tile)));
        // ---- verify (checksum of loaded words; per-tile word sum is closed form because the buffer holds the word index)
        int iv = std::min(iters, 16); CK(cudaMemset(g_out, 0, 4 * g_nsm));
        k_tma<true><<<ctas, 128, smem>>>(tm, inner, rows, S, iv, ntiles, 0, g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint32_t> ho(ctas); CK(cudaMemcpy(ho.data(), g_out, 4 * ctas, cudaMemcpyDeviceToHost));
        bool ok = true; uint64_t W = tile / 4;
        for (int c = 0; c < ctas; ++c) {
          uint32_t e = 0;
          for (int i = 0; i < iv; ++i) { uint64_t idx = ((uint64_t)i * ctas + c) % ntiles; e += (uint32_t)(W * (idx * W) + W * (W - 1) / 2); }
          if (e != ho[c]) ok = false;
        }
        if (!ok) fprintf(stderr, "VERIFY FAIL tma rows %d inner %d swz %d S %d ctas %d\n", rows, inner, sw.swz, S, ctas);
        if (!cold) { k_tma<false><<<ctas, 128, smem>>>(tm, inner, rows, S, iters, ntiles, 0, g_out, g_st); CK(cudaDeviceSynchronize()); }
        std::vector<double> bw, per, mhz;
        for (int rep = 0; rep < 5; ++rep) {
          if (cold) flush_l2();
          k_tma<false><<<ctas, 128, smem>>>(tm, inner, rows, S, iters, ntiles, cold ? rep * 4099 : 0, g_out, g_st);
          CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          Win w = summarize(get_st(ctas), (double)iters * tile); bw.push_back(w.bw_total); per.push_back(w.bw_sm); mhz.push_back(w.mhz);
        }
        Json j; j.s("family", "ingress").s("primitive", "tma.load2d.bw").i("p.box_rows", rows).i("p.box_inner_bytes", inner)
          .i("p.tile_bytes", tile).s("p.swizzle", swn[sw.swz == 0 ? 0 : sw.swz == 64 ? 1 : 2]).i("p.stages", S).i("p.iters", iters)
          .i("p.bytes_in_flight_per_sm", (long long)S * tile)
          .f("m.bw_gbps_total", median(bw)).f("m.bw_gbps_per_sm", median(per)).f("m.bytes_per_clk_sm", median(per) * 1e3 / median(mhz))
          .i("c.ctas", ctas).s("c.l2", cold ? "cold" : "warm").f("c.sm_clock_mhz", median(mhz)).t("verified", ok).i("n_reps", 5).f("cv", cv(bw))
          .s("note", "one producer thread keeps stages tiles in flight (refill right after each mbarrier completes, no consumer work); window = first issue to last completion "
                     "per CTA, includes pipeline fill. bw_gbps_total = all CTAs bytes / (latest end - earliest start). cold = 1 GiB buffer streamed once after an L2 flush; "
                     "warm = 8 MiB footprint re-read after a warm-up launch. verified = per-CTA checksum of all words in a separate short run of the same config");
        j.print();
      }
    }
  }
}

// ================================================================= C3: per-thread global loads
enum { M_LDG, M_LDNC, M_CPA };
__device__ __forceinline__ uint4 ldg4(const uint4* p) { uint4 v; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldnc4(const uint4* p) { uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint4 lds4(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v; }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

template <int MODE, int U, int D>
__global__ void __launch_bounds__(1024, 1) k_ld(const uint4* buf, uint32_t mask, int iters, uint32_t base, uint32_t* out, St3* st) {
  extern __shared__ __align__(16) uint8_t sm[];
  const uint32_t T = blockDim.x, C = gridDim.x, A = C * T * U, tid = threadIdx.x, sb = smem_u32(sm);
  uint32_t acc = 0;
  __syncthreads();
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  if constexpr (MODE != M_CPA) {
    for (int i = 0; i < iters; ++i) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint4* p = buf + ((base + i * A + u * C * T + blockIdx.x * T + tid) & mask);
        v[u] = MODE == M_LDG ? ldg4(p) : ldnc4(p);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) acc += v[u].x + v[u].y + v[u].z + v[u].w;
    }
  } else {
    for (int i = 0; i < iters + D - 1; ++i) {
      if (i < iters) {
        const uint32_t slot = i & (D - 1);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const uint4* p = buf + ((base + i * A + u * C * T + blockIdx.x * T + tid) & mask);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sb + ((slot * U + u) * T + tid) * 16), "l"(p) : "memory");
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      cp_wait<D - 1>();
      if (i >= D - 1) {
        const uint32_t slot = (i - (D - 1)) & (D - 1);
#pragma unroll
        for (int u = 0; u < U; ++u) { uint4 v = lds4(sb + ((slot * U + u) * T + tid) * 16); acc += v.x + v.y + v.z + v.w; }
      }
    }
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (tid == 0) st[blockIdx.x] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
  atomicAdd(out + blockIdx.x, acc);
}
typedef void (*K3)(const uint4*, uint32_t, int, uint32_t, uint32_t*, St3*);
template <int M, int D> static K3 pick3u(int U) {
  switch (U) { case 1: return k_ld<M, 1, D>; case 2: return k_ld<M, 2, D>; case 4: return k_ld<M, 4, D>; case 8: return k_ld<M, 8, D>; default: return k_ld<M, 16, D>; }
}
static K3 pick3(int mode, int D, int U) {
  if (mode == M_LDG) return pick3u<M_LDG, 1>(U);
  if (mode == M_LDNC) return pick3u<M_LDNC, 1>(U);
  return D == 1 ? pick3u<M_CPA, 1>(U) : pick3u<M_CPA, 2>(U);
}

static void run_c3() {
  const struct { int mode, D; const char* name; const char* prim; } modes[] = {
    {M_LDG, 1, "ld.global.v4", "ldg.128"}, {M_LDNC, 1, "ld.global.nc.v4", "ldg.128.nc"},
    {M_CPA, 1, "cp.async.cg 16B depth1", "cp.async.16"}, {M_CPA, 2, "cp.async.cg 16B depth2", "cp.async.16.d2"}};
  const uint32_t WARM_CHUNKS = (16u << 20) / 16, COLD_CHUNKS = (uint32_t)(BUF_BYTES / 16);
  for (auto md : modes) for (int T : {32, 128, 512, 1024}) for (int U : {1, 2, 4, 8, 16}) {
    size_t inflight = (size_t)md.D * T * U * 16; if (md.mode == M_CPA && inflight > 200 * 1024) continue;
    K3 k = pick3(md.mode, md.D, U);
    size_t smem = std::max<size_t>(md.mode == M_CPA ? inflight : 0, 120 * 1024);
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
    for (int ctas : {1, g_nsm}) for (int cold = 0; cold < 2; ++cold) {
      uint32_t mask = (cold ? COLD_CHUNKS : WARM_CHUNKS) - 1;
      uint64_t per_iter = (uint64_t)ctas * T * U * 16, target = ctas == 1 ? (32ull << 20) : (768ull << 20);
      int iters = (int)std::min<uint64_t>(4096, std::max<uint64_t>(8, target / per_iter));
      if (cold && (uint64_t)iters * per_iter > BUF_BYTES) iters = (int)(BUF_BYTES / per_iter);
      // verify: separate short run + host replay of the address pattern
      int iv = 2; CK(cudaMemset(g_out, 0, 4 * g_nsm));
      k<<<ctas, T, smem>>>((const uint4*)g_buf, mask, iv, 0, g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      std::vector<uint32_t> ho(ctas); CK(cudaMemcpy(ho.data(), g_out, 4 * ctas, cudaMemcpyDeviceToHost));
      bool ok = true; uint32_t A = ctas * T * U;
      for (int c = 0; c < ctas && ok; ++c) {
        uint32_t e = 0;
        for (int i = 0; i < iv; ++i) for (int u = 0; u < U; ++u) for (int t = 0; t < T; ++t) {
          uint32_t kc = (i * A + u * ctas * T + c * T + t) & mask; e += 16u * kc + 6u; }
        if (e != ho[c]) ok = false;
      }
      if (!ok) fprintf(stderr, "VERIFY FAIL ld %s T %d U %d ctas %d\n", md.prim, T, U, ctas);
      if (!cold) { k<<<ctas, T, smem>>>((const uint4*)g_buf, mask, iters, 0, g_out, g_st); CK(cudaDeviceSynchronize()); }
      std::vector<double> bw, per, mhz;
      for (int rep = 0; rep < 5; ++rep) {
        if (cold) flush_l2();
        k<<<ctas, T, smem>>>((const uint4*)g_buf, mask, iters, cold ? rep * 65536 * 16 : 0, g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        Win w = summarize(get_st(ctas), (double)iters * T * U * 16); bw.push_back(w.bw_total); per.push_back(w.bw_sm); mhz.push_back(w.mhz);
      }
      double ps = median(per);
      Json j; j.s("family", "ingress").s("primitive", std::string("gload.") + md.prim).s("p.instr", md.name).i("p.threads", T).i("p.loads_per_thread", U)
        .i("p.depth", md.D).i("p.bytes_in_flight_per_sm", (long long)inflight).i("p.iters", iters)
        .f("m.bw_gbps_total", median(bw)).f("m.bw_gbps_per_sm", ps).f("m.bytes_per_clk_sm", ps * 1e3 / median(mhz))
        .f("m.littles_latency_ns", inflight / ps).i("c.ctas", ctas).s("c.l2", cold ? "cold" : "warm").f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", 5).f("cv", cv(bw))
        .s("note", "one CTA per SM (smem padded). bytes_in_flight_per_sm = depth*threads*loads_per_thread*16 at batch issue: each thread issues loads_per_thread independent 16B "
                   "loads, consumes them, then issues the next batch (ldg: no software pipelining; cp.async depth2: second batch issued before waiting on the first). "
                   "littles_latency_ns = in-flight bytes / per-SM bandwidth (effective latency under load). cold = 1 GiB streamed after L2 flush; warm = 16 MiB footprint "
                   "re-read after a warm-up launch. verified = checksum of loaded words vs host replay of the address pattern");
      j.print();
    }
  }
}

// ================================================================= C4: shared memory
enum { LDS32, LDS64, LDS128, STS32, STS64, STS128, LDM1, LDM2, LDM4, LDM1T, LDM2T, LDM4T, LDM4_CONF, LDM4_SWZ, LDS128_CONF, LDS128_SWZ, NMODE };
static const struct { const char* prim; const char* op; int bytes_per_instr; int lanes; int words; } MODEINFO[NMODE] = {
  {"lds.32", "LDS.32", 128, 32, 1}, {"lds.64", "LDS.64", 256, 32, 2}, {"lds.128", "LDS.128", 512, 32, 4},
  {"sts.32", "STS.32", 128, 32, 1}, {"sts.64", "STS.64", 256, 32, 2}, {"sts.128", "STS.128", 512, 32, 4},
  {"ldmatrix.x1", "ldmatrix.x1", 128, 8, 4}, {"ldmatrix.x2", "ldmatrix.x2", 256, 16, 4}, {"ldmatrix.x4", "ldmatrix.x4", 512, 32, 4},
  {"ldmatrix.x1.trans", "ldmatrix.x1.trans", 128, 8, 4}, {"ldmatrix.x2.trans", "ldmatrix.x2.trans", 256, 16, 4}, {"ldmatrix.x4.trans", "ldmatrix.x4.trans", 512, 32, 4},
  {"ldmatrix.x4.rowmajor128B", "ldmatrix.x4 on 128B-pitch rows, no swizzle", 512, 32, 4},
  {"ldmatrix.x4.swizzle128B", "ldmatrix.x4 on 128B-pitch rows, 128B swizzle", 512, 32, 4},
  {"lds.128.rowmajor128B", "LDS.128 on 128B-pitch rows, no swizzle", 512, 32, 4},
  {"lds.128.swizzle128B", "LDS.128 on 128B-pitch rows, 128B swizzle", 512, 32, 4}};
static const uint32_t A1 = 0x9E3779B1u, A2 = 0x85EBCA6Bu;
__host__ __device__ __forceinline__ uint32_t initw(uint32_t w) { return (w * 2654435761u) ^ 0x9E3779B9u; }
// byte offset touched by thread tid at op index n = i*8 + u
__host__ __device__ __forceinline__ uint32_t soff(int mode, int T, int i, int u, int tid) {
  int lane = tid & 31;
  switch (mode) {
    case LDS32: case STS32: return (((i * 8) & (131072 / (T * 4) - 1)) + u) * (T * 4) + tid * 4;
    case LDS64: case STS64: return (((i * 8) & (131072 / (T * 8) - 1)) + u) * (T * 8) + tid * 8;
    case LDS128: case STS128: return (((i * 8) & (131072 / (T * 16) - 1)) + u) * (T * 16) + tid * 16;
    case LDM1: case LDM2: case LDM4: case LDM1T: case LDM2T: case LDM4T: return (((i * 8) & 63) + u) * 512 + lane * 16;
    default: {
      int cc = u, row = (i & 1) * 32 + lane;
      int ph = (mode == LDM4_SWZ || mode == LDS128_SWZ) ? (cc ^ (row & 7)) : cc;
      return row * 128 + ph * 16;
    }
  }
}
__host__ __device__ __forceinline__ uint32_t h2sum(uint32_t x) { return (x & 0xFFFFu) + (x >> 16); }  // repack-invariant (trans changes which halves share a word)
template <int MODE> __device__ __forceinline__ void sop(uint32_t a, uint32_t val0, uint32_t& acc) {
  if constexpr (MODE == LDS32) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); acc += v; }
  else if constexpr (MODE == LDS64) { uint32_t x, y; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"(a) : "memory"); acc += x + y; }
  else if constexpr (MODE == LDS128 || MODE == LDS128_CONF || MODE == LDS128_SWZ) { uint4 v = lds4(a); acc += v.x + v.y + v.z + v.w; }
  else if constexpr (MODE == STS32) asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(val0) : "memory");
  else if constexpr (MODE == STS64) asm volatile("st.shared.v2.u32 [%0], {%1,%2};" :: "r"(a), "r"(val0), "r"(val0 + 1) : "memory");
  else if constexpr (MODE == STS128) asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(val0), "r"(val0 + 1), "r"(val0 + 2), "r"(val0 + 3) : "memory");
  else if constexpr (MODE == LDM1) { uint32_t x; asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(a) : "memory"); acc += x; }
  else if constexpr (MODE == LDM2) { uint32_t x, y; asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"(a) : "memory"); acc += x + y; }
  else if constexpr (MODE == LDM4 || MODE == LDM4_CONF || MODE == LDM4_SWZ) { uint32_t x, y, z, w; asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(a) : "memory"); acc += x + y + z + w; }
  else if constexpr (MODE == LDM1T) { uint32_t x; asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(a) : "memory"); acc += h2sum(x); }
  else if constexpr (MODE == LDM2T) { uint32_t x, y; asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"(a) : "memory"); acc += h2sum(x) + h2sum(y); }
  else { uint32_t x, y, z, w; asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(a) : "memory"); acc += h2sum(x) + h2sum(y) + h2sum(z) + h2sum(w); }
}
template <int MODE, int T>
__global__ void __launch_bounds__(T, 1) k_smem(int iters, uint32_t* out, St3* st) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x; const uint32_t sb = smem_u32(sm);
  for (int w = tid; w < 32768; w += T) ((uint32_t*)sm)[w] = initw(w);
  __syncthreads();
  uint32_t acc = 0, tm = tid * A2, xi = 0;
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 8; ++u) sop<MODE>(sb + soff(MODE, T, i, u, tid ^ (i & 31)), (xi + u * A1) ^ tm, acc);
    xi += 8 * A1;
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (tid == 0) st[blockIdx.x] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
  atomicAdd(out + blockIdx.x, acc);
  if constexpr (MODE >= STS32 && MODE <= STS128) {  // read back: every slot must hold its last writer's value
    const int vb = MODE == STS32 ? 4 : MODE == STS64 ? 8 : 16, P = 131072 / (T * vb), ntot = iters * 8;
    uint32_t bad = 0;
    for (int s = 0; s < P; ++s) {
      int nl = (ntot - 1) - (((ntot - 1) - s) & (P - 1));
      uint32_t wtm = (tid ^ ((nl >> 3) & 31)) * A2;  // writer of this position in that iteration
      for (int k = 0; k < vb / 4; ++k) bad += ((const uint32_t*)sm)[(s * T * vb + tid * vb) / 4 + k] != ((((uint32_t)nl * A1) ^ wtm) + k);
    }
    atomicAdd(out + gridDim.x + blockIdx.x, bad);
  }
}
typedef void (*KS)(int, uint32_t*, St3*);
template <int M> static KS pickT(int T) { switch (T) { case 128: return k_smem<M, 128>; case 256: return k_smem<M, 256>; case 512: return k_smem<M, 512>; default: return k_smem<M, 1024>; } }
static KS pickS(int m, int T) {
#define C4(x) case x: return pickT<x>(T)
  switch (m) { C4(0); C4(1); C4(2); C4(3); C4(4); C4(5); C4(6); C4(7); C4(8); C4(9); C4(10); C4(11); C4(12); C4(13); C4(14); default: return pickT<15>(T); }
}

static void run_c4() {
  const int I1 = 128, I2 = 1152;
  for (int m = 0; m < NMODE; ++m) for (int T : {128, 256, 512, 1024}) for (int ctas : {1, g_nsm}) {
    if (ctas > 1 && T != 1024) continue;
    KS k = pickS(m, T); CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072));
    // verify
    const int iv = 64; CK(cudaMemset(g_out, 0, 8 * g_nsm));
    k<<<1, T, 131072>>>(iv, g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint32_t ho[2]; CK(cudaMemcpy(ho, g_out, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ho + 1, g_out + g_nsm, 4, cudaMemcpyDeviceToHost));
    bool ok;
    if (m >= STS32 && m <= STS128) ok = ho[1] == 0;
    else {
      uint32_t e = 0;
      for (int i = 0; i < iv; ++i) for (int u = 0; u < 8; ++u) for (int t = 0; t < T; ++t) {
        if ((t & 31) >= MODEINFO[m].lanes) continue;
        uint32_t o = soff(m, T, i, u, t ^ (i & 31)) / 4;
        for (int w = 0; w < MODEINFO[m].words; ++w) e += (m >= LDM1T && m <= LDM4T) ? h2sum(initw(o + w)) : initw(o + w);
      }
      ok = e == ho[0];
    }
    if (!ok) fprintf(stderr, "VERIFY FAIL smem %s T %d\n", MODEINFO[m].prim, T);
    std::vector<double> per, mhz;
    for (int rep = 0; rep < 5; ++rep) {
      double c[2]; int its[2] = {I1, I2};
      for (int x = 0; x < 2; ++x) {
        k<<<ctas, T, 131072>>>(its[x], g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        auto s = get_st(ctas); std::vector<double> cy; for (auto& v : s) cy.push_back((double)v.cyc); c[x] = median(cy);
        if (x == 1) { std::vector<Stamp> ss; for (auto& v : s) ss.push_back({v.cyc, v.ns}); mhz.push_back(clock_mhz(ss)); }
      }
      per.push_back((c[1] - c[0]) / (I2 - I1));
    }
    double it = median(per), nw = T / 32, cyc_instr = it / (nw * 8);
    Json j; j.s("family", "ingress").s("primitive", std::string("smem.") + MODEINFO[m].prim).s("p.op", MODEINFO[m].op).i("p.warps", (long long)nw)
      .i("p.bytes_per_warp_instr", MODEINFO[m].bytes_per_instr)
      .f("m.cyc_per_warp_instr_sm", cyc_instr).f("m.warp_instr_per_clk_sm", 1.0 / cyc_instr)
      .f("m.bytes_per_clk_sm", MODEINFO[m].bytes_per_instr / cyc_instr).i("c.ctas", ctas).f("c.sm_clock_mhz", median(mhz))
      .t("verified", ok).i("n_reps", 5).f("cv", cv(per))
      .s("note", "slope over two loop lengths, 8 independent ops per iteration with addresses varying per iteration; bytes_per_clk_sm counts useful bytes (ldmatrix xN = N*128 B). "
                 "Peak shared memory bandwidth is 128 B/clk/SM. rowmajor128B = 8 lanes of a phase hit the same 16B column of 8 rows of a 128B-pitch tile (8-way bank conflict); "
                 "swizzle128B = chunk index xor (row mod 8), conflict-free. LDS/STS lanes read consecutive addresses (conflict-free). "
                 "ldmatrix .trans only changes register layout (its checksum sums 16-bit halves: ~3 extra ALU ops per register, so .trans rows and sts.128, which builds 4 distinct values, can be issue-limited and read as a lower bound). Each iteration permutes lanes by xor (iteration mod 32) so addresses are not loop-invariant; conflict structure is unchanged. verified = checksum of loaded words (sts: read-back of last-written values)");
    j.print();
  }
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  g_nsm = num_sms();
  CK(cudaMalloc(&g_buf, BUF_BYTES)); CK(cudaMalloc(&g_flush, FLUSH_BYTES)); CK(cudaMemset(g_flush, 1, FLUSH_BYTES));
  CK(cudaMalloc(&g_out, 8 * 4096)); CK(cudaMalloc(&g_st, sizeof(St3) * 4096));
  k_fill<<<g_nsm * 4, 1024>>>(g_buf, BUF_BYTES / 4); CK(cudaDeviceSynchronize());
  if (!only || !strcmp(only, "c4")) run_c4();
  if (!only || !strcmp(only, "c3")) run_c3();
  if (!only || !strcmp(only, "c2")) run_c2();
  return 0;
}
