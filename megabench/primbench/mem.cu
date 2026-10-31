// Memory-system primitives on sm_100a (family "mem"):
//   E1  HBM streaming read bandwidth vs number of CTAs and bytes in flight per SM
//   E2  L2: read / write bandwidth and pointer-chase latency vs working set (1 MiB .. 256 MiB)
//   E3  DSMEM: cluster size 2/4/8, remote ld/st latency and bandwidth (mapa + ld/st.shared::cluster)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -I megabench/primbench mem.cu -o mem
// Usage: ./mem [e1|e2|e3]   (JSON lines on stdout, diagnostics on stderr)
#include "pb_common.cuh"
#include <cstring>
#include <random>

struct St3 { long long cyc; unsigned long long ns, g0; };
__device__ __forceinline__ uint4 ldg4(const uint4* p) { uint4 v; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ void stg4(uint4* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory"); }

static uint32_t* g_buf;  // 1 GiB, word w holds w
static uint4* g_flush;   // 512 MiB, read to evict L2
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

struct Win { double bw_total, bw_sm, mhz; };
static Win summarize(const std::vector<St3>& v, double bytes_per_cta) {
  unsigned long long mn = ~0ull, mx = 0; std::vector<double> per; std::vector<Stamp> s;
  for (auto& x : v) { mn = std::min(mn, x.g0); mx = std::max(mx, x.g0 + x.ns); per.push_back(bytes_per_cta / (double)x.ns); s.push_back({x.cyc, x.ns}); }
  return {bytes_per_cta * v.size() / (double)(mx - mn), median(per), clock_mhz(s)};
}
static std::vector<St3> get_st(int n) { std::vector<St3> h(n); CK(cudaMemcpy(h.data(), g_st, sizeof(St3) * n, cudaMemcpyDeviceToHost)); return h; }

// Streaming read: each thread issues U independent 16B loads (warp-contiguous), sums them, repeats. One CTA per SM (smem padded).
template <int U>
__global__ void __launch_bounds__(1024, 1) k_rd(const uint4* buf, uint32_t mask, int iters, uint32_t* out, St3* st) {
  const uint32_t T = blockDim.x, C = gridDim.x, A = C * T * U, tid = threadIdx.x;
  uint32_t acc = 0;
  __syncthreads();
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ldg4(buf + ((i * A + u * C * T + blockIdx.x * T + tid) & mask));
#pragma unroll
    for (int u = 0; u < U; ++u) acc += v[u].x + v[u].y + v[u].z + v[u].w;
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (tid == 0) st[blockIdx.x] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
  atomicAdd(out + blockIdx.x, acc);
}
// Streaming write: chunk kc gets words 4kc..4kc+3 (its own index), so the final buffer is checkable without knowing the pass count.
template <int U>
__global__ void __launch_bounds__(1024, 1) k_wr(uint4* buf, uint32_t mask, int iters, St3* st) {
  const uint32_t T = blockDim.x, C = gridDim.x, A = C * T * U, tid = threadIdx.x;
  __syncthreads();
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < U; ++u) { uint32_t kc = (i * A + u * C * T + blockIdx.x * T + tid) & mask; stg4(buf + kc, 4 * kc, 4 * kc + 1, 4 * kc + 2, 4 * kc + 3); }
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (tid == 0) st[blockIdx.x] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
}
__global__ void k_check(const uint32_t* p, size_t n, uint32_t* bad) {
  uint32_t b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b += p[i] != (uint32_t)i;
  if (b) atomicAdd(bad, b); }

typedef void (*KR)(const uint4*, uint32_t, int, uint32_t*, St3*);
typedef void (*KW)(uint4*, uint32_t, int, St3*);
static KR pickR(int U) { switch (U) { case 1: return k_rd<1>; case 2: return k_rd<2>; case 4: return k_rd<4>; case 8: return k_rd<8>; default: return k_rd<16>; } }
static KW pickW(int U) { switch (U) { case 1: return k_wr<1>; case 2: return k_wr<2>; case 4: return k_wr<4>; case 8: return k_wr<8>; default: return k_wr<16>; } }
static const size_t PAD_SMEM = 120 * 1024;  // forces one CTA per SM
static std::vector<uint32_t> expect_rd(int C, int T, int U, int iters, uint32_t mask) {
  std::vector<uint32_t> e(C); uint32_t A = C * T * U;
  for (int c = 0; c < C; ++c) { uint32_t s = 0;
    for (int i = 0; i < iters; ++i) for (int u = 0; u < U; ++u) for (int t = 0; t < T; ++t) s += 16u * ((i * A + u * C * T + c * T + t) & mask) + 6u;
    e[c] = s; }
  return e;
}

// ---------------------------------------------------------------- E1
static void run_e1() {
  const struct { int T, U; } cfg[] = {{256, 1}, {256, 2}, {256, 4}, {256, 8}, {512, 8}, {1024, 8}, {1024, 16}};
  for (int ctas : {8, 16, 32, 64, 96, 128, g_nsm}) for (auto c : cfg) {
    KR k = pickR(c.U); CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PAD_SMEM));
    uint32_t mask = (uint32_t)(BUF_BYTES / 16) - 1;
    size_t inflight = (size_t)c.T * c.U * 16;
    int iters = (int)std::max<size_t>(2, (6ull << 20) / inflight);
    std::vector<double> bw, per, mhz; bool ok = true;
    for (int rep = 0; rep < 5; ++rep) {
      flush_l2(); CK(cudaMemset(g_out, 0, 4 * g_nsm));
      k<<<ctas, c.T, PAD_SMEM>>>((const uint4*)g_buf, mask, iters, g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      Win w = summarize(get_st(ctas), (double)iters * inflight); bw.push_back(w.bw_total); per.push_back(w.bw_sm); mhz.push_back(w.mhz);
      if (rep == 0) {
        std::vector<uint32_t> ho(ctas); CK(cudaMemcpy(ho.data(), g_out, 4 * ctas, cudaMemcpyDeviceToHost));
        auto e = expect_rd(ctas, c.T, c.U, iters, mask); for (int i = 0; i < ctas; ++i) if (e[i] != ho[i]) ok = false;
        if (!ok) fprintf(stderr, "VERIFY FAIL e1 ctas %d T %d U %d\n", ctas, c.T, c.U);
      }
    }
    Json j; j.s("family", "mem").s("primitive", "hbm.read_stream").i("p.ctas", ctas).i("p.threads", c.T).i("p.loads_per_thread", c.U)
      .i("p.bytes_in_flight_per_sm", (long long)inflight).i("p.bytes_per_cta", (long long)iters * inflight)
      .f("m.bw_gbps_total", median(bw)).f("m.bw_gbps_per_sm", median(per)).f("m.bytes_per_clk_sm", median(per) * 1e3 / median(mhz))
      .f("m.littles_latency_ns", inflight / median(per)).i("c.ctas", ctas).s("c.l2", "cold").f("c.sm_clock_mhz", median(mhz))
      .t("verified", ok).i("n_reps", 5).f("cv", cv(bw))
      .s("note", "ld.global.v4 streaming, every byte read once from a 1 GiB buffer after an L2 flush (reads of 6 MiB per CTA, so each point is a short burst: window includes ramp-up). "
                 "bytes_in_flight_per_sm = threads*loads_per_thread*16 at batch issue (no software pipelining across batches); sequential streams may be prefetched in L2, so littles_latency_ns can sit below DRAM latency. bw_gbps_total = all bytes / (latest end - earliest start). "
                 "Peak HBM read bandwidth is the plateau of bw_gbps_total over CTAs and bytes in flight. verified = per-CTA checksum vs host replay");
    j.print();
  }
}

// ---------------------------------------------------------------- E2
__global__ void k_chase(const char* buf, uint32_t start, int hops, uint32_t* out, St3* st) {
  uint32_t idx = start; long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int h = 0; h < hops; ++h) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(buf + (size_t)idx * 128) : "memory");
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  out[0] = idx; st[0] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
}

static void run_e2() {
  const int T = 1024, U = 8; const size_t WS_MAX = 256ull << 20;
  KR kr = pickR(U); KW kw = pickW(U);
  CK(cudaFuncSetAttribute(kr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PAD_SMEM));
  CK(cudaFuncSetAttribute(kw, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PAD_SMEM));
  uint32_t* dbad; CK(cudaMalloc(&dbad, 4));
  std::vector<uint32_t> host(WS_MAX / 4);
  for (size_t ws = 1ull << 20; ws <= WS_MAX; ws <<= 1) {
    uint32_t mask = (uint32_t)(ws / 16) - 1; int ctas = g_nsm;
    uint64_t per_iter = (uint64_t)ctas * T * U * 16;
    int iters = (int)std::max<uint64_t>(4, std::max<uint64_t>(1ull << 30, 2 * ws) / per_iter);
    // ---- read bandwidth (warm: one warm-up pass first; for ws beyond L2 capacity the data streams from HBM)
    {
      CK(cudaMemset(g_out, 0, 4 * g_nsm));
      kr<<<ctas, T, PAD_SMEM>>>((const uint4*)g_buf, mask, iters, g_out, g_st); CK(cudaDeviceSynchronize());
      std::vector<double> bw, per, mhz; bool ok = true;
      for (int rep = 0; rep < 5; ++rep) {
        CK(cudaMemset(g_out, 0, 4 * g_nsm));
        kr<<<ctas, T, PAD_SMEM>>>((const uint4*)g_buf, mask, iters, g_out, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        Win w = summarize(get_st(ctas), (double)iters * T * U * 16); bw.push_back(w.bw_total); per.push_back(w.bw_sm); mhz.push_back(w.mhz);
        if (rep == 0) {
          std::vector<uint32_t> ho(ctas); CK(cudaMemcpy(ho.data(), g_out, 4 * ctas, cudaMemcpyDeviceToHost));
          auto e = expect_rd(ctas, T, U, iters, mask); for (int i = 0; i < ctas; ++i) if (e[i] != ho[i]) ok = false;
          if (!ok) fprintf(stderr, "VERIFY FAIL e2 read ws %zu\n", ws);
        }
      }
      Json j; j.s("family", "mem").s("primitive", "l2.read_bw").i("p.working_set_bytes", (long long)ws).i("p.threads", T).i("p.loads_per_thread", U)
        .i("p.traffic_bytes", (long long)iters * per_iter).f("m.bw_gbps_total", median(bw)).f("m.bw_gbps_per_sm", median(per))
        .f("m.bytes_per_clk_sm", median(per) * 1e3 / median(mhz)).i("c.ctas", ctas).s("c.l2", "warm").f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", 5).f("cv", cv(bw))
        .s("note", "all SMs re-read a working set (wrapping index) after one warm-up launch, 128 KiB in flight per SM; traffic = max(1 GiB, 2 x working set). "
                   "Working sets larger than the L2 capacity stream from HBM: the drop in bw_gbps_total marks the effective capacity. verified = per-CTA checksum vs host replay");
      j.print();
    }
    // ---- write bandwidth
    {
      CK(cudaMemset(g_buf, 0xFF, ws));
      kw<<<ctas, T, PAD_SMEM>>>((uint4*)g_buf, mask, iters, g_st); CK(cudaDeviceSynchronize());
      CK(cudaMemset(dbad, 0, 4)); k_check<<<g_nsm * 2, 1024>>>(g_buf, ws / 4, dbad); CK(cudaDeviceSynchronize());
      uint32_t bad; CK(cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost)); bool ok = bad == 0;
      if (!ok) fprintf(stderr, "VERIFY FAIL e2 write ws %zu bad %u\n", ws, bad);
      std::vector<double> bw, per, mhz;
      for (int rep = 0; rep < 5; ++rep) {
        kw<<<ctas, T, PAD_SMEM>>>((uint4*)g_buf, mask, iters, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        Win w = summarize(get_st(ctas), (double)iters * T * U * 16); bw.push_back(w.bw_total); per.push_back(w.bw_sm); mhz.push_back(w.mhz);
      }
      Json j; j.s("family", "mem").s("primitive", "l2.write_bw").i("p.working_set_bytes", (long long)ws).i("p.threads", T).i("p.stores_per_thread", U)
        .i("p.traffic_bytes", (long long)iters * per_iter).f("m.bw_gbps_total", median(bw)).f("m.bw_gbps_per_sm", median(per))
        .f("m.bytes_per_clk_sm", median(per) * 1e3 / median(mhz)).i("c.ctas", ctas).s("c.l2", "warm").f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", 5).f("cv", cv(bw))
        .s("note", "all SMs st.global.v4 over a working set with wrapping index (write-back L2; beyond capacity dirty lines are evicted to HBM). "
                   "verified = buffer was set to 0xFF, then every word must equal its own index after the first launch (all chunks covered)");
      j.print();
    }
  }
  CK(cudaMemcpy(g_buf, host.data(), 0, cudaMemcpyHostToDevice));
  k_fill<<<g_nsm * 4, 1024>>>(g_buf, BUF_BYTES / 4); CK(cudaDeviceSynchronize());  // restore index pattern
  // ---- pointer-chase latency over a random single-cycle permutation of 128B lines (one word per line)
  char* dchain = (char*)g_buf;  // reuse the big buffer
  std::mt19937 rng(12345);
  for (size_t ws = 1ull << 20; ws <= WS_MAX; ws <<= 1) {
    size_t lines = ws / 128;
    std::vector<uint32_t> perm(lines), nxt(lines); for (size_t i = 0; i < lines; ++i) perm[i] = (uint32_t)i;
    for (size_t i = lines - 1; i > 0; --i) std::swap(perm[i], perm[rng() % i]);  // Sattolo: single cycle
    for (size_t i = 0; i < lines; ++i) nxt[perm[i]] = perm[(i + 1) % lines];
    std::fill(host.begin(), host.begin() + ws / 4, 0u);
    for (size_t i = 0; i < lines; ++i) host[i * 32] = nxt[i];
    CK(cudaMemcpy(dchain, host.data(), ws, cudaMemcpyHostToDevice));
    uint32_t* dout = g_out; uint32_t cur = 0, h_out;
    k_chase<<<1, 1>>>(dchain, cur, (int)lines, dout, g_st); CK(cudaDeviceSynchronize());  // warm: one full traversal (returns to start)
    CK(cudaMemcpy(&h_out, dout, 4, cudaMemcpyDeviceToHost)); bool ok = h_out == cur;
    int hops = (int)std::min<size_t>(lines, 131072);
    std::vector<double> cyc, ns, mhz;
    for (int rep = 0; rep < 5; ++rep) {
      k_chase<<<1, 1>>>(dchain, cur, hops, dout, g_st); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h_out, dout, 4, cudaMemcpyDeviceToHost));
      for (int h = 0; h < hops; ++h) cur = nxt[cur];
      if (h_out != cur) ok = false;
      auto s = get_st(1); cyc.push_back((double)s[0].cyc / hops); ns.push_back((double)s[0].ns / hops); mhz.push_back(clock_mhz({{s[0].cyc, s[0].ns}}));
    }
    if (!ok) fprintf(stderr, "VERIFY FAIL e2 chase ws %zu\n", ws);
    Json j; j.s("family", "mem").s("primitive", "l2.chase_latency").i("p.working_set_bytes", (long long)ws).i("p.hops", hops)
      .f("m.latency_cyc", median(cyc)).f("m.latency_ns", median(ns)).i("c.ctas", 1).s("c.l2", "warm").f("c.sm_clock_mhz", median(mhz))
      .t("verified", ok).i("n_reps", 5).f("cv", cv(ns))
      .s("note", "one thread, dependent ld.global.cg.u32 (L1 bypassed) along a random single-cycle permutation of 128B lines covering the working set; one full warm traversal first, "
                 "then timed hops continue around the cycle. Includes TLB and cross-partition effects of the footprint; beyond L2 capacity hops are DRAM latency. "
                 "verified = final index equals host walk of the permutation");
    j.print();
  }
  k_fill<<<g_nsm * 4, 1024>>>(g_buf, BUF_BYTES / 4); CK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------- E3: DSMEM
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ uint32_t crank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t cnrank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void csync() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory"); }
__host__ __device__ __forceinline__ uint32_t initw(uint32_t w) { return (w * 2654435761u) ^ 0x9E3779B9u; }
static const uint32_t A1 = 0x9E3779B1u, A2 = 0x85EBCA6Bu;

// dependent load chain through the smem of `target` (mode 1: ld.shared::cluster via mapa, mode 0: plain ld.shared)
__global__ void k_dsm_chase(int mode, uint32_t target, int iters, uint32_t* out, St3* st) {
  __shared__ uint32_t chain[1024];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) chain[i] = (i + 37) & 1023;
  csync();
  if (crank() == 0 && threadIdx.x == 0) {
    uint32_t a = smem_u32(chain); if (mode) a = mapa(a, target);
    uint32_t idx = 0; long long t0 = clock64(); uint64_t g0 = globaltimer();
    if (mode) for (int h = 0; h < iters; ++h) asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(idx) : "r"(a + idx * 4) : "memory");
    else for (int h = 0; h < iters; ++h) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(idx) : "r"(a + idx * 4) : "memory");
    long long t1 = clock64(); uint64_t g1 = globaltimer();
    out[0] = idx; st[0] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
  }
  csync();
}
// round trip: rank 0 stores flag i to `partner`, partner echoes it back
__global__ void k_dsm_pp(uint32_t partner, int iters, uint32_t* out, St3* st) {
  __shared__ uint32_t flag;
  if (threadIdx.x == 0) flag = 0;
  csync();
  uint32_t fa = smem_u32(&flag), r = crank();
  if (threadIdx.x == 0 && r == 0) {
    uint32_t ra = mapa(fa, partner), v; long long t0 = clock64(); uint64_t g0 = globaltimer();
    for (int i = 1; i <= iters; ++i) {
      asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" :: "r"(ra), "r"(i) : "memory");
      do { asm volatile("ld.acquire.cluster.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(fa) : "memory"); } while (v != (uint32_t)i);
    }
    long long t1 = clock64(); uint64_t g1 = globaltimer();
    out[0] = v; st[0] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
  } else if (threadIdx.x == 0 && r == partner) {
    uint32_t ra = mapa(fa, 0), v;
    for (int i = 1; i <= iters; ++i) {
      do { asm volatile("ld.acquire.cluster.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(fa) : "memory"); } while (v != (uint32_t)i);
      asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" :: "r"(ra), "r"(i) : "memory");
    }
    out[1] = v;
  }
  csync();
}
// bandwidth: mode 0 local ld.shared, 1 read from rank+1 (ld.shared::cluster.v4), 2 write to rank+1 (st.shared::cluster.v4)
template <int MODE>
__global__ void __launch_bounds__(1024, 1) k_dsm_bw(int iters, uint32_t* out, St3* st) {
  extern __shared__ __align__(16) uint8_t sm[];
  const uint32_t tid = threadIdx.x, sb = smem_u32(sm), r = crank(), CS = cnrank();
  for (uint32_t w = tid; w < 32768; w += 1024) ((uint32_t*)sm)[w] = initw(r * 32768 + w);
  csync();
  const uint32_t q = (r + 1) % CS, base = MODE == 0 ? sb : mapa(sb, q);
  uint32_t acc = 0, xi = 0, wtm0 = tid * A2;
  long long t0 = clock64(); uint64_t g0 = globaltimer();
  for (int i = 0; i < iters; ++i) {
    const uint32_t a0 = base + (tid ^ (i & 31)) * 16;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if constexpr (MODE == 2) {
        uint32_t v = ((xi + u * A1) ^ wtm0) + q * 0x1000193u;
        asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a0 + u * 16384), "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3) : "memory");
      } else {
        uint4 v;
        if constexpr (MODE == 1) asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a0 + u * 16384) : "memory");
        else asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a0 + u * 16384) : "memory");
        acc += v.x + v.y + v.z + v.w;
      }
    }
    xi += 8 * A1;
  }
  __syncthreads();
  long long t1 = clock64(); uint64_t g1 = globaltimer();
  if (tid == 0) st[blockIdx.x] = {t1 - t0, (unsigned long long)(g1 - g0), g0};
  csync();  // remote traffic complete and visible before checking / exiting
  if (MODE != 2) atomicAdd(out + blockIdx.x, acc);
  else {  // my smem was written by rank r-1: slot u must hold its last writer's value
    const uint32_t pq = (r + CS - 1) % CS; const int ntot = iters * 8; uint32_t bad = 0;
    for (int u = 0; u < 8; ++u) {
      int nl = (ntot - 1) - (((ntot - 1) - u) & 7); uint32_t wtm = (tid ^ ((nl >> 3) & 31)) * A2;
      for (int k = 0; k < 4; ++k) bad += ((const uint32_t*)sm)[(u * 16384 + tid * 16) / 4 + k] != ((((uint32_t)nl * A1) ^ wtm) + k + (uint32_t)(r) * 0x1000193u + 0 * pq);
    }
    atomicAdd(out + blockIdx.x, bad);
  }
}

template <typename K> static cudaError_t launch_cluster(K kern, int ctas, int cs, int threads, size_t smem, void** args) {
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(ctas); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  return cudaLaunchKernelExC(&cfg, (const void*)kern, args);
}

static void run_e3() {
  const int T = 1024, ITERS = 256; const size_t SMEM = 131072;
  for (int cs : {2, 4, 8}) {
    // ---- latency: dependent loads (rank 0 -> target) and store round trip (rank 0 <-> partner)
    std::vector<uint32_t> targets = {0, 1}; if (cs - 1 > 1) targets.push_back(cs - 1);
    for (int mode = 0; mode < 1 + (int)targets.size(); ++mode) {
      uint32_t tgt = mode == 0 ? 0 : targets[mode - 1]; int md = mode ? 1 : 0; int iters = 4096;
      std::vector<double> cyc, ns, mhz; bool ok = true;
      for (int rep = 0; rep < 5; ++rep) {
        void* args[] = {&md, &tgt, &iters, &g_out, &g_st};
        CK(launch_cluster(k_dsm_chase, cs, cs, 32, 0, args)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint32_t h; CK(cudaMemcpy(&h, g_out, 4, cudaMemcpyDeviceToHost)); if (h != (uint32_t)(37ull * iters) % 1024) ok = false;
        auto s = get_st(1); cyc.push_back((double)s[0].cyc / iters); ns.push_back((double)s[0].ns / iters); mhz.push_back(clock_mhz({{s[0].cyc, s[0].ns}}));
      }
      Json j; j.s("family", "mem").s("primitive", md ? "dsmem.ld_latency" : "smem.ld_latency_local").i("p.cluster_size", cs).s("p.access", md ? "ld.shared::cluster.u32 via mapa" : "ld.shared.u32")
        .i("p.target_rank", md ? tgt : 0).f("m.latency_cyc", median(cyc)).f("m.latency_ns", median(ns)).i("c.ctas", cs).f("c.sm_clock_mhz", median(mhz))
        .t("verified", ok).i("n_reps", 5).f("cv", cv(cyc))
        .s("note", "rank 0 thread 0, dependent load chain (next address = loaded value) through the smem of target_rank; target_rank 0 is rank 0 reading its own smem through the cluster window. "
                   "ld_latency_local is the plain ld.shared reference. verified = final chain index");
      j.print();
    }
    for (uint32_t partner : targets) {
      if (partner == 0) continue;
      int iters = 2048; std::vector<double> cyc, ns, mhz; bool ok = true;
      for (int rep = 0; rep < 5; ++rep) {
        CK(cudaMemset(g_out, 0, 8));
        void* args[] = {&partner, &iters, &g_out, &g_st};
        CK(launch_cluster(k_dsm_pp, cs, cs, 32, 0, args)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint32_t h[2]; CK(cudaMemcpy(h, g_out, 8, cudaMemcpyDeviceToHost)); if (h[0] != (uint32_t)iters || h[1] != (uint32_t)iters) ok = false;
        auto s = get_st(1); cyc.push_back((double)s[0].cyc / iters / 2); ns.push_back((double)s[0].ns / iters / 2); mhz.push_back(clock_mhz({{s[0].cyc, s[0].ns}}));
      }
      Json j; j.s("family", "mem").s("primitive", "dsmem.st_oneway_latency").i("p.cluster_size", cs).i("p.target_rank", partner)
        .f("m.latency_cyc", median(cyc)).f("m.latency_ns", median(ns)).i("c.ctas", cs).f("c.sm_clock_mhz", median(mhz)).t("verified", ok).i("n_reps", 5).f("cv", cv(cyc))
        .s("note", "half of the round trip of a ping-pong: st.release.cluster.shared::cluster to the partner flag, partner spins with ld.acquire.cluster and echoes. "
                   "Includes release/acquire cost on both sides, so it is store-to-observed latency, not raw store issue. verified = both flags reached the iteration count");
      j.print();
    }
    // ---- bandwidth
    for (int ncl : {1, 0}) {
      int ctas;
      if (ncl) ctas = cs;
      else {  // as many clusters as can be co-resident (capped by SM count)
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(cs); cfg.blockDim = dim3(T); cfg.dynamicSmemBytes = SMEM;
        cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim = {(unsigned)cs, 1, 1}; cfg.attrs = at; cfg.numAttrs = 1;
        CK(cudaFuncSetAttribute(k_dsm_bw<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        int mx = 0; CK(cudaOccupancyMaxActiveClusters(&mx, (const void*)k_dsm_bw<1>, &cfg));
        ctas = std::min(mx, g_nsm / cs) * cs; if (ctas == cs) continue;
      }
      for (int mode = 0; mode < 3; ++mode) {
        const void* kf = mode == 0 ? (const void*)k_dsm_bw<0> : mode == 1 ? (const void*)k_dsm_bw<1> : (const void*)k_dsm_bw<2>;
        CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        std::vector<double> bw, per, mhz; bool ok = true;
        for (int rep = 0; rep < 5; ++rep) {
          CK(cudaMemset(g_out, 0, 4 * g_nsm)); int it = ITERS; void* args[] = {&it, &g_out, &g_st};
          cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(ctas); cfg.blockDim = dim3(T); cfg.dynamicSmemBytes = SMEM;
          cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim = {(unsigned)cs, 1, 1}; cfg.attrs = at; cfg.numAttrs = 1;
          CK(cudaLaunchKernelExC(&cfg, kf, args)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
          Win w = summarize(get_st(ctas), (double)ITERS * 8 * T * 16); bw.push_back(w.bw_total); per.push_back(w.bw_sm); mhz.push_back(w.mhz);
          if (rep == 0) {
            std::vector<uint32_t> ho(ctas); CK(cudaMemcpy(ho.data(), g_out, 4 * ctas, cudaMemcpyDeviceToHost));
            if (mode == 2) { for (int c = 0; c < ctas; ++c) if (ho[c]) ok = false; }
            else {
              std::vector<uint32_t> e(cs);
              for (int r = 0; r < cs; ++r) { uint32_t q = mode == 0 ? r : (r + 1) % cs, s = 0;
                for (int i = 0; i < ITERS; ++i) for (int u = 0; u < 8; ++u) for (int t = 0; t < T; ++t) {
                  uint32_t o = (u * 16384 + (t ^ (i & 31)) * 16) / 4; for (int k = 0; k < 4; ++k) s += initw(q * 32768 + o + k); }
                e[r] = s; }
              for (int c = 0; c < ctas; ++c) if (ho[c] != e[c % cs]) ok = false;
            }
            if (!ok) fprintf(stderr, "VERIFY FAIL e3 bw cs %d mode %d\n", cs, mode);
          }
        }
        const char* pr[3] = {"smem.ld_bw_local", "dsmem.ld_bw", "dsmem.st_bw"};
        Json j; j.s("family", "mem").s("primitive", pr[mode]).i("p.cluster_size", cs).i("p.threads", T).i("p.ops_per_thread_iter", 8)
          .f("m.bw_gbps_total", median(bw)).f("m.bw_gbps_per_sm", median(per)).f("m.bytes_per_clk_sm", median(per) * 1e3 / median(mhz))
          .i("c.ctas", ctas).i("c.clusters", ctas / cs).f("c.sm_clock_mhz", median(mhz)).t("verified", ok).i("n_reps", 5).f("cv", cv(bw))
          .s("note", "every CTA of a cluster accesses the smem of rank (rank+1) mod cluster_size with 16B vector ops, 1024 threads, 8 independent ops per iteration (lane xor per iteration so addresses vary); "
                     "local mode reads its own smem with plain ld.shared as the reference. Per-SM rate counts bytes moved by that CTA. verified = checksum of loaded words vs host replay "
                     "(store mode: read-back of the last-written values in the neighbour). multicast not measured");
        j.print();
      }
    }
  }
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  g_nsm = num_sms();
  CK(cudaMalloc(&g_buf, BUF_BYTES)); CK(cudaMalloc(&g_flush, FLUSH_BYTES)); CK(cudaMemset(g_flush, 1, FLUSH_BYTES));
  CK(cudaMalloc(&g_out, 8 * 4096)); CK(cudaMalloc(&g_st, sizeof(St3) * 4096));
  k_fill<<<g_nsm * 4, 1024>>>(g_buf, BUF_BYTES / 4); CK(cudaDeviceSynchronize());
  if (!only || !strcmp(only, "e1")) run_e1();
  if (!only || !strcmp(only, "e2")) run_e2();
  if (!only || !strcmp(only, "e3")) run_e3();
  return 0;
}
