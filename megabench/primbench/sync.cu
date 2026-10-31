// Synchronization primitive costs on sm_100a: mbarrier (arrive/try_wait, warp ping-pong, expect_tx + bulk copy
// completion, cluster-remote arrive), named barriers / __syncthreads vs warp count, fence.proxy.async.
// Every point = slope of clock64 between two loop lengths (removes loop/launch overhead), median of 9 launches.
// Usage: ./sync        all points as JSON lines (no args)
#include "pb_common.cuh"

enum Mode { M_ARRIVE_WAIT, M_PINGPONG, M_TX_BULK, M_TX_SMEM, M_REMOTE, M_BAR, M_SYNCTHREADS, M_FENCE, M_ARRIVE_ONLY };

__device__ __forceinline__ void mb_init(uint32_t a, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mb_expect(uint32_t a, uint32_t b) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(b) : "memory"); }
__device__ __forceinline__ void mb_complete(uint32_t a, uint32_t b) { asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(b) : "memory"); }
__device__ __forceinline__ void mb_wait(uint32_t a, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" :: "r"(a), "r"(ph) : "memory");
}
__device__ __forceinline__ void mb_wait_cl(uint32_t a, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" :: "r"(a), "r"(ph) : "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ void mb_arrive_remote(uint32_t a) { asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void cl_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

__global__ void __launch_bounds__(1024, 1) k_sync(int mode, int iters, int arg, Stamp* out, const uint8_t* gsrc, unsigned* sink) {
  __shared__ __align__(1024) uint8_t buf[8192];
  __shared__ __align__(8) uint64_t bars[4];
  const uint32_t b0 = smem_u32(&bars[0]), b1 = smem_u32(&bars[1]);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) { mb_init(b0, 1); mb_init(b1, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (mode == M_REMOTE) cl_sync(); else __syncthreads();
  long long c0 = 0, c1 = 0; unsigned long long g0 = 0, g1 = 0; uint32_t acc = 0;
  bool timer = tid == 0;
  switch (mode) {
    case M_ARRIVE_WAIT:   // arrive (count 1) then try_wait on the completed phase, one thread
      if (tid == 0) { c0 = clock64(); g0 = globaltimer();
        for (int i = 0; i < iters; ++i) { mb_arrive(b0); mb_wait(b0, i & 1); }
        c1 = clock64(); g1 = globaltimer(); }
      break;
    case M_ARRIVE_ONLY:   // arrive on a barrier that needs 2^20-1 arrivals: pure arrive issue cost, never completes (count 2^20-1)
      if (tid == 0) { mb_init(b1, (1u << 20) - 1); c0 = clock64(); g0 = globaltimer();
        for (int i = 0; i < iters; ++i) mb_arrive(b1);
        c1 = clock64(); g1 = globaltimer(); }
      break;
    case M_PINGPONG:      // warp 0 lane 0 (P) <-> warp 1 lane 0 (C) through two barriers; one iteration = 2 hand-offs
      if (warp == 0 && lane == 0) { c0 = clock64(); g0 = globaltimer();
        for (int i = 0; i < iters; ++i) { mb_arrive(b1); mb_wait(b0, i & 1); }
        c1 = clock64(); g1 = globaltimer(); }
      else if (warp == 1 && lane == 0) {
        for (int i = 0; i < iters; ++i) { mb_wait(b1, i & 1); mb_arrive(b0); } }
      break;
    case M_TX_BULK:       // arrive.expect_tx(S) + cp.async.bulk global->shared (L2-warm source) + wait: serial latency
      if (tid == 0) {
        c0 = clock64(); g0 = globaltimer();
        for (int i = 0; i < iters; ++i) {
          mb_expect(b0, arg);
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(smem_u32(buf)), "l"(gsrc), "r"(arg), "r"(b0) : "memory");
          mb_wait(b0, i & 1);
        }
        c1 = clock64(); g1 = globaltimer(); acc = buf[0];
      }
      break;
    case M_TX_SMEM:       // arrive.expect_tx(S) then complete_tx(S) by the same thread, no copy: barrier tx bookkeeping only
      if (tid == 0) { c0 = clock64(); g0 = globaltimer();
        for (int i = 0; i < iters; ++i) { mb_expect(b0, arg); mb_complete(b0, arg); mb_wait(b0, i & 1); }
        c1 = clock64(); g1 = globaltimer(); }
      break;
    case M_REMOTE: {      // cluster of 2: CTA0 arrives on CTA1's barrier, CTA1 on CTA0's; 2 hand-offs per iteration
      uint32_t rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
      uint32_t peer = mapa(b0, rank ^ 1);
      if (tid == 0 && rank == 0) { c0 = clock64(); g0 = globaltimer();
        for (int i = 0; i < iters; ++i) { mb_arrive_remote(peer); mb_wait_cl(b0, i & 1); }
        c1 = clock64(); g1 = globaltimer(); }
      else if (tid == 0) {
        for (int i = 0; i < iters; ++i) { mb_wait_cl(b0, i & 1); mb_arrive_remote(peer); } }
      cl_sync();
      break; }
    case M_BAR: case M_SYNCTHREADS: {  // blockDim = 32 * warps, all threads participate
      int n = blockDim.x;
      if (tid == 0) { c0 = clock64(); g0 = globaltimer(); }
      if (mode == M_BAR) for (int i = 0; i < iters; ++i) bar_named(1, n);
      else for (int i = 0; i < iters; ++i) __syncthreads();
      if (tid == 0) { c1 = clock64(); g1 = globaltimer(); }
      break; }
    case M_FENCE:         // arg 0: fence.proxy.async by thread 0; 1: st.shared + fence by thread 0; 2: every thread of the block fences
      if (arg == 2 || tid == 0) {
        if (tid == 0) { c0 = clock64(); g0 = globaltimer(); }
        for (int i = 0; i < iters; ++i) {
          if (arg == 1) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_u32(buf)), "r"(i) : "memory"); }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (tid == 0) { c1 = clock64(); g1 = globaltimer(); }
      }
      break;
  }
  if (timer && c1) { out[blockIdx.x] = Stamp{c1 - c0, g1 - g0}; if (acc == 0x7fffffff) sink[0] = acc; }
  else if (mode == M_REMOTE && tid == 0 && blockIdx.x == 0 && c1) out[0] = Stamp{c1 - c0, g1 - g0};
}

static void launch(int mode, int iters, int arg, int threads, Stamp* d_out, const uint8_t* src, unsigned* sink) {
  if (mode == M_REMOTE) {
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(2); cfg.blockDim = dim3(threads);
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, k_sync, mode, iters, arg, d_out, src, sink));
  } else k_sync<<<1, threads>>>(mode, iters, arg, d_out, src, sink);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}

struct Pt { const char* prim; int mode; int arg; int threads; int per_iter; const char* extra_key; long long extra_val; const char* note; };

static void measure(const Pt& pt, const uint8_t* src, unsigned* sink, Stamp* d_out) {
  const int I1 = 512, I2 = 4608;
  std::vector<double> slopes, mhz;
  for (int rep = 0; rep < 9; ++rep) {
    Stamp s1, s2;
    launch(pt.mode, I1, pt.arg, pt.threads, d_out, src, sink); CK(cudaMemcpy(&s1, d_out, sizeof s1, cudaMemcpyDeviceToHost));
    launch(pt.mode, I2, pt.arg, pt.threads, d_out, src, sink); CK(cudaMemcpy(&s2, d_out, sizeof s2, cudaMemcpyDeviceToHost));
    if (rep == 0) continue;  // warm-up
    slopes.push_back((double)(s2.cyc - s1.cyc) / (I2 - I1) / pt.per_iter);
    if (s2.ns) mhz.push_back((double)s2.cyc / s2.ns * 1e3);
  }
  double cyc = median(slopes), f = median(mhz);
  Json j; j.s("family", "sync").s("primitive", pt.prim);
  if (pt.extra_key) j.i(pt.extra_key, pt.extra_val);
  j.key("verified"); j.b += "null";  // no data output to check; a wrong protocol would hang
  j.i("p.threads", pt.threads).f("m.cyc", cyc).f("m.ns", f > 0 ? cyc / f * 1e3 : NAN)
   .f("c.sm_clock_mhz", f).i("c.ctas", pt.mode == M_REMOTE ? 2 : 1).i("c.threads", pt.threads)
   .i("n_reps", (long long)slopes.size()).f("cv", cv(slopes)).s("note", pt.note).print();
}

int main() {
  Stamp* d_out; uint8_t* src; unsigned* sink;
  CK(cudaMalloc(&d_out, 4 * sizeof(Stamp))); CK(cudaMalloc(&src, 1 << 16)); CK(cudaMalloc(&sink, 64));
  CK(cudaMemset(src, 1, 1 << 16));
  // warm L2 line set once (bulk-copy source stays L2-resident across iterations)
  std::vector<Pt> pts;
  pts.push_back({"sync.mbarrier_arrive_wait", M_ARRIVE_WAIT, 0, 32, 1, nullptr, 0,
    "one thread: mbarrier.arrive (count 1) then try_wait.parity on the phase it completes; cyc = full round trip, shared::cta"});
  pts.push_back({"sync.mbarrier_arrive_issue", M_ARRIVE_ONLY, 0, 32, 1, nullptr, 0,
    "back-to-back mbarrier.arrive on a barrier that never completes (count 2^20-1): issue cost per arrive from one thread"});
  pts.push_back({"sync.mbarrier_pingpong", M_PINGPONG, 0, 64, 2, nullptr, 0,
    "two warps (lane 0 each) hand off through two mbarriers with phase flip; cyc = cost per one-way hand-off (arrive -> peer's try_wait returns)"});
  for (int S : {16, 64, 256, 1024, 4096, 8192})
    pts.push_back({"sync.mbarrier_expect_tx_bulk", M_TX_BULK, S, 32, 1, "p.bytes", S,
      "arrive.expect_tx(bytes) + cp.async.bulk global->shared (L2-warm source) + try_wait; serial latency per copy"});
  for (int S : {16, 4096})
    pts.push_back({"sync.mbarrier_expect_tx_complete_tx", M_TX_SMEM, S, 32, 1, "p.bytes", S,
      "arrive.expect_tx(bytes) then mbarrier.complete_tx(bytes) by the same thread, then try_wait; no copy"});
  pts.push_back({"sync.mbarrier_remote_arrive", M_REMOTE, 0, 32, 2, "p.cluster", 2,
    "cluster of 2: CTA0 and CTA1 hand off by mapa + mbarrier.arrive.release.cluster.shared::cluster and try_wait.acquire.cluster; cyc = cost per one-way hand-off"});
  for (int W : {2, 3, 4, 6, 8, 12, 16, 24, 32}) {
    pts.push_back({"sync.bar_sync_named", M_BAR, 0, 32 * W, 1, "p.warps", W, "bar.sync 1, 32*warps, all warps of the CTA participate; cyc per barrier"});
    pts.push_back({"sync.syncthreads", M_SYNCTHREADS, 0, 32 * W, 1, "p.warps", W, "__syncthreads() with blockDim = 32*warps; cyc per barrier"});
  }
  pts.push_back({"sync.fence_proxy_async", M_FENCE, 0, 32, 1, "p.variant", 0, "fence.proxy.async.shared::cta back-to-back, one thread; variant 0"});
  pts.push_back({"sync.fence_proxy_async", M_FENCE, 1, 32, 1, "p.variant", 1, "st.shared.u32 + fence.proxy.async.shared::cta, one thread; variant 1"});
  pts.push_back({"sync.fence_proxy_async", M_FENCE, 2, 128, 1, "p.variant", 2, "fence.proxy.async.shared::cta by all 128 threads concurrently, thread 0 timed; variant 2"});
  for (auto& pt : pts) measure(pt, src, sink, d_out);
  return 0;
}
