// Inter-workgroup synchronization and coherence on CDNA (gfx90a / gfx942 / gfx950).
//   map    : which XCD each workgroup of a launch lands on (s_getreg HW_REG_XCC_ID)
//   gbar   : grid-barrier protocols, cost per barrier and a cross-XCD visibility check
//   flag   : one-way flag hand-off latency between two workgroups, same XCD and other XCD
//   fence  : cost of one agent-scope release / acquire fence against dirty bytes in L2
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx942 -o sync_amd sync_amd.hip
// Run:   ./sync_amd [map|gbar|flag|fence]...   (no argument runs all)
#include "pb_hip.h"
#include <hip/hip_cooperative_groups.h>
#include <cstring>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------- map
__global__ void k_map(unsigned* xcc, unsigned* hw) {
  if (threadIdx.x == 0) { xcc[blockIdx.x] = xcc_id(); hw[blockIdx.x] = hw_id(); }
}

static void run_map() {
  const int ncu = num_cus();
  for (int per_cu : {1, 2}) {
    const int nb = ncu * per_cu;
    unsigned *dx, *dh; CK(hipMalloc(&dx, nb * 4)); CK(hipMalloc(&dh, nb * 4));
    hipLaunchKernelGGL(k_map, dim3(nb), dim3(256), 0, 0, dx, dh);
    CK(hipDeviceSynchronize());
    std::vector<unsigned> x(nb); CK(hipMemcpy(x.data(), dx, nb * 4, hipMemcpyDeviceToHost));
    unsigned nx = 0; for (unsigned v : x) nx = std::max(nx, v + 1);
    int rr = 0; for (int b = 0; b < nb; ++b) rr += (x[b] == (unsigned)(b % nx));
    std::vector<int> per(nx, 0); for (unsigned v : x) per[v]++;
    Json j; j.s("family", "sync").s("primitive", "xcd.block_map").dev("sync_amd.hip");
    j.i("p.blocks", nb).i("p.threads", 256).i("m.xcds", nx)
     .f("m.frac_round_robin", (double)rr / nb)
     .i("m.min_blocks_per_xcd", *std::min_element(per.begin(), per.end()))
     .i("m.max_blocks_per_xcd", *std::max_element(per.begin(), per.end()))
     .t("verified", true).i("n_reps", 1).f("cv", 0)
     .s("note", "fraction of workgroups with xcc_id == blockIdx.x % xcds (round-robin dispatch)").print();
    CK(hipFree(dx)); CK(hipFree(dh));
  }
}

// ---------------------------------------------------------------- gbar
enum Proto { P_RELAXED, P_THREADFENCE, P_ACQ_POLL, P_REL_FENCE, P_REL_FENCE_SLEEP, P_ALL_FENCE, P_CG,
             P_COHERENT, P_XCD2, NPROTO };
static const char* proto_name[] = {"relaxed_no_fence", "threadfence", "release_add_acquire_poll",
                                   "release_add_relaxed_poll_acquire_fence",
                                   "release_add_relaxed_poll_sleep_acquire_fence",
                                   "every_thread_threadfence", "cooperative_groups_grid_sync",
                                   "coherent_data_no_fence", "per_xcd_two_level_release_acquire"};
static const char* proto_note[] = {
  "thread 0: relaxed agent-scope atomic add, relaxed poll, no fences (cost floor; not a valid barrier for data)",
  "thread 0: __threadfence(); relaxed add; relaxed poll; __threadfence() (common CUDA port)",
  "thread 0: add with release (agent); poll with acquire loads (agent): one buffer_inv per poll",
  "thread 0: add with release (agent); poll with relaxed loads; one acquire fence (agent) after the poll",
  "as release_add_relaxed_poll_acquire_fence with s_sleep 1 in the poll loop",
  "every thread: __threadfence() before and after; thread 0 relaxed add and poll",
  "cooperative_groups::this_grid().sync() (cooperative launch)",
  "no fences: cross-block data written and read with relaxed agent-scope atomics (sc1, bypass the non-coherent L2); "
  "barrier = s_waitcnt + __syncthreads + relaxed counter",
  "thread 0: release add on one counter per XCD (blockIdx % 8, own 128 B line); the last arrival of an XCD "
  "adds to a top counter; relaxed poll on the top counter; one acquire fence"};

template <int P>
__device__ __forceinline__ void gbar(unsigned* ctr, unsigned target) {
  if (P == P_CG) { cg::this_grid().sync(); return; }
  if (P == P_XCD2) {
    // ctr[0] = top counter, ctr[32 * (1 + x)] = XCD x counter (128 B apart); target counts blocks
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned nb = gridDim.x, x = blockIdx.x % 8;
      const unsigned in_x = nb / 8 + (x < nb % 8 ? 1u : 0u);
      const unsigned round = target / nb;
      unsigned old = __hip_atomic_fetch_add(ctr + 32 * (1 + x), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      if (old + 1 == round * in_x) __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < round * 8) { asm volatile("" ::: "memory"); }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    return;
  }
  if (P == P_ALL_FENCE) __threadfence();
  if (P == P_COHERENT) __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (P == P_THREADFENCE) __threadfence();
    if (P == P_REL_FENCE || P == P_REL_FENCE_SLEEP || P == P_ACQ_POLL)
      __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    else
      __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (P == P_ACQ_POLL) {
      while (__hip_atomic_load(ctr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) { asm volatile("" ::: "memory"); }
    } else {
      while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
        if (P == P_REL_FENCE_SLEEP) __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");
      }
    }
    if (P == P_THREADFENCE) __threadfence();
    if (P == P_REL_FENCE || P == P_REL_FENCE_SLEEP) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  if (P == P_ALL_FENCE) __threadfence();
}

// Each iteration: every block writes `dirty` words of its own scratch (plain stores) and a tag
// word, crosses the barrier, then reads the tag of block (b + 1) % nb (another XCD under
// round-robin dispatch). A tag older than this iteration is a stale read.
template <int P>
__global__ void k_gbar(unsigned* ctr, unsigned* tag, unsigned* scratch, unsigned* errs,
                       int iters, int dirty_words) {
  const unsigned nb = gridDim.x, b = blockIdx.x;
  unsigned target = 0, err = 0;
  unsigned* mine = scratch + (size_t)b * dirty_words;
  for (int it = 1; it <= iters; ++it) {
    for (int j = threadIdx.x; j < dirty_words; j += blockDim.x) mine[j] = it + j;
    if (threadIdx.x == 0) {
      if (P == P_COHERENT) __hip_atomic_store(tag + b * 16, (unsigned)it, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      else tag[b * 16] = it;
    }
    target += nb;
    gbar<P>(ctr, target);
    if (threadIdx.x == 0) {
      unsigned* src = tag + ((b + 1) % nb) * 16;
      unsigned v = P == P_COHERENT ? __hip_atomic_load(src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : *src;
      err += (v < (unsigned)it);
    }
  }
  if (threadIdx.x == 0) errs[b] = err;
}

template <int P>
static void launch_gbar(int nb, int threads, unsigned* ctr, unsigned* tag, unsigned* scr,
                        unsigned* errs, int iters, int dirty) {
  CK(hipMemsetAsync(ctr, 0, 2048));
  void* args[] = {&ctr, &tag, &scr, &errs, &iters, &dirty};
  CK(hipLaunchCooperativeKernel((const void*)k_gbar<P>, dim3(nb), dim3(threads), args, 0, 0));
}

typedef void (*GbarFn)(int, int, unsigned*, unsigned*, unsigned*, unsigned*, int, int);
static GbarFn gbar_fns[] = {launch_gbar<0>, launch_gbar<1>, launch_gbar<2>, launch_gbar<3>,
                            launch_gbar<4>, launch_gbar<5>, launch_gbar<6>, launch_gbar<7>,
                            launch_gbar<8>};
static const void* gbar_kernels[] = {(const void*)k_gbar<0>, (const void*)k_gbar<1>,
                                     (const void*)k_gbar<2>, (const void*)k_gbar<3>,
                                     (const void*)k_gbar<4>, (const void*)k_gbar<5>,
                                     (const void*)k_gbar<6>, (const void*)k_gbar<7>,
                                     (const void*)k_gbar<8>};

static void run_gbar() {
  const int ncu = num_cus();
  struct G { int per_cu_num, per_cu_den, threads; };
  const G geoms[] = {{1, 1, 256}, {1, 1, 512}, {1, 1, 1024}, {2, 1, 256}, {1, 4, 256}};
  const int dirties[] = {0, 256, 4096};  // words (x4 bytes) per block per iteration
  const int maxnb = 2 * ncu;
  unsigned *ctr, *tag, *scr, *errs;
  CK(hipMalloc(&ctr, 2048)); CK(hipMalloc(&tag, maxnb * 64)); CK(hipMemset(tag, 0, maxnb * 64));
  CK(hipMalloc(&scr, (size_t)maxnb * 4096 * 4)); CK(hipMalloc(&errs, maxnb * 4));
  for (const G& g : geoms) {
    const int nb = ncu * g.per_cu_num / g.per_cu_den;
    for (int dirty : dirties) {
      if (g.threads != 256 && dirty == 4096) continue;
      for (int p = 0; p < NPROTO; ++p) {
        int occ = 0;
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gbar_kernels[p], g.threads, 0));
        if (occ * ncu < nb) continue;
        const int I1 = 100, I2 = 600;
        std::vector<double> per;
        long long stale = 0;
        for (int rep = 0; rep < 7; ++rep) {
          CK(hipMemset(tag, 0, maxnb * 64));
          auto t1 = time_ms([&] { gbar_fns[p](nb, g.threads, ctr, tag, scr, errs, I1, dirty); }, 1, rep == 0 ? 1 : 0);
          CK(hipMemset(tag, 0, maxnb * 64));
          auto t2 = time_ms([&] { gbar_fns[p](nb, g.threads, ctr, tag, scr, errs, I2, dirty); }, 1, 0);
          std::vector<unsigned> e(nb); CK(hipMemcpy(e.data(), errs, nb * 4, hipMemcpyDeviceToHost));
          for (unsigned v : e) stale += v;
          per.push_back((t2[0] - t1[0]) * 1e3 / (I2 - I1));
        }
        const double us = median(per);
        Json j; j.s("family", "sync").s("primitive", std::string("grid_barrier.") + proto_name[p]).dev("sync_amd.hip");
        j.i("p.blocks", nb).i("p.threads", g.threads).i("p.dirty_bytes_per_block", dirty * 4)
         .f("m.us", us).i("m.stale_reads", stale).i("m.checked_reads", 7LL * (I2 + I1) * nb)
         .t("verified", stale == 0).i("n_reps", (long long)per.size()).f("cv", cv(per))
         .s("s.timer", "hipEvent slope").s("note", std::string(proto_note[p]) +
            "; us per iteration of (write dirty bytes + tag, barrier, read neighbour tag); "
            "stale_reads counts neighbour tags older than the iteration (cross-XCD visibility)").print();
      }
    }
  }
  CK(hipFree(ctr)); CK(hipFree(tag)); CK(hipFree(scr)); CK(hipFree(errs));
}

// ---------------------------------------------------------------- flag
// Blocks `a` and `b` ping-pong through two flags; out = wall ticks for `iters` round trips.
template <bool ACQREL>
__global__ void k_flag(unsigned* f, int a, int b, int iters, unsigned long long* out, unsigned* xcc) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  if (me != a && me != b) return;
  xcc[me == a ? 0 : 1] = xcc_id();
  unsigned* ping = f; unsigned* pong = f + 64;
  const int ord_st = ACQREL ? __ATOMIC_RELEASE : __ATOMIC_RELAXED;
  const int ord_ld = ACQREL ? __ATOMIC_ACQUIRE : __ATOMIC_RELAXED;
  if (me == a) {
    unsigned long long t0 = wall_clock64();
    for (int i = 1; i <= iters; ++i) {
      __hip_atomic_store(ping, (unsigned)i, ord_st, __HIP_MEMORY_SCOPE_AGENT);
      while (__hip_atomic_load(pong, ord_ld, __HIP_MEMORY_SCOPE_AGENT) != (unsigned)i) {}
    }
    out[0] = wall_clock64() - t0;
  } else {
    for (int i = 1; i <= iters; ++i) {
      while (__hip_atomic_load(ping, ord_ld, __HIP_MEMORY_SCOPE_AGENT) != (unsigned)i) {}
      __hip_atomic_store(pong, (unsigned)i, ord_st, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

static void run_flag() {
  const int nb = 16, iters = 4000;
  unsigned *f, *xcc; unsigned long long* out;
  CK(hipMalloc(&f, 1024)); CK(hipMalloc(&xcc, 8)); CK(hipMalloc(&out, 8));
  const double mhz = wall_mhz();
  for (int acq = 0; acq < 2; ++acq)
    for (int peer : {1, 8}) {
      std::vector<double> ns; unsigned hx[2] = {0, 0};
      for (int rep = 0; rep < 9; ++rep) {
        CK(hipMemset(f, 0, 1024));
        void* args[] = {&f, (void*)nullptr, (void*)nullptr, (void*)&iters, &out, &xcc};
        int a = 0, b = peer; args[1] = &a; args[2] = &b;
        CK(hipLaunchCooperativeKernel(acq ? (const void*)k_flag<true> : (const void*)k_flag<false>,
                                      dim3(nb), dim3(64), args, 0, 0));
        CK(hipDeviceSynchronize());
        unsigned long long t; CK(hipMemcpy(&t, out, 8, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hx, xcc, 8, hipMemcpyDeviceToHost));
        if (rep) ns.push_back(t / mhz * 1e3 / iters / 2);
      }
      Json j; j.s("family", "sync").s("primitive", acq ? "flag_handoff.release_acquire" : "flag_handoff.relaxed")
        .dev("sync_amd.hip");
      j.i("p.peer_block", peer).t("p.same_xcd", hx[0] == hx[1]).i("c.xcc_a", hx[0]).i("c.xcc_b", hx[1])
       .f("m.ns", median(ns)).null("verified").i("n_reps", (long long)ns.size()).f("cv", cv(ns))
       .s("s.timer", "wall_clock64").s("note", "one-way hand-off = round trip / 2 between block 0 and peer block; "
          "agent-scope atomic store and polling load by one thread each").print();
    }
  CK(hipFree(f)); CK(hipFree(xcc)); CK(hipFree(out));
}

// ---------------------------------------------------------------- fence
// Every block dirties `dirty` words, waits for its stores, then thread 0 times one fence.
template <int KIND>
__global__ void k_fence(unsigned* scratch, int dirty, int iters, unsigned long long* out) {
  unsigned* mine = scratch + (size_t)blockIdx.x * dirty;
  unsigned long long acc = 0;
  for (int it = 0; it < iters; ++it) {
    for (int j = threadIdx.x; j < dirty; j += blockDim.x) mine[j] = it + j;
    __syncthreads();
    if (threadIdx.x == 0) {
      __builtin_amdgcn_s_waitcnt(0);
      unsigned long long t0 = wall_clock64();
      if (KIND == 0) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      if (KIND == 1) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      if (KIND == 2) __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
      if (KIND == 3) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      __builtin_amdgcn_s_waitcnt(0);
      acc += wall_clock64() - t0;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

static void run_fence() {
  const int ncu = num_cus(), nb = ncu, iters = 200;
  const char* kinds[] = {"release.agent", "acquire.agent", "seq_cst.agent", "release.workgroup"};
  void (*ks[])(unsigned*, int, int, unsigned long long*) = {k_fence<0>, k_fence<1>, k_fence<2>, k_fence<3>};
  unsigned* scr; unsigned long long* out;
  CK(hipMalloc(&scr, (size_t)nb * 16384 * 4)); CK(hipMalloc(&out, nb * 8));
  const double mhz = wall_mhz();
  for (int k = 0; k < 4; ++k)
    for (int dirty : {0, 64, 1024, 16384}) {
      std::vector<double> med;
      for (int rep = 0; rep < 5; ++rep) {
        hipLaunchKernelGGL(ks[k], dim3(nb), dim3(256), 0, 0, scr, dirty, iters, out);
        CK(hipDeviceSynchronize());
        std::vector<unsigned long long> o(nb); CK(hipMemcpy(o.data(), out, nb * 8, hipMemcpyDeviceToHost));
        std::vector<double> v; for (auto t : o) v.push_back(t / mhz * 1e3 / iters);
        med.push_back(median(v));
      }
      Json j; j.s("family", "sync").s("primitive", std::string("fence.") + kinds[k]).dev("sync_amd.hip");
      j.i("p.blocks", nb).i("p.threads", 256).i("p.dirty_bytes_per_block", dirty * 4)
       .f("m.ns", median(med)).null("verified").i("n_reps", (long long)med.size()).f("cv", cv(med))
       .s("s.timer", "wall_clock64").s("note", "median over blocks of one fence by thread 0 after every block "
          "wrote dirty_bytes_per_block with plain stores; all blocks fence at about the same time").print();
    }
  CK(hipFree(scr)); CK(hipFree(out));
}

int main(int argc, char** argv) {
  bool all = argc < 2;
  auto want = [&](const char* m) { if (all) return true; for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], m)) return true; return false; };
  if (want("map")) run_map();
  if (want("flag")) run_flag();
  if (want("fence")) run_fence();
  if (want("gbar")) run_gbar();
  return 0;
}
