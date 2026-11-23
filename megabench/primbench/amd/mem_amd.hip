// Streaming read bandwidth on CDNA (weight streaming in a decode megakernel).
// Varies the load width per lane, independent loads in flight per lane, nontemporal hints,
// the working-set size (HBM vs Infinity Cache vs L2) and the number of active CUs.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx942 -o mem_amd mem_amd.hip
#include "pb_hip.h"

template <int W> struct Vec;
template <> struct Vec<4> { typedef unsigned T; };
typedef unsigned u2v __attribute__((ext_vector_type(2)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
template <> struct Vec<8> { typedef u2v T; };
template <> struct Vec<16> { typedef u4v T; };

__device__ __forceinline__ unsigned fold(unsigned v) { return v; }
__device__ __forceinline__ unsigned fold(u2v v) { return v.x ^ v.y; }
__device__ __forceinline__ unsigned fold(u4v v) { return v.x ^ v.y ^ v.z ^ v.w; }

// Each block owns a contiguous slice of `n` vectors (the static split a megakernel GEMV uses),
// reads it U vectors per lane per step, and repeats over `passes`.
template <int W, int U, bool NT>
__global__ void k_read(const typename Vec<W>::T* __restrict__ p, size_t n_per_block, int passes,
                       unsigned* sink) {
  typedef typename Vec<W>::T T;
  const T* base = p + (size_t)blockIdx.x * n_per_block;
  unsigned acc = 0;
  for (int pass = 0; pass < passes; ++pass) {
    for (size_t i = threadIdx.x; i < n_per_block; i += (size_t)blockDim.x * U) {
      T v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        size_t k = i + (size_t)u * blockDim.x;
        if (k < n_per_block) v[u] = NT ? __builtin_nontemporal_load(base + k) : base[k];
        else v[u] = T{};
      }
#pragma unroll
      for (int u = 0; u < U; ++u) acc ^= fold(v[u]);
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

template <int W, int U, bool NT>
static void point(const void* buf, size_t bytes, int blocks, int threads, int passes, unsigned* sink,
                  const char* level) {
  typedef typename Vec<W>::T T;
  const size_t per_block = bytes / W / blocks;
  auto ms = time_ms([&] {
    hipLaunchKernelGGL((k_read<W, U, NT>), dim3(blocks), dim3(threads), 0, 0, (const T*)buf,
                       per_block, passes, sink);
  }, 7, 2);
  const double gbs = (double)per_block * W * blocks * passes / (median(ms) * 1e-3) / 1e9;
  Json j; j.s("family", "mem").s("primitive", NT ? "mem.stream_read_nontemporal" : "mem.stream_read").dev("mem_amd.hip");
  j.i("p.width_bytes", W).i("p.loads_in_flight", U).i("p.blocks", blocks).i("p.threads", threads)
   .i("p.bytes", (long long)(per_block * W * blocks)).i("p.passes", passes).s("c.level", level)
   .f("m.gbps", gbs).null("verified").i("n_reps", (long long)ms.size()).f("cv", cv(ms))
   .s("s.timer", "hipEvent").s("note", "each block streams its own contiguous slice; GB/s over all passes "
      "(level hbm: buffer larger than the Infinity Cache, one pass; mall/l2: warm repeated passes)").print();
}

int main() {
  const int ncu = num_cus();
  const size_t big = (size_t)2 << 30;   // 2 GiB: beyond the 256 MB Infinity Cache
  void* buf; CK(hipMalloc(&buf, big)); CK(hipMemset(buf, 1, big));
  unsigned* sink; CK(hipMalloc(&sink, 64));
  // HBM: width x loads in flight at one full-device geometry
  point<4, 1, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<4, 4, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<8, 1, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<8, 4, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<16, 1, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<16, 2, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<16, 4, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<16, 8, false>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  point<16, 4, true>(buf, big, ncu * 4, 256, 1, sink, "hbm");
  // HBM: geometry (one persistent block per CU is the usual megakernel launch)
  for (int t : {256, 512, 1024}) {
    point<16, 1, false>(buf, big, ncu, t, 1, sink, "hbm");
    point<16, 4, false>(buf, big, ncu, t, 1, sink, "hbm");
  }
  point<16, 4, false>(buf, big, ncu * 2, 512, 1, sink, "hbm");
  // HBM: fraction of CUs active (phases where only some CUs have work)
  for (int frac : {8, 4, 2}) {
    point<16, 4, false>(buf, big / 4, ncu / frac, 256, 1, sink, "hbm");
    point<16, 4, false>(buf, big / 4, ncu / frac, 1024, 1, sink, "hbm");
  }
  // Warm working sets: 128 MiB (fits the Infinity Cache on MI300X/MI350X), 16 MiB (fits L2 summed over XCDs)
  for (size_t ws : {(size_t)128 << 20, (size_t)16 << 20}) {
    const char* lvl = ws > ((size_t)64 << 20) ? "mall" : "l2";
    point<16, 4, false>(buf, ws, ncu, 256, 20, sink, lvl);
    point<16, 4, false>(buf, ws, ncu, 1024, 20, sink, lvl);
    point<16, 4, true>(buf, ws, ncu, 1024, 20, sink, lvl);
  }
  CK(hipFree(buf)); CK(hipFree(sink));
  return 0;
}
