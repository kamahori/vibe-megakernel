// Primitives for the Waypoint megakernel (sm_100a): mbarrier, TMA, tcgen05, grid barrier.
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W_%=;\n}\n" ::"r"(a), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map, uint32_t dst, uint32_t mbar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
      "l"(map), "r"(mbar), "r"(x), "r"(y)
      : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

// ---- tcgen05
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr, uint32_t sbo_bytes = 1024, uint32_t lbo_bytes = 16) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)(lbo_bytes >> 4) << 16;
  d |= (uint64_t)(sbo_bytes >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;  // SWIZZLE_128B
  return d;
}
// bf16 x bf16 -> f32; a_mn/b_mn: operand is MN-major (transposed) when 1.
__host__ __device__ __forceinline__ uint32_t make_idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void mma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void tc_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// 32 lanes x 32 columns of fp32
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* r) {
  uint32_t v[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
        "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]),
        "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
        "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
      : "r"(taddr));
  tmem_wait_ld();
#pragma unroll
  for (int i = 0; i < 32; ++i) r[i] = __uint_as_float(v[i]);
}

// ---- grid barrier (monotonic counter; counter is always a multiple of nblocks between launches)
__device__ __forceinline__ void grid_barrier(unsigned long long* counter) {
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long n = gridDim.x, old;
    asm volatile("fence.proxy.async.global;" ::: "memory");
    asm volatile("atom.add.release.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(counter) : "memory");
    unsigned long long target = (old / n + 1) * n, cur;
    do {
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(counter) : "memory");
    } while (cur < target);
  }
  __syncthreads();
  fence_proxy_async_global();
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }
