// Thin PTX wrappers used by the loadstream primitives (sm_90+; tuned for sm_100a).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>

namespace ls {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(smem_u32(bar)), "r"(parity)
        : "memory");
    return ok != 0;
}

// ------------------------------------------------------- bulk async copies
// Policy for streamed-once data (weights): evict_first keeps L2 for
// activations and data you will read again.
__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t policy_evict_normal() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// Global -> this CTA's shared memory, completing `bytes` on `bar`.
// dst, src 16-byte aligned; bytes a multiple of 16.
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar,
                                         uint64_t policy) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1], %2, [%3], %4;" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
        : "memory");
}
// Fire-and-forget L2 prefetch (no smem, no completion). bytes multiple of 16.
__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// ------------------------------------------------------- gpu-scope memory
__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void red_release_add(uint32_t* p, uint32_t v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint32_t smid() {
    uint32_t r;
    asm volatile("mov.u32 %0, %smid;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void nanosleep(uint32_t ns) { asm volatile("nanosleep.u32 %0;" ::"r"(ns)); }

// Named barrier over a subset of warps (id 1..15; id 0 is __syncthreads).
__device__ __forceinline__ void named_sync(uint32_t id, uint32_t threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// Loads for data written earlier in the same kernel by other SMs. They
// bypass L1 (which is not coherent across SMs). Never use __ldg or
// `const __restrict__` pointers for such data.
__device__ __forceinline__ float4 ld_cg(const float4* p) { return __ldcg(p); }
__device__ __forceinline__ float ld_cg(const float* p) { return __ldcg(p); }
__device__ __forceinline__ uint4 ld_cg(const uint4* p) { return __ldcg(p); }

// Weight loads that bypass L1 allocation (streamed once).
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

}  // namespace ls
