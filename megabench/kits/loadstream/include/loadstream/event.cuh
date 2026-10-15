// Events: monotonically increasing ready counters in global memory that
// replace grid barriers. A producer of an activation slice adds to the
// counter after writing it; a consumer waits until the counter reaches the
// value for this call.
//
// Counters are never reset. Pass a 1-based call index `epoch` as a kernel
// argument (the host increments it each call). If every call adds exactly
// `per_call` to a counter, the target for this call is epoch * per_call.
// Comparisons are wrap-safe, so 32-bit counters never need clearing.
//
// The kernel must be launched cooperatively (all CTAs co-resident), or a
// waiting CTA can starve the CTA it waits for.
//
// Build with -DLS_WATCHDOG to turn hangs into a printed diagnostic and a
// trap (an "unspecified launch failure"), instead of a benchmark timeout.
#pragma once
#include <cstdio>
#include "ptx.cuh"

#ifndef LS_WATCHDOG_NS
#define LS_WATCHDOG_NS 2000000000ull  // 2 s
#endif

namespace ls {

__device__ __forceinline__ bool reached(uint32_t value, uint32_t target) {
    return static_cast<int32_t>(value - target) >= 0;
}

__device__ __forceinline__ void spin_mbar(uint64_t* bar, uint32_t parity, const char* what) {
#ifdef LS_WATCHDOG
    uint64_t start = 0;
    uint32_t polls = 0;
    while (!mbar_try_wait(bar, parity)) {
        if (++polls == 1024) start = globaltimer();
        if (polls > 1024 && (polls & 1023) == 0 && globaltimer() - start > LS_WATCHDOG_NS) {
            printf("[loadstream watchdog] block %d thread %d stuck on %s parity %u\n", blockIdx.x,
                   threadIdx.x, what, parity);
            __trap();
        }
    }
#else
    (void)what;
    while (!mbar_try_wait(bar, parity)) {
    }
#endif
}

// Wait until *ctr reaches target. Call from ONE thread (e.g. lane 0), then
// share the result with __syncwarp() / a named barrier. `id` labels the
// event in watchdog output.
__device__ __forceinline__ void wait_geq(const uint32_t* ctr, uint32_t target, int id = -1) {
#ifdef LS_WATCHDOG
    uint64_t start = globaltimer();
    uint32_t polls = 0;
    uint32_t v;
    while (!reached(v = ld_acquire(ctr), target)) {
        if ((++polls & 255) == 0 && globaltimer() - start > LS_WATCHDOG_NS) {
            printf("[loadstream watchdog] block %d thread %d: event %d at %u, waiting for %u\n", blockIdx.x,
                   threadIdx.x, id, v, target);
            __trap();
        }
    }
#else
    (void)id;
    while (!reached(ld_acquire(ctr), target)) {
    }
#endif
}

// Warp-collective wait: lane 0 polls, the warp proceeds together.
__device__ __forceinline__ void warp_wait_geq(const uint32_t* ctr, uint32_t target, int id = -1) {
    if ((threadIdx.x & 31) == 0) wait_geq(ctr, target, id);
    __syncwarp();
}

// Publish: every lane of the warp has finished its global writes; lane 0
// adds `n` with release semantics. Use after the warp's last store.
// red.release.gpu is cumulative: writes ordered before it by __syncwarp /
// bar.sync (by any thread of the group) become visible to a reader whose
// ld.acquire sees the new count, so no __threadfence is needed (the same
// pattern as CUTLASS's GenericBarrier). A fence here costs ~0.5 us per hop.
__device__ __forceinline__ void warp_signal(uint32_t* ctr, uint32_t n) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) red_release_add(ctr, n);
}

// Same for a group of warps that already synchronised with named_sync():
// call from exactly one thread of the group, after the barrier.
__device__ __forceinline__ void signal(uint32_t* ctr, uint32_t n) {
    red_release_add(ctr, n);
}

// Grid-wide barrier on a dedicated counter (fallback when a dataflow event
// is awkward). Every CTA calls it the same number of times per launch;
// `index` is the 0-based barrier count within this launch and
// `per_call` = barriers per launch. Call from all threads.
__device__ __forceinline__ void grid_barrier(uint32_t* ctr, uint32_t epoch, uint32_t per_call, uint32_t index) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t n = gridDim.x;
        signal(ctr, 1);
        wait_geq(ctr, ((epoch - 1) * per_call + index + 1) * n, -2);
    }
    __syncthreads();
}

}  // namespace ls
