// Device-side timeline stamps: per-CTA records of (globaltimer, tag, value)
// written to a preallocated buffer, so profiling adds no kernel launches.
//
// Buffer layout: int64 [gridDim.x][capacity][2]; word 0 = globaltimer ns,
// word 1 = (tag << 32) | value. Slot 0 of each CTA row holds the count.
// Analyse with `python -m loadstream.timeline` (see loadstream/timeline.py).
//
// Stamps compile to nothing unless LS_PROFILE is defined, so the graded
// build can keep the calls. Stamp from one thread per role (e.g. producer
// lane 0, consumer warp 0 lane 0) to keep overhead at ~20 ns per stamp.
#pragma once
#include "ptx.cuh"

namespace ls {

struct Stamps {
    long long* buf;  // nullptr disables
    int capacity;    // records per CTA, including the count slot
};

#ifdef LS_PROFILE
__device__ __forceinline__ void stamp(const Stamps& s, uint32_t tag, uint32_t value = 0) {
    if (s.buf == nullptr) return;
    long long* row = s.buf + static_cast<long long>(blockIdx.x) * s.capacity * 2;
    unsigned long long* count = reinterpret_cast<unsigned long long*>(row);
    unsigned long long i = atomicAdd(count, 1ull) + 1;  // CTA-local slot 0 is the counter
    if (i < static_cast<unsigned long long>(s.capacity)) {
        row[2 * i] = static_cast<long long>(globaltimer());
        row[2 * i + 1] = (static_cast<long long>(tag) << 32) | value;
    }
}
#else
__device__ __forceinline__ void stamp(const Stamps&, uint32_t, uint32_t = 0) {}
#endif

}  // namespace ls
