// WeightRing: a shared-memory ring of byte slots filled by one producer warp
// with cp.async.bulk and drained, in the same order, by every consumer warp.
//
// The producer never waits for activations, only for free slots, so weights
// for later ops stream in while consumers wait on events. Producer and
// consumers must walk the *same* sequence of tiles; the usual way is one
// schedule function templated on the role (see examples/mlp_chain).
#pragma once
#include "ptx.cuh"
#include "event.cuh"

namespace ls {

// Shared-memory control block; the slot bytes live elsewhere (dynamic smem).
template <int SLOTS>
struct RingBarriers {
    uint64_t full[SLOTS];
    uint64_t empty[SLOTS];
};

template <int SLOTS, int SLOT_BYTES>
struct WeightRing {
    static_assert(SLOT_BYTES % 128 == 0, "slots must be 128-byte multiples");
    static constexpr int kSlots = SLOTS;
    static constexpr int kSlotBytes = SLOT_BYTES;
    uint8_t* data;                 // SLOTS * SLOT_BYTES, 128-byte aligned
    RingBarriers<SLOTS>* bars;
    int slot = 0;
    uint32_t phase = 0;

    __device__ WeightRing(uint8_t* data_, RingBarriers<SLOTS>* bars_) : data(data_), bars(bars_) {}

    // Call from every thread of the CTA, before any role starts, then
    // __syncthreads(). `consumer_warps` = warps that will release each slot.
    __device__ static void init(RingBarriers<SLOTS>* bars, uint32_t consumer_warps) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < SLOTS; ++i) {
                mbar_init(&bars->full[i], 1);
                mbar_init(&bars->empty[i], consumer_warps);
            }
            mbar_fence_init();
        }
    }

    __device__ uint8_t* slot_ptr() const { return data + slot * SLOT_BYTES; }
    __device__ void advance() {
        if (++slot == SLOTS) {
            slot = 0;
            phase ^= 1;
        }
    }
};

// ------------------------------------------------------------- producer
// Used by the whole producer warp (32 lanes). Copies are spread across lanes.
template <int SLOTS, int SLOT_BYTES>
struct Producer {
    WeightRing<SLOTS, SLOT_BYTES> ring;
    uint64_t policy;
    uint32_t lane;
    uint64_t bytes_issued = 0;

    __device__ Producer(uint8_t* data, RingBarriers<SLOTS>* bars, uint64_t policy_)
        : ring(data, bars), policy(policy_), lane(threadIdx.x & 31) {}

    // Wait for a free slot and announce `total` bytes on its full barrier.
    __device__ uint8_t* begin(uint32_t total) {
#ifdef LS_WATCHDOG
        if (total == 0 || total > SLOT_BYTES || (total & 15)) {
            if (lane == 0) printf("[loadstream] block %d: bad tile size %u (slot %d bytes)\n", blockIdx.x, total, SLOT_BYTES);
            __trap();
        }
#endif
        // A fresh barrier completes "phase -1" (parity 1), so the first lap
        // passes immediately.
        if (lane == 0) {
            spin_mbar(&ring.bars->empty[ring.slot], ring.phase ^ 1, "ring.empty");
            mbar_arrive_expect_tx(&ring.bars->full[ring.slot], total);
        }
        __syncwarp();
        bytes_issued += total;
        return ring.slot_ptr();
    }
    // One contiguous copy, issued by lane 0.
    __device__ void copy(uint32_t dst_offset, const void* src, uint32_t bytes) {
        if (lane == 0) bulk_g2s(ring.slot_ptr() + dst_offset, src, bytes, &ring.bars->full[ring.slot], policy);
    }
    // `rows` copies of `row_bytes` each: src rows `src_pitch` bytes apart,
    // landing `dst_pitch` bytes apart (pad dst_pitch by 16 to avoid
    // ldmatrix bank conflicts). Lanes issue in parallel.
    __device__ void copy_rows(uint32_t dst_offset, const void* src, uint32_t rows, uint32_t row_bytes,
                              uint64_t src_pitch, uint32_t dst_pitch) {
        const uint8_t* s = static_cast<const uint8_t*>(src);
#ifdef LS_WATCHDOG
        if (rows && dst_offset + (rows - 1) * dst_pitch + row_bytes > SLOT_BYTES) {
            if (lane == 0) printf("[loadstream] block %d: copy_rows overflows the slot\n", blockIdx.x);
            __trap();
        }
#endif
        for (uint32_t r = lane; r < rows; r += 32)
            bulk_g2s(ring.slot_ptr() + dst_offset + r * dst_pitch, s + r * src_pitch, row_bytes,
                     &ring.bars->full[ring.slot], policy);
    }
    __device__ void commit() {
        __syncwarp();
        ring.advance();
    }
    // Convenience: one tile of `rows` full rows that are contiguous in global memory.
    __device__ void tile_contiguous(const void* src, uint32_t bytes) {
        begin(bytes);
        copy(0, src, bytes);
        commit();
    }
};

// ------------------------------------------------------------- consumer
// One per consumer warp. Every consumer warp must wait() and release()
// every slot, in order, even if it does no work on that slot.
template <int SLOTS, int SLOT_BYTES>
struct Consumer {
    WeightRing<SLOTS, SLOT_BYTES> ring;
    __device__ Consumer(uint8_t* data, RingBarriers<SLOTS>* bars) : ring(data, bars) {}

    __device__ const uint8_t* wait() {
        spin_mbar(&ring.bars->full[ring.slot], ring.phase, "ring.full");
        return ring.slot_ptr();
    }
    // Call after this warp's last read of the slot.
    __device__ void release() {
        __syncwarp();
        if ((threadIdx.x & 31) == 0) mbar_arrive(&ring.bars->empty[ring.slot]);
        ring.advance();
    }
};

}  // namespace ls
