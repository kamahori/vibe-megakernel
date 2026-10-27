// Per-block phase timestamps for MegaBench candidates, in FlashInfer's profiler layout.
//
// Probes compile to nothing unless MEGABENCH_TIMELINE is defined, so the scored build
// carries none. With it defined, each writer (one thread per block and group) records
// begin/end stamps of named phases into a uint64 buffer that
// `megabench.integrations.timeline` decodes. The layout is the one FlashInfer's
// include/flashinfer/profiler.cuh writes, so a candidate may use either.
//
//   word 0           : int32 num_blocks, int32 num_groups (little endian)
//   word 1 + w + k*S : k-th event of writer w = block * num_groups + group,
//                      S = num_blocks * num_groups;
//                      low 32 bits  = type (bits 0-1) | event (2-11) | block (12-23) | sm (24-31)
//                      high 32 bits = %globaltimer_lo, nanoseconds
//
// Size the buffer as megabench_timeline_words(blocks, groups, events per writer) and zero
// it before the launch; writes past the end are dropped.
//
//   MB_TIMELINE_INIT(buffer, words, group, num_groups, is_writer);  // once, every thread
//   MB_TIMELINE_BEGIN(event); ... MB_TIMELINE_END(event);           // event < 1024
//   MB_TIMELINE_INSTANT(event);
#pragma once

#include <cstdint>

__host__ __device__ constexpr uint64_t megabench_timeline_words(uint32_t blocks, uint32_t groups,
                                                                uint32_t events) {
  return 1 + uint64_t(blocks) * groups * events;
}

#ifdef MEGABENCH_TIMELINE

namespace megabench_timeline {

struct Writer {
  uint64_t* next;
  uint64_t* end;
  uint32_t stride;
  uint32_t tag;
  bool on;

  __device__ void init(uint64_t* buffer, uint64_t words, uint32_t group, uint32_t num_groups,
                       bool is_writer) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
      buffer[0] = uint64_t(gridDim.x) | (uint64_t(num_groups) << 32);
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    next = buffer + 1 + uint64_t(blockIdx.x) * num_groups + group;
    end = buffer + words;
    stride = gridDim.x * num_groups;
    tag = ((sm & 0xFFu) << 24) | ((blockIdx.x & 0xFFFu) << 12);
    on = is_writer;
  }

  __device__ void record(uint32_t event, uint32_t type) {
    if (!on) return;
    uint32_t now;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(now));
    if (next < end) *next = (uint64_t(now) << 32) | tag | ((event & 0x3FFu) << 2) | type;
    next += stride;
  }
};

}  // namespace megabench_timeline

#define MB_TIMELINE_INIT(buffer, words, group, num_groups, is_writer) \
  megabench_timeline::Writer mb_timeline_writer;                       \
  mb_timeline_writer.init((buffer), (words), (group), (num_groups), (is_writer))
#define MB_TIMELINE_BEGIN(event) mb_timeline_writer.record((event), 0u)
#define MB_TIMELINE_END(event) mb_timeline_writer.record((event), 1u)
#define MB_TIMELINE_INSTANT(event) mb_timeline_writer.record((event), 2u)

#else

#define MB_TIMELINE_INIT(buffer, words, group, num_groups, is_writer) \
  do {                                                                 \
    (void)(buffer);                                                    \
    (void)(words);                                                     \
  } while (0)
#define MB_TIMELINE_BEGIN(event) \
  do {                           \
  } while (0)
#define MB_TIMELINE_END(event) \
  do {                         \
  } while (0)
#define MB_TIMELINE_INSTANT(event) \
  do {                             \
  } while (0)

#endif
