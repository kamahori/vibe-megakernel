// loadstream: primitives for load-centric decode megakernels.
//
//   ring.cuh   WeightRing / Producer / Consumer: TMA-bulk byte ring, producer runs ahead
//   event.cuh  epoch counters (wait_geq / warp_signal / grid_barrier) with a watchdog
//   tap.cuh    deterministic GEMV tiles (BF16, FP8, int8, int4), small-M mma.sync tap, reductions
//   stamp.cuh  globaltimer stamps for the timeline report (LS_PROFILE)
//   plan.cuh   byte-balanced row bands
#pragma once
#include "ptx.cuh"
#include "event.cuh"
#include "ring.cuh"
#include "tap.cuh"
#include "stamp.cuh"
#include "plan.cuh"
