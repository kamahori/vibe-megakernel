#pragma once
#include "gemm.cuh"

__device__ __forceinline__ void store_bf16x32(bf16* dst, const float* v) {
  uint4 o[4];
  uint32_t* w = reinterpret_cast<uint32_t*>(o);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    __nv_bfloat162 t = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    w[i] = *reinterpret_cast<uint32_t*>(&t);
  }
  uint4* d = reinterpret_cast<uint4*>(dst);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = o[i];
}

// fast SiLU: x * rcp.approx(1 + ex2.approx(-x*log2e)); caller does the single bf16 round at store
__device__ __forceinline__ float silu_fast(float x) {
  float e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(x * -1.4426950408889634f));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.f + e));
  return x * r;
}

__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// EP_RED drain on all 8 warps, BN = 128. Warp w reads TMEM lane quarter q = w & 3 (rows m0 + 32q .. +31), columns
// [64h, 64h + 64) with h = w >> 2, and writes them to a 32 x 128 fp32 smem tile per quarter (idle mainloop stage smem,
// 16 KB per quarter, 16-byte chunks XOR-swizzled by row & 7: conflict-free both ways). After a 64-thread named barrier
// per quarter, warp h reduces rows h*16 .. h*16+15: one red.v4 warp instruction per row covers 128 contiguous columns,
// lane l adds columns 4l .. 4l+3.
__device__ __forceinline__ void epi_red_colv4_impl(const GemmArgs& g, uint32_t taddr, uint8_t* sbase, int m0, int n0, int warp, int lane) {
  const int q = warp & 3, h = warp >> 2;
  float4* tile = reinterpret_cast<float4*>(sbase + q * 16384);
  // the stage smem was last accessed by TMA writes / MMA reads (async proxy), complete at tfull
  fence_proxy_async();
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    float v[32];
    tmem_ld32(taddr + 64 * h + 32 * b, v);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = 16 * h + 8 * b + i;
      tile[lane * 32 + (c ^ (lane & 7))] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
    }
  }
  asm volatile("bar.sync %0, 64;" ::"r"(12 + q) : "memory");
  float* dst = g.red + (size_t)(m0 + 32 * q + 16 * h) * g.ld_out + n0 + 4 * lane;
  float4 r[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int row = 16 * h + i;
    r[i] = tile[row * 32 + (lane ^ (row & 7))];
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) red_add_v4(dst + (size_t)i * g.ld_out, r[i]);
  // generic smem accesses above must be ordered before the next phase's TMA writes into the stage area
  fence_proxy_async();
}

// store / red / silu / fuse epilogues (one thread = one output row, 32 columns per TMEM load)
struct EpiSimple {
  __device__ __forceinline__ void red_colv4(const GemmArgs& g, uint32_t taddr, uint8_t* sbase, int m0, int n0, int warp, int lane) const {
    epi_red_colv4_impl(g, taddr, sbase, m0, n0, warp, lane);
  }
  template <int BN, int EP>
  __device__ __forceinline__ void run(const GemmArgs& g, uint32_t taddr, int row, int n0, int nt) const {
#pragma unroll 1
    for (int c = 0; c < BN; c += 32) {
      float v[32];
      tmem_ld32(taddr + c, v);
      const size_t o = (size_t)row * g.ld_out + n0 + c;
      if (EP == EP_RED) {
#pragma unroll
        for (int i = 0; i < 32; ++i) atomicAdd(g.red + o + i, v[i]);
      } else {
        if (EP == EP_SILU) {
#pragma unroll
          for (int i = 0; i < 32; ++i) v[i] = silu_fast(bfr(v[i]));
        } else if (EP == EP_FUSE) {
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            float y = bfr(v[i]);
            float z = bfr(y + __bfloat162float(g.aux[n0 + c + i]));
            v[i] = bfr(silu_f(z));
          }
        }
        store_bf16x32(g.out + o, v);
      }
    }
  }
};
