// tcgen05 GEMM phase: C[512, N] = A[512, K] * B[N, K]^T (both K-major, bf16, TMA SWIZZLE_128B, 64-wide K stages).
// One work item per CTA: (m tile 128, n tile BN, k split). Accumulator in TMEM cols [0, BN).
#pragma once
#include "prim.cuh"

// Split-K completion epilogue layout (EP_RED, BN 128): 1 = columns-across-lanes red.global.add.v4.f32 through an smem
// transpose, 8 warps (default; EpiSimple::red_colv4 in epi_simple.cuh), 0 = the round-2 row-per-thread scalar atomicAdd form.
#ifndef MK_RED_COLV4
#define MK_RED_COLV4 1
#endif

enum { EP_STORE = 0, EP_RED = 1, EP_SILU = 2, EP_FUSE = 3, EP_QKV = 4 };

struct GemmArgs {
  const CUtensorMap* A;
  const CUtensorMap* B[3];   // up to 3 B maps (qkv: q,k,v), selected per n tile
  int b_nt_split[2];         // n-tile index where B[1], B[2] begin
  int b_row_off;             // row offset of the layer inside the stacked weight map (per B map: given per map by caller)
  int b_row_off1, b_row_off2;
  int n_tiles;               // number of BN tiles in N
  int k_iters;               // K iterations of 64 per item (after split)
  int ksplit;
  int ld_out;                // row stride (elements) of output / accumulators
  bf16* out;
  float* red;
  const bf16* aux;           // EP_FUSE: per-column vector
};

constexpr int GEMM_STAGE_SMEM = 192 * 1024;

template <int BN>
struct GemmCfg {
  static constexpr int A_BYTES = 128 * 128, B_BYTES = BN * 128, STAGE = A_BYTES + B_BYTES;
  static constexpr int STAGES = GEMM_STAGE_SMEM / STAGE > 8 ? 8 : GEMM_STAGE_SMEM / STAGE;
};

// Smem layout of the mbarrier area: full[8], empty[8], tmem_full
struct SmemBar {
  uint64_t full[8], empty[8], tfull;
};

template <int BN, int EP, class Epi>
__device__ __forceinline__ void gemm_phase(const GemmArgs& g, uint8_t* sbase, SmemBar* bar, uint32_t tmem, const Epi& epi) {
  using C = GemmCfg<BN>;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nitems = 4 * g.n_tiles * g.ksplit;
  __syncthreads();
  if (tid == 0) {
    for (int s = 0; s < C::STAGES; ++s) {
      mbar_init(smem_u32(&bar->full[s]), 1);
      mbar_init(smem_u32(&bar->empty[s]), 1);
    }
    mbar_init(smem_u32(&bar->tfull), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  const int item = blockIdx.x;
  if (item < nitems) {
    const int mt = item & 3, rest = item >> 2;
    const int ksi = rest % g.ksplit, nt = rest / g.ksplit;
    const int m0 = mt * 128, n0 = nt * BN;
    const int kbeg = ksi * g.k_iters;
    if (warp == 0 && lane == 0) {
      const CUtensorMap* bm = g.B[0];
      int brow = g.b_row_off, nloc = n0;
      if (g.b_nt_split[0] > 0 && nt >= g.b_nt_split[0]) {
        if (g.b_nt_split[1] > 0 && nt >= g.b_nt_split[1]) { bm = g.B[2]; brow = g.b_row_off2; nloc = n0 - g.b_nt_split[1] * BN; }
        else { bm = g.B[1]; brow = g.b_row_off1; nloc = n0 - g.b_nt_split[0] * BN; }
      }
      for (int kk = 0; kk < g.k_iters; ++kk) {
        const int s = kk % C::STAGES, ph = (kk / C::STAGES) & 1;
        mbar_wait(smem_u32(&bar->empty[s]), ph ^ 1);
        const uint32_t fb = smem_u32(&bar->full[s]);
        mbar_expect_tx(fb, C::STAGE);
        const uint32_t sa = smem_u32(sbase + s * C::STAGE);
        tma_load_2d(g.A, sa, fb, (kbeg + kk) * 64, m0);
        tma_load_2d(bm, sa + C::A_BYTES, fb, (kbeg + kk) * 64, brow + nloc);
      }
    } else if (warp == 1 && lane == 0) {
      const uint32_t idesc = make_idesc(128, BN);
      for (int kk = 0; kk < g.k_iters; ++kk) {
        const int s = kk % C::STAGES, ph = (kk / C::STAGES) & 1;
        mbar_wait(smem_u32(&bar->full[s]), ph);
        tc_fence_after();
        const uint32_t sa = smem_u32(sbase + s * C::STAGE), sb = sa + C::A_BYTES;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ss(tmem, make_sdesc(sa + 32 * j), make_sdesc(sb + 32 * j), idesc, (kk > 0 || j > 0) ? 1u : 0u);
        tc_commit(smem_u32(&bar->empty[s]));
      }
      tc_commit(smem_u32(&bar->tfull));
    } else if (EP != EP_SILU && !(EP == EP_RED && BN == 128 && MK_RED_COLV4) && warp >= 4) {
      mbar_wait(smem_u32(&bar->tfull), 0);
      tc_fence_after();
      const int w4 = warp & 3;
      const int row = m0 + w4 * 32 + lane;
      epi.template run<BN, EP>(g, tmem + ((uint32_t)(w4 * 32) << 16), row, n0, nt);
    }
    if constexpr (EP == EP_RED && BN == 128 && MK_RED_COLV4) {
      // split-K completion: 8-warp columns-across-lanes red.v4 drain (EpiSimple::red_colv4)
      __syncwarp();
      mbar_wait(smem_u32(&bar->tfull), 0);
      tc_fence_after();
      epi.red_colv4(g, tmem + ((uint32_t)((warp & 3) * 32) << 16), sbase, m0, n0, warp, lane);
    }
    if (EP == EP_SILU) {
      // 8-warp drain: warps w and w+4 share TMEM lane quarter (w & 3) and split the BN columns in halves.
      __syncwarp();
      mbar_wait(smem_u32(&bar->tfull), 0);
      tc_fence_after();
      const int w4 = warp & 3, half = (warp >> 2) * (BN / 2);
      const int row = m0 + w4 * 32 + lane;
      epi.template run<BN / 2, EP>(g, tmem + ((uint32_t)(w4 * 32) << 16) + half, row, n0 + half, nt);
    }
  }
  __syncthreads();
}
