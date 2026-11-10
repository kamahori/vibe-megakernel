// tcgen05 flash attention phase. Work item = (q head, 128-query tile): 32 x 4 = 128 items, full KV per item.
// Intra-CTA KV split: softmax warpgroup A (warps 4-7) takes the even KV tiles, warpgroup B (warps 0-3) the odd ones.
// Each stream g (A=0, B=1) has its own S buffer (TMEM cols g*128), P smem buffer, O_part (TMEM cols 256+g*64), m/l/O
// registers and barriers; lane 0 of warp 4 (A) / warp 2 (B) issues its stream's TMA loads and MMAs inline.
// S = Q K^T (M128 N128 K64), P (bf16) -> smem, O_part = P V (M128 N64 K128, V MN-major). One LSE merge at the end.
#pragma once
#include "prim.cuh"

struct AttnBar {
  uint64_t qfull, kfull[4], vfull[3], sfull[2], sfree[2], pfull[2], ofull[2], mrg;
  int vrow[16];  // cache row base of each visible slot (layer*slots + vis[i]) * 512
};

struct AttnArgs {
  const CUtensorMap *Q, *Kc, *Vc, *Kb, *Vb;  // Q: [512, 2048]; caches [L*S*512, 1024]; current k/v [512, 1024]
  int layer, slots, nvis;
  const int* vis;                              // visible slot ids
  bf16* out;                                   // [512, 2048]
};

#ifndef ATTN_NTILE
#define ATTN_NTILE(a) (((a).nvis + 1) * 4)
#endif

__device__ __forceinline__ float att_ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// packed fp32x2 (FFMA2 / FADD2): d = a * b + c on two lanes with a shared b, c
__device__ __forceinline__ void att_fma2(float& d0, float& d1, float a0, float a1, float b, float c) {
  asm("{\n.reg .b64 a, b, c, d;\nmov.b64 a, {%2, %3};\nmov.b64 b, {%4, %4};\nmov.b64 c, {%5, %5};\n"
      "fma.rn.ftz.f32x2 d, a, b, c;\nmov.b64 {%0, %1}, d;\n}\n"
      : "=f"(d0), "=f"(d1) : "f"(a0), "f"(a1), "f"(b), "f"(c));
}
__device__ __forceinline__ void att_add2(float& d0, float& d1, float a0, float a1) {
  asm("{\n.reg .b64 a, b;\nmov.b64 a, {%0, %1};\nmov.b64 b, {%2, %3};\nadd.rn.ftz.f32x2 a, a, b;\nmov.b64 {%0, %1}, a;\n}\n"
      : "+f"(d0), "+f"(d1) : "f"(a0), "f"(a1));
}
__device__ __forceinline__ float att_max3(float a, float b, float c) {
  float y;
  asm("max.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
  return y;
}
__device__ __forceinline__ uint32_t att_pack(float lo, float hi) {
  uint32_t y;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
  return y;
}

__device__ __forceinline__ void att_st128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ float4 att_ld128(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void att_sts32(uint32_t addr, float v) { asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory"); }
__device__ __forceinline__ float att_ld32(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
  return v;
}

// 32 lanes x 32 columns fp32, no wait (caller issues tmem_wait_ld)
__device__ __forceinline__ void att_ld32_nw(uint32_t taddr, float* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]),
        "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]), "=f"(r[17]), "=f"(r[18]),
        "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]), "=f"(r[24]), "=f"(r[25]), "=f"(r[26]), "=f"(r[27]),
        "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void att_st32(uint32_t taddr, const float* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "f"(r[0]), "f"(r[1]), "f"(r[2]), "f"(r[3]), "f"(r[4]), "f"(r[5]), "f"(r[6]), "f"(r[7]), "f"(r[8]), "f"(r[9]),
      "f"(r[10]), "f"(r[11]), "f"(r[12]), "f"(r[13]), "f"(r[14]), "f"(r[15]), "f"(r[16]), "f"(r[17]), "f"(r[18]),
      "f"(r[19]), "f"(r[20]), "f"(r[21]), "f"(r[22]), "f"(r[23]), "f"(r[24]), "f"(r[25]), "f"(r[26]), "f"(r[27]),
      "f"(r[28]), "f"(r[29]), "f"(r[30]), "f"(r[31]) : "memory");
}
__device__ __forceinline__ bool att_test_wait(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred P;\nmbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}\n"
               : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return __shfl_sync(0xffffffffu, ok, 0) != 0;  // warp-uniform
}
__device__ __forceinline__ void att_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// Issue helpers for a converged warp: one elected lane issues (operands are warp-uniform).
// S = Q K^T for one KV tile: 4 x (M128 N128 K16), Q/K K-major SW128; one asm block (descriptor address field += 32 B).
__device__ __forceinline__ void att_mma_s_e(uint32_t d_tmem, uint64_t a0, uint64_t b0, uint32_t idesc) {
  asm volatile(
      "{\n.reg .pred e;\n.reg .b64 a, b;\nelect.sync _|e, 0xffffffff;\n"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
      "add.s64 a, %1, 2;\nadd.s64 b, %2, 2;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 4;\nadd.s64 b, %2, 4;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 6;\nadd.s64 b, %2, 6;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n}\n" ::"r"(d_tmem),
      "l"(a0), "l"(b0), "r"(idesc));
}
// O (+)= P V for one KV tile: 8 x (M128 N64 K16); P K-major SW128 in two 16 KB column halves, V MN-major (2 KB per K16).
__device__ __forceinline__ void att_mma_pv_e(uint32_t d_tmem, uint64_t a0, uint64_t b0, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred e, p;\n.reg .b64 a, b;\nelect.sync _|e, 0xffffffff;\nsetp.ne.b32 p, %4, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
      "add.s64 a, %1, 2;\nadd.s64 b, %2, 128;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 4;\nadd.s64 b, %2, 256;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 6;\nadd.s64 b, %2, 384;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 1024;\nadd.s64 b, %2, 512;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 1026;\nadd.s64 b, %2, 640;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 1028;\nadd.s64 b, %2, 768;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n"
      "add.s64 a, %1, 1030;\nadd.s64 b, %2, 896;\n@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, 1;\n}\n" ::"r"(d_tmem),
      "l"(a0), "l"(b0), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void att_commit_e(uint32_t mbar) {
  asm volatile(
      "{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n"
      "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}\n" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ void att_tma_e(const CUtensorMap* map, uint32_t dst, uint32_t mbar, int x, int y) {
  asm volatile(
      "{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\n"
      "@e mbarrier.arrive.expect_tx.shared::cta.b64 _, [%2], 16384;\n"
      "@e cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n}\n" ::"r"(dst),
      "l"(map), "r"(mbar), "r"(x), "r"(y) : "memory");
}

// smem: Q 16 KB | K ring 4 x 16 KB (tile j -> slot j%4) | V ring 3 x 16 KB (tile j -> slot j%3) | P_A 32 KB | P_B 32 KB = 192 KB
constexpr int ATT_Q = 0, ATT_K = 16384, ATT_V = 81920, ATT_P = 131072, ATT_T = 16384, ATT_PSZ = 32768;

// K or V tile j of the item into dst (called by a whole converged warp)
__device__ __forceinline__ void attn_load(const AttnArgs& a, const AttnBar* bar, int kvh, int j, bool isv, uint32_t dst, uint32_t mb) {
  const CUtensorMap* m;
  int row;
  if (j < a.nvis * 4) {
    row = bar->vrow[j >> 2] + (j & 3) * 128;
    m = isv ? a.Vc : a.Kc;
  } else {
    row = (j - a.nvis * 4) * 128;
    m = isv ? a.Vb : a.Kb;
  }
  att_tma_e(m, dst, mb, kvh * 64, row);
}

__device__ __forceinline__ void attn_phase(const AttnArgs& a, uint8_t* sbase, AttnBar* bar, uint32_t tmem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __syncthreads();
  if (tid == 0) {
    mbar_init(smem_u32(&bar->qfull), 1);
    mbar_init(smem_u32(&bar->mrg), 128);
    for (int s = 0; s < 4; ++s) mbar_init(smem_u32(&bar->kfull[s]), 1);
    for (int s = 0; s < 3; ++s) mbar_init(smem_u32(&bar->vfull[s]), 1);
    for (int s = 0; s < 2; ++s) {
      mbar_init(smem_u32(&bar->sfull[s]), 1);
      mbar_init(smem_u32(&bar->sfree[s]), 4);
      mbar_init(smem_u32(&bar->pfull[s]), 128);
      mbar_init(smem_u32(&bar->ofull[s]), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (tid < a.nvis && tid < 16) bar->vrow[tid] = (a.layer * a.slots + a.vis[tid]) * 512;
  __syncthreads();
  const int item = blockIdx.x;
  const int ntile = ATTN_NTILE(a);
  if (item < 128) {
    const int head = item >> 2, qt = item & 3, kvh = head >> 1;
    const int g = warp >= 4 ? 0 : 1;                       // stream: A (warps 4-7) even tiles, B (warps 0-3) odd tiles
    // Issue duties are spread over the 4 warps of each stream (whole warp, one elected lane issues):
    // role 0: PV MMA, role 1: S MMA, role 2: K loads, role 3: V loads. Warp 4 also issues the prologue loads.
    // B's roles are rotated by 2 so the two MMA-issuing warps of each stream sit on different SM sub-partitions.
    const int role = (warp + 2 * g) & 3;
    const uint32_t sb = smem_u32(sbase);
    const uint32_t q0 = sb + ATT_Q, pg = sb + ATT_P + g * ATT_PSZ;
    const uint32_t b_sfull = smem_u32(&bar->sfull[g]), b_sfree = smem_u32(&bar->sfree[g]);
    const uint32_t b_pfull = smem_u32(&bar->pfull[g]), b_ofull = smem_u32(&bar->ofull[g]);
    const uint32_t b_kfull = smem_u32(&bar->kfull[0]), b_vfull = smem_u32(&bar->vfull[0]);
    const uint32_t idS = make_idesc(128, 128), idO = make_idesc(128, 64, 0, 1);
    const uint32_t tS = tmem + g * 128, tO = tmem + 256 + g * 64;
    auto issue_S = [&](int j) {
      mbar_wait(b_kfull + 8 * (j & 3), (j >> 2) & 1);
      tc_fence_after();
      const uint32_t k0 = sb + ATT_K + (j & 3) * ATT_T;
      att_mma_s_e(tS, make_sdesc(q0), make_sdesc(k0), idS);
      att_commit_e(b_sfull);
    };
    {
      if (warp == 4) {
        att_tma_e(a.Q, q0, smem_u32(&bar->qfull), head * 64, qt * 128);
        for (int t = 0; t < 4 && t < ntile; ++t) attn_load(a, bar, kvh, t, false, sb + ATT_K + t * ATT_T, b_kfull + 8 * t);
        for (int t = 0; t < 3 && t < ntile; ++t) attn_load(a, bar, kvh, t, true, sb + ATT_V + t * ATT_T, b_vfull + 8 * t);
      }
      if (role == 1 && g < ntile) {
        mbar_wait(smem_u32(&bar->qfull), 0);
        issue_S(g);
      }
    }
    __syncwarp();
    const int w4 = warp & 3, r = w4 * 32 + lane;
    const uint32_t lsel = (uint32_t)(w4 * 32) << 16;
    const float sc = 0.125f * 1.4426950408889634f;
    // O accumulates in TMEM (tO); the running max m is only raised when a row max exceeds it by more than 8 (log2 units),
    // and then the warp rescales its O rows in TMEM. P <= 2^8 keeps bf16 P and fp32 l/O safe.
    float m = -INFINITY, l = 0.f;
    const uint32_t prow = pg + r * 128;
    const int rsw = r & 7;
    int k = 0;
    for (int j = g; j < ntile; j += 2, ++k) {
      mbar_wait(b_sfull, k & 1);
      tc_fence_after();
      float s[128];
#pragma unroll
      for (int c = 0; c < 4; ++c) att_ld32_nw(tS + lsel + c * 32, s + c * 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(b_sfree);
      __syncwarp();
      // S(j) is complete: its K slot is free for tile j+4 (same stream)
      if (role == 2 && j + 4 < ntile) attn_load(a, bar, kvh, j + 4, false, sb + ATT_K + (j & 3) * ATT_T, b_kfull + 8 * (j & 3));
      if (role == 1 && j + 2 < ntile) {
        mbar_wait(b_sfree, k & 1);
        issue_S(j + 2);
      }
      __syncwarp();
      float mx8[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) mx8[q] = att_max3(s[q], s[q + 8], s[q + 16]);
#pragma unroll
      for (int i = 24; i + 16 <= 128; i += 16)
#pragma unroll
        for (int q = 0; q < 8; ++q) mx8[q] = att_max3(mx8[q], s[i + q], s[i + 8 + q]);
#pragma unroll
      for (int q = 0; q < 8; ++q) mx8[q] = fmaxf(mx8[q], s[120 + q]);
      const float mxs = sc * fmaxf(att_max3(mx8[0], mx8[1], mx8[2]), att_max3(att_max3(mx8[3], mx8[4], mx8[5]), mx8[6], mx8[7]));
      // PV(j-2) complete -> V slot (j+1)%3 is free: load V(j+1) for the other stream. Done here if PV(j-2) has already
      // completed, else after the exponentials (where all warps wait for it anyway).
      bool vpend = role == 3 && k > 0 && j + 1 < ntile;
      if (vpend && att_test_wait(b_ofull, (k - 1) & 1)) {
        attn_load(a, bar, kvh, j + 1, true, sb + ATT_V + ((j + 1) % 3) * ATT_T, b_vfull + 8 * ((j + 1) % 3));
        vpend = false;
      }
      __syncwarp();
      const bool up = mxs > m + 8.f;
      float alpha = 1.f;
      if (up) {
        alpha = att_ex2(m - mxs);
        m = mxs;
      }
      const float nm = -m;
      float sm4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 128; i += 2) {
        float x0, x1;
        att_fma2(x0, x1, s[i], s[i + 1], sc, nm);
        s[i] = att_ex2(x0);
        s[i + 1] = att_ex2(x1);
        if ((i >> 1) & 1) att_add2(sm4[2], sm4[3], s[i], s[i + 1]);
        else att_add2(sm4[0], sm4[1], s[i], s[i + 1]);
      }
      l = fmaf(l, alpha, (sm4[0] + sm4[1]) + (sm4[2] + sm4[3]));
      if (k > 0) {
        mbar_wait(b_ofull, (k - 1) & 1);
        if (vpend) attn_load(a, bar, kvh, j + 1, true, sb + ATT_V + ((j + 1) % 3) * ATT_T, b_vfull + 8 * ((j + 1) % 3));
        tc_fence_after();
        if (__any_sync(0xffffffffu, up)) {
#pragma unroll 1
          for (int c = 0; c < 2; ++c) {
            float part[32];
            att_ld32_nw(tO + lsel + c * 32, part);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 32; ++i) part[i] *= alpha;
            att_st32(tO + lsel + c * 32, part);
          }
          att_wait_st();
        }
        tc_fence_before();
      }
#pragma unroll
      for (int c16 = 0; c16 < 16; ++c16) {
        const float* x = s + c16 * 8;
        att_st128(prow + (c16 >> 3) * 16384 + (((c16 & 7) ^ rsw) << 4), att_pack(x[0], x[1]), att_pack(x[2], x[3]),
                  att_pack(x[4], x[5]), att_pack(x[6], x[7]));
      }
      fence_proxy_async();
      mbar_arrive(b_pfull);
      if (role == 0) {
        mbar_wait(b_pfull, k & 1);
        mbar_wait(b_vfull + 8 * (j % 3), (j / 3) & 1);
        tc_fence_after();
        const uint32_t v0 = sb + ATT_V + (j % 3) * ATT_T;
        att_mma_pv_e(tO, make_sdesc(pg), make_sdesc(v0, 1024, 16), idO, k > 0);
        att_commit_e(b_ofull);
      }
      __syncwarp();
    }
    float o[64];
    if (k > 0) {
      mbar_wait(b_ofull, (k - 1) & 1);
      tc_fence_after();
      att_ld32_nw(tO + lsel, o);
      att_ld32_nw(tO + lsel + 32, o + 32);
      tmem_wait_ld();
      tc_fence_before();
    } else {
#pragma unroll
      for (int i = 0; i < 64; ++i) o[i] = 0.f;
    }
    // merge staging: B's normalized O rows in P_B (row r: 16 x 16 B chunks, chunk c at (c ^ (r & 15))), B's LSE in K slot 1
    const uint32_t mo = sb + ATT_P + ATT_PSZ + r * 256, ml = sb + ATT_K + ATT_T + r * 4;
    if (g == 1) {
      float inv = 0.f, lse = -INFINITY;
      if (l > 0.f) {
        inv = __frcp_rn(l);
        lse = m + __log2f(l);
      }
#pragma unroll
      for (int c = 0; c < 16; ++c)
        att_st128(mo + ((c ^ (r & 15)) << 4), __float_as_uint(o[4 * c] * inv), __float_as_uint(o[4 * c + 1] * inv),
                  __float_as_uint(o[4 * c + 2] * inv), __float_as_uint(o[4 * c + 3] * inv));
      att_sts32(ml, lse);
      mbar_arrive(smem_u32(&bar->mrg));
    } else {
      mbar_wait(smem_u32(&bar->mrg), 0);
      const float lb = att_ld32(ml);
      const float mm = fmaxf(m, lb);
      const float fa = att_ex2(m - mm), fb = att_ex2(lb - mm);
      const float inv = __frcp_rn(fmaf(l, fa, fb));
      const float ca = fa * inv, cb = fb * inv;
      bf16* op = a.out + (size_t)(qt * 128 + r) * 2048 + head * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const float4 u = att_ld128(mo + (((2 * c8) ^ (r & 15)) << 4)), w = att_ld128(mo + (((2 * c8 + 1) ^ (r & 15)) << 4));
        const float* x = o + 8 * c8;
        uint4 pk;
        pk.x = att_pack(fmaf(x[0], ca, u.x * cb), fmaf(x[1], ca, u.y * cb));
        pk.y = att_pack(fmaf(x[2], ca, u.z * cb), fmaf(x[3], ca, u.w * cb));
        pk.z = att_pack(fmaf(x[4], ca, w.x * cb), fmaf(x[5], ca, w.y * cb));
        pk.w = att_pack(fmaf(x[6], ca, w.z * cb), fmaf(x[7], ca, w.w * cb));
        reinterpret_cast<uint4*>(op)[c8] = pk;
      }
    }
  }
  __syncthreads();
}
