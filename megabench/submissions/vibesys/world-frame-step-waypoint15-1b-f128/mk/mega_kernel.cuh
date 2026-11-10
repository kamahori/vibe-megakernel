#pragma once
#include "mega.cuh"
#include "../timeline_probe.cuh"

// timeline events
enum { EV_COND = 0, EV_PATCH, EV_RMS, EV_QKV, EV_ATTN, EV_WO, EV_FUSION, EV_FC1, EV_FC2, EV_HEAD, EV_NUM };

__device__ __constant__ float c_delta[4] = {0.8984375f - 1.0f, 0.75f - 0.8984375f, 0.30078125f - 0.75f, 0.0f - 0.30078125f};

__global__ void __launch_bounds__(256, 1) mega_kernel(const __grid_constant__ Params P) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* sbase = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  SmemBar* gbar = (SmemBar*)(sbase + GEMM_STAGE_SMEM);
  AttnBar* abar = (AttnBar*)(sbase + GEMM_STAGE_SMEM);
  uint32_t* tslot = (uint32_t*)(sbase + GEMM_STAGE_SMEM + 1024);
  const int tid = threadIdx.x, warp = tid >> 5;
  MB_TIMELINE_INIT((uint64_t*)P.tl, megabench_timeline_words(148, 1, 8192), 0, 1, tid == 0);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(smem_u32(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  unsigned long long* gb = P.bar;
#define BARRIER() grid_barrier(gb)
#ifdef MK_DEBUG
#define STOP(k) if (P.stop_layer == l && P.stop == (k)) { __syncthreads(); tc_fence_before(); if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem)); return; }
#else
#define STOP(k)
#endif
#define MODP(p, l, kind, i) (P.mod + ((size_t)((p) * 48 + (l) * 2 + (kind)) * 6144 + (i) * 2048))

  // ---- conditioning (all passes at once)
  MB_TIMELINE_BEGIN(EV_COND);
  embed_phase1(P, sbase);
  BARRIER();
  embed_phase2(P, sbase);
  BARRIER();
  gemv_phase(P, sbase);
  BARRIER();
  MB_TIMELINE_END(EV_COND);

  EpiSimple epiS;
  for (int pass = 0; pass < 5; ++pass) {
    const bool kv = pass == 4;
#ifdef MK_DEBUG
    if (P.only_pass >= 0 && pass != P.only_pass) continue;
#endif
    MB_TIMELINE_BEGIN(EV_PATCH);
    {
      GemmArgs g = {};
      g.A = &P.tmA_patch; g.B[0] = &P.tmB_patch; g.n_tiles = 16; g.k_iters = 2; g.ksplit = 1; g.ld_out = DM; g.out = P.x;
      gemm_phase<128, EP_STORE>(g, sbase, gbar, tmem, epiS);
    }
    BARRIER();
    MB_TIMELINE_END(EV_PATCH);
    MB_TIMELINE_BEGIN(EV_RMS);
    row_pass<0, 1>(P, nullptr, MODP(pass, 0, 0, 0), MODP(pass, 0, 0, 1));
    BARRIER();
    MB_TIMELINE_END(EV_RMS);
    for (int l = 0; l < NL; ++l) {
      // QKV
      MB_TIMELINE_BEGIN(EV_QKV);
      {
        GemmArgs g = {};
        g.A = &P.tmA_h; g.B[0] = &P.tmB_wq; g.B[1] = &P.tmB_wk; g.B[2] = &P.tmB_wv;
        g.b_nt_split[0] = 16; g.b_nt_split[1] = 24;
        g.b_row_off = l * 2048; g.b_row_off1 = l * 1024; g.b_row_off2 = l * 1024;
        g.n_tiles = 32; g.k_iters = 32; g.ksplit = 1; g.ld_out = 0;
        EpiQkv e;
        e.P = &P; e.layer = l;
        e.kdst = P.kbuf; e.vdst = P.vbuf;
        e.kout = kv ? P.k_write + (size_t)l * 512 * 1024 : nullptr;
        e.vout = kv ? P.v_write + (size_t)l * 512 * 1024 : nullptr;
        gemm_phase<128, EP_QKV>(g, sbase, gbar, tmem, e);
      }
      BARRIER();
      MB_TIMELINE_END(EV_QKV);
      STOP(1)
      if (kv && l == NL - 1) break;
      // attention
      MB_TIMELINE_BEGIN(EV_ATTN);
      {
        AttnArgs a;
        a.Q = &P.tmQ; a.Kc = &P.tmKc; a.Vc = &P.tmVc; a.Kb = &P.tmKb; a.Vb = &P.tmVb;
        a.layer = l; a.slots = 16; a.nvis = P.nvis[l]; a.vis = P.vis + l * 16; a.out = P.att;
        attn_phase(a, sbase, abar, tmem);
      }
      BARRIER();
      MB_TIMELINE_END(EV_ATTN);
      STOP(2)
      // Wo (split-K 2, red.add)
      MB_TIMELINE_BEGIN(EV_WO);
      {
        GemmArgs g = {};
        g.A = &P.tmA_att; g.B[0] = &P.tmB_wo; g.b_row_off = l * 2048;
        g.n_tiles = 16; g.k_iters = 16; g.ksplit = 2; g.ld_out = DM; g.red = P.acc;
        gemm_phase<128, EP_RED>(g, sbase, gbar, tmem, epiS);
      }
      BARRIER();
      MB_TIMELINE_END(EV_WO);
      MB_TIMELINE_BEGIN(EV_RMS);
      const bool control = (l % 3) == 0;
      if (control) {
        row_pass<1, 0>(P, MODP(pass, l, 0, 2), nullptr, nullptr);
        BARRIER();
        STOP(3)
        MB_TIMELINE_END(EV_RMS);
        MB_TIMELINE_BEGIN(EV_FUSION);
        const int fi = l / 3;
        {
          GemmArgs g = {};
          g.A = &P.tmA_h; g.B[0] = &P.tmB_fx; g.b_row_off = fi * 2048;
          g.n_tiles = 32; g.k_iters = 32; g.ksplit = 1; g.ld_out = DM; g.out = P.att; g.aux = P.cvec + fi * 2048;
          gemm_phase<64, EP_FUSE>(g, sbase, gbar, tmem, epiS);
        }
        BARRIER();
        STOP(4)
        {
          GemmArgs g = {};
          g.A = &P.tmA_att; g.B[0] = &P.tmB_fo; g.b_row_off = fi * 2048;
          g.n_tiles = 16; g.k_iters = 16; g.ksplit = 2; g.ld_out = DM; g.red = P.acc;
          gemm_phase<128, EP_RED>(g, sbase, gbar, tmem, epiS);
        }
        BARRIER();
        MB_TIMELINE_END(EV_FUSION);
        MB_TIMELINE_BEGIN(EV_RMS);
        row_pass<2, 1>(P, nullptr, MODP(pass, l, 1, 0), MODP(pass, l, 1, 1));
        BARRIER();
        STOP(5)
        MB_TIMELINE_END(EV_RMS);
      } else {
        row_pass<1, 1>(P, MODP(pass, l, 0, 2), MODP(pass, l, 1, 0), MODP(pass, l, 1, 1));
        BARRIER();
        MB_TIMELINE_END(EV_RMS);
      }
      // MLP
      MB_TIMELINE_BEGIN(EV_FC1);
      {
        GemmArgs g = {};
        g.A = &P.tmA_h; g.B[0] = &P.tmB_fc1; g.b_row_off = l * 8192;
        g.n_tiles = 32; g.k_iters = 32; g.ksplit = 1; g.ld_out = 8192; g.out = P.mlp;
        gemm_phase<256, EP_SILU>(g, sbase, gbar, tmem, epiS);
      }
      BARRIER();
      MB_TIMELINE_END(EV_FC1);
      STOP(6)
      MB_TIMELINE_BEGIN(EV_FC2);
      {
        GemmArgs g = {};
        g.A = &P.tmA_mlp; g.B[0] = &P.tmB_fc2; g.b_row_off = l * 2048;
        g.n_tiles = 16; g.k_iters = 64; g.ksplit = 2; g.ld_out = DM; g.red = P.acc;
        gemm_phase<128, EP_RED>(g, sbase, gbar, tmem, epiS);
      }
      BARRIER();
      MB_TIMELINE_END(EV_FC2);
      STOP(7)
      MB_TIMELINE_BEGIN(EV_RMS);
      if (l + 1 < NL) {
        row_pass<1, 1>(P, MODP(pass, l, 1, 2), MODP(pass, l + 1, 0, 0), MODP(pass, l + 1, 0, 1));
        BARRIER();
        MB_TIMELINE_END(EV_RMS);
      } else {
        MB_TIMELINE_END(EV_RMS);
        MB_TIMELINE_BEGIN(EV_HEAD);
        unpatch_prefetch(P, sbase);
        row_pass<1, 2>(P, MODP(pass, l, 1, 2), P.outmod + pass * 4096, P.outmod + pass * 4096 + 2048);
        BARRIER();
        unpatch_phase(P, sbase, c_delta[pass]);
        BARRIER();
        MB_TIMELINE_END(EV_HEAD);
      }
    }
  }
  __syncthreads();
  tc_fence_before();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
}
