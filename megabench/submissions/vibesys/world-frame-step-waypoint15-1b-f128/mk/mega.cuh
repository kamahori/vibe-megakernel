#pragma once
#include "gemm.cuh"
#include "epi_simple.cuh"
#include "attn.cuh"

#define NL 24
#define DM 2048
#define RMS_EPS_F 1.1920928955078125e-07f

struct __align__(64) Params {
  CUtensorMap tmA_h, tmA_att, tmA_mlp, tmA_patch, tmQ;
  CUtensorMap tmB_wq, tmB_wk, tmB_wv, tmB_wo, tmB_fc1, tmB_fc2, tmB_fx, tmB_fo, tmB_patch;
  CUtensorMap tmKc, tmVc, tmKb, tmVb;
  // inputs
  const bf16 *noise, *mouse, *button, *scroll;
  const float *noise_fc1, *noise_fc2;
  const bf16 *ctrl_fc1, *ctrl_fc2, *attn_cond_bias, *attn_cond_w, *mlp_cond_bias, *mlp_cond_w, *v_lamb, *fuse_c, *out_norm_w,
      *unpatch_w, *unpatch_b;
  // outputs
  bf16 *latent, *k_write, *v_write;
  // scratch
  bf16 *x, *h, *att, *mlp, *patch_in, *qbuf, *kbuf, *vbuf, *vfirst;
  float* acc;
  const float *fourier, *rope_cos, *rope_sin;
  float* n_h1;
  bf16 *cond, *ctrl_h1, *ctrl, *mod, *outmod, *cvec;
  const int *vis, *nvis;
  unsigned long long* bar;
  unsigned long long* tl;   // timeline buffer (unused in production)
  int stop, stop_layer, only_pass;   // debug (defaults -1: off)
};

// -------- vector helpers
__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) { float2 t = __bfloat1622float2(p[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 u;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---------------- QKV epilogue
struct EpiQkv {
  const Params* P;
  int layer;
  bf16 *kdst, *vdst;   // scratch (read by attention)
  bf16 *kout, *vout;   // optional extra destinations (k_write / v_write in the KV pass)
  template <int BN, int EP>
  __device__ __forceinline__ void run(const GemmArgs& g, uint32_t taddr, int row, int n0, int nt) const {
#pragma unroll 1
    for (int hh = 0; hh < 2; ++hh) {
      float r[64];
      tmem_ld32(taddr + hh * 64, r);
      tmem_ld32(taddr + hh * 64 + 32, r + 32);
      uint4 o[8];
      if (nt < 24) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) { r[i] = bfr(r[i]); ss += r[i] * r[i]; }
        const float inv = rsqrtf(ss * (1.f / 64.f) + RMS_EPS_F);
        const float* cs = P->rope_cos + row * 32;
        const float* sn = P->rope_sin + row * 32;
        float out[64];
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const float e = bfr(r[2 * i] * inv), od = bfr(r[2 * i + 1] * inv);
          const float c = cs[i], s = sn[i];
          out[i] = e * c - od * s;
          out[32 + i] = od * c + e * s;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = pack8(out + 8 * i);
        bf16* dst;
        if (nt < 16) dst = P->qbuf + (size_t)row * 2048 + (nt * 2 + hh) * 64;
        else dst = kdst + (size_t)row * 1024 + ((nt - 16) * 2 + hh) * 64;
#pragma unroll
        for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(dst)[i] = o[i];
        if (nt >= 16 && kout) {
          bf16* d2 = kout + (size_t)row * 1024 + ((nt - 16) * 2 + hh) * 64;
#pragma unroll
          for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(d2)[i] = o[i];
        }
      } else {
        const int col = ((nt - 24) * 2 + hh) * 64;
        bf16* vf = P->vfirst + (size_t)row * 1024 + col;
        const float w = __bfloat162float(P->v_lamb[layer]);
        float out[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) r[i] = bfr(r[i]);
        if (layer == 0) {
#pragma unroll
          for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(vf)[i] = pack8(r + 8 * i);
#pragma unroll
          for (int i = 0; i < 64; ++i) out[i] = r[i];
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float f[8];
            unpack8(reinterpret_cast<const uint4*>(vf)[i], f);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const float s = r[8 * i + j], e = f[j], d = e - s;
              out[8 * i + j] = (w < 0.5f) ? (s + w * d) : (e - d * (1.f - w));
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = pack8(out + 8 * i);
        bf16* dst = vdst + (size_t)row * 1024 + col;
#pragma unroll
        for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(dst)[i] = o[i];
        if (vout) {
          bf16* d2 = vout + (size_t)row * 1024 + col;
#pragma unroll
          for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(d2)[i] = o[i];
        }
      }
    }
  }
};

// ---------------- row pass: residual (opt) + rms (+ modulate | head silu)
// RES: 0 none, 1 gated (x = bf(bf(acc)*g) + x), 2 plain (x = bf(acc)+x); MOD: 0 rms only, 1 modulate, 2 head
// Two warps per row: warp w (0..3) owns columns [0,1024) and warp w+4 owns [1024,2048) of row
// (w)*gridDim.x + blockIdx.x. The pair joins its sum of squares through 8 floats of smem in the
// reserved bytes after the TMEM slot (sbase + GEMM_STAGE_SMEM + 1024 + 64) and named barrier 1+w.
template <int RES, int MOD>
__device__ __forceinline__ void row_pass(const Params& P, const bf16* g, const bf16* s, const bf16* b) {
  extern __shared__ uint8_t smem_raw[];
  float* pss = (float*)((((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023) + GEMM_STAGE_SMEM + 1024 + 64);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wq = warp & 3, half = warp >> 2;
  const int row = wq * gridDim.x + blockIdx.x;
  if (row < 512) {
    bf16* xr = P.x + (size_t)row * DM + half * 1024;
    float* ar = P.acc + (size_t)row * DM + half * 1024;
    if (RES == 1) g += half * 1024;
    if (MOD) { s += half * 1024; b += half * 1024; }
    float xv[32];
    float ss = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int c = t * 256 + lane * 8;
      float f[8];
      unpack8(*reinterpret_cast<const uint4*>(xr + c), f);
      if (RES) {
        float4 a0 = *reinterpret_cast<const float4*>(ar + c), a1 = *reinterpret_cast<const float4*>(ar + c + 4);
        float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float gg[8];
        if (RES == 1) unpack8(*reinterpret_cast<const uint4*>(g + c), gg);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float tq = bfr(a[i]);
          if (RES == 1) tq = bfr(tq * gg[i]);
          f[i] = bfr(tq + f[i]);
        }
        *reinterpret_cast<uint4*>(xr + c) = pack8(f);
        *reinterpret_cast<float4*>(ar + c) = make_float4(0, 0, 0, 0);
        *reinterpret_cast<float4*>(ar + c + 4) = make_float4(0, 0, 0, 0);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) { xv[t * 8 + i] = f[i]; ss += f[i] * f[i]; }
    }
    // modulation vectors do not depend on the reduction: issue their loads before the join
    float sv[4][8], bv[4][8];
    if (MOD) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int c = t * 256 + lane * 8;
        unpack8(*reinterpret_cast<const uint4*>(s + c), sv[t]);
        unpack8(*reinterpret_cast<const uint4*>(b + c), bv[t]);
      }
    }
    ss = warp_sum(ss);
    if (lane == 0) pss[warp] = ss;
    asm volatile("bar.sync %0, 64;" ::"r"(1 + wq) : "memory");
    const float tot = pss[wq] + pss[wq + 4];
    const float inv = rsqrtf(tot * (1.f / 2048.f) + RMS_EPS_F);
    bf16* hr = P.h + (size_t)row * DM + half * 1024;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int c = t * 256 + lane * 8;
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float y = bfr(xv[t * 8 + i] * inv);
        if (MOD) {
          y = bfr(y * bfr(1.f + sv[t][i]));
          y = bfr(y + bv[t][i]);
          if (MOD == 2) y = bfr(silu_f(y));
        }
        o[i] = y;
      }
      *reinterpret_cast<uint4*>(hr + c) = pack8(o);
    }
  }
}

// ---------------- GEMV segments (modulation vectors, out head, fuse_c)
// global row space: 48 segments of 6144 rows (attn/mlp cond per layer), out_norm 4096 rows, fuse_c 16384 rows
// Tensor-core GEMV: mma.sync m16n8k16 bf16, A = 16 weight rows (straight from global, ld.global.nc.v4),
// B = the (up to 5) conditioning vectors as bf16 [8][2048] in smem (exact: they are bf16-rounded), fp32 accum.
// K is permuted consistently in A and B: lane (g = lane/4, q = lane%4) loads 8 consecutive k of rows g and g+8.
#define GV_TOTAL (48 * 6144 + 4096 + 16384)
#define GV_TILES (GV_TOTAL / 16)
#define GV_VS 2080  // smem row stride (bf16 elems) of the vector matrix: 64 B shift per n row

__device__ __forceinline__ void ldnc4(uint4& v, const bf16* p) {
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}
__device__ __forceinline__ void mma_bf16_16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void gemv_phase(const Params& P, uint8_t* sbase) {
  bf16* vec = reinterpret_cast<bf16*>(sbase);  // [8][GV_VS]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q = lane & 3;
  const int per = (GV_TILES + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * per, hi = min(GV_TILES, lo + per);
  int pos = lo;  // in tiles of 16 rows
  while (pos < hi) {
    int sg, segbase, segrows;
    const int prow = pos * 16;
    if (prow < 48 * 6144) { sg = prow / 6144; segbase = sg * 6144; segrows = 6144; }
    else if (prow < 48 * 6144 + 4096) { sg = 48; segbase = 48 * 6144; segrows = 4096; }
    else { sg = 49; segbase = 48 * 6144 + 4096; segrows = 16384; }
    const int end = min(hi, (segbase + segrows) / 16);
    const int nv = (sg == 49) ? 1 : 5;
    __syncthreads();
    if (sg < 48) {
      const int l = sg >> 1, kind = sg & 1;
      const bf16* bias = (kind ? P.mlp_cond_bias : P.attn_cond_bias) + l * DM;
      for (int i = tid; i < 8 * DM; i += 256) {
        const int p = i >> 11, k = i & (DM - 1);
        float val = 0.f;
        if (p < 5) {
          const float v = bfr(__bfloat162float(P.cond[p * DM + k]) + __bfloat162float(bias[k]));
          val = silu_f(v);
        }
        vec[p * GV_VS + k] = __float2bfloat16_rn(val);
      }
    } else if (sg == 48) {
      for (int i = tid; i < 8 * DM; i += 256) {
        const int p = i >> 11, k = i & (DM - 1);
        vec[p * GV_VS + k] = __float2bfloat16_rn(p < 5 ? silu_f(__bfloat162float(P.cond[i])) : 0.f);
      }
    } else {
      // ctrl_n = bf16(rms(ctrl))
      __shared__ float red[8];
      float ss = 0.f;
      for (int i = tid; i < DM; i += 256) { float v = __bfloat162float(P.ctrl[i]); ss += v * v; }
      ss = warp_sum(ss);
      if (lane == 0) red[warp] = ss;
      __syncthreads();
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) tot += red[i];
      const float inv = rsqrtf(tot * (1.f / 2048.f) + RMS_EPS_F);
      for (int i = tid; i < 8 * DM; i += 256) {
        const int p = i >> 11, k = i & (DM - 1);
        vec[p * GV_VS + k] = __float2bfloat16_rn(p == 0 ? __bfloat162float(P.ctrl[k]) * inv : 0.f);
      }
    }
    __syncthreads();
    const bf16* W;
    if (sg < 48) W = (sg & 1 ? P.mlp_cond_w : P.attn_cond_w) + (size_t)(sg >> 1) * 6144 * DM;
    else if (sg == 48) W = P.out_norm_w;
    else W = P.fuse_c;
    const int t0 = pos - segbase / 16, t1 = end - segbase / 16;
    const bf16* vr = vec + g * GV_VS + q * 8;
    for (int t = t0 + warp; t < t1; t += 8) {
      const bf16* wr = W + (size_t)(t * 16 + g) * DM + q * 8;
      const bf16* wr8 = wr + 8 * DM;
      float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
      uint4 a[2][8], a8[2][8];
#pragma unroll
      for (int j = 0; j < 8; ++j) { ldnc4(a[0][j], wr + j * 32); ldnc4(a8[0][j], wr8 + j * 32); }
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const int cur = b & 1;
        if (b + 1 < 8) {
#pragma unroll
          for (int j = 0; j < 8; ++j) { ldnc4(a[cur ^ 1][j], wr + ((b + 1) * 8 + j) * 32); ldnc4(a8[cur ^ 1][j], wr8 + ((b + 1) * 8 + j) * 32); }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint4 v = *reinterpret_cast<const uint4*>(vr + (b * 8 + j) * 32);
          mma_bf16_16816(c0, a[cur][j].x, a8[cur][j].x, a[cur][j].y, a8[cur][j].y, v.x, v.y);
          mma_bf16_16816(c1, a[cur][j].z, a8[cur][j].z, a[cur][j].w, a8[cur][j].w, v.z, v.w);
        }
      }
      // c[0]: (row g, n 2q), c[1]: (row g, n 2q+1), c[2]: (row g+8, n 2q), c[3]: (row g+8, n 2q+1)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int n = 2 * q + (e & 1), rr = t * 16 + g + (e >> 1) * 8;
        if (n < nv) {
          const bf16 val = __float2bfloat16_rn(c0[e] + c1[e]);
          if (sg < 48) P.mod[(size_t)(n * 48 + sg) * 6144 + rr] = val;
          else if (sg == 48) P.outmod[n * 4096 + rr] = val;
          else P.cvec[rr] = val;
        }
      }
    }
    pos = end;
  }
  __syncthreads();
}

// ---------------- noise / control embedding phases
__device__ __forceinline__ void embed_phase1(const Params& P, uint8_t* sbase) {
  float* fv = reinterpret_cast<float*>(sbase);       // [5][512]
  float* cx = fv + 5 * 512;                           // [259]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __syncthreads();
  for (int i = tid; i < 5 * 512; i += 256) fv[i] = P.fourier[i];
  for (int i = tid; i < 259; i += 256) {
    cx[i] = __bfloat162float(i < 2 ? P.mouse[i] : (i < 258 ? P.button[i - 2] : P.scroll[0]));
  }
  __syncthreads();
  const int gw = blockIdx.x * 8 + warp, nw = gridDim.x * 8;
  for (int row = gw; row < 8192; row += nw) {
    const float4* w = reinterpret_cast<const float4*>(P.noise_fc1 + (size_t)row * 512);
    float a[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float4 wv = __ldg(w + lane + 32 * t);
#pragma unroll
      for (int p = 0; p < 5; ++p) {
        const float4 v = *reinterpret_cast<const float4*>(fv + p * 512 + (lane + 32 * t) * 4);
        a[p] += wv.x * v.x + wv.y * v.y + wv.z * v.z + wv.w * v.w;
      }
    }
#pragma unroll
    for (int p = 0; p < 5; ++p) {
      a[p] = warp_sum(a[p]);
      if (lane == 0) P.n_h1[p * 8192 + row] = silu_f(a[p]);
    }
    float c = 0.f;
    {
      bf16 cw[9];
#pragma unroll
      for (int i = 0; i < 9; ++i) { const int k = lane + 32 * i; cw[i] = (k < 259) ? P.ctrl_fc1[(size_t)row * 259 + k] : __float2bfloat16_rn(0.f); }
#pragma unroll
      for (int i = 0; i < 9; ++i) { const int k = lane + 32 * i; if (k < 259) c += cx[k] * __bfloat162float(cw[i]); }
    }
    c = warp_sum(c);
    if (lane == 0) P.ctrl_h1[row] = __float2bfloat16_rn(silu_f(bfr(c)));
  }
  // latent / patch_in init from noise
  for (int i = blockIdx.x * 256 + tid; i < 32 * 32 * 64; i += gridDim.x * 256) {
    const bf16 v = P.noise[i];
    P.latent[i] = v;
    const int c = i >> 11, yy = (i >> 6) & 31, xx = i & 63;
    const int n = (yy >> 1) * 32 + (xx >> 1), j = c * 4 + (yy & 1) * 2 + (xx & 1);
    P.patch_in[n * 128 + j] = v;
  }
}

__device__ __forceinline__ void embed_phase2(const Params& P, uint8_t* sbase) {
  float* nh = reinterpret_cast<float*>(sbase);  // [5][8192]
  float* ch = nh + 5 * 8192;                    // [8192]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __syncthreads();
  for (int i = tid; i < 5 * 8192 / 4; i += 256) reinterpret_cast<float4*>(nh)[i] = reinterpret_cast<const float4*>(P.n_h1)[i];
  for (int i = tid; i < 8192; i += 256) ch[i] = __bfloat162float(P.ctrl_h1[i]);
  __syncthreads();
  const int gw = blockIdx.x * 8 + warp, nw = gridDim.x * 8;
  for (int row = gw; row < 2048; row += nw) {
    const float4* w = reinterpret_cast<const float4*>(P.noise_fc2 + (size_t)row * 8192);
    float a[5] = {0, 0, 0, 0, 0};
    const uint4* cw = reinterpret_cast<const uint4*>(P.ctrl_fc2 + (size_t)row * 8192);
    float c = 0.f;
    // 16 batches: 4 x 16 B noise_fc2 + 2 x 16 B ctrl_fc2 loads in flight per lane, then the FMAs (same per-lane order)
#pragma unroll 1
    for (int bt = 0; bt < 16; ++bt) {
      float4 wv[4];
      uint4 cv[2];
#pragma unroll
      for (int j = 0; j < 4; ++j) wv[j] = __ldg(w + lane + 32 * (bt * 4 + j));
#pragma unroll
      for (int j = 0; j < 2; ++j) cv[j] = __ldg(cw + lane + 32 * (bt * 2 + j));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int t = bt * 4 + j;
#pragma unroll
        for (int p = 0; p < 5; ++p) {
          const float4 v = *reinterpret_cast<const float4*>(nh + p * 8192 + (lane + 32 * t) * 4);
          a[p] += wv[j].x * v.x + wv[j].y * v.y + wv[j].z * v.z + wv[j].w * v.w;
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int t = bt * 2 + j;
        float f[8];
        unpack8(cv[j], f);
        const int k = (lane + 32 * t) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) c += f[i] * ch[k + i];
      }
    }
#pragma unroll
    for (int p = 0; p < 5; ++p) {
      a[p] = warp_sum(a[p]);
      if (lane == 0) P.cond[p * DM + row] = __float2bfloat16_rn(a[p]);
    }
    c = warp_sum(c);
    if (lane == 0) P.ctrl[row] = __float2bfloat16_rn(c);
  }
}

// ---------------- unpatchify (+ latent update); h = silu head output [512, 2048]
// Split-K across CTAs (primdb 1a786bf646): CTA b < 128 = (16-token group g = b >> 2, k slice kb = (b & 3) * 512).
// unpatch_prefetch (start of the head window, before the head row pass) bulk-copies the CTA's contiguous W k-slice
// (512 rows x 128 bf16 = 128 KB) into stage smem in 8 chunks of 16 KB, one mbarrier each (barrier block); W does not depend on h, so
// the load hides behind the row pass and the barrier. unpatch_phase stages its h slice, warp w computes k rows
// [64w, 64w + 64) x 16 tokens x 4 columns per lane from smem, reduces the 8 warps in smem and red.adds the partial
// Y[16][128] (fp32) into acc[0 : 65536), which the head row pass (before the barrier) left zeroed. The last of the 4
// CTAs of a group (counter in acc[65536 + g], also zero) applies bias + latent update and re-zeroes its Y rows and
// counter, so acc is all-zero at phase exit as before. No grid barrier is added.
#define UNP_W_BYTES (512 * 128 * 2)
#define UNP_XS_OFF UNP_W_BYTES
#define UNP_FLAG_OFF (UNP_W_BYTES + 16 * 512 * 4)
// The 8 chunk mbarriers live in spare words of the 1 KB barrier block (GemmBar/AttnBar use < 256 B, tslot at +1024):
// stage smem is a TMA destination in the next phases, and mbarriers left there (never invalidated) hang on re-init.
#define UNP_BAR_OFF (GEMM_STAGE_SMEM + 512)
__device__ __forceinline__ void unpatch_prefetch(const Params& P, uint8_t* sbase) {
  if (blockIdx.x >= 128) return;
  fence_proxy_async();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t bar0 = smem_u32(sbase + UNP_BAR_OFF);
#pragma unroll 1
    for (int c = 0; c < 8; ++c) mbar_init(bar0 + 8 * c, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const bf16* src = P.unpatch_w + (size_t)(blockIdx.x & 3) * 512 * 128;
#pragma unroll 1
    for (int c = 0; c < 8; ++c) {
      mbar_expect_tx(bar0 + 8 * c, 16384);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                       smem_u32(sbase + c * 16384)),
                   "l"(src + c * 64 * 128), "r"(16384), "r"(bar0 + 8 * c)
                   : "memory");
    }
  }
}

__device__ __forceinline__ void unpatch_phase(const Params& P, uint8_t* sbase, float delta) {
  constexpr int TPC = 16, KS = 4, KL = DM / KS;
  const bf16* ws = reinterpret_cast<const bf16*>(sbase);          // [512][128] (W k-slice)
  float* xs = reinterpret_cast<float*>(sbase + UNP_XS_OFF);         // [16][512]
  float* red = reinterpret_cast<float*>(sbase);                     // [8][16][128], aliases ws after the compute
  int* sflag = reinterpret_cast<int*>(sbase + UNP_FLAG_OFF);
  float* Y = P.acc;
  unsigned int* cnt = reinterpret_cast<unsigned int*>(P.acc + 512 * 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __syncthreads();
  if (blockIdx.x < 128) {
    const int g = blockIdx.x >> 2, kb = (blockIdx.x & 3) * KL, n0 = g * TPC;
    for (int i = tid; i < TPC * KL / 8; i += 256) {
      const int t = i >> 6, c = (i & 63) * 8;
      float f[8];
      unpack8(__ldcg(reinterpret_cast<const uint4*>(P.h + (size_t)(n0 + t) * DM + kb + c)), f);
      float4* d = reinterpret_cast<float4*>(xs + t * KL + c);
      d[0] = make_float4(f[0], f[1], f[2], f[3]);
      d[1] = make_float4(f[4], f[5], f[6], f[7]);
    }
    __syncthreads();
    mbar_wait(smem_u32(sbase + UNP_BAR_OFF + 8 * warp), 0);
    float a[TPC][4];
#pragma unroll
    for (int t = 0; t < TPC; ++t) { a[t][0] = 0.f; a[t][1] = 0.f; a[t][2] = 0.f; a[t][3] = 0.f; }
    const int k0 = warp * 64;
#pragma unroll 1
    for (int k = 0; k < 64; k += 4) {
      float w[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint2 r = *reinterpret_cast<const uint2*>(ws + (size_t)(k0 + k + u) * 128 + 4 * lane);
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
        w[u][0] = lo.x; w[u][1] = lo.y; w[u][2] = hi.x; w[u][3] = hi.y;
      }
#pragma unroll
      for (int t = 0; t < TPC; ++t) {
        const float4 x = *reinterpret_cast<const float4*>(xs + t * KL + k0 + k);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          a[t][c] += x.x * w[0][c];
          a[t][c] += x.y * w[1][c];
          a[t][c] += x.z * w[2][c];
          a[t][c] += x.w * w[3][c];
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < TPC; ++t)
      reinterpret_cast<float4*>(red + (warp * TPC + t) * 128)[lane] = make_float4(a[t][0], a[t][1], a[t][2], a[t][3]);
    __syncthreads();
    // 16 x 128 partial outputs = 512 float4, 2 per thread, 8 warp partials each
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = tid + q * 256, t = i >> 5, c4 = (i & 31) * 4;
      float4 s = *reinterpret_cast<const float4*>(red + t * 128 + c4);
#pragma unroll
      for (int r = 1; r < 8; ++r) {
        const float4 v = *reinterpret_cast<const float4*>(red + (r * TPC + t) * 128 + c4);
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      red_add_v4(Y + (size_t)(n0 + t) * 128 + c4, s);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) *sflag = (atomicAdd(cnt + g, 1u) == KS - 1);
    __syncthreads();
    if (*sflag) {
      // last arriver: 16 x 128 outputs as 512 groups of 4 columns (j = 4c' .. 4c'+3 = one channel c, (ph, pw) in
      // {00, 01, 10, 11}: latent pairs li, li + 1 and li + 64, li + 65); 2 groups per thread, all loads before stores.
      __threadfence();
      float4 yv[2];
      __nv_bfloat162 l0[2], l1[2];
      float bias[2];
      int li[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int i = tid + q * 256, t = i >> 5, c = i & 31, n = n0 + t, yy = n >> 5, xx = n & 31;
        li[q] = c * 2048 + (2 * yy) * 64 + 2 * xx;
        yv[q] = __ldcg(reinterpret_cast<const float4*>(Y + (size_t)n * 128 + 4 * c));
        l0[q] = __ldcg(reinterpret_cast<const __nv_bfloat162*>(P.latent + li[q]));
        l1[q] = __ldcg(reinterpret_cast<const __nv_bfloat162*>(P.latent + li[q] + 64));
        bias[q] = __bfloat162float(P.unpatch_b[c]);
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int i = tid + q * 256, t = i >> 5, c = i & 31, n = n0 + t;
        reinterpret_cast<float4*>(Y + (size_t)n * 128 + 4 * c)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        const float y4[4] = {yv[q].x, yv[q].y, yv[q].z, yv[q].w};
        const float2 a = __bfloat1622float2(l0[q]), b = __bfloat1622float2(l1[q]);
        const float lo[4] = {a.x, a.y, b.x, b.y};
        __nv_bfloat16 nl[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float v = bfr(y4[r] + bias[q]);
          nl[r] = __float2bfloat16_rn(lo[r] + bfr(delta * v));
        }
        *reinterpret_cast<__nv_bfloat162*>(P.latent + li[q]) = __halves2bfloat162(nl[0], nl[1]);
        *reinterpret_cast<__nv_bfloat162*>(P.latent + li[q] + 64) = __halves2bfloat162(nl[2], nl[3]);
        uint2 pk;
        *reinterpret_cast<__nv_bfloat162*>(&pk.x) = __halves2bfloat162(nl[0], nl[1]);
        *reinterpret_cast<__nv_bfloat162*>(&pk.y) = __halves2bfloat162(nl[2], nl[3]);
        *reinterpret_cast<uint2*>(P.patch_in + (size_t)n * 128 + 4 * c) = pk;
      }
      if (tid == 0) cnt[g] = 0u;
    }
    fence_proxy_async();
  }
}
