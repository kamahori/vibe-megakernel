// Small-token GEMM on the tcgen05 tensor core, swap-AB: Y[M][N] = X[M][K] * W[N][K]^T, M = tokens (1..64).
// Companion of gemm_small.cu (same shapes, token counts, cold / warm L2 conditions, reference, tolerance and
// graph timing; this file includes it for that harness). W rows are the MMA M = 128 operand, the tokens are the
// MMA N, padded to 16 / 32 / 64. One CTA = 256 threads, 1 CTA / SM, accumulator in TMEM.
//   bf16 W   cp.async 16 B (all threads) global -> swizzled smem (SWIZZLE_128B, K-major), ring of S stages of
//            128 k; X rows (the same 128 k) are cp.async'd into the B slot of every stage.
//   int8 W   int8 rows are cp.async'd raw into a staging ring, each thread converts its own 16 B chunks to bf16
//            (PRMT + 2^23 FADD magic, exact) into the swizzled A slot, fence.proxy.async, then the same bf16 MMA.
//            The per-row fp32 scale (= TMEM lane = W row) is applied to the fp32 accumulator in the epilogue.
//            kind::i8 with int8 activations is NOT measured here (it needs quantised X, a different numeric contract;
//            tc_mma.cu has the kind::i8 instruction rates).
// Work partition over CTAs (the item this file also measures). The K space is KI = K / 128 iterations per output
// tile (128 W rows); the flattened space has ntiles x KI iterations. A CTA b of G owns the contiguous range
// [b T / G, (b + 1) T / G) of it and walks it as segments (one segment = one tile, a k range):
//   output_tile  G = ntiles          every segment is a whole tile, direct bf16 store, no reduction
//   split_k      G = ntiles x ks     equal k slices aligned to tiles (ks | KI)
//   stream_k     G = 148 (SM count)  ranges unaligned to tiles; a tile can have 2+ contributors
// Reduction of partial tiles (any tile with more than one contributor):
//   fixup_last_arriver   fp32 partials to a workspace [tile][contributor][token][128 rows], threadfence, per-tile
//                        counter; the last CTA to arrive sums the partials, scales, stores bf16, resets the counter
//   atomic_red_add       red.global.add.f32 into a zeroed fp32 workspace [token][N], then a second kernel (one CTA per
//                        multi-contributor tile) converts to bf16 (scale) and re-zeroes the workspace
// Every candidate is checked against the fp32 reference before timing; failures are dropped.
// Rows: primitive gemm.primbench (best candidate over everything, comparable with the ffma / mma_sync arms) and
// gemm.partition (best candidate per partition class and reduction).
// Usage: ./gemm_small_tc [shape_idx]
#define GEMM_SMALL_LIB
#include "gemm_small.cu"
#include <map>

constexpr int KC = 128;  // k per pipeline stage

__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
// kind::f16, bf16 x bf16 -> fp32, K-major A and B
__device__ __forceinline__ uint32_t make_idesc_bf16(int M, int N) {
  return 1u << 4 | 1u << 7 | 1u << 10 | (uint32_t)(N >> 3) << 17 | (uint32_t)(M >> 4) << 24;
}
__device__ __forceinline__ void mma_bf16_ss(uint32_t dt, uint64_t ad, uint64_t bd, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(dt), "l"(ad), "l"(bd), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void mbar_wait_p(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n"
               :: "r"(mbar), "r"(phase));
}
__device__ __forceinline__ void tc_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
}
template <int N> __device__ __forceinline__ void cp_wait_n() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void commit_to(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar));
}

// CTA b of G owns flattened iterations [start(b), start(b + 1)); T = ntiles * KI
__host__ __device__ __forceinline__ long long part_start(long long b, long long T, int G) { return b * T / G; }
// CTAs whose range touches tile t: first one and the count
__host__ __device__ __forceinline__ void tile_contrib(int t, int KI, long long T, int G, int& b0, int& nc) {
  long long lo = (long long)t * KI, hi = lo + KI;
  int b = (int)(lo * G / T);
  if (b > 0) --b;
  while (part_start(b + 1, T, G) <= lo) ++b;
  b0 = b;
  int e = b;
  while (e < G && part_start(e, T, G) < hi) ++e;
  nc = e - b;
}

template <int WT, int NP, int S>
struct TcCfg {
  static constexpr int A_BYTES = 128 * KC * 2, B_BYTES = NP * KC * 2, STAGE = A_BYTES + B_BYTES;
  static constexpr int RAW = WT == WI8 ? 128 * KC : 0;
  static constexpr size_t SMEM = 1024 + (size_t)S * (STAGE + RAW) + 256;
};

template <int WT, int NP, int S>
__global__ void __launch_bounds__(256, 1)
k_tc(const uint8_t* __restrict__ W, const float* __restrict__ scale, const bf16* __restrict__ X, bf16* __restrict__ Y,
     float* ws, int* cnt, int M, int N, int K, int G, int maxc, int atomic_red) {
  using C = TcCfg<WT, NP, S>;
  constexpr int ESZ = WT == WBF16 ? 2 : 1;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* raw0 = base + (size_t)S * C::STAGE;
  uint64_t* mb = (uint64_t*)(base + (size_t)S * (C::STAGE + C::RAW));  // free[S], done
  uint32_t* tslot = (uint32_t*)(mb + S + 1);
  int* flag = (int*)(tslot + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = blockIdx.x;
  const int KI = K / KC, ntiles = N / 128;
  const long long T = (long long)ntiles * KI;
  const long long g0 = part_start(b, T, G), g1 = part_start(b + 1, T, G);
  const int n = (int)(g1 - g0);

  if (tid == 0) {
    for (int s = 0; s <= S; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(mb + s)));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" :: "r"(smem_u32(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_sync();
  const uint32_t tmem = *tslot;
  const uint32_t idesc = make_idesc_bf16(128, NP);

  auto stage_a = [&](int s) { return base + (size_t)s * C::STAGE; };
  auto stage_b = [&](int s) { return base + (size_t)s * C::STAGE + C::A_BYTES; };
  auto issue = [&](int j) {
    if (j < n) {
      const int g = (int)(g0 + j), tile = g / KI, kc = g % KI, s = j % S;
      const uint8_t* Wt = W + ((size_t)tile * 128 * K + (size_t)kc * KC) * ESZ;
      if constexpr (WT == WBF16) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int i = tid + 256 * q, row = i >> 4, c16 = i & 15, atom = c16 >> 3, c = c16 & 7;
          cp_async16(stage_a(s) + atom * 16384 + row * 128 + ((c ^ (row & 7)) << 4), Wt + (size_t)row * K * 2 + c16 * 16);
        }
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int i = tid + 256 * q, row = i >> 3, j8 = i & 7;
          cp_async16(raw0 + (size_t)s * C::RAW + i * 16, Wt + (size_t)row * K + j8 * 16);
        }
      }
      for (int i = tid; i < NP * 16; i += 256) {
        const int row = i >> 4, c16 = i & 15, atom = c16 >> 3, c = c16 & 7;
        cp_async16(stage_b(s) + atom * (NP * 128) + row * 128 + ((c ^ (row & 7)) << 4), X + (size_t)row * K + (size_t)kc * KC + c16 * 8);
      }
    }
    cp_commit();
  };
  for (int j = 0; j < S - 1; ++j) issue(j);

  int seg_k0 = 0, dph = 0;
  for (int j = 0; j < n; ++j) {
    const int g = (int)(g0 + j), tile = g / KI, kc = g % KI, s = j % S;
    cp_wait_n<S - 2>();
    if constexpr (WT == WI8) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int i = tid + 256 * q, row = i >> 3, j8 = i & 7;
        const uint4 r = *reinterpret_cast<const uint4*>(raw0 + (size_t)s * C::RAW + i * 16);
        uint4 v0, v1;
        i8x4_bf16(r.x, v0.x, v0.y); i8x4_bf16(r.y, v0.z, v0.w);
        i8x4_bf16(r.z, v1.x, v1.y); i8x4_bf16(r.w, v1.z, v1.w);
        const int c = (2 * j8) & 7, atom = j8 >> 2;
        uint8_t* d = stage_a(s) + atom * 16384 + row * 128;
        *reinterpret_cast<uint4*>(d + ((c ^ (row & 7)) << 4)) = v0;
        *reinterpret_cast<uint4*>(d + (((c + 1) ^ (row & 7)) << 4)) = v1;
      }
    }
    asm volatile("fence.proxy.async.shared::cta;");
    tc_sync();
    const bool seg_start = (j == 0) || kc == 0, seg_end = (kc == KI - 1) || (j == n - 1);
    if (seg_start) seg_k0 = kc;
    if (tid == 0) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint64_t ad = make_sdesc(smem_u32(stage_a(s)) + (kk >> 2) * 16384 + (kk & 3) * 32);
        const uint64_t bd = make_sdesc(smem_u32(stage_b(s)) + (kk >> 2) * (NP * 128) + (kk & 3) * 32);
        mma_bf16_ss(tmem, ad, bd, idesc, (seg_start && kk == 0) ? 0u : 1u);
      }
      commit_to(smem_u32(mb + s));
      if (seg_end) commit_to(smem_u32(mb + S));
    }
    // refill the slot of iteration j - 1 (its MMAs must be done) with iteration j + S - 1
    if (j + S - 1 < n) {
      if (j >= 1) mbar_wait_p(smem_u32(mb + (j - 1) % S), ((j - 1) / S) & 1);
    }
    issue(j + S - 1);
    if (seg_end) {
      mbar_wait_p(smem_u32(mb + S), dph);
      dph ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;");
      const bool full = seg_k0 == 0 && kc == KI - 1;
      int b0, nc;
      tile_contrib(tile, KI, T, G, b0, nc);
      const int slot = b - b0;
      if (warp < 4) {
        const int rowl = warp * 32 + lane, row = tile * 128 + rowl;
        const float sc = WT == WI8 ? scale[row] : 1.f;
#pragma unroll
        for (int c = 0; c < NP / 8; ++c) {
          uint32_t r[8];
          const uint32_t ta = tmem + ((uint32_t)(warp * 32) << 16) + c * 8;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                       : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(ta));
          asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int t = c * 8 + e;
            const float v = __uint_as_float(r[e]);
            if (t < M) {
              if (full) Y[(size_t)t * N + row] = __float2bfloat16_rn(v * sc);
              else if (atomic_red) atomicAdd(ws + (size_t)t * N + row, v);
              else ws[((size_t)(tile * maxc + slot) * NP + t) * 128 + rowl] = v;
            }
          }
        }
      }
      if (!full && !atomic_red) {
        __threadfence();
        __syncthreads();
        if (tid == 0) { const int old = atomicAdd(cnt + tile, 1); *flag = (old == nc - 1); if (old == nc - 1) cnt[tile] = 0; }
        __syncthreads();
        if (*flag) {
          __threadfence();
          if (tid < 128) {
            const int row = tile * 128 + tid;
            const float sc = WT == WI8 ? scale[row] : 1.f;
            for (int t = 0; t < M; ++t) {
              float a = 0.f;
              for (int q = 0; q < nc; ++q) a += __ldcg(ws + ((size_t)(tile * maxc + q) * NP + t) * 128 + tid);
              Y[(size_t)t * N + row] = __float2bfloat16_rn(a * sc);
            }
          }
        }
      }
    }
  }
  tc_sync();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(tmem));
}

// atomic_red_add second pass: one CTA per tile with more than one contributor
__global__ void __launch_bounds__(128) k_finalize(const float* __restrict__ scale, float* ws, bf16* Y, int M, int N, int K, int G, int wi8) {
  const int tile = blockIdx.x, KI = K / KC;
  const long long T = (long long)(N / 128) * KI;
  int b0, nc;
  tile_contrib(tile, KI, T, G, b0, nc);
  if (nc <= 1) return;
  const int row = tile * 128 + threadIdx.x;
  const float sc = wi8 ? scale[row] : 1.f;
  for (int t = 0; t < M; ++t) {
    const float v = ws[(size_t)t * N + row];
    ws[(size_t)t * N + row] = 0.f;
    Y[(size_t)t * N + row] = __float2bfloat16_rn(v * sc);
  }
}

// ---------------------------------------------------------------- candidates
struct TCand : Cand { std::string part, red; int ks = 0; };
struct TcMem { std::vector<void*> ptrs; ~TcMem() { for (auto p : ptrs) cudaFree(p); } };

template <int WT, int NP, int S>
static void reg_tc(std::vector<TCand>& v, const Ctx& cx, TcMem& mem, int G, int ks, const char* part, int atomic_red) {
  using C = TcCfg<WT, NP, S>;
  const int ntiles = cx.N / 128, KI = cx.K / KC;
  const long long T = (long long)ntiles * KI;
  if (cx.K % KC || cx.N % 128 || G > T || C::SMEM > 227 * 1024) return;
  int maxc = 1;
  for (int t = 0; t < ntiles; ++t) { int b0, nc; tile_contrib(t, KI, T, G, b0, nc); maxc = std::max(maxc, nc); }
  if (atomic_red && maxc == 1) return;  // nothing to reduce: same as the fixup candidate
  auto fn = k_tc<WT, NP, S>;
  CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM));
  cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, fn));
  float* ws = nullptr; int* cnt = nullptr;
  if (maxc > 1) {
    size_t wsb = atomic_red ? (size_t)64 * cx.N * 4 : (size_t)ntiles * maxc * NP * 128 * 4;
    CK(cudaMalloc(&ws, wsb)); CK(cudaMemset(ws, 0, wsb)); mem.ptrs.push_back(ws);
    CK(cudaMalloc(&cnt, ntiles * 4)); CK(cudaMemset(cnt, 0, ntiles * 4)); mem.ptrs.push_back(cnt);
  }
  TCand c;
  c.part = part; c.ks = ks; c.red = maxc == 1 ? "none" : atomic_red ? "atomic_red_add+finalize_kernel" : "fixup_last_arriver";
  c.cfg = std::string("tcgen05 swapAB M128xN") + std::to_string(NP) + " kstage=128 stages=" + std::to_string(S) + " ctas=" + std::to_string(G) +
          " part=" + part + " ks=" + std::to_string(ks) + " red=" + c.red;
  c.sched = std::string("swap_ab cp.async_ring") + (WT == WI8 ? "+i8_to_bf16_smem" : "") + " tmem_acc";
  c.rows = 128; c.tile_n = NP; c.grid = G; c.threads = 256; c.smem = C::SMEM; c.regs = a.numRegs; c.local = (int)a.localSizeBytes;
  const int wi8 = WT == WI8;
  c.launch = [=](cudaStream_t st, const void* Wp) {
    fn<<<G, 256, C::SMEM, st>>>((const uint8_t*)Wp, cx.scale, cx.X, cx.Y, ws, cnt, cx.M, cx.N, cx.K, G, maxc, atomic_red);
    if (atomic_red && maxc > 1) k_finalize<<<ntiles, 128, 0, st>>>(cx.scale, ws, cx.Y, cx.M, cx.N, cx.K, G, wi8);
  };
  v.push_back(c);
}

template <int WT, int NP, int S>
static void reg_tc_all(std::vector<TCand>& v, const Ctx& cx, TcMem& mem) {
  const int ntiles = cx.N / 128, KI = cx.K / KC;
  reg_tc<WT, NP, S>(v, cx, mem, ntiles, 1, "output_tile", 0);
  for (int ks : {2, 3, 4, 6, 8, 12, 16}) {
    if (KI % ks || ntiles * ks > 4 * 148) continue;
    for (int at = 0; at < 2; ++at) reg_tc<WT, NP, S>(v, cx, mem, ntiles * ks, ks, "split_k", at);
  }
  for (int at = 0; at < 2; ++at) reg_tc<WT, NP, S>(v, cx, mem, 148, 0, "stream_k", at);
}
template <int WT>
static std::vector<TCand> tc_candidates(const Ctx& cx, TcMem& mem) {
  std::vector<TCand> v;
  constexpr int S = WT == WBF16 ? 4 : 3;
  if (cx.M <= 16) reg_tc_all<WT, 16, S>(v, cx, mem);
  else if (cx.M <= 32) reg_tc_all<WT, 32, S>(v, cx, mem);
  else reg_tc_all<WT, 64, S>(v, cx, mem);
  return v;
}

int main(int argc, char** argv) {
  const int NS = 4, shapes[NS][2] = {{4096, 4096}, {8192, 8192}, {28672, 4096}, {4096, 14336}};
  const int Ms[] = {1, 2, 4, 8, 16, 32, 64};
  int only = argc > 1 ? atoi(argv[1]) : -1;
  const double mhz = measure_sm_clock();
  fprintf(stderr, "sm clock (spin) %.0f MHz\n", mhz);
  for (int si = 0; si < NS; ++si) {
    if (only >= 0 && si != only) continue;
    const int N = shapes[si][0], K = shapes[si][1];
    for (int wt = 0; wt < 2; ++wt) {
      const size_t wbytes = (size_t)N * K * (wt == WBF16 ? 2 : 1);
      const int ncopy = (int)((576ull << 20) / wbytes + 1);
      std::vector<void*> copies(ncopy);
      for (auto& p : copies) CK(cudaMalloc(&p, wbytes));
      k_init_w<<<2048, 256>>>(copies[0], wbytes / (wt == WBF16 ? 2 : 1), wt);
      for (int i = 1; i < ncopy; ++i) CK(cudaMemcpy(copies[i], copies[0], wbytes, cudaMemcpyDeviceToDevice));
      bf16* X; float *scale, *refd; bf16* Y;
      CK(cudaMalloc(&X, (size_t)64 * K * 2)); CK(cudaMalloc(&scale, N * 4)); CK(cudaMalloc(&refd, (size_t)64 * N * 4));
      CK(cudaMalloc(&Y, (size_t)64 * N * 2));
      k_init_x<<<256, 256>>>(X, (size_t)64 * K); k_init_scale<<<(N + 255) / 256, 256>>>(scale, N);
      if (wt == WBF16) k_ref<WBF16><<<N, 256>>>(copies[0], scale, X, refd, N, K);
      else k_ref<WI8><<<N, 256>>>(copies[0], scale, X, refd, N, K);
      CK(cudaDeviceSynchronize());
      std::vector<float> ref((size_t)64 * N); CK(cudaMemcpy(ref.data(), refd, ref.size() * 4, cudaMemcpyDeviceToHost));
      const char* wname = wt == WBF16 ? "bf16" : "int8";
      for (int M : Ms) {
        Ctx cx{scale, X, Y, M, N, K};
        TcMem mem;
        std::vector<TCand> cs = wt == WBF16 ? tc_candidates<WBF16>(cx, mem) : tc_candidates<WI8>(cx, mem);
        std::vector<TCand> ok; std::vector<double> errs;
        std::vector<uint16_t> y((size_t)M * N);
        for (auto& c : cs) {
          CK(cudaMemset(Y, 0xFF, (size_t)M * N * 2));
          c.launch(0, copies[0]);
          cudaError_t e0 = cudaGetLastError(), e1 = cudaDeviceSynchronize();
          if (e0 != cudaSuccess || e1 != cudaSuccess) { fprintf(stderr, "CUDA error in %s: %s %s\n", c.cfg.c_str(), cudaGetErrorString(e0), cudaGetErrorString(e1)); exit(1); }
          CK(cudaMemcpy(y.data(), Y, y.size() * 2, cudaMemcpyDeviceToHost));
          double e; bool pass = verify(y, ref, M, N, e);
          if (!pass) fprintf(stderr, "DROP tcgen05 %s M=%d %dx%d: %s err=%g\n", wname, M, N, K, c.cfg.c_str(), e);
          else { ok.push_back(c); errs.push_back(e); }
        }
        for (int l2 = 0; l2 < 2; ++l2) {
          std::vector<void*> cp = l2 ? std::vector<void*>{copies[0]} : copies;
          int L = l2 ? 16 : std::max(ncopy, 12);
          std::vector<double> ts(ok.size()); std::vector<std::vector<double>> reps(ok.size());
          for (size_t i = 0; i < ok.size(); ++i) ts[i] = time_cand(ok[i], cp, L, reps[i]);
          // rows: overall best (gemm.primbench), best per (partition, reduction) (gemm.partition)
          std::map<std::string, int> bestc; int best = -1;
          for (size_t i = 0; i < ok.size(); ++i) {
            if (best < 0 || ts[i] < ts[best]) best = (int)i;
            std::string k = ok[i].part + "|" + ok[i].red;
            if (!bestc.count(k) || ts[i] < ts[bestc[k]]) bestc[k] = (int)i;
          }
          if (best < 0) { fprintf(stderr, "NO valid candidate tcgen05 %s M=%d %dx%d\n", wname, M, N, K); continue; }
          auto emit = [&](int i, const char* prim) {
            const TCand& c = ok[i]; const double bt = ts[i];
            double gbps = (double)wbytes / (bt * 1e-6) / 1e9, tflops = 2.0 * M * N * K / (bt * 1e-6) / 1e12;
            bool sane = l2 == 1 || gbps < 8000;
            if (!sane) fprintf(stderr, "SUSPECT cold run above HBM peak: tcgen05 %s M=%d %dx%d %.0f GB/s\n", wname, M, N, K, gbps);
            Json j;
            j.s("family", "gemm").s("primitive", prim).s("p.engine", "tcgen05_swapab").s("p.w_dtype", wname).s("p.x_dtype", "bf16")
                .i("p.M", M).i("p.N", N).i("p.K", K).s("p.config", c.cfg).i("p.tile_m", 128).i("p.tile_n", c.tile_n)
                .s("p.sched", c.sched).s("p.partition", c.part).i("p.ks", c.ks).s("p.reduction", c.red)
                .s("p.dequant", wt == WI8 ? "i8_to_bf16_smem_prmt_magic" : "none")
                .f("m.us", bt).f("m.tflops", tflops).f("m.hbm_gbps", gbps).f("m.max_abs_err", errs[i])
                .i("m.regs", c.regs).i("m.local_bytes", c.local)
                .s("c.l2", l2 ? "warm" : "cold").f("c.sm_clock_mhz", mhz).i("c.ctas", c.grid).i("c.threads", c.threads)
                .i("c.smem", (long long)c.smem)
                .t("verified", sane).i("n_reps", (long long)reps[i].size()).f("cv", cv(reps[i]))
                .s("s.bench", "primbench/gemm_small_tc.cu").s("s.timer", "cudaEvent_graph")
                .s("note", std::string("Y[M][N] = X[M][K] W[N][K]^T, M = tokens padded to N=16/32/64 of the MMA; tcgen05.mma kind::f16 swap-AB (W rows = MMA M 128). " "Simple stage handshake, not tuned (no warp specialization, no overlap of the epilogue): a lower bound on what tcgen05 reaches here; compare primitive gemm.cutlass_tcgen05 for a tuned kernel. ") +
                               (wt == WI8 ? "int8 W with per-row fp32 scale in the fp32 epilogue; int8 -> bf16 by PRMT+2^23 FADD into swizzled smem before the MMA (kind::i8 not used). " : "") +
                               "partition / ks / reduction: see gemm_small_tc.cu header (CTA = contiguous range of the flattened k-iteration space). "
                               "us = CUDA-graph average over back-to-back launches (includes the inter-kernel gap, ~1-2 us; the atomic reduction adds a second kernel). " +
                               (l2 ? "L2 warm: same W every launch (W <= ~100 MB can be L2 resident, then hbm_gbps is not HBM)."
                                   : "L2 cold: W copies rotate over >= 576 MB."));
            j.print();
          };
          emit(best, "gemm.primbench");
          for (auto& kv : bestc) emit(kv.second, "gemm.partition");
        }
      }
      for (auto p : copies) cudaFree(p);
      cudaFree(X); cudaFree(scale); cudaFree(refd); cudaFree(Y);
      fprintf(stderr, "done %dx%d %s\n", N, K, wname);
    }
  }
  return 0;
}
