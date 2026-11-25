---
name: cdna-matrix-core-design
description: Use when planning or reviewing a matrix-shaped megakernel phase (batch-1 GEMV, several tokens from speculative decoding, MoE expert batches or batched decode, attention products, dequantized weights) on AMD CDNA (MI300X gfx942, MI210 gfx90a, MI350X gfx950), to put MFMA, VALU and dot-instruction designs into the candidate set and decide among them by measurement.
---

# CDNA matrix-core and vector-engine design

Use this skill with `megakernel-design` (section 5) on the rocm backend. It lists options and
the questions that decide them. Rows quoted below are in `./primdb` (filter with
`arch=gfx942` or `arch=gfx90a`); the probes are `primbench/amd/mfma_amd.hip` and
`primbench/amd/gemv_amd.hip`. Every GEMV row was checked against an fp32 CPU reference.

## Instructions available

| | gfx90a (MI210) | gfx942 (MI300X) | gfx950 (MI350X, not measured here) |
|---|---|---|---|
| bf16/f16 MFMA | 32x32x8, 16x16x16, 4x4x4 (16 blocks) | same | adds 32x32x16, 16x16x32 |
| fp8 MFMA | none | 16x16x32, 32x32x16, **e4m3fnuz** (1.0 = 0x40) | OCP e4m3 (1.0 = 0x38), plus scaled f8f6f4 |
| int8 MFMA | 16x16x16 | 16x16x32 | 16x16x64 |
| `v_dot2_f32_f16` (`__builtin_amdgcn_fdot2`) | yes | yes | yes |
| `v_dot2_f32_bf16` | no | **no** (does not compile) | yes |
| packed fp32 `v_pk_fma_f32` | yes | yes | yes |

- An MFMA takes A and B fragments from VGPRs of all 64 lanes and accumulates in
  AGPRs or VGPRs. The fragment layouts are in the AMD matrix instruction calculator and the CDNA3
  ISA guide; `gemv_amd.hip` has worked layouts for 16x16x16 and 4x4x4. Test a layout with
  distinctive data, not all-ones.
- A row and a column of a 16x16x16 tile come from lanes `l % 16` and `l / 16`, so an x vector
  broadcast to all 16 columns wastes 15/16 of the MFMA at batch 1.

Measured instruction costs (`./primdb q mfma arch=gfx942`; one wave per SIMD; `latency_cyc`
is a dependent chain, `issue_cyc` four independent accumulators):

<!-- MI300X_MFMA_TABLE -->
| instruction | MI300X latency cyc | MI300X issue cyc | MI300X TFLOP/s | MI210 latency cyc | MI210 issue cyc | MI210 TFLOP/s |
|---|---|---|---|---|---|---|
| `v_mfma_f32_32x32x8_bf16` | 44 | 32 | 1216 | 64 | 64 | 169 |
| `v_mfma_f32_16x16x16_bf16` | 44 | 18 | 1039 | 44 | 32 | 169 |
| `v_mfma_f32_4x4x4_16b_bf16` | 40 | 14 | 346 | 40 | 14 | 92 |
| `v_mfma_f32_32x32x8_f16` | - | 32 | 1218 | - | 64 | 169 |
| `v_mfma_f32_16x16x16_f16` | - | 18 | 940 | - | 32 | 169 |
| `v_mfma_f32_16x16x32_fp8_fp8` | 44 | 18 | 2167 | - | - | - |
| `v_mfma_f32_32x32x16_fp8_fp8` | 44 | 32 | 2333 | - | - | - |
| `v_mfma_i32_16x16x32_i8` | - | 18 | 1827 | - | - | - |

`wave_sum.shfl_xor` on MI210: 452 cycles

`wave_sum.dpp_readlane` on MI210: 108 cycles

`wave_sum.shfl_xor` on MI300X: 452 cycles

`wave_sum.dpp_readlane` on MI300X: 128 cycles
<!-- /MI300X_MFMA_TABLE -->

## Batch-1 GEMV: engine choice

A decode GEMV reads each weight once, so its floor is bytes / bandwidth. The engine matters
through what it costs per byte: instruction issue, reduction, register pressure, and how many
waves have work.

Measured cold single-GEMV time on MI300X (`./primdb q gemv arch=gfx942 M=1`, one block per
CU, best thread count per engine):

<!-- MI300X_GEMV_TABLE -->
| N x K (bf16) | HBM floor us | valu_r1 | valu_r4 | mfma16 | mfma4 |
|---|---|---|---|---|---|
| 2048 x 768 | 0.6 | **3.3 (512)** | 3.8 (512) | 11.7 (256) | 5.3 (256) |
| 1024 x 2048 | 0.8 | **4.2 (256)** | 5.3 (512) | 28.0 (256) | 10.6 (512) |
| 1536 x 2048 | 1.2 | **4.5 (512)** | 5.7 (512) | 28.3 (256) | 10.7 (512) |
| 1024 x 3072 | 1.2 | **5.2 (256)** | 6.2 (256) | 40.5 (256) | 14.2 (256) |
| 4096 x 1024 | 1.6 | **4.7 (512)** | 5.3 (512) | 15.4 (256) | 7.2 (512) |
| 6144 x 1024 | 2.4 | **5.9 (512)** | 6.2 (512) | 15.7 (256) | 7.5 (512) |
| 6144 x 4096 | 9.5 | 18.0 (1024) | **18.0 (512)** | 57.2 (256) | 23.0 (512) |
| 151936 x 1024 | 58.7 | 80.8 (1024) | **79.5 (512)** | 109.3 (1024) | 81.7 (1024) |

us = one cold GEMV, in-kernel span (no launch overhead); (threads per block); bold = fastest.
<!-- /MI300X_GEMV_TABLE -->

On MI300X the wave-per-row VALU engine was fastest on every M = 1 shape: `mfma4` was
1.0-2.6x slower and `mfma16` 1.4-8x slower (worst at N = 1024). Small shapes stay 3-5x above
the HBM floor for every engine; this is fixed cost (work distribution, reduction, tail), not
bandwidth.

Engines in the rows:
- `valu_r1` / `valu_r4`: one wave per 1 or 4 rows, 16 B loads per lane, fp32 FMA, DPP
  reduction across the wave.
- `mfma16`: one wave per 16-row tile, `v_mfma_f32_16x16x16_bf16` with x broadcast.
- `mfma4`: one wave per 4-row tile, `v_mfma_f32_4x4x4_16b_bf16`, the 16 blocks splitting K.

Questions:
- How many work units does the phase have per CU? A 16-row tile per wave leaves most waves idle
  when N is small (N = 1024 is 64 tiles for 304 CUs). The rows show this as the largest losses.
- Does the engine remove work that VALU pays for? MFMA needs no cross-lane reduction; VALU
  needs one per row (DPP: `./primdb q wave_sum.`).
- What does the engine do to registers? MFMA accumulators can live in AGPRs. At 256 threads one
  wave per SIMD has up to 512 VGPR + AGPR; at 1024 threads the cap is 128 and earlier kernels
  spilled.
- Earlier MI300X runs: MFMA (4x4x4 tiles) for gate/up gained 10-15% per byte in isolation and
  about 0.07 ms in the kernel; MFMA for QKV lost 0.05-0.09 ms in every variant. Tile balance
  (the most tiles any wave gets) decided it.

## Several tokens (speculative decoding, MoE expert batches, batched decode: M = 2..128)

With M tokens, each weight byte feeds M multiply-adds. VALU cost grows with M; MFMA cost
does not until M fills the tile.

Measured on MI300X (`./primdb q gemv arch=gfx942 M=4,8,16,32,64,128`; MI210 rows: `arch=gfx90a`):

<!-- MI300X_SMALLM_TABLE -->
| N x K | M | valu_r1 us | valu_r4 us | mfma16 us | mfma_lds_sk us | mfma_lds_db us |
|---|---|---|---|---|---|---|
| 1536 x 2048 | 1 | **4.5** | 5.7 | 28.3 | 6.3 | 6.1 |
| 1536 x 2048 | 4 | **6.0** | 8.5 | 30.1 | 6.4 | 6.3 |
| 1536 x 2048 | 8 | 6.4 | - | 31.5 | **6.4** | 6.5 |
| 1536 x 2048 | 16 | 9.4 | 17.0 | 32.6 | 8.2 | **7.8** |
| 1536 x 2048 | 32 | 16.3 | 35.1 | 42.9 | 12.5 | **12.4** |
| 1536 x 2048 | 64 | 29.9 | 60.1 | 77.0 | 20.8 | **20.7** |
| 1536 x 2048 | 128 | 55.0 | 115.2 | 99.0 | **38.2** | 50.0 |
| 2048 x 768 | 1 | **3.3** | 3.8 | 11.7 | 4.2 | 4.4 |
| 2048 x 768 | 4 | **3.9** | 6.0 | 13.0 | 4.6 | 4.6 |
| 2048 x 768 | 8 | **4.6** | - | 13.0 | 5.2 | 5.5 |
| 2048 x 768 | 16 | 6.6 | 11.2 | 13.5 | 5.1 | **4.9** |
| 2048 x 768 | 32 | 10.6 | 21.5 | 17.5 | 6.7 | **6.0** |
| 2048 x 768 | 64 | 19.3 | 38.4 | 29.8 | **10.7** | 11.8 |
| 2048 x 768 | 128 | 34.9 | 74.0 | 39.1 | **19.6** | 36.1 |
| 4096 x 1024 | 1 | **4.7** | 5.3 | 15.4 | 6.4 | 6.7 |
| 4096 x 1024 | 4 | **6.3** | 7.9 | 15.9 | 6.9 | 6.8 |
| 4096 x 1024 | 8 | 7.6 | - | 16.9 | 7.2 | **7.0** |
| 4096 x 1024 | 16 | 11.6 | 14.1 | 17.4 | 9.3 | **9.2** |
| 4096 x 1024 | 32 | 21.2 | 26.8 | 22.6 | **14.8** | 15.1 |
| 4096 x 1024 | 64 | 39.1 | 48.2 | 39.7 | **26.2** | 26.7 |
| 4096 x 1024 | 128 | 73.9 | 86.0 | 51.5 | **49.0** | 84.9 |
| 6144 x 4096 | 1 | 18.0 | **18.0** | 57.2 | 21.5 | 22.6 |
| 6144 x 4096 | 4 | 26.6 | 23.3 | 65.0 | **22.3** | 23.4 |
| 6144 x 4096 | 8 | 31.4 | - | 61.9 | **24.2** | 24.9 |
| 6144 x 4096 | 16 | 52.2 | 45.0 | 63.3 | 27.5 | **26.9** |
| 6144 x 4096 | 32 | 100.2 | 81.7 | 83.4 | 38.9 | **33.9** |
| 6144 x 4096 | 64 | 191.3 | 153.8 | 151.1 | 62.5 | **50.2** |
| 6144 x 4096 | 128 | 354.7 | 302.4 | 247.5 | 105.7 | **88.4** |

us = one cold GEMV, best thread count; bold = fastest. `mfma_lds_*`: x staged in LDS per k-chunk, 16x16x16 MFMA, split-K with fp32 atomics; `_db` double-buffers the chunk (64-k chunks at M = 128).
<!-- /MI300X_SMALLM_TABLE -->

The MFMA layout decides the answer, not the instruction:
- `mfma16` reads x from global in every wave and gives each wave one 16-row tile, so the
  1536- and 2048-row shapes keep most waves idle. Against it VALU looks better up to M = 16
  on MI300X.
- `mfma_lds_sk` stages x[0..M) for one 128-k chunk in LDS once per block, and splits K across
  blocks until there is at least one task per block, adding partials with fp32 atomics. On
  MI300X it passes VALU at M = 4 (6144 x 4096), M = 8 (4096 x 1024, 1536 x 2048) and
  M = 16 (2048 x 768). At M = 16 it is 1.7x faster than VALU on 6144 x 4096; at M = 128 it is
  1.4-1.8x faster on the small shapes and 3.4x on 6144 x 4096.
- VALU stays fastest at M = 1, by about 30% on MI300X.
- On MI210 the LDS engine crosses at M = 4-8 and is 5-6.6x faster than VALU at M = 128;
  MI300X's VALU is relatively much stronger.
- Double-buffering the LDS chunk (`mfma_lds_db`) gained 13-20% only on 6144 x 4096 at
  M >= 32 and was neutral elsewhere. At M = 128 it needs 64-k chunks to fit 64 KB of LDS, and
  there it lost badly on the small shapes.
- 32x32x8 MFMA tiles (32 rows x 32 tokens per wave) were measured and were 1.3-2.3x slower
  than 16x16 on both GPUs: no more FLOPs per cycle on CDNA, half the row tiles, so K has to
  be split more.
- The best LDS row reaches about 73 TFLOP/s on MI300X (about 6% of the bf16 peak) and 55 on
  MI210 (about 30%), so a tuned large-M kernel can do better than these rows.

For MoE with 128 experts and top-8 routing, each expert sees about batch / 16 tokens: MFMA pays
for gate/up from about batch 128 and for down from about batch 256 on MI300X. Measure your
layout at your M before choosing.

Questions:
- At your M, is the VALU path still bound by bytes? Compare its time at M against M = 1.
- Do the tokens' activations stay in registers or LDS across the K loop? Each token is one more
  x load per weight load on VALU; on MFMA it is a column of B, and reading it from global in
  every wave made the plain `mfma16` engine lose.
- Does every wave have a tile? With N / 16 tiles below the number of waves, split K.
- Do earlier runs' symptoms fit? A 5-token fp32-activation GEMV reached only 1.6-2.6 TB/s on
  MI300X: the VALU issue rate, not HBM, was the limit.

## Wave64 reductions

- DPP (`__builtin_amdgcn_update_dpp`: quad_perm 0xb1, 0x4e, row_half_mirror 0x141, row_mirror
  0x140, then `v_readlane` of lanes 0/16/32/48) against `__shfl_xor` (ds_bpermute): see
  `wave_sum.*` rows (MI300X: 128 vs 452 cycles; MI210: 108 vs 452). In earlier kernels DPP won 2-6% end to end.
- No single DPP op crosses the two halves of the wave on gfx942; use readlane or `row_bcast31`.
- Butterflies over several rows at once (one reduction per lane group) cut a QKV phase from 118
  to 68 us in one run.

## Dequantized weights (int4 / int8)

- Count the unpack instructions per weight byte in the ISA: in earlier int4 kernels the
  `v_perm`/`v_and`/`v_lshr` unpack was about 3x the dot instructions.
- int4 into f16 without converts: `0x6400 | n` is the f16 value 1024 + n; build pairs with
  `v_perm_b32`, use `v_dot2_f32_f16`, and subtract 1024 * sum(x) once per row.
  This gave 14.5 -> 12.0 us per layer in a benchmark and 1-2% in the kernel.
- int8: XOR bias + `v_cvt_f32_ubyteN` + fma was 2.4x faster than the naive path in a benchmark,
  and 8.35 -> 7.79 ms in the kernel. A 256-entry LDS lookup table was slower.
- `v_and_or_b32` is not formed from plain C in every case; inline asm cut 12 -> 10.8 us per layer.
  A VOP3 instruction takes at most one SGPR operand.

## Verify before integrating

- Check numerics against an fp32 reference with distinctive data, so that a wrong fragment
  layout shows.
- Run `codegen_gate_amd.py`: VGPR/AGPR counts, spills, scratch. A change that adds any
  value live across a barrier at the 256-VGPR cap spills.
- Compare engines at the real shape in the full kernel (or a compile-time variant of it), with
  the neighbouring reduction and barrier included. A row is an empty-kernel number.
- Record per phase which engine won, at which N, K, M and thread count, so later rounds reuse it.
