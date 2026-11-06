# primdb row schema

One JSON object per line. `seed.jsonl` is the trusted, read-only seed; workspace
measurements go to `primdb/local.jsonl` (or `$PRIMDB_LOCAL`). Queries read both.

| field | type | meaning |
|---|---|---|
| `id` | str | stable hash of primitive, params, cond and source file/job |
| `kind` | `measure` \| `model` \| `fact` | a measured point, a fitted cost term, or a static fact |
| `family` | str | group: `tc_mma`, `mma_sync`, `simt`, `tmem`, `ingress`, `dequant`, `sync`, `mem`, `sched`, `gemm`, `gemm_phase`, `model`, `model_check` |
| `primitive` | str | dotted name, e.g. `tcgen05.mma`, `mma.sync`, `sync.grid_barrier` |
| `params` | obj | what was varied: shape (`M`, `N`, `K`), `kind`/`dtype`, `cta_group`, `a_src`, `issuers`, `warps`, `ctas`, `engine`, ... |
| `metrics` | obj | results; units in the name: `_cyc`, `_ns`, `us`, `_gbps`, `_tflops`, `_per_clk_sm` |
| `cond` | obj | conditions held fixed: `sm_clock_mhz` (measured), `l2` (`cold`/`warm`), `ctas`, `threads`, `smem` |
| `verified` | bool \| null | output checked against a reference before timing; null when not checked |
| `n_reps`, `cv` | int, float | repetitions and coefficient of variation of the reported metric |
| `source` | obj | `bench` (source file), `file` (raw output), `job`, `timer` (`clock64`, `globaltimer`, `cudaEvent`) |
| `model` | obj | `kind=model` only: `formula`, `coeffs`, `units`, `range`, `fit_rmse`, `jobs` |
| `note` | str | how to read the row |

Benchmarks print flat JSON lines and `primdb add` normalizes them: keys `p.<x>`,
`m.<x>`, `c.<x>` and `s.<x>` go to `params`, `metrics`, `cond` and `source`;
`family` and `primitive` are required.

```
{"family":"mma_sync","primitive":"mma.sync","p.shape":"m16n8k16","p.dtype":"bf16","p.warps":8,"m.cyc_per_instr_sm":4.1,"c.sm_clock_mhz":1965,"verified":true}
```

Cycle metrics are SM clocks; convert with the row's `sm_clock_mhz`, which drops
under sustained tensor-core load, so compare engines at their own clocks.

`gemm` / `gemm_phase` rows may carry `p.partition` (`output_tile`, `split_k`, `stream_k`), `p.ks`
and `p.reduction`; `p.x_ingress` / `p.w_ingress` are `tma`, `cp_async` or `ld`. `model_check` rows
compare a prediction built from other rows with a measured point: `m.pred_us`, `m.meas_us`,
`m.residual` = meas - pred, `m.rel_err` = pred / meas - 1; `model.rows` names the rows used.
