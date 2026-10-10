"""Out-of-process MPK worker for the Method 1 study (runs in the MPK venv).

Modes:

* ``offline``: build native MPK Qwen3-0.6B megakernels (upstream ``Qwen3Builder``,
  MegaBench synthetic weights) in ``offline`` mode, one per ``--msl`` value, with
  a fixed 128-token prompt and EOS disabled. Each timed rep resets the request
  (``init_request_func``), then times ``pk()`` with CUDA events. In offline mode
  the calling stream waits for the worker and scheduler streams, so the end
  event marks completion. The MPK scheduler stops a request once
  ``step + 1 >= msl`` (persistent_kernel.cuh:319), and a decode iteration at
  step ``s`` processes position ``s`` over ``s + 1`` KV positions, so
  T(msl=130) - T(msl=129) is exactly the decode step at position 128 over 129
  positions, the MegaBench ``b1-s128`` step.
* ``notoken``: the PR #10 native probe (``online_notoken``, step 128, one call =
  one decode step) with weights and KV loaded once outside timing; each rep
  times ``pk()`` plus the host sync it needs (the mode does not join its
  streams back to the caller).

Rows are written as JSONL (one ``mpk_build`` row per build, one ``mpk_timing``
row per build and round, one ``mpk_check`` row for the notoken correctness
trial). Every row carries the GPU, the MPK commit, and NVML clocks when present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MPK_SRC = REPO / "reproductions" / "mirage-mpk"
CASE_ID = "dense-step-qwen3-06b-b1-s128"
PROMPT_LEN = 128
PAGE_SIZE = 256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mpk_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(MPK_SRC), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class _Clock:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nv = None

    def snap(self) -> dict:
        if self.nv is None:
            return {}
        try:
            nv, h = self.nv, self.h
            return {"sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                    "throttle_mask": int(nv.nvmlDeviceGetCurrentClocksEventReasons(h))}
        except Exception:
            return {}


def _import_builder():
    """Qwen3Builder from the pinned source tree (the wheel may omit qwen3/)."""
    import mirage.mpk.models as model_package
    source_models = MPK_SRC / "python" / "mirage" / "mpk" / "models"
    if str(source_models) not in model_package.__path__:
        model_package.__path__.append(str(source_models))
    from mirage.mpk.models.qwen3.builder import Qwen3Builder
    return Qwen3Builder


def _rope_tables(seq: int, dim: int, device: str):
    """The PR #10 probe's tables ([1, seq, dim] cos/sin, BF16), as Qwen3Builder expects."""
    from megabench.probes.mpk_native_qwen3_probe import _rope_tables as probe_tables
    return probe_tables(seq, dim, device)


def _state_dict(case, values: dict, gate_up_groups: int, shuffle_tensors) -> dict:
    """MegaBench weight names -> the HF-style names Qwen3Builder consumes (PR #10 layout)."""
    p = case.params
    state = {"model.embed_tokens.weight": values["embed"].clone(),
             "lm_head.weight": values["embed"].clone(),
             "model.norm.weight": values["fnorm"].clone()}
    for layer in range(p["layers"]):
        pre = f"model.layers.{layer}."
        state[pre + "input_layernorm.weight"] = values["ln1"][layer].clone()
        state[pre + "self_attn.qkv_proj.weight"] = shuffle_tensors(
            [values[n][layer] for n in ("wq", "wk", "wv")], p["kv_heads"], 0)
        state[pre + "self_attn.q_norm.weight"] = values["qn"][layer].clone()
        state[pre + "self_attn.k_norm.weight"] = values["kn"][layer].clone()
        state[pre + "self_attn.o_proj.weight"] = values["wo"][layer].clone()
        state[pre + "post_attention_layernorm.weight"] = values["ln2"][layer].clone()
        state[pre + "mlp.gate_up_proj.weight"] = shuffle_tensors(
            [values["wg"][layer], values["wu"][layer]], gate_up_groups, 0)
        state[pre + "mlp.down_proj.weight"] = values["wd"][layer].clone()
    return state


class OfflineMPK:
    """One compiled offline-mode MPK kernel for a fixed (msl, mbt, cutlass)."""

    def __init__(self, case, values: dict, prompt: "torch.Tensor", msl: int, mbt: int,
                 cutlass: bool, compile_root: Path):
        import torch
        import mirage
        from mirage.mpk.models.graph_builder import MirageModelConfig
        from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer, shuffle_tensors
        from mirage.mpk.persistent_kernel import PersistentKernel
        Qwen3Builder = _import_builder()

        if msl > PAGE_SIZE:
            raise ValueError(f"msl {msl} needs more than one {PAGE_SIZE}-token page")
        p = case.params
        self.msl, self.mbt, self.cutlass = msl, mbt, cutlass
        self.vocab = p["vocab"]
        gate_up_groups = grid_for_rmsnorm_linear_layer(2 * p["intermediate"]) // 2
        self.kcache = torch.zeros((p["layers"], 1, PAGE_SIZE, p["kv_heads"], p["head_dim"]),
                                  device="cuda", dtype=torch.bfloat16)
        self.vcache = torch.zeros_like(self.kcache)
        self.state = _state_dict(case, values, gate_up_groups, shuffle_tensors)
        self.prompt = prompt.to(device="cuda", dtype=torch.int64)
        z32 = lambda n: torch.zeros(n, device="cuda", dtype=torch.int32)  # noqa: E731
        self.meta = {
            "step": z32(1),
            "tokens": torch.zeros((1, msl), device="cuda", dtype=torch.int64),
            "input_tokens": torch.zeros((mbt, 1), device="cuda", dtype=torch.int64),
            "output_tokens": torch.zeros((mbt, 1), device="cuda", dtype=torch.int64),
            "num_new_tokens": torch.ones(1, device="cuda", dtype=torch.int32),
            "prompt_lengths": torch.full((1,), PROMPT_LEN, device="cuda", dtype=torch.int32),
            "qo_indptr_buffer": z32(2),
            "paged_kv_indptr_buffer": z32(2),
            "paged_kv_indices_buffer": z32(1),
            "paged_kv_last_page_len_buffer": z32(1),
            "paged_kv_indices_snapshot": z32(1),
        }
        self.meta["tokens"][0, :PROMPT_LEN] = self.prompt
        workers, schedulers = mirage.get_configurations_from_gpu(0)
        self.workers, self.schedulers = workers, schedulers
        params = PersistentKernel.get_default_init_parameters()
        params.update(mode="offline", num_workers=workers, num_local_schedulers=schedulers,
                      max_seq_length=msl, max_num_batched_tokens=mbt,
                      max_num_batched_requests=1, max_num_pages=1, page_size=PAGE_SIZE,
                      meta_tensors=self.meta, use_cutlass_kernel=cutlass, eos_token_id=-1)
        self.pk = PersistentKernel(**params)
        self.builder = Qwen3Builder(self.pk)
        self.builder.mpk_gate_up_groups = gate_up_groups
        cos, sin = _rope_tables(4096, p["head_dim"], "cuda")
        config = MirageModelConfig(
            hidden_size=p["hidden"], intermediate_size=p["intermediate"],
            vocab_size=p["vocab"], local_num_q_heads=p["q_heads"],
            local_num_kv_heads=p["kv_heads"], head_dim=p["head_dim"],
            num_layers=p["layers"], k_cache=self.kcache, v_cache=self.vcache,
            position_embeddings=(cos, sin), state_dict=self.state, with_lm_head=True)
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            self.builder.build_from_config(config)
        finally:
            torch.set_default_dtype(old)
        compile_root.mkdir(parents=True, exist_ok=True)
        self.compile_dir = Path(tempfile.mkdtemp(
            prefix=f"mpk_offline_msl{msl}_mbt{mbt}_{'cut' if cutlass else 'nocut'}_",
            dir=compile_root))
        t0 = time.perf_counter()
        self.pk.compile(output_dir=str(self.compile_dir))
        self.compile_ms = (time.perf_counter() - t0) * 1e3
        self._reset = self.pk.init_func.__self__.init_request_func
        self.stop = msl

    def set_stop(self, stop: int) -> None:
        """Re-run the launcher's init with a runtime max_seq_length <= the compiled one.

        The scheduler's stop test reads ``config.max_seq_length``, a host-side
        runtime field set only by ``init_persistent_kernel``; the compiled
        ``MPK_MAX_SEQ_LENGTH`` is just the token-buffer stride. Re-initializing
        the same build changes where requests stop while code, weights, KV
        cache and intermediate buffers stay identical (the runtime's own queues
        are reallocated and the old ones leak, which is harmless here).
        """
        if stop > self.msl or stop <= PROMPT_LEN:
            raise ValueError(f"stop {stop} outside ({PROMPT_LEN}, {self.msl}]")
        if stop == self.stop:
            return
        pk = self.pk
        order = ["step", "tokens", "input_tokens", "output_tokens", "num_new_tokens",
                 "prompt_lengths", "qo_indptr_buffer"]
        for g in range(len(pk.kv_groups)):
            order += [f"paged_kv_indptr_buffer_{g}", f"paged_kv_indices_buffer_{g}",
                      f"paged_kv_last_page_len_buffer_{g}", f"paged_kv_indices_snapshot_{g}"]
        meta_ptrs = [pk.meta_tensors[k].data_ptr() for k in order]
        names = list(pk._model_tensors.keys())
        ptrs = [t.data_ptr() for t in pk._model_tensors.values()]
        pk.init_func(meta_ptrs, 0, pk.mpi_rank, pk.num_workers, pk.num_local_schedulers,
                     pk.num_remote_schedulers, stop, pk.total_num_requests, pk.eos_token_id,
                     pk.allocate_nvshmem_teams, names, ptrs,
                     str(self.compile_dir / f"task_graph_rank{pk.mpi_rank}.json"),
                     [g.block_size for g in pk.kv_groups],
                     [g.window_size for g in pk.kv_groups])
        self.stop = stop

    def reset(self) -> None:
        """Restore the request to "prompt only, nothing generated"."""
        self.meta["step"].zero_()
        self.meta["tokens"][0, PROMPT_LEN:].zero_()
        self._reset()

    def run_once(self) -> float:
        import torch
        self.reset()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.pk()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    def generated(self) -> dict:
        step = int(self.meta["step"][0].item())
        toks = self.meta["tokens"][0, PROMPT_LEN:step + 1].tolist()
        return {"final_step": step, "generated": step + 1 - PROMPT_LEN, "tokens": toks}


def _case_and_values(seed: int):
    import torch
    from megabench.cases import select_cases
    from megabench.tasks.workloads import make_inputs
    case = select_cases("all", [CASE_ID])[0]
    values = make_inputs(case, seed, "cuda")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    prompt = torch.randint(0, case.params["vocab"], (PROMPT_LEN,), generator=gen)
    return case, values, prompt


def _write(fh, row: dict) -> None:
    fh.write(json.dumps(row) + "\n")
    fh.flush()


def run_offline(args, fh, base: dict) -> int:
    """Time T(stop) for every stop length.

    With ``--stops`` one kernel is compiled at the largest stop and each stop is
    set at runtime (``set_stop``), so every point shares code, weights and
    buffers. Without it each ``--msl`` value is its own build (the build-to-build
    offset then adds to every difference).
    """
    import torch
    case, values, prompt = _case_and_values(args.seed)
    clock = _Clock()
    builds = [max(args.stops)] if args.stops else list(args.msl)
    kernels: dict[int, OfflineMPK] = {}
    status = 0
    for msl in builds:
        try:
            k = OfflineMPK(case, values, prompt, msl, args.mbt, args.cutlass == "on",
                           Path(args.compile_root))
        except Exception as exc:
            _write(fh, dict(base, kind="mpk_build", mode="offline", msl=msl, mbt=args.mbt,
                            cutlass=args.cutlass, ok=False,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-4000:]))
            status = 1
            continue
        k.run_once()
        _write(fh, dict(base, kind="mpk_build", mode="offline", msl=msl, mbt=args.mbt,
                        cutlass=args.cutlass, ok=True, compile_ms=k.compile_ms,
                        compile_dir=str(k.compile_dir), workers=k.workers,
                        schedulers=k.schedulers, prompt_len=PROMPT_LEN,
                        expected_generated=msl - PROMPT_LEN, **k.generated()))
        kernels[msl] = k
    if not kernels:
        return 1
    if args.stops:
        k = next(iter(kernels.values()))
        units = [(stop, k) for stop in args.stops]
    else:
        units = list(kernels.items())
    for stop, k in units:
        k.set_stop(stop)
        for _ in range(args.warmup):
            k.run_once()
    rng = random.Random(args.seed)
    for rnd in range(args.rounds):
        rng.shuffle(units)
        for stop, k in units:
            k.set_stop(stop)
            k.run_once()  # discard: first run after a re-init
            c0 = clock.snap()
            samples = [k.run_once() for _ in range(args.reps)]
            gen = k.generated()
            _write(fh, dict(base, kind="mpk_timing", mode="offline", msl=stop,
                            compile_msl=k.msl, stop_method="reinit" if args.stops else "build",
                            mbt=args.mbt, cutlass=args.cutlass, round=rnd, samples_ms=samples,
                            median_ms=statistics.median(samples),
                            final_step=gen["final_step"], generated=gen["generated"],
                            tokens_head=gen["tokens"][:8],
                            clock_before=c0, clock_after=clock.snap()))
    del kernels, units
    torch.cuda.synchronize()
    return status


def run_notoken(args, fh, base: dict) -> int:
    import torch
    from megabench.probes.mpk_native_qwen3_probe import NativeQwen3
    from megabench.tasks.workloads import make_inputs, reference
    from megabench.cases import select_cases
    case = select_cases("all", [CASE_ID])[0]
    clock = _Clock()
    values = make_inputs(case, args.seed, "cuda")
    t0 = time.perf_counter()
    cand = NativeQwen3(case, values, case.params["layers"], Path(args.compile_root))
    compile_ms = (time.perf_counter() - t0) * 1e3
    _write(fh, dict(base, kind="mpk_build", mode="notoken", ok=True, compile_ms=compile_ms,
                    compile_dir=str(cand.compile_dir)))
    # Correctness trial on the timing seed (the probe's MegaBench comparison).
    expected = reference(case, values)
    with torch.inference_mode():
        got = cand(values)
    torch.cuda.synchronize()
    outs = {}
    for name, want in expected.items():
        have = got[name]
        if want.is_floating_point():
            d = (have.float() - want.float()).abs()
            rtol = case.bf16_rtol if want.dtype == torch.bfloat16 else case.rtol
            bad = ~torch.isclose(have.float(), want.float(), atol=case.atol, rtol=rtol)
            outs[name] = {"mismatched": int(bad.sum()), "max_abs_error": float(d.max())}
        else:
            outs[name] = {"mismatched": int((have != want).sum()),
                          "got": int(have.item()), "want": int(want.item())}
    _write(fh, dict(base, kind="mpk_check", mode="notoken", seed=args.seed, outputs=outs,
                    strict_pass=all(o["mismatched"] == 0 for o in outs.values())))
    cand.load(values)  # weights, KV and metadata once; timing excludes copies

    def once() -> float:
        # online_notoken leaves metadata to the host; restore step-128 metadata in place.
        cand.meta["step"].fill_(cand.context)
        cand.meta["paged_kv_last_page_len_buffer"].fill_(cand.context + 1)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        cand.pk()
        torch.cuda.synchronize()  # the mode does not join its streams to the caller
        e.record()
        e.synchronize()
        return s.elapsed_time(e)

    for _ in range(args.warmup):
        once()
    for rnd in range(args.rounds):
        c0 = clock.snap()
        samples = [once() for _ in range(args.reps)]
        _write(fh, dict(base, kind="mpk_timing", mode="notoken", round=rnd,
                        samples_ms=samples, median_ms=statistics.median(samples),
                        clock_before=c0, clock_after=clock.snap()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("offline", "notoken"), required=True)
    ap.add_argument("--msl", default="129,130", help="comma list, one build each (offline)")
    ap.add_argument("--stops", default="",
                    help="comma list of runtime stop lengths on ONE build (offline); "
                         "overrides --msl")
    ap.add_argument("--mbt", type=int, default=1)
    ap.add_argument("--cutlass", choices=("on", "off"), default="on")
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--compile-root", default=os.environ.get(
        "MPK_COMPILE_ROOT", "/raid/garv901/.cache/megabench_sota/mpk_builds"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.msl = [int(x) for x in args.msl.split(",") if x]
    args.stops = [int(x) for x in args.stops.split(",") if x]
    import torch
    base = {"t": _now(), "tag": args.tag, "gpu": torch.cuda.get_device_name(0),
            "mpk_commit": _mpk_commit(), "torch": torch.__version__,
            "python": sys.version.split()[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        try:
            with torch.inference_mode():
                return run_offline(args, fh, base) if args.mode == "offline" \
                    else run_notoken(args, fh, base)
        except Exception as exc:
            _write(fh, dict(base, kind="mpk_error", mode=args.mode,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-4000:]))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
