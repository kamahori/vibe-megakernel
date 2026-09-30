"""Evaluate an agent's submitted megakernels against independent workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cases import Case, select_cases


ROOT = Path(__file__).resolve().parents[1]


def _load_submission(path: Path, case: Case) -> Callable:
    spec = importlib.util.spec_from_file_location("megabench_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import submission {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if not callable(build):
        raise TypeError("submission must export build(case: dict) -> callable")
    runner = build(case.to_dict())
    if not callable(runner):
        raise TypeError("build(case) must return a callable run(inputs)")
    return runner


def _compare(expected: dict, actual: dict, case: Case, device: str) -> dict:
    import torch

    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise AssertionError(f"output keys: expected {sorted(expected)}, got "
                             f"{sorted(actual) if isinstance(actual, dict) else type(actual)}")
    details = {}
    for name, want in expected.items():
        got = actual[name]
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f"{name}: output must be a Tensor")
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(f"{name}: expected {tuple(want.shape)} {want.dtype}, "
                                 f"got {tuple(got.shape)} {got.dtype}")
        if got.device.type != torch.device(device).type:
            raise AssertionError(f"{name}: output is on {got.device}, expected {device}")
        if want.is_floating_point():
            delta = (got.float() - want.float()).abs()
            close = torch.isclose(got.float(), want.float(),
                                  atol=case.atol, rtol=case.rtol, equal_nan=False)
            details[name] = {"max_abs_error": float(delta.max().item()),
                             "mismatched_elements": int((~close).sum().item())}
            if not bool(close.all()):
                raise AssertionError(f"{name}: {details[name]}")
        else:
            mismatch = int((got != want).sum().item())
            details[name] = {"mismatched_elements": mismatch}
            if mismatch:
                raise AssertionError(f"{name}: {mismatch} integer mismatches")
    return details


def _measure(fn: Callable, inputs: dict, device: str, warmup: int,
             reps: int) -> dict:
    import torch

    cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        fn(inputs)
    if cuda:
        torch.cuda.synchronize(device)
    host_ms: list[float] = []
    event_ms: list[float] = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        fn(inputs)
        if cuda:
            end_event.record()
            end_event.synchronize()
            event_ms.append(start_event.elapsed_time(end_event))
        host_ms.append((time.perf_counter() - start) * 1000)
    def p95(values: list[float]) -> float:
        ordered = sorted(values)
        at = (len(ordered) - 1) * 0.95
        lo = int(at)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] -
                              ordered[lo]) * (at - lo)

    return {"host_ms": host_ms, "host_p50_ms": statistics.median(host_ms),
            "host_p95_ms": p95(host_ms),
            "cuda_event_ms": event_ms,
            "cuda_event_p50_ms": statistics.median(event_ms) if cuda else None,
            "cuda_event_p95_ms": p95(event_ms) if cuda else None}


def _audit_launches(fn: Callable, inputs: dict, device: str,
                    launch_budget: int) -> dict:
    import torch

    if torch.device(device).type != "cuda":
        return {"status": "not_applicable_cpu"}
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn(inputs)
            torch.cuda.synchronize(device)
        gpu_events = [event for event in prof.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA]
        cpu_names = [event.name for event in prof.events()
                     if event.device_type == torch.autograd.DeviceType.CPU]
        count = len(gpu_events)
        graph_hint = any("cudagraphlaunch" in name.lower() for name in cpu_names)
        return {"status": "within_budget" if 1 <= count <= launch_budget else
                "launch_budget_failed", "gpu_kernel_count": count,
                "max_gpu_launches": launch_budget,
                "gpu_kernel_names": [event.name for event in gpu_events],
                "cuda_graph_runtime_hint": graph_hint,
                "review_required": True}
    except Exception as exc:
        return {"status": "unverified", "reason": f"{type(exc).__name__}: {exc}",
                "review_required": True}


def evaluate_case(case: Case, submission: Path, *, device: str,
                  trials: int, warmup: int, reps: int,
                  graph_baseline: bool = True) -> dict:
    import torch

    from .workloads import make_inputs, reference

    record = {"case": case.to_dict(), "submission": str(submission),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "hardware": {"hostname": platform.node(), "torch": torch.__version__,
                           "cuda": torch.version.cuda, "device": device,
                           "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}}
    source = submission.read_bytes()
    record["submission_sha256"] = hashlib.sha256(source).hexdigest()
    source_text = source.decode("utf-8", errors="replace")
    record["source_hints_advisory"] = {
        "mentions_cuda_graph": "CUDAGraph" in source_text or "cuda_graph" in source_text,
        "mentions_torch_compile": "torch.compile" in source_text,
        "mentions_oracle_module": "megabench.workloads" in source_text,
    }
    if not case.ready:
        return record | {"status": "not_implemented", "reason": case.note}
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            return record | {"status": "unavailable", "reason": "CUDA not visible"}
        props = torch.cuda.get_device_properties(device)
        record["hardware"].update(gpu_name=props.name,
                                  gpu_memory_bytes=props.total_memory)
    if case.gpus != 1:
        return record | {"status": "not_implemented", "reason":
                         "multi-GPU submission protocol is not implemented"}

    build_start = time.perf_counter()
    candidate = _load_submission(submission, case)
    record["build_ms"] = (time.perf_counter() - build_start) * 1000
    record["correctness"] = {"status": "pass", "trials": []}
    trial_seeds = [secrets.randbits(32) for _ in range(trials)]
    with torch.inference_mode():
        for trial_index, seed in enumerate(trial_seeds):
            inputs = make_inputs(case, seed, device)
            originals = {name: value.clone() for name, value in inputs.items()}
            expected = reference(case, originals)
            first_start = time.perf_counter() if trial_index == 0 else None
            actual = candidate(inputs)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            if first_start is not None:
                record["first_call_ms_including_jit"] = (
                    time.perf_counter() - first_start) * 1000
            for name in inputs:
                if not torch.equal(inputs[name], originals[name]):
                    raise AssertionError(f"candidate mutated input {name}")
            details = _compare(expected, actual, case, device)
            record["correctness"]["trials"].append({"seed": seed,
                                                      "outputs": details})

        perf_inputs = make_inputs(case, secrets.randbits(32), device)
        record["candidate_timing"] = _measure(candidate, perf_inputs, device,
                                              warmup, reps)
        record["reference_timing"] = _measure(
            lambda values: reference(case, values), perf_inputs, device,
            warmup, reps)
        if graph_baseline and torch.device(device).type == "cuda":
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_output = reference(case, perf_inputs)
                record["graph_baseline_timing"] = _measure(
                    lambda _: graph.replay(), perf_inputs, device, warmup, reps)
                # Keep captured tensors alive until replay timing is complete.
                del captured_output
            except Exception as exc:
                record["graph_baseline_unavailable"] = (
                    f"{type(exc).__name__}: {exc}")
        record["launch_audit"] = _audit_launches(candidate, perf_inputs, device,
                                                 case.max_gpu_launches)
        cand = record["candidate_timing"]["host_p50_ms"]
        base = record["reference_timing"]["host_p50_ms"]
        record["speedup_vs_eager_host"] = base / cand
        if torch.device(device).type == "cuda":
            cand_event = record["candidate_timing"]["cuda_event_p50_ms"]
            base_event = record["reference_timing"]["cuda_event_p50_ms"]
            record["speedup_vs_eager_cuda_event"] = base_event / cand_event
            best = base_event
            if "graph_baseline_timing" in record:
                graph_event = record["graph_baseline_timing"]["cuda_event_p50_ms"]
                record["speedup_vs_graph_cuda_event"] = graph_event / cand_event
                best = min(best, graph_event)
            record["speedup_vs_best_baseline_cuda_event"] = best / cand_event
        if case.family == "spec_verify":
            spec = reference(case, perf_inputs)
            accepted = int(spec["accepted_count"].sum().item())
            committed = int(spec["committed_count"].sum().item())
            proposed = case.params["batch"] * case.params["draft_len"]
            record["spec_accounting"] = {"proposed_draft_tokens": proposed,
                                         "accepted_draft_tokens": accepted,
                                         "committed_tokens": committed,
                                         "acceptance_fraction": accepted / proposed}
    audit = record["launch_audit"]["status"]
    record["status"] = ("correctness_only" if torch.device(device).type == "cpu" else
                        "ok_provisional" if audit == "within_budget" else
                        "non_megakernel" if audit == "launch_budget_failed" else
                        "authenticity_unverified")
    return record


def _worker(args: argparse.Namespace) -> int:
    case = select_cases("all", [args.case])[0]
    try:
        result = evaluate_case(case, Path(args.submission).resolve(),
                               device=args.device, trials=args.trials,
                               warmup=args.warmup, reps=args.reps,
                               graph_baseline=args.graph_baseline)
    except Exception as exc:
        result = {"case": case.to_dict(), "submission": args.submission,
                  "status": "incorrect" if isinstance(exc, AssertionError) else "error",
                  "reason": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(limit=12)}
    with Path(args.result).open("x", encoding="utf-8") as file:
        json.dump(result, file)
    return 0


def _run_one(args: argparse.Namespace, case: Case, tempdir: Path) -> dict:
    if not case.ready:
        return {"case": case.to_dict(), "status": "not_implemented", "reason": case.note}
    result_path = tempdir / f"{case.id}.json"
    cmd = [sys.executable, "-m", "megabench", "_worker", "--case", case.id,
           "--submission", str(Path(args.submission).resolve()), "--device", args.device,
           "--trials", str(args.trials), "--warmup", str(args.warmup),
           "--reps", str(args.reps), "--result", str(result_path)]
    if not args.graph_baseline:
        cmd.append("--no-graph-baseline")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=args.timeout)
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if proc.returncode:
                result["worker_exit_code"] = proc.returncode
        else:
            result = {"case": case.to_dict(), "status": "worker_failed",
                      "reason": f"worker exited {proc.returncode}",
                      "stderr_tail": proc.stderr[-4000:]}
        return result
    except subprocess.TimeoutExpired:
        return {"case": case.to_dict(), "status": "timeout",
                "reason": f"worker exceeded {args.timeout}s"}


def _output_path(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parent / "runs" / f"{stamp}-{os.getpid()}.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m megabench")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list agent-writing challenge cells")
    listing.add_argument("--suite", choices=("smoke", "core", "planned", "all"),
                         default="all")
    listing.add_argument("--json", action="store_true")
    evaluation = sub.add_parser("evaluate", help="evaluate a submission, one process/case")
    evaluation.add_argument("--submission", required=True)
    evaluation.add_argument("--suite", choices=("smoke", "core", "planned", "all"),
                            default="smoke")
    evaluation.add_argument("--case", action="append", dest="ids")
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--trials", type=int, default=3)
    evaluation.add_argument("--warmup", type=int, default=2)
    evaluation.add_argument("--reps", type=int, default=5)
    evaluation.add_argument("--timeout", type=int, default=180)
    evaluation.add_argument("--no-graph-baseline", dest="graph_baseline",
                            action="store_false")
    evaluation.add_argument("--output")
    worker = sub.add_parser("_worker")
    worker.add_argument("--case", required=True)
    worker.add_argument("--submission", required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--trials", type=int, required=True)
    worker.add_argument("--warmup", type=int, required=True)
    worker.add_argument("--reps", type=int, required=True)
    worker.add_argument("--result", required=True)
    worker.add_argument("--no-graph-baseline", dest="graph_baseline",
                        action="store_false")
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return _worker(args)
    if args.command == "list":
        cases = select_cases(args.suite)
        if args.json:
            print(json.dumps([case.to_dict() for case in cases], indent=2))
        else:
            for case in cases:
                state = "ready" if case.ready else "planned"
                print(f"{case.id:30} {case.category:24} {case.params} [{state}]")
        return 0
    if min(args.trials, args.reps, args.timeout) < 1 or args.warmup < 0:
        parser.error("trials, reps, timeout must be positive; warmup >= 0")
    submission = Path(args.submission).resolve()
    if not submission.is_file():
        parser.error(f"submission does not exist: {submission}")
    try:
        cases = select_cases(args.suite, args.ids)
    except ValueError as exc:
        parser.error(str(exc))
    path = _output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="megabench-worker-") as directory:
        with path.open("x", encoding="utf-8") as output:
            for case in cases:
                result = _run_one(args, case, Path(directory))
                results.append(result)
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(f"{case.id}: {result['status']}" +
                      (f" ({result['speedup_vs_best_baseline_cuda_event']:.2f}x "
                       "GPU-event vs best available baseline)"
                       if "speedup_vs_best_baseline_cuda_event" in result else ""))
    complete = [item for item in results if item["status"] == "ok_provisional"]
    score = (math.exp(sum(math.log(item["speedup_vs_best_baseline_cuda_event"])
                          for item in complete) / len(complete))
             if len(complete) == len(cases) and complete else None)
    print(json.dumps({"results": str(path), "provisional_geomean_speedup": score,
                      "passed": len(complete), "total": len(cases)}))
    return 0 if len(complete) == len(cases) else 1
