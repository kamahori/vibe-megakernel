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
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..cases import Case, select_cases
from ..workloads import make_inputs, reference
from .benchmark import _audit_launches, _measure
from .correctness import check_trials


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_FILES = (
    "megabench/cases.py", "megabench/runner.py", "megabench/workloads.py",
    "megabench/harness/runner.py", "megabench/harness/correctness.py",
    "megabench/harness/benchmark.py", "megabench/tasks/workloads.py",
    "megabench/tasks/dense.py", "megabench/tasks/common.py", "megabench/tasks/moe.py",
    "megabench/tasks/gemma.py", "megabench/tasks/eagle3.py",
    "megabench/tasks/references/qwen3.py",
    "megabench/tasks/references/moe.py",
)


def _contract_digest() -> str:
    digest = hashlib.sha256()
    for name in CONTRACT_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


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


def evaluate_case(case: Case, submission: Path, *, device: str,
                  trials: int, warmup: int, reps: int,
                  graph_baseline: bool = True) -> dict:
    import torch

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
        "mentions_oracle_module": any(
            name in source_text for name in
            ("megabench.workloads", "megabench.tasks.workloads",
             "megabench.tasks.dense", "megabench.tasks.moe",
             "megabench.tasks.gemma", "megabench.tasks.eagle3",
             "megabench.tasks.references")),
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
    with torch.inference_mode():
        record["correctness"], first_call_ms = check_trials(
            case, candidate, device, trials)
        if first_call_ms is not None:
            record["first_call_ms_including_jit"] = first_call_ms

        perf_inputs = make_inputs(case, secrets.randbits(32), device)
        if case.family == "spec_target_step":
            perf_output = reference(case, perf_inputs)
            record["spec_accounting"] = {
                "proposed_draft_tokens": case.params["draft_depth"],
                "accepted_draft_tokens": int(perf_output["accepted_count"].item()),
                "committed_tokens": int(perf_output["committed_count"].item()),
            }
            del perf_output
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


def _docker_worker_command(args: argparse.Namespace, case: Case,
                           tempdir: Path, result_path: Path) -> tuple[list[str], str, str]:
    """Run only this case in a disposable, unprivileged container."""
    submission = Path(args.submission).resolve()
    run_id = uuid.uuid4().hex
    name = f"megabench-{run_id}"
    label = f"org.megabench.run={run_id}"
    if submission.is_relative_to(ROOT):
        container_submission = Path("/workspace") / submission.relative_to(ROOT)
        extra_mount: list[str] = []
    else:
        container_submission = Path("/submission") / submission.name
        extra_mount = ["--volume", f"{submission.parent}:/submission:ro"]
    cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name,
           "--label", label, "--network", "none", "--read-only",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--pids-limit", "1024", "--shm-size", "1g",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "--tmpfs", "/tmp:rw,exec,nosuid,size=8g",
           "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
           "--env", "TRITON_CACHE_DIR=/tmp/triton",
           "--env", "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions",
           "--workdir", "/workspace",
           "--volume", f"{ROOT}:/workspace:ro",
           "--volume", f"{tempdir}:/results:rw", *extra_mount]
    for group in os.getgroups():
        cmd.extend(["--group-add", str(group)])
    if args.docker_gpus:
        cmd.extend(["--gpus", args.docker_gpus])
    cmd.extend([args.docker_image, "python", "-m", "megabench", "_worker",
                "--case", case.id, "--submission", str(container_submission),
                "--device", args.device, "--trials", str(args.trials),
                "--warmup", str(args.warmup), "--reps", str(args.reps),
                "--result", f"/results/{result_path.name}"])
    if not args.graph_baseline:
        cmd.append("--no-graph-baseline")
    return cmd, name, run_id


def _stop_owned_container(name: str, run_id: str) -> str | None:
    """On timeout, stop only the container carrying our unique run label."""
    try:
        inspected = subprocess.run(
            ["docker", "inspect", "--format",
             '{{ index .Config.Labels "org.megabench.run" }}', name],
            capture_output=True, text=True, timeout=10)
        if inspected.returncode or inspected.stdout.strip() != run_id:
            return "timed-out container could not be verified for cleanup"
        stopped = subprocess.run(["docker", "stop", "--time", "1", name],
                                 capture_output=True, text=True, timeout=15)
        if stopped.returncode:
            return f"could not stop timed-out container: {stopped.stderr[-500:]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not stop timed-out container: {type(exc).__name__}: {exc}"
    return None


def _run_one(args: argparse.Namespace, case: Case, tempdir: Path) -> dict:
    if not case.ready:
        return {"case": case.to_dict(), "status": "not_implemented", "reason": case.note}
    submission = str(Path(args.submission).resolve())
    result_path = tempdir / f"{case.id}.json"
    container = None
    if args.docker_image:
        cmd, name, run_id = _docker_worker_command(args, case, tempdir, result_path)
        container = (name, run_id)
    else:
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
        result["submission"] = submission
        if container:
            result["evaluation_runtime"] = {"kind": "docker", "image": args.docker_image,
                                            "gpus": args.docker_gpus}
        return result
    except subprocess.TimeoutExpired:
        result = {"case": case.to_dict(), "submission": submission,
                  "status": "timeout",
                  "reason": f"worker exceeded {args.timeout}s"}
        if container:
            warning = _stop_owned_container(*container)
            if warning:
                result["cleanup_warning"] = warning
        return result
    except OSError as exc:
        return {"case": case.to_dict(), "submission": submission,
                "status": "worker_failed",
                "reason": f"could not start worker: {type(exc).__name__}: {exc}"}


def _output_path(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "runs" / f"{stamp}-{os.getpid()}.jsonl"


def aggregate_sessions(paths: list[Path], suite: str, method: str) -> dict:
    """Combine one independently produced result per case for one method."""
    expected = {case.id: case for case in select_cases(suite) if case.ready}
    if not expected:
        raise ValueError(f"suite {suite} has no ready cases")
    results: dict[str, dict] = {}
    session_ids: set[str] = set()
    source_dirs: set[Path] = set()
    contract_sha256 = _contract_digest()
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected exactly one case result")
        row = rows[0]
        case_id = row.get("case", {}).get("id")
        session = row.get("session")
        if case_id not in expected:
            raise ValueError(f"{path}: unexpected case {case_id}")
        if row["case"] != expected[case_id].to_dict():
            raise ValueError(f"{path}: case contract differs from current {case_id}")
        if row.get("contract_sha256") != contract_sha256:
            raise ValueError(f"{path}: benchmark contract digest differs")
        if case_id in results:
            raise ValueError(f"duplicate case {case_id}")
        if not isinstance(session, dict) or session.get("case_id") != case_id:
            raise ValueError(f"{path}: missing matching session metadata")
        if session.get("method") != method or not session.get("id"):
            raise ValueError(f"{path}: method or session ID mismatch")
        if session["id"] in session_ids:
            raise ValueError(f"duplicate session ID {session['id']}")
        source_dir = Path(row["submission"]).resolve().parent
        if source_dir in source_dirs:
            raise ValueError(f"shared candidate directory {source_dir}")
        session_ids.add(session["id"])
        source_dirs.add(source_dir)
        results[case_id] = row
    missing = sorted(expected.keys() - results.keys())
    passed = [row for row in results.values() if row["status"] == "ok_provisional"]
    score = (math.exp(sum(math.log(row["speedup_vs_best_baseline_cuda_event"])
                          for row in passed) / len(passed))
             if not missing and len(passed) == len(expected) else None)
    return {"suite": suite, "method": method,
            "contract_sha256": contract_sha256, "cases": results,
            "missing_cases": missing, "passed": len(passed),
            "total": len(expected), "provisional_geomean_speedup": score}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m megabench")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list agent-writing challenge cells")
    listing.add_argument("--suite", choices=("core", "p0", "p1", "p2", "p3", "planned", "all"),
                         default="all")
    listing.add_argument("--json", action="store_true")
    evaluation = sub.add_parser("evaluate", help="evaluate one case from one agent session")
    evaluation.add_argument("--submission", required=True)
    evaluation.add_argument("--case", required=True)
    evaluation.add_argument("--method", required=True)
    evaluation.add_argument("--session-id", required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--trials", type=int, default=3)
    evaluation.add_argument("--warmup", type=int, default=2)
    evaluation.add_argument("--reps", type=int, default=5)
    evaluation.add_argument("--timeout", type=int, default=180)
    evaluation.add_argument("--docker-image", help="run each case in this existing local image")
    evaluation.add_argument("--docker-gpus", help="Docker GPU selection, e.g. device=6")
    evaluation.add_argument("--no-graph-baseline", dest="graph_baseline",
                            action="store_false")
    evaluation.add_argument("--output")
    aggregate = sub.add_parser("aggregate", help="combine independent case sessions")
    aggregate.add_argument("--suite", choices=("core", "p0", "p1", "p2", "p3"),
                           default="p0")
    aggregate.add_argument("--method", required=True)
    aggregate.add_argument("--input", action="append", required=True)
    aggregate.add_argument("--output")
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
                print(f"{case.id:44} {case.suite.upper():3} {state:7} "
                      f"{case.model} / {case.phase}")
        return 0
    if args.command == "aggregate":
        try:
            summary = aggregate_sessions([Path(name) for name in args.input],
                                         args.suite, args.method)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        if args.output:
            destination = Path(args.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as output:
                json.dump(summary, output, indent=2)
        print(json.dumps(summary))
        return 0 if summary["provisional_geomean_speedup"] is not None else 1
    if min(args.trials, args.reps, args.timeout) < 1 or args.warmup < 0:
        parser.error("trials, reps, timeout must be positive; warmup >= 0")
    if args.docker_gpus and not args.docker_image:
        parser.error("--docker-gpus requires --docker-image")
    if args.docker_image and args.docker_image.startswith("-"):
        parser.error("--docker-image must be an image name or digest")
    if args.docker_image and args.device.startswith("cuda") and not args.docker_gpus:
        parser.error("CUDA Docker evaluation requires explicit --docker-gpus")
    submission = Path(args.submission).resolve()
    if not submission.is_file():
        parser.error(f"submission does not exist: {submission}")
    try:
        cases = select_cases("all", [args.case])
    except ValueError as exc:
        parser.error(str(exc))
    path = _output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    case = cases[0]
    results = []
    with tempfile.TemporaryDirectory(prefix="megabench-worker-") as directory:
        with path.open("x", encoding="utf-8") as output:
            result = _run_one(args, case, Path(directory))
            result["session"] = {"id": args.session_id, "method": args.method,
                                 "case_id": case.id}
            result["contract_sha256"] = _contract_digest()
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
             if complete else None)
    print(json.dumps({"results": str(path), "provisional_geomean_speedup": score,
                      "passed": len(complete), "total": 1}))
    return 0 if complete else 1
