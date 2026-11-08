"""Give agent commands a GPU through one Slurm job per command.

The agent sandbox has no GPU and cannot submit Slurm jobs. ``serve`` runs on
the host, outside the sandbox, and listens on a Unix socket that the sandbox
can reach. ``run`` is the client (the workspace's ``./gpu_run``): it sends the
command, its directory, its environment and its stdout/stderr descriptors to
the broker. The broker starts ``srun <args> <job python> gpu_job.py run ...``,
so the job writes straight to the caller's stdout/stderr, and returns the exit
status. ``gpu_job.py`` builds the agent's sandbox again inside the job.

If the client goes away (for example a tool timeout), the broker cancels the
job. At most ``--max-jobs`` jobs from one run are queued or running at a time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import select
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from . import gpu_noise

SECRET_ENV = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
_HEADER = struct.Struct("!Q")


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("client closed the connection")
        data += chunk
    return data


def _address(path: Path) -> tuple[int, str]:
    """A short name for *path*: Unix socket paths are limited to 108 bytes."""
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    return fd, f"/proc/self/fd/{fd}/{path.name}"


def _client_gone(conn: socket.socket) -> bool:
    readable, _, _ = select.select([conn], [], [], 0)
    return bool(readable) and not conn.recv(1, socket.MSG_PEEK)


def request(socket_path: Path, argv: list[str]) -> int:
    """Client: run *argv* on a GPU and return its exit status."""
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    payload = json.dumps({"argv": argv, "cwd": os.getcwd(), "env": env}).encode()
    fd, address = _address(socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(address)
        os.close(fd)
        socket.send_fds(conn, [_HEADER.pack(len(payload))],
                        [sys.stdout.fileno(), sys.stderr.fileno()])
        conn.sendall(payload)
        reply = json.loads(_recv_exact(conn, _HEADER.unpack(_recv_exact(conn, _HEADER.size))[0]))
    if "error" in reply:
        print(f"gpu_run: {reply['error']}", file=sys.stderr)
        return 125
    return reply["returncode"]


def _minutes(value: str) -> float:
    """Minutes of a Slurm time limit (MM, HH:MM:SS, MM:SS or D-HH:MM:SS)."""
    days, _, rest = value.rpartition("-")
    parts = [float(x) for x in rest.split(":")]
    if len(parts) == 3:
        minutes = parts[0] * 60 + parts[1] + parts[2] / 60
    elif len(parts) == 2:
        minutes = parts[0] + parts[1] / 60
    else:
        minutes = parts[0]
    return minutes + (float(days) * 1440 if days else 0)


def missing_command(argv: list[str], cwd: str, env: dict) -> str | None:
    """Why *argv* cannot start, found on the host before a job waits in the queue."""
    import shutil

    if not argv:
        return "empty command"
    program = argv[0]
    if "/" in program:
        path = Path(program) if Path(program).is_absolute() else Path(cwd) / program
        if not path.exists():
            return f"{program}: no such file (in {cwd})"
        if not os.access(path, os.X_OK):
            return f"{program}: not executable"
        return None
    if shutil.which(program, path=env.get("PATH", "")) is None:
        return f"{program}: command not found on PATH"
    return None


def choose_partition(windows: dict, gpus: int, minutes: float,
                     order: tuple[str, ...] = ("main", "priority")) -> str | None:
    """The first partition of *order* where a job of *gpus* and *minutes* starts now.

    *windows* is the JSON of ``slurm-windows --json``. None when no partition starts it now.
    """
    from datetime import datetime

    now = datetime.fromisoformat(windows["now"])
    for name in order:
        part = next((p for p in windows.get("partitions", []) if p["partition"] == name), None)
        if part is None:
            continue
        for group in part["groups"]:
            if not group["gpus_min"] <= gpus <= group["gpus_max"]:
                continue
            for option in group["options"]:
                if option["max_time"] != "unlimited" and _minutes(option["max_time"]) < minutes:
                    continue
                start = option["start"]
                if start == "now" or datetime.fromisoformat(start) <= now:
                    return name
                break  # the first option that fits the time limit decides
    return None


class Broker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.slots = threading.BoundedSemaphore(args.max_jobs)
        self.count = itertools.count(1)
        self.tag = uuid.uuid4().hex[:8]
        self.env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
        self.sampler = (gpu_noise.NodeSampler(args.sample_ms).start()
                        if args.sample_ms > 0 else None)
        try:
            self.project: Path | None = Path(json.loads(args.spec.read_text())["project"])
        except (OSError, ValueError, KeyError):
            self.project = None

    def partitioned(self, srun: list[str]) -> list[str]:
        """With --partition-policy auto: main when it starts the job now, else priority
        when that starts it now, else main (the job then waits in main)."""
        if self.args.partition_policy != "auto":
            return srun
        gpus, minutes = 1, None
        for arg in srun:
            if arg.startswith("--gres=gpu:"):
                gpus = int(arg.rsplit(":", 1)[1])
            elif arg.startswith("--time="):
                minutes = _minutes(arg.split("=", 1)[1])
        if minutes is None:
            return srun
        try:
            out = subprocess.run(["slurm-windows", "--json"], capture_output=True, text=True,
                                 timeout=30, env=self.env)
            chosen = choose_partition(json.loads(out.stdout), gpus, minutes) or "main"
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            return srun
        return [arg if not arg.startswith("--partition=") else f"--partition={chosen}"
                for arg in srun]

    def gpu_report(self, name: str, path: Path, t0: float, t1: float, fd: int, code: int,
                   argv: list[str]) -> None:
        """Tell the caller (and reports/gpu-jobs.jsonl) the GPU state during the job."""
        gpu_file = path.with_suffix(".gpu")
        try:
            index = int(gpu_file.read_text().split(",")[0])
        except (OSError, ValueError):
            return
        finally:
            gpu_file.unlink(missing_ok=True)
        summary = gpu_noise.summarize_window(
            self.sampler.window(t0, t1) if self.sampler else [], index)
        line = gpu_noise.format_window(name, summary)
        self.log(line.removeprefix("gpu_run: "))
        try:
            os.write(fd, (line + "\n").encode())
        except OSError:
            pass
        if self.project is not None and (self.project / "reports").is_dir():
            row = {"job": name, "argv": argv, "exit": code, "seconds": round(t1 - t0, 1),
                   "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), **summary}
            with (self.project / "reports/gpu-jobs.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")

    def log(self, message: str) -> None:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)

    def handle(self, conn: socket.socket) -> None:
        fds: list[int] = []
        try:
            header, fds, _, _ = socket.recv_fds(conn, _HEADER.size, 2)
            if len(header) < _HEADER.size or len(fds) != 2:
                raise ValueError("bad request header")
            body = json.loads(_recv_exact(conn, _HEADER.unpack(header)[0]))
            reply = self.run(conn, body, fds)
        except Exception as exc:  # the client gets the reason; the broker keeps serving
            reply = {"error": str(exc)}
        finally:
            for fd in fds:
                os.close(fd)
        try:
            data = json.dumps(reply).encode()
            conn.sendall(_HEADER.pack(len(data)) + data)
        except OSError:
            pass
        conn.close()

    def run(self, conn: socket.socket, body: dict, fds: list[int]) -> dict:
        problem = missing_command(body.get("argv", []), body.get("cwd", "."), body.get("env", {}))
        if problem is not None:  # no GPU job: it would fail at once after its queue wait
            return {"error": f"{problem}; no GPU job was started"}
        while not self.slots.acquire(timeout=1):
            if _client_gone(conn):
                return {"error": "client went away while waiting for a slot"}
        try:
            number = next(self.count)
            name = f"vs-gpu-{self.tag}-{number}"
            path = self.args.requests / f"{name}.json"
            path.write_text(json.dumps(body))
            command = [*self.partitioned(self.args.srun), f"--job-name={name}",
                       str(self.args.job_python),
                       str(self.args.job_script), "run", str(self.args.spec), str(path)]
            self.log(f"{name} start {json.dumps(body['argv'])}")
            started, t0 = time.monotonic(), time.time()
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=fds[0],
                                    stderr=fds[1], env=self.env)
            while True:
                try:
                    code = proc.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if _client_gone(conn):
                        self.log(f"{name} client gone; cancel")
                        subprocess.run(["scancel", f"--name={name}"], check=False, env=self.env)
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return {"error": "client went away; job cancelled"}
            self.log(f"{name} exit {code} after {time.monotonic() - started:.0f}s")
            self.gpu_report(name, path, t0, time.time(), fds[1], code, body["argv"])
            path.unlink(missing_ok=True)
            return {"returncode": code}
        finally:
            self.slots.release()


def serve(args: argparse.Namespace) -> int:
    args.requests.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    broker = Broker(args)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        fd, address = _address(args.socket)
        server.bind(address)
        os.close(fd)
        os.chmod(args.socket, 0o600)
        server.listen(16)
        broker.log(f"listening on {args.socket}; srun {' '.join(args.srun[1:])}")
        while True:
            conn, _ = server.accept()
            threading.Thread(target=broker.handle, args=(conn,), daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    client = sub.add_parser("run", help="run a command on a GPU (inside the sandbox)")
    client.add_argument("--socket", type=Path, required=True)
    client.add_argument("command", nargs=argparse.REMAINDER)
    server = sub.add_parser("serve", help="serve GPU requests (on the host)")
    server.add_argument("--socket", type=Path, required=True)
    server.add_argument("--requests", type=Path, required=True,
                        help="directory for request files, not visible to the agent")
    server.add_argument("--spec", type=Path, required=True, help="sandbox spec for gpu_job.py")
    server.add_argument("--job-python", type=Path, required=True,
                        help="Python with the VibeSys sandbox packages")
    server.add_argument("--job-script", type=Path,
                        default=Path(__file__).with_name("gpu_job.py"))
    server.add_argument("--max-jobs", type=int, default=1)
    server.add_argument("--partition-policy", choices=("auto", "fixed"), default="auto",
                        help="auto: main if it starts the job now, else priority if that "
                             "does, else main; fixed: the srun --partition as given")
    server.add_argument("--sample-ms", type=int, default=200,
                        help="nvidia-smi sampling period for the per-job GPU report (0: off)")
    server.add_argument("srun", nargs=argparse.REMAINDER,
                        help="-- srun and its arguments, for example: -- srun --gres=gpu:1")
    args = parser.parse_args(argv)
    if args.mode == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("give a command, for example: ./gpu_run nvidia-smi")
        return request(args.socket, command)
    args.srun = args.srun[1:] if args.srun[:1] == ["--"] else args.srun
    if not args.srun:
        parser.error("give the srun command after --")
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
