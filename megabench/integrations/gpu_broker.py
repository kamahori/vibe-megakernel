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

With ``--hold-s N`` the broker does not start one Slurm job per command. It gets an
allocation with ``salloc --no-shell`` (the same options) and runs each command as a step
(``srun --jobid``) in it. After a command ends, the broker keeps the allocation for N
seconds, so that the next command of the run starts at once instead of waiting in the
queue again. It releases the allocation after N idle seconds, or when less than
``--min-left-s`` of its time limit is left.
"""

from __future__ import annotations

import argparse
import itertools
import re
import tempfile
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


def _bytes(value: str) -> float:
    """Bytes of a Slurm memory value such as 48254344K or 64G (0 if not one)."""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)[nc]?", value.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * 1024 ** "_KMGT".index(m.group(2) or "_")


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


def _option(args: list[str], name: str) -> str | None:
    return next((a.split("=", 1)[1] for a in args if a.startswith(f"--{name}=")), None)


class Alloc:
    """A Slurm allocation that the broker holds between commands."""

    def __init__(self, jobid: str, limit_s: float) -> None:
        self.jobid, self.limit_s = jobid, limit_s
        self.granted = time.monotonic()
        self.idle_since: float | None = None

    def left(self) -> float:
        return self.granted + self.limit_s - time.monotonic()


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
        # Queue wait: seconds in which a job of this run waits in the Slurm queue and none
        # runs. A time limit on the agents' work (the microbench phase) adds it, so that a
        # full node does not use up the time of a phase that measures on the GPU.
        self.states: dict[str, str] = {}
        self.state_lock = threading.Lock()
        self.queue_wait = 0.0
        socket = getattr(args, "socket", None)  # in-process users (harness_model) have none
        self.wait_file = Path(socket).parent / "queue_wait_s" if socket else None
        threading.Thread(target=self._count_wait, daemon=True).start()
        # Allocations held between commands (--hold-s).
        self.idle: list[Alloc] = []
        self.held: dict[str, Alloc] = {}  # every allocation the broker has, idle or busy
        self.pool_lock = threading.Lock()
        def tool(name: str) -> str:  # the Slurm tool next to the given srun, else on PATH
            beside = Path(args.srun[0]).with_name(name)
            return str(beside) if "/" in args.srun[0] and beside.exists() else name
        self.salloc, self.scancel, self.squeue = tool("salloc"), tool("scancel"), tool("squeue")
        self.sacct = tool("sacct")
        if getattr(args, "hold_s", 0) > 0:
            threading.Thread(target=self._reap, daemon=True).start()

    def _min_left(self) -> float:
        if getattr(self.args, "min_left_s", None) is not None:
            return self.args.min_left_s
        limit = _option(self.args.srun, "time")
        return 0.5 * _minutes(limit) * 60 if limit else 300.0

    def _release(self, alloc: Alloc, why: str) -> None:
        with self.pool_lock:
            if self.held.pop(alloc.jobid, None) is None:
                return
        self.log(f"release allocation {alloc.jobid} ({why})")
        subprocess.run([self.scancel, alloc.jobid], check=False, env=self.env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _reap(self, tick: float = 2.0) -> None:
        while True:
            time.sleep(tick)
            now, drop = time.monotonic(), []
            with self.pool_lock:
                for alloc in list(self.idle):
                    if now - (alloc.idle_since or now) >= self.args.hold_s:
                        drop.append((alloc, "idle"))
                    elif alloc.left() < self._min_left():
                        drop.append((alloc, "time limit near"))
                    else:
                        continue
                    self.idle.remove(alloc)
            for alloc, why in drop:
                self._release(alloc, why)

    def _take(self) -> Alloc | None:
        with self.pool_lock:
            while self.idle:
                alloc = self.idle.pop()
                if alloc.left() >= self._min_left():
                    alloc.idle_since = None
                    return alloc
                threading.Thread(target=self._release, args=(alloc, "time limit near"),
                                 daemon=True).start()
        return None

    def _alive(self, alloc: Alloc) -> bool:
        try:
            out = subprocess.run([self.squeue, "-h", "-j", alloc.jobid, "-o", "%T"],
                                 capture_output=True, text=True, timeout=30, env=self.env)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return out.stdout.strip() == "RUNNING"

    def release_all(self) -> None:
        with self.pool_lock:
            held, self.idle = list(self.held.values()), []
        for alloc in held:
            self._release(alloc, "broker stops")

    def _give_back(self, alloc: Alloc) -> None:
        if alloc.left() < self._min_left():
            self._release(alloc, "time limit near")
            return
        if not self._alive(alloc):
            self.log(f"allocation {alloc.jobid} is no longer running; dropped")
            with self.pool_lock:
                self.held.pop(alloc.jobid, None)
            return
        alloc.idle_since = time.monotonic()
        with self.pool_lock:
            self.idle.append(alloc)

    def _allocate(self, conn: socket.socket, name: str) -> Alloc | dict:
        """A new allocation with the srun options, or an error reply."""
        opts = [a for a in self.partitioned(self.args.srun)[1:]
                if not a.startswith("--kill-on-bad-exit")]
        limit = _option(opts, "time")
        with tempfile.TemporaryFile("w+") as err:
            proc = subprocess.Popen([self.salloc, "--no-shell", f"--job-name={name}", *opts],
                                    stdin=subprocess.DEVNULL, stdout=err, stderr=err,
                                    env=self.env)
            self._state(name, "pending")
            try:
                while True:
                    try:
                        code = proc.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if _client_gone(conn):
                            self.log(f"{name} client gone while waiting for a GPU; cancel")
                            proc.terminate()
                            subprocess.run([self.scancel, f"--name={name}"], check=False,
                                           env=self.env)
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            return {"error": "client went away; job cancelled"}
            finally:
                self._state(name, None)
            err.seek(0)
            text = err.read()
        found = re.search(r"Granted job allocation (\d+)", text)
        if code != 0 or not found:
            tail = "\n".join(text.strip().splitlines()[-5:])
            return {"error": f"salloc failed (exit {code}): {tail}"}
        self.log(f"{name} allocation {found.group(1)} granted")
        alloc = Alloc(found.group(1), _minutes(limit) * 60 if limit else 1200.0)
        with self.pool_lock:
            self.held[alloc.jobid] = alloc
        return alloc

    def _count_wait(self, tick: float = 1.0) -> None:
        last_write = 0.0
        while True:
            time.sleep(tick)
            with self.state_lock:
                states = set(self.states.values())
            if "pending" in states and "running" not in states:
                self.queue_wait += tick
            if self.wait_file and time.monotonic() - last_write >= 5:
                last_write = time.monotonic()
                try:
                    tmp = self.wait_file.with_suffix(".tmp")
                    tmp.write_text(f"{self.queue_wait:.0f}\n")
                    tmp.replace(self.wait_file)
                except OSError:
                    pass

    def _state(self, name: str, state: str | None) -> None:
        with self.state_lock:
            if state is None:
                self.states.pop(name, None)
            else:
                self.states[name] = state

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

    def oom_report(self, name: str, fd: int, jobid: str | None = None, tries: int = 5) -> bool:
        """If Slurm stopped job or step *name* for host memory, tell the caller how and why.

        Without this the caller sees only an exit code and a slurmstepd line; agents then
        rebuilt framework tools to fit the limit instead of reporting the limit.
        """
        rows = []
        for attempt in range(tries):  # accounting can lag the end of the job by seconds
            command = [self.sacct, "-n", "-P", "-o", "JobName,State,MaxRSS,ReqMem"]
            command += ["-j", jobid] if jobid else [f"--name={name}", "-S", "now-2days"]
            try:
                out = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                     env=self.env).stdout
            except (OSError, subprocess.TimeoutExpired):
                return False
            rows = [r.split("|") for r in out.splitlines() if r.count("|") == 3]
            rows = [r for r in rows if r[0] == name or (not jobid and r[0] in ("batch", "extern", ""))]
            if any("OUT_OF_MEMORY" in r[1] for r in rows):
                break
            if attempt + 1 < tries:
                time.sleep(2)
        hit = [r for r in rows if "OUT_OF_MEMORY" in r[1]]
        if not hit:
            return False
        peak = max((r[2] for r in rows if r[2]), key=_bytes, default="?")
        limit = next((r[3] for r in rows if r[3]), "?")
        line = (f"gpu_run: OUT OF MEMORY: Slurm stopped {name}: it used more host memory than "
                f"the job's limit ({limit}; peak seen {peak}). This is a limit of the run's setup, "
                "not of the candidate. Report it as a harness fault in your result. A score from a "
                "changed copy of a framework tool is not evidence for the gate.")
        self.log(line.removeprefix("gpu_run: "))
        try:
            os.write(fd, (line + "\n").encode())
        except OSError:
            pass
        return True

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
            if getattr(self.args, "hold_s", 0) > 0:
                return self._run_held(conn, body, fds)
            number = next(self.count)
            name = f"vs-gpu-{self.tag}-{number}"
            path = self.args.requests / f"{name}.json"
            srun = self.partitioned(self.args.srun)
            limit = next((a.split("=", 1)[1] for a in srun if a.startswith("--time=")), None)
            if limit:
                body["job_limit_s"] = _minutes(limit) * 60  # gpu_job stops a hang before it
            path.write_text(json.dumps(body))
            command = [*srun, f"--job-name={name}",
                       str(self.args.job_python),
                       str(self.args.job_script), "run", str(self.args.spec), str(path)]
            self.log(f"{name} start {json.dumps(body['argv'])}")
            started, t0 = time.monotonic(), time.time()
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=fds[0],
                                    stderr=fds[1], env=self.env)
            # gpu_job writes <request>.gpu when the job starts on its GPU.
            self._state(name, "pending")
            while True:
                if self.states.get(name) == "pending" and path.with_suffix(".gpu").exists():
                    self._state(name, "running")
                try:
                    code = proc.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if _client_gone(conn):
                        self.log(f"{name} client gone; cancel")
                        subprocess.run([self.scancel, f"--name={name}"], check=False, env=self.env)
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        self._state(name, None)
                        return {"error": "client went away; job cancelled"}
            self._state(name, None)
            self.log(f"{name} exit {code} after {time.monotonic() - started:.0f}s")
            self.gpu_report(name, path, t0, time.time(), fds[1], code, body["argv"])
            if code != 0:
                self.oom_report(name, fds[1])
            path.unlink(missing_ok=True)
            return {"returncode": code}
        finally:
            self.slots.release()

    def _run_held(self, conn: socket.socket, body: dict, fds: list[int]) -> dict:
        """Run the command as a step of a held allocation (a new one if none is idle)."""
        number = next(self.count)
        name = f"vs-gpu-{self.tag}-{number}"
        path = self.args.requests / f"{name}.json"
        alloc = self._take()
        if alloc is None:
            got = self._allocate(conn, name)
            if isinstance(got, dict):
                return got
            alloc = got
        else:
            self.log(f"{name} reuses allocation {alloc.jobid} ({alloc.left():.0f}s left)")
        body["job_limit_s"] = max(60.0, alloc.left() - 30)  # gpu_job stops a hang before it
        path.write_text(json.dumps(body))
        step = [self.args.srun[0], f"--jobid={alloc.jobid}", "--ntasks=1", "--nodes=1",
                *(a for a in self.args.srun if a.startswith(("--cpus-per-task=", "--gres="))),
                "--kill-on-bad-exit=1", f"--job-name={name}",
                str(self.args.job_python), str(self.args.job_script), "run",
                str(self.args.spec), str(path)]
        self.log(f"{name} start {json.dumps(body['argv'])}")
        started, t0 = time.monotonic(), time.time()
        proc = subprocess.Popen(step, stdin=subprocess.DEVNULL, stdout=fds[0], stderr=fds[1],
                                env=self.env)
        self._state(name, "running")
        try:
            while True:
                try:
                    code = proc.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if _client_gone(conn):
                        self.log(f"{name} client gone; cancel")
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        self._release(alloc, "client gone")
                        return {"error": "client went away; job cancelled"}
        finally:
            self._state(name, None)
        self.log(f"{name} exit {code} after {time.monotonic() - started:.0f}s")
        self.gpu_report(name, path, t0, time.time(), fds[1], code, body["argv"])
        if code != 0:
            self.oom_report(name, fds[1], jobid=alloc.jobid)
        path.unlink(missing_ok=True)
        self._give_back(alloc)
        return {"returncode": code}


def serve(args: argparse.Namespace) -> int:
    args.requests.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    broker = Broker(args)
    import signal

    def stop(signum, frame):  # held allocations must not outlive the broker
        broker.release_all()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
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
    server.add_argument("--hold-s", type=float, default=0,
                        help="keep a Slurm allocation this many idle seconds after a command "
                             "and run the next command in it (0: one job per command)")
    server.add_argument("--min-left-s", type=float, default=None,
                        help="reuse a held allocation only with this much time left "
                             "(default: half its time limit)")
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
