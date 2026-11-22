"""CPU checks for the containerized Claude Code / Codex runs (container_run)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from ..integrations import container_run

MEGABENCH = Path(__file__).resolve().parents[1]
CASE = "dense-step-qwen3-06b-b1-s128"


def _proxy_module():
    spec = importlib.util.spec_from_file_location("egress_proxy",
                                                  MEGABENCH / "docker/egress_proxy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EgressProxyTests(unittest.TestCase):
    def test_allowlist_matches_exact_names_and_dot_suffixes(self) -> None:
        allowed = _proxy_module().allowed
        rules = ["api.anthropic.com", ".openai.com"]
        self.assertTrue(allowed("api.anthropic.com", rules))
        self.assertTrue(allowed("API.Anthropic.com.", rules))
        self.assertFalse(allowed("evil-api.anthropic.com", rules))
        self.assertFalse(allowed("anthropic.com", rules))
        self.assertTrue(allowed("openai.com", rules))
        self.assertTrue(allowed("auth.openai.com", rules))
        self.assertFalse(allowed("notopenai.com", rules))
        self.assertFalse(allowed("github.com", rules))

    def test_refuses_other_hosts_other_ports_and_plain_http(self) -> None:
        proxy = _proxy_module()

        async def scenario() -> list[bytes]:
            log = open(os.devnull, "w")
            handler = proxy.Proxy(["api.anthropic.com"], log).handle
            server = await asyncio.start_server(handler, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            replies = []
            for request in (b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com\r\n\r\n",
                            b"CONNECT api.anthropic.com:22 HTTP/1.1\r\n\r\n",
                            b"GET http://api.anthropic.com/ HTTP/1.1\r\n\r\n"):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(request)
                await writer.drain()
                replies.append(await reader.read(100))
                writer.close()
            server.close()
            await server.wait_closed()
            log.close()
            return replies

        for reply in asyncio.run(scenario()):
            self.assertTrue(reply.startswith(b"HTTP/1.1 403"), reply)


    def test_forward_reaches_the_proxy_through_its_unix_socket(self) -> None:
        proxy = _proxy_module()

        async def scenario() -> bytes:
            with tempfile.TemporaryDirectory() as tmp:
                sock = str(Path(tmp) / "egress.sock")
                log = open(os.devnull, "w")
                server = await asyncio.start_unix_server(proxy.Proxy(["example.invalid"], log).handle, sock)
                import argparse
                forward = asyncio.create_task(proxy.forward(argparse.Namespace(unix=sock, port=0)))
                await asyncio.sleep(0)  # port 0 cannot be dialled: test the handler directly
                forward.cancel()
                reader, writer = await asyncio.open_unix_connection(sock)
                writer.write(b"CONNECT github.com:443 HTTP/1.1\r\n\r\n")
                await writer.drain()
                reply = await reader.read(100)
                writer.close()
                server.close()
                log.close()
                return reply

        self.assertTrue(asyncio.run(scenario()).startswith(b"HTTP/1.1 403"))


class PrepareTests(unittest.TestCase):
    def prepare(self, tmp: Path, *extra: str) -> Path:
        key = tmp / "key"
        key.write_text("not-a-real-token\n")
        key.chmod(0o600)
        run = tmp / "run"
        argv = ["prepare", "--case", CASE, "--run-dir", str(run),
                "--image", "localhost/megabench-agent:test", "--credential-file", str(key), *extra]
        self.assertEqual(container_run.main(argv), 0)
        return run

    def test_claude_run_isolates_home_and_points_trusted_python_at_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp), "--agent", "claude",
                               "--credential-kind", "claude-oauth-token", "--model", "opus")
            self.assertEqual(os.readlink(run / "trusted/.venv"), container_run.IMAGE_VENV)
            self.assertFalse((run / "trusted/megabench/submissions").exists())
            workspace = run / CASE
            self.assertTrue((workspace / "TASK.md").is_file())
            self.assertFalse((workspace / "gpu_run").exists())  # the GPU is in the container
            check = (workspace / "check_candidate").read_text()
            self.assertIn(str(run / "trusted/.venv/bin/python"), check)
            settings = json.loads((run / "home/.claude/settings.json").read_text())
            self.assertEqual(set(settings["permissions"]["deny"]), {"WebFetch", "WebSearch"})
            state = json.loads((run / "home/.claude.json").read_text())
            self.assertTrue(state["projects"][str(workspace)]["hasTrustDialogAccepted"])
            env = dict(line.split("=", 1) for line in (run / "run.env").read_text().splitlines())
            env = {k: shlex.split(v)[0] if v else "" for k, v in env.items()}
            self.assertEqual(env["CREDENTIAL_ENV"], "CLAUDE_CODE_OAUTH_TOKEN")
            self.assertEqual(env["ALLOW_HOSTS"], "api.anthropic.com")
            self.assertEqual(env["MODE"], "headless")
            self.assertNotIn("not-a-real-token", (run / "run.env").read_text())
            self.assertIn(str(run / "state"), env["FORBID"].split())
            goal = (run / "control/goal.txt").read_text()
            self.assertNotIn("\n", goal.strip())
            self.assertTrue(os.access(run / "round.sh", os.X_OK))
            self.assertTrue(os.access(run / "control/entry.sh", os.X_OK))
            self.assertTrue((run / "control/egress_proxy.py").is_file())

    def test_codex_run_disables_web_search_and_defaults_to_tmux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp), "--agent", "codex", "--credential-kind",
                               "openai-api-key", "--model", "gpt-5.5", "--effort", "high")
            config = tomllib.loads((run / "home/.codex/config.toml").read_text())
            self.assertEqual(config["web_search"], "disabled")
            self.assertEqual(config["model"], "gpt-5.5")
            self.assertEqual(config["projects"][str(run / CASE)]["trust_level"], "trusted")
            meta = json.loads((run / "config.json").read_text())
            self.assertEqual(meta["mode"], "tmux")
            self.assertIn("api.openai.com", meta["allow_hosts"])

    def test_rejects_wrong_credential_kind_an_open_key_and_too_long_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            key = tmp / "key"
            key.write_text("x")
            base = ["prepare", "--case", CASE, "--image", "i", "--credential-file", str(key)]
            key.chmod(0o644)
            with self.assertRaises(SystemExit):
                container_run.main([*base, "--run-dir", str(tmp / "a"), "--agent", "claude",
                                    "--credential-kind", "claude-oauth-token"])
            key.chmod(0o600)
            with self.assertRaises(SystemExit):
                container_run.main([*base, "--run-dir", str(tmp / "b"), "--agent", "claude",
                                    "--credential-kind", "openai-api-key"])
            with self.assertRaises(SystemExit):
                container_run.main([*base, "--run-dir", str(tmp / "c"), "--agent", "claude",
                                    "--credential-kind", "claude-oauth-token",
                                    "--agent-minutes", "110"])
            self.assertFalse(any((tmp / name).exists() for name in "abc"))


class ScriptTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for name in ("container_round.sh", "container_entry.sh"):
            path = MEGABENCH / "integrations" / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            self.assertTrue(os.access(path, os.X_OK), name)

    def test_containers_have_no_network_and_skip_the_banner_entrypoint(self) -> None:
        text = (MEGABENCH / "integrations/container_round.sh").read_text()
        runs = [line for line in text.splitlines() if "podman run" in line]
        self.assertGreaterEqual(len(runs), 3)
        for line in runs:
            if '"${common[@]}"' not in line:
                self.assertIn("--entrypoint=", line)
                self.assertIn("--network none", line)
        self.assertIn("common=(--network none --userns=keep-id --entrypoint=", text)
        self.assertNotIn("podman network", text)  # the shared rootless uplink dies with its job

    def test_round_script_never_runs_git_on_the_host(self) -> None:
        text = (MEGABENCH / "integrations/container_round.sh").read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("git ", "git\t")):
                self.fail(f"host-side git on the agent's repository: {line}")


if __name__ == "__main__":
    unittest.main()


class ExperimentToolsTests(unittest.TestCase):
    """Overlays, ./time_left, check snapshots, the round budget, harvest and report."""

    def prepare(self, tmp: Path, *extra: str) -> Path:
        key = tmp / "key"
        key.write_text("not-a-real-token\n")
        key.chmod(0o600)
        overlay = tmp / "overlay"
        (overlay / "TASK.d").mkdir(parents=True)
        (overlay / "TASK.d/arm.md").write_text("arm notes\n")
        run = tmp / "run"
        argv = ["prepare", "--agent", "claude", "--credential-kind", "claude-oauth-token",
                "--case", CASE, "--run-dir", str(run), "--image", "localhost/megabench-agent:test",
                "--credential-file", str(key), "--overlay", str(overlay), "--rounds", "3",
                "--agent-minutes", "10", "--budget-minutes", "25", *extra]
        self.assertEqual(container_run.main(argv), 0)
        return run

    def test_workspace_tools_and_snapshot_of_a_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp))
            ws = run / CASE
            self.assertEqual((ws / "TASK.d/arm.md").read_text(), "arm notes\n")
            status = subprocess.run(["git", "-C", str(ws), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout
            self.assertEqual(status, "")  # overlays and tools are committed
            self.assertIn("25", (run / "control/kickoff.txt").read_text())
            check = (ws / "check_candidate").read_text()
            self.assertIn(".harness/pending/$run_id.tgz", check)
            # The inserted line, run on its own, tars the sources but not reports/ or .git.
            line = next(l for l in check.splitlines() if l.startswith("mkdir -p .harness"))
            (ws / "reports/check-x.jsonl").write_text("{}\n")
            (ws / "mk.cu").write_text("// kernel\n")
            subprocess.run(["sh", "-c", "run_id=t1\n" + line], cwd=ws, check=True)
            names = subprocess.run(["tar", "-tzf", str(ws / ".harness/pending/t1.tgz")],
                                   capture_output=True, text=True, check=True).stdout.split()
            self.assertIn("./mk.cu", names)
            self.assertIn("./submission.py", names)
            self.assertFalse(any(n.startswith(("./reports", "./.git/", "./.harness")) for n in names))
            env = dict(line.split("=", 1) for line in (run / "run.env").read_text().splitlines())
            self.assertEqual(shlex.split(env["BUDGET_SECONDS"])[0], "1500")

    def test_remove_drops_stock_paths_before_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp), "--remove", "megakernel_skills", "--remove", "primdb")
            ws = run / CASE
            self.assertFalse((ws / "megakernel_skills").exists())
            self.assertFalse((ws / "primdb").exists())
            self.assertTrue((ws / "check_candidate").exists())
            self.assertEqual(json.loads((run / "config.json").read_text())["removed"],
                             ["megakernel_skills", "primdb"])
            status = subprocess.run(["git", "-C", str(ws), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout
            self.assertEqual(status, "")
            for bad in ("check_candidate", "../x", "missing_tool"):
                with self.subTest(bad=bad), tempfile.TemporaryDirectory() as other:
                    with self.assertRaises(SystemExit):
                        self.prepare(Path(other), "--remove", bad)

    def test_trusted_hide_gives_the_agent_a_stripped_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp), "--trusted-hide", "megabench/skills",
                               "--trusted-hide", "megabench/primdb")
            agent, full = run / "trusted-agent", run / "trusted"
            self.assertFalse((agent / "megabench/skills").exists())
            self.assertFalse((agent / "megabench/primdb").exists())
            self.assertTrue((full / "megabench/skills").is_dir())     # grading keeps everything
            self.assertTrue((agent / "megabench/harness").is_dir())
            self.assertEqual(os.readlink(agent / ".venv"), os.readlink(full / ".venv"))
            env = dict(line.split("=", 1) for line in (run / "run.env").read_text().splitlines())
            self.assertEqual(shlex.split(env["AGENT_TRUSTED"])[0], str(agent))
            round_sh = (run / "round.sh").read_text()
            self.assertIn('-v "${AGENT_TRUSTED:-$TRUSTED}:$TRUSTED:ro"', round_sh)
            self.assertIn('-v "$TRUSTED:$TRUSTED:ro" -v "$1:$1"', round_sh)  # grade_dir: full snapshot
            for bad in ("megabench/harness", "../x", "megabench/nope"):
                with self.subTest(bad=bad), tempfile.TemporaryDirectory() as other:
                    with self.assertRaises(SystemExit):
                        self.prepare(Path(other), "--trusted-hide", bad)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp))
            self.assertFalse((run / "trusted-agent").exists())

    def test_budget_harvest_and_report(self) -> None:
        harvest = _harvest_module()
        with tempfile.TemporaryDirectory() as tmp:
            run = self.prepare(Path(tmp))
            ws = run / CASE
            # Round 1 gets a full 10-minute slot, round 3 only what is left of 25 minutes.
            self.assertEqual(harvest.budget(run, 1), 600)
            (run / "state/round-1.started").touch()
            (run / "state/round-1.elapsed").write_text("600 124\n")
            (run / "state/round-2.elapsed").write_text("600 124\n")
            self.assertEqual(harvest.budget(run, 3), 300)
            state = json.loads((run / "control/round.json").read_text())
            self.assertTrue(state["last_slot"])
            left = subprocess.run([str(ws / "time_left")], capture_output=True, text=True).stdout
            self.assertTrue(left.startswith("4 minutes left") or left.startswith("5 minutes left"), left)
            (run / "state/round-3.elapsed").write_text("300 124\n")
            self.assertEqual(harvest.budget(run, 4), 0)
            # A passing and a failing check: only the passing one becomes a snapshot.
            pending = ws / ".harness/pending"
            pending.mkdir(parents=True)
            for check_id, status, p50 in (("a", "ok_provisional", 40.0), ("b", "incorrect", None)):
                (pending / f"{check_id}.tgz").write_bytes(b"tar")
                (ws / f"reports/check-{check_id}.jsonl").write_text(json.dumps(
                    {"status": status, "candidate_timing": {"cuda_event_p50_ms": p50}}) + "\n")
            (pending / "c.tgz").write_bytes(b"tar")  # still running: no report yet
            self.assertEqual(harvest.harvest(run), 1)
            self.assertEqual(harvest.harvest(run), 0)
            self.assertEqual([p.name for p in harvest.candidates(run)], ["001-a.tgz"])
            ids = [json.loads(l)["id"] for l in (run / "logs/checks.jsonl").read_text().splitlines()]
            self.assertEqual(ids, ["a", "b"])
            # Rescore of the snapshot: 3 passing checks with one launch per rank.
            rescored = run / "rescore/001-a-99"
            (rescored / "src").mkdir(parents=True)
            (rescored / "review.json").write_text("[]\n")
            for k, p50 in enumerate((41.0, 40.0, 42.0), 1):
                (rescored / f"check-{k}.jsonl").write_text(json.dumps({
                    "status": "ok_provisional", "candidate_timing": {"cuda_event_p50_ms": p50},
                    "launch_audit": {"status": "within_budget", "per_rank": [
                        {"gpu_kernel_count": 1, "max_gpu_launches": 1}] * 4}}) + "\n")
            (run / "logs/agent-round-1.out").write_text("\n".join(json.dumps(e) for e in (
                {"type": "assistant", "message": {"content": [{"type": "tool_use"}]}},
                {"type": "user", "message": {"content": "Stop hook feedback: not yet"}},
                {"type": "result", "total_cost_usd": 1.5})) + "\n")
            (run / "logs/agent-round-1.times").write_text("\n".join(
                str((run / "state/round-1.started").stat().st_mtime + t) for t in (30, 40, 50)) + "\n")
            self.assertEqual(container_run.main(["report", "--run-dir", str(run)]), 0)
            result = json.loads((run / "result.json").read_text())
            self.assertEqual(result["L_ms"], 41.0)
            self.assertEqual(result["goal_reprompts"], 1)
            self.assertEqual(result["cost_usd"], 1.5)
            self.assertEqual(result["checks"], {"n": 2, "passing": 1})

    def test_review_flags_loaded_tensors_and_reference_imports(self) -> None:
        harvest = _harvest_module()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            (src / "submission.py").write_text(
                "import torch\nimport helper\nx = torch.load('answers.pt')\nsrc = HERE / 'csrc' / 'mk.cu'\n")
            (src / "csrc").mkdir()
            (src / "csrc/mk.cu").write_text('#include "inc.cuh"\n')
            (src / "csrc/inc.cuh").write_text("// reads oracle values\n")
            (src / "helper.py").write_text("from megabench.tasks import dsv41\n")
            (src / "tools").mkdir()
            (src / "tools/dbg.py").write_text("from megabench.workloads import oracle\n")
            problems = harvest.review(src)
            # tools/dbg.py is not reached from submission.py, so it is not reviewed.
            self.assertEqual(sorted(problems), ["csrc/inc.cuh: mentions an oracle",
                                                "helper.py: imports megabench", "submission.py: torch.load"])


def _harvest_module():
    from ..integrations import container_harvest
    return container_harvest
