"""HTTP CONNECT proxy that lets the agent container reach only allowlisted hosts.

The agent container has no network (``--network none``). ``serve`` runs in the
Slurm job outside the container and listens on a Unix socket whose directory is
bind-mounted into the container; ``forward`` runs inside the container and
exposes that socket as 127.0.0.1:3128, so the agent's only way out is
HTTPS_PROXY=http://127.0.0.1:3128. The proxy tunnels CONNECT requests to port
443 of hosts in --allow (exact names, or ``.suffix`` for a domain and its
subdomains) and refuses everything else, plain HTTP included. Every decision is
logged, one JSON line per request.

(Podman's rootless networks are not used: their uplink is one process per user
shared by all jobs, and it dies with the job that started it.)

Standard library only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time


def allowed(host: str, rules: list[str]) -> bool:
    host = host.lower().rstrip(".")
    for rule in rules:
        rule = rule.lower()
        if rule.startswith("."):
            if host == rule[1:] or host.endswith(rule):
                return True
        elif host == rule:
            return True
    return False


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except OSError:
            pass


class Proxy:
    def __init__(self, rules: list[str], log) -> None:
        self.rules = rules
        self.log = log

    def record(self, **fields) -> None:
        self.log.write(json.dumps({"t": round(time.time(), 3), **fields}) + "\n")
        self.log.flush()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                ConnectionError):
            writer.close()
            return
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            self.record(decision="deny", request=line[:200], why="only CONNECT is allowed")
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        host, _, port = parts[1].rpartition(":")
        host = host.strip("[]")
        if port != "443" or not allowed(host, self.rules):
            self.record(decision="deny", host=host, port=port)
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, 443), 30)
        except (OSError, asyncio.TimeoutError) as exc:
            self.record(decision="error", host=host, why=f"{type(exc).__name__}: {exc}")
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        self.record(decision="allow", host=host)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))


async def serve(args: argparse.Namespace) -> None:
    log = open(args.log, "a") if args.log else sys.stdout
    proxy = Proxy(args.allow, log)
    if args.unix:
        server = await asyncio.start_unix_server(proxy.handle, args.unix)
        os.chmod(args.unix, 0o600)
    else:
        server = await asyncio.start_server(proxy.handle, args.host, args.port)
    proxy.record(decision="start", allow=args.allow, listen=args.unix or args.port)
    async with server:
        await server.serve_forever()


async def forward(args: argparse.Namespace) -> None:
    """Expose the proxy's Unix socket as a TCP port on loopback (inside the container)."""
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(args.unix)
        except OSError:
            writer.close()
            return
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

    server = await asyncio.start_server(handle, "127.0.0.1", args.port)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    srv = sub.add_parser("serve", help="the allowlist proxy (outside the container)")
    srv.add_argument("--unix", help="listen on this Unix socket instead of TCP")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=3128)
    srv.add_argument("--allow", nargs="+", required=True,
                     help="host names; a leading dot allows a domain and its subdomains")
    srv.add_argument("--log", help="JSON-lines decision log (default: stdout)")
    fwd = sub.add_parser("forward", help="127.0.0.1:PORT -> the proxy socket (inside)")
    fwd.add_argument("--unix", required=True)
    fwd.add_argument("--port", type=int, default=3128)
    args = parser.parse_args(argv)
    asyncio.run(serve(args) if args.mode == "serve" else forward(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
