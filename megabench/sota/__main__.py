"""``python -m megabench.sota {run,report,env}``."""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("run", "report", "env"):
        print("usage: python -m megabench.sota {run,report,env} [options]", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        from .run import main as run_main
        return run_main(rest)
    if cmd == "report":
        from .report import main as report_main
        return report_main(rest)
    from .records import env_info
    print(json.dumps(env_info(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
