"""Compatibility entry points for the MegaBench evaluation harness."""

import subprocess  # Retained for existing callers that patch runner.subprocess.run.

from .harness.benchmark import _audit_launches, _measure
from .harness.correctness import _compare
from .harness.runner import (
    CONTRACT_FILES, ROOT, _contract_digest, _docker_worker_command,
    _load_submission, _output_path, _run_one, _stop_owned_container, _worker,
    aggregate_sessions, evaluate_case, main,
)
