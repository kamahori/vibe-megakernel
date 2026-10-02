"""Run the locally testable portion of the ForgeMegakernel reproduction."""

from __future__ import annotations

import argparse
import json

import torch

from .candidate import decode_candidate
from .config import ModelConfig, QWEN3_06B, accessed_bytes
from .oracle import evaluate_cpu
from .reference import empty_cache, random_weights
from .schedule import audit_schedule, compile_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--schedule-sms", type=int, default=8)
    args = parser.parse_args()
    cfg = ModelConfig()
    if args.batch <= 0 or not 1 <= args.context <= cfg.max_seq:
        parser.error("batch must be positive and context inside toy max_seq")
    weights = random_weights(cfg)
    tokens = torch.arange(args.batch, dtype=torch.long) % cfg.vocab
    positions = torch.arange(args.batch, dtype=torch.long) % args.context
    if args.batch > 1:
        positions[-1] = args.context - 1
    result = evaluate_cpu(cfg, weights, tokens, positions,
                          empty_cache(cfg, args.batch), decode_candidate)
    report = result.to_dict()
    report["schedule_model"] = audit_schedule(compile_schedule(cfg, args.schedule_sms), cfg.layers)
    report["qwen3_06b_eq2_example_bytes_b1_s128"] = accessed_bytes(QWEN3_06B, 1, 128)
    print(json.dumps(report, indent=2))
    if result.status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
