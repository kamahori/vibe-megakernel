"""Read-only Qwen3 checkpoint diagnostics against independent CPU decode paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .candidate import decode_candidate
from .checkpoint import checkpoint_sha256, load_qwen3_checkpoint
from .config import accessed_bytes, config_hash
from .oracle import relative_row_error, top_logprob_error
from .reference import decode_reference, empty_cache


def evaluate_checkpoint(snapshot: Path, tokens: list[int], *, hf: bool = False) -> dict[str, object]:
    if not tokens:
        raise ValueError("at least one teacher-forced token is required")
    cfg, weights = load_qwen3_checkpoint(snapshot, max_seq=len(tokens))
    if any(token < 0 or token >= cfg.vocab for token in tokens):
        raise ValueError("token ID outside checkpoint vocabulary")
    caches = {
        "golden": empty_cache(cfg, 1),
        "fp32": empty_cache(cfg, 1),
        "candidate": empty_cache(cfg, 1),
    }
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in caches}
    b_errors, c_errors = [], []
    with torch.no_grad():
        for position, token in enumerate(tokens):
            one_token = torch.tensor([token], dtype=torch.long)
            one_position = torch.tensor([position], dtype=torch.long)
            golden = decode_reference(cfg, weights, one_token, one_position,
                                      caches["golden"], torch.float64)
            fp32 = decode_reference(cfg, weights, one_token, one_position,
                                    caches["fp32"], torch.float32)
            candidate = decode_candidate(cfg, weights, one_token, one_position,
                                         caches["candidate"])
            for name, result in (("golden", golden), ("fp32", fp32), ("candidate", candidate)):
                caches[name] = (result.keys, result.values)
                logits[name].append(result.logits[0].cpu())
            for row_name in ("keys", "values"):
                golden_rows = getattr(golden, row_name)[:, 0, position]
                fp32_rows = getattr(fp32, row_name)[:, 0, position]
                candidate_rows = getattr(candidate, row_name)[:, 0, position]
                b_errors.append(relative_row_error(fp32_rows, golden_rows))
                c_errors.append(relative_row_error(candidate_rows, fp32_rows))
    stacked = {name: torch.stack(items) for name, items in logits.items()}
    kv_b = sum(b_errors) / len(b_errors)
    kv_c = sum(c_errors) / len(c_errors)
    candidate_error, candidate_max = top_logprob_error(stacked["candidate"], stacked["golden"])
    fp32_error, _ = top_logprob_error(stacked["fp32"], stacked["golden"])
    details: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha256(snapshot / "model.safetensors"),
        "config_hash": config_hash(cfg, 1, len(tokens)),
        "tokens": tokens,
        "eq2_bytes_bf16_last_cell": accessed_bytes(cfg, 1, len(tokens)),
        "kv_b_fp32_vs_fp64": kv_b, "kv_c_candidate_vs_fp32": kv_c,
        "kv_bar_ok": kv_c < 2 * kv_b,
        "candidate_logprob_error": candidate_error,
        "candidate_logprob_error_max": candidate_max,
        "fp32_logprob_error": fp32_error,
        "candidate_top1": stacked["candidate"].argmax(-1).tolist(),
        "golden_top1": stacked["golden"].argmax(-1).tolist(),
        "finite_logits": bool(torch.isfinite(stacked["candidate"]).all()),
        "paper_milestones": "abstain: CPU candidate, no GPU launch or SGLang baseline",
    }
    if hf:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        hf_logits = []
        with torch.no_grad():
            for end in range(1, len(tokens) + 1):
                prefix = torch.tensor([tokens[:end]], dtype=torch.long)
                hf_logits.append(model(prefix).logits[0, -1].float().cpu())
        hf_stacked = torch.stack(hf_logits)
        hf_error, hf_max = top_logprob_error(hf_stacked, stacked["golden"])
        details.update(hf_bf16_logprob_error=hf_error,
                       hf_bf16_logprob_error_max=hf_max,
                       candidate_under_5d=candidate_error <= 5 * hf_error,
                       hf_top1=hf_stacked.argmax(-1).tolist())
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--hf", action="store_true", help="also measure Hugging Face bf16 error")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = evaluate_checkpoint(args.snapshot, args.tokens, hf=args.hf)
    print(json.dumps(result, indent=2, sort_keys=True))
    if (not result["kv_bar_ok"] or not result["finite_logits"] or
            (args.hf and not result["candidate_under_5d"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
