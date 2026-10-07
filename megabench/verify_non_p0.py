"""Verify full catalog geometries without treating eager references as candidates.

Run CUDA verification through the cluster scheduler. Reports are exclusive
JSON files so an existing experiment can never be overwritten.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch

from .cases import select_cases
from .harness.benchmark import _measure


MODULES = {"gptoss_step": "gptoss", "hybrid_step": "hybrid",
           "vl_decode_step": "vision", "spec_full_iteration": "speculative"}


def verify(case_id: str, device: str, trials: int) -> dict:
    case = select_cases("all", [case_id])[0]
    module = importlib.import_module("megabench.tasks." + MODULES[case.family])
    report = {"case": case.to_dict(), "device": device, "trials": [],
              "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.device(device).type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
    with torch.inference_mode():
        for index in range(trials):
            seed = 104729 + index
            start = time.monotonic()
            inputs = module.make_inputs(case, seed, device)
            scenario = "seeded"
            if case.family == "spec_full_iteration" and index % 3:
                prefix = case.params["draft_depth"] if index % 3 == 1 else 1
                module.set_acceptance_scenario(case, inputs, prefix)
                scenario = "accept_full" if index % 3 == 1 else "accept_one"
            expected = module.reference(case, inputs)
            repeated = module.reference(case, inputs)
            for name, output in expected.items():
                if output.is_floating_point() and not bool(torch.isfinite(output).all()):
                    raise AssertionError(f"non-finite {name}")
                torch.testing.assert_close(output, repeated[name], rtol=0, atol=0)
            first_logits = expected["logits"].clone()
            inputs["token"] = (inputs["token"] + 1) % case.params["vocab"]
            other = module.reference(case, inputs)
            if not any(not torch.equal(expected[name], other[name]) for name in expected):
                raise AssertionError("outputs did not respond to the runtime token")
            timing = _measure(lambda values: module.reference(case, values), inputs,
                              device, warmup=1, reps=3)
            report["trials"].append({"seed": seed, "scenario": scenario, "outputs": {
                name: {"shape": list(output.shape), "dtype": str(output.dtype)}
                for name, output in expected.items()}, "reference_timing": timing,
                "elapsed_seconds": time.monotonic() - start,
                "input_bytes": sum(value.numel() * value.element_size() for value in inputs.values())})
            print(json.dumps({"case": case.id, "seed": seed, "status": "pass",
                              "reference_cuda_event_p50_ms": timing["cuda_event_p50_ms"]}), flush=True)
            del inputs, expected, repeated, other, first_logits
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    report["status"] = "pass"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    with args.output.open("x") as output:
        report = [verify(case_id, args.device, args.trials) for case_id in args.case]
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
