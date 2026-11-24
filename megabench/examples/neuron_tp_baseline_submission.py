"""Protocol smoke test for the Trainium cases: correct, but NOT a megakernel.

A TP case's ranks each run the eager TP baseline on their shard; a one-core
case runs the eager reference on its NeuronCore. The Neuron launch audit
should reject both: they issue many PyTorch ops (and collectives) per step.
"""

from dataclasses import replace

from megabench.baselines_neuron import build_tp_baseline
from megabench.cases import Case
from megabench.tasks.sharding import TP_FAMILIES
from megabench.tasks.workloads import reference


def build(case: dict):
    challenge = Case(**{key: value for key, value in case.items() if key != "execution"})
    if challenge.gpus == 1 and challenge.family not in TP_FAMILIES:
        single = replace(challenge, backend="cuda")
        return lambda inputs: reference(single, inputs)
    return build_tp_baseline(challenge, case["execution"])
