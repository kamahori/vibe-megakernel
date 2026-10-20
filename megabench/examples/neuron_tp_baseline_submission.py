"""Protocol smoke test for the Trainium TP4 cases: correct, but NOT a megakernel.

Each rank runs the eager TP baseline on its shard. The Neuron launch audit
should reject it: it issues many PyTorch ops and collectives per step.
"""

from megabench.baselines_neuron import build_tp_baseline
from megabench.cases import Case


def build(case: dict):
    challenge = Case(**{key: value for key, value in case.items() if key != "execution"})
    return build_tp_baseline(challenge, case["execution"])
