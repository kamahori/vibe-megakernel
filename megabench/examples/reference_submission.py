"""Protocol smoke test only: correct, but deliberately NOT a megakernel.

The GPU launch audit should reject this multi-operation PyTorch baseline.
"""

from megabench.cases import Case
from megabench.workloads import reference


def build(case: dict):
    challenge = Case(**{key: value for key, value in case.items() if key != "execution"})

    def run(inputs: dict):
        return reference(challenge, inputs)

    return run
