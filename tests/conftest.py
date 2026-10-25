import inspect

import numpy as np
import pytest
from qiskit.circuit import Parameter
from qiskit.circuit.library import standard_gates as sg


def all_standard_gates(symbolic: bool = False, seed: int = 0):
    """Yield (name, gate) for every gate class in qiskit.circuit.library.standard_gates."""
    rng = np.random.default_rng(seed)
    for name in sorted(dir(sg)):
        cls = getattr(sg, name)
        if not (inspect.isclass(cls) and name.endswith("Gate")):
            continue
        sig = list(inspect.signature(cls.__init__).parameters.values())[1:]
        n_params = sum(1 for p in sig if p.default is inspect._empty)

        def val(i):
            return Parameter(f"{name}_{i}") if symbolic else float(rng.uniform(0, 2 * np.pi))

        if name == "MCXGate":
            gate = cls(3)
        elif name in ("MCPhaseGate", "MCU1Gate"):
            gate = cls(val(0), 3)
        else:
            gate = cls(*[val(i) for i in range(n_params)])
        yield name, gate


@pytest.fixture
def rng():
    return np.random.default_rng(42)
