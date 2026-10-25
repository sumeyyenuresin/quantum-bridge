"""Ready-made executors ``f(QuantumCircuit) -> float`` for testing and benchmarks.

They run Qiskit circuits on PennyLane simulators through the bridge, so the
same circuit object can be used for ideal runs, noisy runs and ZNE without
installing a separate noise simulator.
"""

from __future__ import annotations

from typing import Callable

import pennylane as qml
from qiskit.circuit import QuantumCircuit

from .observables import to_pl_observable
from .to_pennylane import to_pennylane


def ideal_executor(observable) -> Callable[[QuantumCircuit], float]:
    """Exact (noise-free, infinite-shot) expectation value."""

    def run(qc: QuantumCircuit) -> float:
        circ = to_pennylane(qc)
        obs = to_pl_observable(observable, circ.num_wires)
        dev = qml.device("default.qubit", wires=circ.wires)

        @qml.qnode(dev)
        def node():
            circ.apply({})
            return qml.expval(obs)

        return float(node())

    return run


def depolarizing_executor(
    observable,
    p1: float = 1e-3,
    p2: float = 1e-2,
    readout: float = 0.0,
    shots: int | None = None,
    seed: int | None = None,
) -> Callable[[QuantumCircuit], float]:
    """Density-matrix simulation with gate-level depolarising noise.

    After every 1-qubit gate a depolarising channel with probability ``p1``
    acts on its qubit; after every multi-qubit gate a channel with ``p2`` acts
    on each participating qubit. ``readout`` adds a symmetric bit-flip before
    measurement. ``shots=None`` gives the exact expectation value.
    """

    def noise(wires):
        p = p1 if len(wires) == 1 else p2
        if p > 0:
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)

    def run(qc: QuantumCircuit) -> float:
        circ = to_pennylane(qc, keep_global_phase=False)
        obs = to_pl_observable(observable, circ.num_wires)
        dev = qml.device("default.mixed", wires=circ.wires, seed=seed)

        @qml.qnode(dev)
        def node():
            circ.apply({}, after_gate=noise)
            if readout > 0:
                for w in circ.wires:
                    qml.BitFlip(readout, wires=w)
            return qml.expval(obs)

        if shots is not None:
            node = qml.set_shots(node, shots=shots)
        return float(node())

    return run
