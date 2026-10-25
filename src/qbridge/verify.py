"""Numerical proof that a translation is exact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


@dataclass
class VerificationResult:
    equivalent: bool
    max_abs_error: float
    method: str  # "unitary" or "statevector"
    up_to_global_phase: bool

    def __bool__(self) -> bool:
        return self.equivalent

    def __str__(self) -> str:
        mark = "EQUIVALENT" if self.equivalent else "NOT EQUIVALENT"
        phase = " (up to global phase)" if self.up_to_global_phase else ""
        return f"{mark}{phase} via {self.method}: max |error| = {self.max_abs_error:.2e}"


def qiskit_unitary(qc: QuantumCircuit) -> np.ndarray:
    """Unitary of a bound Qiskit circuit, ignoring terminal measurements."""
    return Operator(qc.remove_final_measurements(inplace=False)).data


def pennylane_unitary(pl_circuit, values=None, num_wires=None) -> np.ndarray:
    """Unitary of a PennyLane function in **Qiskit (little-endian) ordering**."""
    wires = getattr(pl_circuit, "wires", None)
    if wires is None:
        wires = list(range(num_wires))
    values = values or {}

    def qfunc():
        if hasattr(pl_circuit, "apply"):
            pl_circuit.apply(values)
        else:
            pl_circuit(**values)

    return np.asarray(qml.matrix(qfunc, wire_order=list(reversed(wires)))())


def _align(a, b, up_to_global_phase):
    if not up_to_global_phase:
        return a, b
    idx = np.argmax(np.abs(b))
    if abs(b.flat[idx]) < 1e-12:
        return a, b
    phase = a.flat[idx] / b.flat[idx]
    phase /= abs(phase)
    return a, b * phase


def verify(
    qc: QuantumCircuit,
    pl_circuit=None,
    values: dict | None = None,
    *,
    atol: float = 1e-8,
    up_to_global_phase: bool = False,
    method: str = "auto",
) -> VerificationResult:
    """Check that ``to_pennylane(qc)`` (or a given PennyLane circuit) is exact.

    ``values`` maps parameter names to numbers for parametrised circuits;
    random values in [0, 2pi) are drawn when omitted. ``method="auto"`` uses
    the full unitary up to 10 qubits and the output state vector beyond.
    """
    from .to_pennylane import to_pennylane

    if pl_circuit is None:
        pl_circuit = to_pennylane(qc)

    if values is None and qc.parameters:
        rng = np.random.default_rng(1234)
        values = {p.name: rng.uniform(0, 2 * np.pi) for p in qc.parameters}
    values = values or {}

    bound = qc.assign_parameters({p: values[p.name] for p in qc.parameters}) if qc.parameters else qc
    n = qc.num_qubits
    if method == "auto":
        method = "unitary" if n <= 10 else "statevector"

    if method == "unitary":
        ref = qiskit_unitary(bound)
        got = pennylane_unitary(pl_circuit, values)
    elif method == "statevector":
        ref = Statevector(bound.remove_final_measurements(inplace=False)).data
        dev = qml.device("default.qubit", wires=pl_circuit.wires)

        @qml.qnode(dev)
        def state():
            pl_circuit.apply(values)
            return qml.state()

        # PennyLane is big-endian; reverse qubit axes to match Qiskit
        got = np.asarray(state()).reshape([2] * n).transpose(list(reversed(range(n)))).reshape(-1)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    ref, got = _align(ref, got, up_to_global_phase)
    err = float(np.max(np.abs(ref - got)))
    return VerificationResult(err <= atol, err, method, up_to_global_phase)
