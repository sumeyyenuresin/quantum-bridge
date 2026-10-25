"""Observable conversion (Qiskit Pauli conventions <-> PennyLane)."""

from __future__ import annotations

import numpy as np
import pennylane as qml

_PAULI = {"I": qml.Identity, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def pauli_string_to_pl(label: str, num_wires: int | None = None, wires=None):
    """Pauli string in **Qiskit order** (rightmost char = qubit 0) -> PennyLane.

    ``"ZI"`` therefore means Z on qubit 1, identity on qubit 0.
    """
    label = label.upper().lstrip("+")
    n = len(label)
    if num_wires is not None and n != num_wires:
        raise ValueError(f"Pauli string '{label}' has {n} terms, circuit has {num_wires} qubits.")
    wires = list(range(n)) if wires is None else list(wires)
    factors = [
        _PAULI[ch](wires=wires[q]) for q, ch in enumerate(reversed(label)) if ch != "I"
    ]
    if not factors:
        return qml.Identity(wires=wires[0])
    return factors[0] if len(factors) == 1 else qml.prod(*factors)


def to_pl_observable(obs, num_wires: int, wires=None):
    """Accept str / qiskit Pauli / SparsePauliOp / PennyLane operator."""
    from qiskit.quantum_info import Pauli, SparsePauliOp

    if isinstance(obs, str):
        return pauli_string_to_pl(obs, num_wires, wires)
    if isinstance(obs, Pauli):
        return pauli_string_to_pl(obs.to_label(), num_wires, wires)
    if isinstance(obs, SparsePauliOp):
        coeffs = np.asarray(obs.coeffs)
        if np.max(np.abs(coeffs.imag)) > 1e-12:
            raise ValueError("Observable must be Hermitian (real coefficients).")
        terms = [pauli_string_to_pl(lbl, num_wires, wires) for lbl in obs.paulis.to_labels()]
        return qml.dot(list(coeffs.real), terms)
    if isinstance(obs, qml.operation.Operator):
        return obs
    raise TypeError(f"Unsupported observable type: {type(obs).__name__}")
