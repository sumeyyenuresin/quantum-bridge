"""Gate correspondence tables between Qiskit and PennyLane.

Every entry here is *exact*: it reproduces the Qiskit unitary including the
global phase. This is enforced by ``tests/test_gates.py``, which compares the
full unitary of every mapped gate for random parameters.

Wire convention: a Qiskit instruction acting on qubits ``(q0, q1, ...)`` maps
to a PennyLane operation on ``wires=[q0, q1, ...]`` with the same roles
(control first, target last, ...). Endianness only matters when comparing raw
matrices or state vectors, which ``qbridge.verify`` handles for you.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pennylane as qml

Builder = Callable[[Sequence, Sequence], None]


def _r_gate(p, w):
    # Qiskit R(theta, phi) = exp(-i theta/2 (cos(phi) X + sin(phi) Y))
    #                      = RZ(phi) RX(theta) RZ(-phi)   (exact, no phase)
    theta, phi = p
    qml.RZ(-phi, wires=w)
    qml.RX(theta, wires=w)
    qml.RZ(phi, wires=w)


def _dcx(p, w):
    qml.CNOT(wires=[w[0], w[1]])
    qml.CNOT(wires=[w[1], w[0]])


# Qiskit gate name -> builder(params, wires)
QISKIT_TO_PL: dict[str, Builder] = {
    # --- single-qubit, fixed ---
    "id": lambda p, w: qml.Identity(wires=w),
    "x": lambda p, w: qml.PauliX(wires=w),
    "y": lambda p, w: qml.PauliY(wires=w),
    "z": lambda p, w: qml.PauliZ(wires=w),
    "h": lambda p, w: qml.Hadamard(wires=w),
    "s": lambda p, w: qml.S(wires=w),
    "sdg": lambda p, w: qml.adjoint(qml.S(wires=w)),
    "t": lambda p, w: qml.T(wires=w),
    "tdg": lambda p, w: qml.adjoint(qml.T(wires=w)),
    "sx": lambda p, w: qml.SX(wires=w),
    "sxdg": lambda p, w: qml.adjoint(qml.SX(wires=w)),
    # --- single-qubit, parametrised ---
    "rx": lambda p, w: qml.RX(p[0], wires=w),
    "ry": lambda p, w: qml.RY(p[0], wires=w),
    "rz": lambda p, w: qml.RZ(p[0], wires=w),
    "p": lambda p, w: qml.PhaseShift(p[0], wires=w),
    "u1": lambda p, w: qml.PhaseShift(p[0], wires=w),
    "u2": lambda p, w: qml.U2(p[0], p[1], wires=w),
    "u3": lambda p, w: qml.U3(p[0], p[1], p[2], wires=w),
    "u": lambda p, w: qml.U3(p[0], p[1], p[2], wires=w),
    "r": _r_gate,
    # --- two-qubit ---
    "cx": lambda p, w: qml.CNOT(wires=w),
    "cy": lambda p, w: qml.CY(wires=w),
    "cz": lambda p, w: qml.CZ(wires=w),
    "ch": lambda p, w: qml.CH(wires=w),
    "swap": lambda p, w: qml.SWAP(wires=w),
    "iswap": lambda p, w: qml.ISWAP(wires=w),
    "ecr": lambda p, w: qml.ECR(wires=w),
    "dcx": _dcx,
    "cp": lambda p, w: qml.ControlledPhaseShift(p[0], wires=w),
    "cu1": lambda p, w: qml.ControlledPhaseShift(p[0], wires=w),
    "crx": lambda p, w: qml.CRX(p[0], wires=w),
    "cry": lambda p, w: qml.CRY(p[0], wires=w),
    "crz": lambda p, w: qml.CRZ(p[0], wires=w),
    "rxx": lambda p, w: qml.IsingXX(p[0], wires=w),
    "ryy": lambda p, w: qml.IsingYY(p[0], wires=w),
    "rzz": lambda p, w: qml.IsingZZ(p[0], wires=w),
    # --- three-qubit ---
    "ccx": lambda p, w: qml.Toffoli(wires=w),
    "cswap": lambda p, w: qml.CSWAP(wires=w),
    "ccz": lambda p, w: qml.CCZ(wires=w),
    # --- zero-qubit ---
    # Qiskit GlobalPhaseGate(theta) = exp(+i theta); PennyLane uses exp(-i phi)
    "global_phase": lambda p, w: qml.GlobalPhase(-p[0]),
}

# Instructions that carry no unitary action and are safe to skip.
SKIPPED = {"barrier", "delay"}


def _qk():
    from qiskit.circuit import library as lib

    return lib


def pl_to_qiskit_table():
    """PennyLane operation name -> factory(params) -> qiskit Gate."""
    lib = _qk()
    return {
        "Identity": lambda p: lib.IGate(),
        "PauliX": lambda p: lib.XGate(),
        "PauliY": lambda p: lib.YGate(),
        "PauliZ": lambda p: lib.ZGate(),
        "Hadamard": lambda p: lib.HGate(),
        "S": lambda p: lib.SGate(),
        "T": lambda p: lib.TGate(),
        "SX": lambda p: lib.SXGate(),
        "RX": lambda p: lib.RXGate(p[0]),
        "RY": lambda p: lib.RYGate(p[0]),
        "RZ": lambda p: lib.RZGate(p[0]),
        "PhaseShift": lambda p: lib.PhaseGate(p[0]),
        "U2": lambda p: lib.U2Gate(p[0], p[1]),
        "U3": lambda p: lib.UGate(p[0], p[1], p[2]),
        "CNOT": lambda p: lib.CXGate(),
        "CY": lambda p: lib.CYGate(),
        "CZ": lambda p: lib.CZGate(),
        "CH": lambda p: lib.CHGate(),
        "SWAP": lambda p: lib.SwapGate(),
        "ISWAP": lambda p: lib.iSwapGate(),
        "ECR": lambda p: lib.ECRGate(),
        "ControlledPhaseShift": lambda p: lib.CPhaseGate(p[0]),
        "CRX": lambda p: lib.CRXGate(p[0]),
        "CRY": lambda p: lib.CRYGate(p[0]),
        "CRZ": lambda p: lib.CRZGate(p[0]),
        "IsingXX": lambda p: lib.RXXGate(p[0]),
        "IsingYY": lambda p: lib.RYYGate(p[0]),
        "IsingZZ": lambda p: lib.RZZGate(p[0]),
        "Toffoli": lambda p: lib.CCXGate(),
        "CSWAP": lambda p: lib.CSwapGate(),
        "CCZ": lambda p: lib.CCZGate(),
    }
