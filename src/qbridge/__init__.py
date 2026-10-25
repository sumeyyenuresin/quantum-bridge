"""Quantum-Bridge: lossless Qiskit <-> PennyLane translation and dependency-free ZNE.

Quick start
-----------
>>> import qbridge as qb
>>> node = qb.to_qnode(qiskit_circuit, observable="ZZ")   # Qiskit -> differentiable QNode
>>> qc   = qb.to_qiskit(pennylane_qfunc, *params)          # PennyLane -> Qiskit
>>> qb.verify(qiskit_circuit)                              # numerical proof of exactness
>>> qb.zne(qc, executor)                                   # zero-noise extrapolation
"""

from .exceptions import ParameterBindingError, QBridgeError, UnsupportedOperationError
from .executors import depolarizing_executor, ideal_executor
from .mitigation import (
    ZNEResult,
    actual_scale,
    exponential_extrapolate,
    fold_global,
    linear_extrapolate,
    poly_extrapolate,
    richardson_extrapolate,
    zne,
)
from .observables import pauli_string_to_pl, to_pl_observable
from .report import ConversionReport
from .to_pennylane import PennyLaneCircuit, to_pennylane, to_qnode
from .to_qiskit import to_qiskit
from .verify import VerificationResult, pennylane_unitary, qiskit_unitary, verify

__version__ = "0.1.0"

__all__ = [
    "to_pennylane",
    "to_qnode",
    "to_qiskit",
    "verify",
    "zne",
    "fold_global",
    "actual_scale",
    "richardson_extrapolate",
    "linear_extrapolate",
    "poly_extrapolate",
    "exponential_extrapolate",
    "depolarizing_executor",
    "ideal_executor",
    "pauli_string_to_pl",
    "to_pl_observable",
    "qiskit_unitary",
    "pennylane_unitary",
    "PennyLaneCircuit",
    "ConversionReport",
    "VerificationResult",
    "ZNEResult",
    "QBridgeError",
    "UnsupportedOperationError",
    "ParameterBindingError",
    "__version__",
]
