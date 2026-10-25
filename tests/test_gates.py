"""Every Qiskit standard gate must translate to an *exactly* equal unitary."""

import pytest
from qiskit import QuantumCircuit

import qbridge as qb
from conftest import all_standard_gates

NUMERIC = list(all_standard_gates(symbolic=False))
SYMBOLIC = [(n, g) for n, g in all_standard_gates(symbolic=True) if g.params]


@pytest.mark.parametrize("name,gate", NUMERIC, ids=[n for n, _ in NUMERIC])
def test_numeric_gate_is_exact(name, gate):
    qc = QuantumCircuit(max(gate.num_qubits, 1))
    qc.append(gate, range(gate.num_qubits))
    res = qb.verify(qc)
    assert res.equivalent, f"{name}: {res}"


@pytest.mark.parametrize("name,gate", SYMBOLIC, ids=[n for n, _ in SYMBOLIC])
def test_symbolic_gate_is_exact(name, gate):
    """Unbound parameters must survive translation (they become QNode arguments)."""
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))
    circ = qb.to_pennylane(qc)
    assert circ.num_parameters == len(qc.parameters)
    res = qb.verify(qc, circ)
    assert res.equivalent, f"{name}: {res}"


def test_full_standard_library_is_covered():
    assert len(NUMERIC) >= 50


def test_report_categories():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cs(0, 1)  # generic controlled
    qc.rzx(0.3, 1, 2)  # no PennyLane equivalent -> exact matrix
    rep = qb.to_pennylane(qc).report
    assert rep.by_gate == {"h": "native", "cs": "controlled", "rzx": "matrix"}
    assert "lossless" in str(rep)
