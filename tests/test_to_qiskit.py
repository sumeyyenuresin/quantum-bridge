"""PennyLane -> Qiskit and full round trips."""

import numpy as np
import pennylane as qml
import pytest
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator

import qbridge as qb


def pl_matrix_qiskit_order(qfunc, n, *args):
    return qml.matrix(qfunc, wire_order=list(reversed(range(n))))(*args)


def test_basic_pennylane_function():
    def qfunc(theta):
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.RY(theta, 1)
        qml.adjoint(qml.S(0))
        qml.ctrl(qml.RX(0.4, wires=2), control=[0, 1], control_values=[1, 0])
        qml.IsingZZ(0.2, wires=[1, 2])
        qml.GlobalPhase(0.3)

    qc = qb.to_qiskit(qfunc, 0.9)
    assert np.allclose(Operator(qc).data, pl_matrix_qiskit_order(qfunc, 3, 0.9))


def test_unmapped_operations_use_decomposition_or_matrix():
    def qfunc():
        qml.QFT(wires=[0, 1, 2])
        qml.MultiRZ(0.3, wires=[0, 2])
        qml.QubitUnitary(qml.matrix(qml.Rot(0.1, 0.2, 0.3, wires=0)), wires=1)

    qc, rep = qb.to_qiskit(qfunc, return_report=True)
    assert np.allclose(Operator(qc).data, pl_matrix_qiskit_order(qfunc, 3))
    assert rep.counts["decomposed"] >= 2


def test_qnode_and_measure():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def node(x):
        qml.RX(x, 0)
        qml.CNOT([0, 1])
        return qml.probs(wires=[0, 1])

    qc = qb.to_qiskit(node, 0.5, measure=True)
    assert qc.count_ops()["measure"] == 2


@pytest.mark.parametrize("seed", range(25))
def test_round_trip_qiskit_pennylane_qiskit(seed):
    n = 2 + seed % 4
    qc = random_circuit(n, depth=10, max_operands=3, seed=100 + seed)
    circ = qb.to_pennylane(qc)
    back = qb.to_qiskit(lambda: circ.apply({}), wire_order=range(n))
    assert Operator(back).equiv(Operator(qc))
    # exact, including global phase
    assert np.allclose(Operator(back).data, Operator(qc).data, atol=1e-8)


def test_string_wire_labels():
    def qfunc():
        qml.Hadamard("anc")
        qml.CNOT(["anc", "data"])

    qc = qb.to_qiskit(qfunc, wire_order=["anc", "data"])
    assert qc.num_qubits == 2
