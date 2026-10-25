"""Whole circuits: random circuits, library ansätze, measurements, wires, errors."""

import numpy as np
import pennylane as qml
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import efficient_su2, real_amplitudes, zz_feature_map
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

import qbridge as qb


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_are_exact(seed):
    n = 2 + seed % 5
    qc = random_circuit(n, depth=12, max_operands=3, seed=seed)
    assert qb.verify(qc).equivalent


@pytest.mark.parametrize(
    "builder",
    [
        lambda: real_amplitudes(4, reps=2),
        lambda: efficient_su2(3, reps=2),
        lambda: zz_feature_map(3, reps=2),  # non-linear expressions (pi - x)(pi - y)
    ],
    ids=["real_amplitudes", "efficient_su2", "zz_feature_map"],
)
def test_library_ansatze_symbolic(builder):
    qc = builder()
    assert qb.verify(qc).equivalent
    assert qb.verify(qc.decompose()).equivalent


def test_parameter_binding_styles():
    a, b = Parameter("a"), Parameter("b")
    qc = QuantumCircuit(1)
    qc.rx(a, 0)
    qc.ry(b, 0)
    node = qb.to_qnode(qc, observable="Z")
    expected = np.cos(0.3) * np.cos(0.7)
    assert np.isclose(node(0.3, 0.7), expected)
    assert np.isclose(node(np.array([0.3, 0.7])), expected)
    assert np.isclose(node(a=0.3, b=0.7), expected)
    with pytest.raises(qb.ParameterBindingError):
        node(0.3)


def test_parameter_vector_order():
    x = ParameterVector("x", 12)  # x[10] sorts correctly after x[9] in Qiskit
    qc = QuantumCircuit(1)
    for i in range(12):
        qc.rz(x[i] * (i + 1), 0)
    circ = qb.to_pennylane(qc)
    assert circ.parameters == [p.name for p in qc.parameters]
    assert qb.verify(qc).equivalent


def test_probabilities_use_qiskit_bit_order():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.h(2)
    qc.cx(2, 1)
    probs = qb.to_qnode(qc)()
    ref = Statevector(qc).probabilities()
    assert np.allclose(probs, ref)


def test_terminal_measurements_are_recorded():
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.cx(0, 2)
    qc.measure([0, 2], [0, 1])
    circ = qb.to_pennylane(qc)
    assert circ.report.measured_qubits == [0, 2]
    probs = qb.to_qnode(qc)()
    assert np.allclose(probs, [0.5, 0, 0, 0.5])


def test_custom_wire_labels():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    circ = qb.to_pennylane(qc, wires=["a", "b"])
    dev = qml.device("default.qubit", wires=["a", "b"])

    @qml.qnode(dev)
    def node():
        circ()
        return qml.expval(qml.PauliZ("a") @ qml.PauliZ("b"))

    assert np.isclose(node(), 1.0)


def test_large_circuit_statevector_path():
    qc = random_circuit(12, depth=6, max_operands=2, seed=7)
    res = qb.verify(qc)
    assert res.method == "statevector" and res.equivalent


def test_mid_circuit_measurement_raises():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)
    with pytest.raises(qb.UnsupportedOperationError):
        qb.to_pennylane(qc)


def test_reset_raises():
    qc = QuantumCircuit(1)
    qc.reset(0)
    with pytest.raises(qb.UnsupportedOperationError):
        qb.to_pennylane(qc)


def test_control_flow_raises():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    with pytest.raises(qb.UnsupportedOperationError):
        qb.to_pennylane(qc)


def test_barrier_is_ignored():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    assert qb.verify(qc).equivalent


def test_global_phase_is_preserved():
    qc = QuantumCircuit(1, global_phase=0.77)
    qc.h(0)
    assert qb.verify(qc).equivalent
    # without the phase it is only equal up to phase
    circ = qb.to_pennylane(qc, keep_global_phase=False)
    assert not qb.verify(qc, circ).equivalent
    assert qb.verify(qc, circ, up_to_global_phase=True).equivalent
