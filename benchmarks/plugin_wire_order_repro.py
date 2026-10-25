"""Minimal reproducer: wrong probabilities after qml.from_qiskit for unmapped gates.

Observed with pennylane-qiskit 0.45.0 / PennyLane 0.45.1 / Qiskit 2.3.0.
Gates without a named PennyLane counterpart (e.g. DCX, RZX, CSX, RCCX, CU) are
emitted as qml.QubitUnitary on the qubits in Qiskit's order, but Qiskit's
matrix is little-endian while PennyLane's QubitUnitary is big-endian. The
measurable probabilities therefore differ from Qiskit's own simulation.

Suitable for an upstream issue. Quantum-Bridge reverses the wires
(qml.QubitUnitary(U, wires=reversed(qargs))) and verifies the result.
"""
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit.library import DCXGate
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.ry(0.4, 0)
qc.ry(1.3, 1)
qc.append(DCXGate(), [0, 1])

ref = Statevector(qc).probabilities()  # Qiskit integer order (q0 = LSB)

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def plugin():
    qml.from_qiskit(qc)()
    return qml.probs(wires=[1, 0])  # same (little-endian) order as Qiskit


print("qiskit :", np.round(ref, 4))
print("plugin :", np.round(plugin(), 4))

try:
    import qbridge as qb

    print("qbridge:", np.round(qb.to_qnode(qc)(), 4))
except ImportError:
    pass
