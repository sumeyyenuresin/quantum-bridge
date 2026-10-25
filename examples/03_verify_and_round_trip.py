"""Prove a translation is exact, inspect what happened, and go back to Qiskit."""
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

import qbridge as qb

theta = Parameter("θ")
qc = QuantumCircuit(3, global_phase=0.25)
qc.h(0)
qc.cs(0, 1)                 # no direct PennyLane class -> generic controlled S
qc.rzx(2 * theta + 0.1, 1, 2)  # no PennyLane class + symbolic -> exact decomposition
qc.append(XXPlusYYGate(0.3, 0.2), [0, 2])  # numeric, no PennyLane class -> exact unitary
qc.measure_all()

circuit = qb.to_pennylane(qc)
print(circuit.report)
print(qb.verify(qc))  # unitary equality incl. global phase, random θ

# ... and back again: PennyLane function -> Qiskit circuit
def pl_function(x):
    qml.QFT(wires=[0, 1, 2])
    qml.ctrl(qml.RY(x, wires=2), control=[0, 1], control_values=[1, 0])

back = qb.to_qiskit(pl_function, 0.7)
print(back.draw(output="text", fold=100))
