"""AFTER: the same task with Quantum-Bridge (any Qiskit gate, any expression)."""
import pennylane as qml
import qbridge as qb
from pennylane import numpy as np
from qiskit.circuit.library import real_amplitudes

node = qb.to_qnode(real_amplitudes(3, reps=2), observable="ZZI")

theta = np.array([0.1 * i for i in range(9)], requires_grad=True)
print("value:", node(theta))
print("grad :", qml.grad(node)(theta))
