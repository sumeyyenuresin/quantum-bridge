"""BEFORE: using a Qiskit ansatz inside PennyLane by hand.

Every gate has to be re-mapped manually, parameter names resolved, and the
Pauli-string ordering (Qiskit is little-endian) translated by the author.
Any new gate type in the ansatz silently breaks this code.
"""
import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit.library import real_amplitudes

qc = real_amplitudes(3, reps=2)  # note: .decompose() would emit "r" gates and break this
dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit(theta):
    values = {p.name: theta[i] for i, p in enumerate(qc.parameters)}
    for inst in qc.data:
        name = inst.operation.name
        wires = [qc.find_bit(q).index for q in inst.qubits]
        if name == "ry":
            qml.RY(values[inst.operation.params[0].name], wires=wires)
        elif name == "cx":
            qml.CNOT(wires=wires)
        else:
            raise NotImplementedError(name)
    return qml.expval(qml.PauliZ(2) @ qml.PauliZ(1))  # Qiskit "ZZI"


theta = np.array([0.1 * i for i in range(qc.num_parameters)], requires_grad=True)
print("value:", circuit(theta))
print("grad :", qml.grad(circuit)(theta))
