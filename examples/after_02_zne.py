"""AFTER: the same zero-noise extrapolation with Quantum-Bridge."""
from qiskit.circuit.library import efficient_su2
import qbridge as qb

qc = efficient_su2(3, reps=3).assign_parameters([0.3 * i for i in range(24)])
noisy = qb.depolarizing_executor("ZZZ", p1=0.002, p2=0.02)

print(qb.zne(qc, noisy, return_result=True))

print("ideal reference:", qb.ideal_executor("ZZZ")(qc))
