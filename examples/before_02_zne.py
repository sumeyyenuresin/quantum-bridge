"""BEFORE: zero-noise extrapolation written by hand for a Qiskit circuit."""
import numpy as np
from qiskit.circuit.library import efficient_su2

import qbridge as qb  # only for a noisy executor to make the example runnable

qc = efficient_su2(3, reps=3).assign_parameters([0.3 * i for i in range(24)])
noisy = qb.depolarizing_executor("ZZZ", p1=0.002, p2=0.02)

body = qc.remove_final_measurements(inplace=False)
scales, values = [], []
for k in (0, 1, 2):  # scale factors 1, 3, 5
    folded = body.copy()
    for _ in range(k):
        folded.compose(body.inverse(), inplace=True)
        folded.compose(body, inplace=True)
    scales.append(len(folded.data) / len(body.data))
    values.append(noisy(folded))
# Richardson extrapolation (Lagrange polynomial evaluated at 0)
x, y = np.array(scales), np.array(values)
mitigated = sum(y[i] * np.prod([x[j] / (x[j] - x[i]) for j in range(len(x)) if j != i])
                for i in range(len(x)))
print("unmitigated:", values[0], " ZNE:", mitigated)
