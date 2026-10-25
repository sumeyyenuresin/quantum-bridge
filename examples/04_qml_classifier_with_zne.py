"""Train a Qiskit-defined quantum classifier with PennyLane autodiff, then
evaluate it on a noisy simulator with and without zero-noise extrapolation.

    python examples/04_qml_classifier_with_zne.py
"""
import numpy as onp
import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit.library import real_amplitudes, zz_feature_map

import qbridge as qb

rng = onp.random.default_rng(7)


def make_data(n):
    """Two interleaved half-moons scaled to [0, pi]."""
    t = rng.uniform(0, onp.pi, n)
    lab = rng.integers(0, 2, n)
    x = onp.where(lab[:, None] == 0,
                  onp.c_[onp.cos(t), onp.sin(t)],
                  onp.c_[1 - onp.cos(t), 0.5 - onp.sin(t)])
    x += rng.normal(0, 0.1, x.shape)
    x = (x - x.min(0)) / (x.max(0) - x.min(0)) * onp.pi
    return x, 2 * lab - 1


# --- model: built entirely with Qiskit ---------------------------------------
feature_map = zz_feature_map(2, reps=1)
ansatz = real_amplitudes(2, reps=2)
model_qc = feature_map.compose(ansatz)
x_names = [p.name for p in feature_map.parameters]
w_names = [p.name for p in ansatz.parameters]

# --- training: PennyLane autograd on the translated circuit -------------------
node = qb.to_qnode(model_qc, observable="ZZ", diff_method="backprop")
order = node.qbridge_circuit.parameters


def f(x, w):
    vals = dict(zip(x_names, x)) | dict(zip(w_names, w))
    return node(*[vals[n] for n in order])


def loss(params, X, Y):
    w, b = params[:-1], params[-1]
    return sum((f(x, w) + b - y) ** 2 for x, y in zip(X, Y)) / len(X)


Xtr, Ytr = make_data(40)
Xte, Yte = make_data(40)
params = np.array(onp.r_[rng.uniform(0, 2 * onp.pi, len(w_names)), 0.0], requires_grad=True)
opt = qml.AdamOptimizer(0.1)
for step in range(30):
    params = opt.step(lambda p: loss(p, Xtr, Ytr), params)
w, b = onp.asarray(params[:-1]), float(params[-1])

# --- evaluation on a noisy device -----------------------------------------
noisy = qb.depolarizing_executor("ZZ", p1=0.005, p2=0.04)
ideal = qb.ideal_executor("ZZ")
rows = {"ideal": [], "noisy": [], "zne": []}
for x in Xte:
    bound = model_qc.assign_parameters(
        {p: v for p, v in zip(feature_map.parameters, x)}
        | {p: v for p, v in zip(ansatz.parameters, w)}
    )
    rows["ideal"].append(ideal(bound))
    rows["noisy"].append(noisy(bound))
    rows["zne"].append(qb.zne(bound, noisy, (1, 3, 5), "richardson"))

for k, v in rows.items():
    acc = onp.mean(onp.sign(onp.array(v) + b) == Yte)
    err = onp.mean(onp.abs(onp.array(v) - onp.array(rows["ideal"])))
    print(f"{k:>6}: accuracy={acc:.3f}  mean|f - f_ideal|={err:.4f}")
