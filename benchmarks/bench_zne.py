"""Error-mitigation benchmarks.

B5  ZNE accuracy on hardware-efficient ansaetze under gate-level depolarising
    noise, cross-validated against PennyLane's built-in mitigate_with_zne.
B6  Shot-noise trade-off: bias vs variance of ZNE with finite shots.
B7  A Qiskit-defined QML classifier trained via PennyLane autodiff, evaluated
    on noisy simulators with and without ZNE.

Writes benchmarks/results/zne.json.
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as onp
import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit.library import efficient_su2, real_amplitudes, zz_feature_map

import qbridge as qb

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RNG = onp.random.default_rng(11)
SCALES = (1, 3, 5)
METHODS = ("linear", "richardson", "exponential")


# ----------------------------------------------------------------------------- B5
def pennylane_builtin_zne(qc, obs_label, p1, p2):
    """Reference: PennyLane's own fold_global + Richardson on an add_noise device."""
    circ = qb.to_pennylane(qc, keep_global_phase=False)
    obs = qb.to_pl_observable(obs_label, circ.num_wires)

    c1 = qml.BooleanFn(lambda op, **kw: len(op.wires) == 1)
    c2 = qml.BooleanFn(lambda op, **kw: len(op.wires) >= 2)

    def n1(op, **kw):
        for w in op.wires:
            qml.DepolarizingChannel(p1, wires=w)

    def n2(op, **kw):
        for w in op.wires:
            qml.DepolarizingChannel(p2, wires=w)

    model = qml.NoiseModel({c1: n1, c2: n2})
    dev = qml.add_noise(qml.device("default.mixed", wires=circ.wires), model)

    @partial(qml.noise.mitigate_with_zne, scale_factors=list(SCALES),
             folding=qml.noise.fold_global, extrapolate=qml.noise.richardson_extrapolate)
    @qml.qnode(dev)
    def node():
        circ.apply({})
        return qml.expval(obs)

    return float(node())


def bench_accuracy(n_instances=20):
    families = {
        "RealAmplitudes(3,2)": real_amplitudes(3, reps=2),
        "EfficientSU2(3,2)": efficient_su2(3, reps=2),
        "RealAmplitudes(4,2)": real_amplitudes(4, reps=2),
    }
    noise_levels = [(1e-3, 1e-2), (2e-3, 2e-2)]
    out = []
    for fam, ansatz in families.items():
        n = ansatz.num_qubits
        obs = "Z" * n
        ideal = qb.ideal_executor(obs)
        for p1, p2 in noise_levels:
            noisy = qb.depolarizing_executor(obs, p1=p1, p2=p2)
            errs = {"unmitigated": [], **{m: [] for m in METHODS}, "pennylane_builtin_richardson": []}
            agree = []
            for _ in range(n_instances):
                qc = ansatz.assign_parameters(RNG.uniform(0, 2 * onp.pi, ansatz.num_parameters))
                exact = ideal(qc)
                folded_vals = None
                for m in METHODS:
                    r = qb.zne(qc, noisy, SCALES, m, return_result=True)
                    errs[m].append(abs(r.value - exact))
                    folded_vals = r
                errs["unmitigated"].append(abs(folded_vals.unmitigated - exact))
                ours_r = qb.richardson_extrapolate(folded_vals.scale_factors, folded_vals.noisy_values)
                if n <= 3:
                    ref = pennylane_builtin_zne(qc, obs, p1, p2)
                    errs["pennylane_builtin_richardson"].append(abs(ref - exact))
                    agree.append(abs(ref - ours_r))
            row = {
                "family": fam, "n_qubits": n, "p1": p1, "p2": p2, "instances": n_instances,
                "n_ops": len(ansatz.data),
                "mae": {k: float(onp.mean(v)) for k, v in errs.items() if v},
                "max_abs_diff_vs_pennylane_builtin": float(onp.max(agree)) if agree else None,
            }
            row["reduction_factor"] = {
                m: row["mae"]["unmitigated"] / row["mae"][m] for m in METHODS
            }
            print(row)
            out.append(row)
    return out


# ----------------------------------------------------------------------------- B6
def bench_shots(repeats=30, shots=4096):
    ansatz = real_amplitudes(3, reps=2)
    qc = ansatz.assign_parameters(onp.linspace(0.2, 2.8, ansatz.num_parameters))
    obs = "ZZZ"
    exact = qb.ideal_executor(obs)(qc)
    out = {"shots_per_circuit": shots, "repeats": repeats, "exact": exact}
    vals = {"unmitigated": [], "linear": [], "richardson": []}
    for r in range(repeats):
        noisy = qb.depolarizing_executor(obs, p1=2e-3, p2=2e-2, shots=shots, seed=1000 + r)
        res = qb.zne(qc, noisy, SCALES, "richardson", return_result=True)
        vals["unmitigated"].append(res.unmitigated)
        vals["richardson"].append(res.value)
        vals["linear"].append(qb.linear_extrapolate(res.scale_factors, res.noisy_values))
    for k, v in vals.items():
        v = onp.array(v)
        out[k] = {"bias": float(v.mean() - exact), "std": float(v.std(ddof=1)),
                  "rmse": float(onp.sqrt(onp.mean((v - exact) ** 2)))}
    print(out)
    return out


# ----------------------------------------------------------------------------- B7
def make_moons(n, rng):
    t = rng.uniform(0, onp.pi, n)
    lab = rng.integers(0, 2, n)
    x = onp.where(lab[:, None] == 0, onp.c_[onp.cos(t), onp.sin(t)],
                  onp.c_[1 - onp.cos(t), 0.5 - onp.sin(t)])
    x += rng.normal(0, 0.1, x.shape)
    x = (x - x.min(0)) / (x.max(0) - x.min(0)) * onp.pi
    return x, 2 * lab - 1


def bench_qml(n_train=60, n_test=60, steps=60):
    rng = onp.random.default_rng(5)
    fm, an = zz_feature_map(2, reps=1), real_amplitudes(2, reps=3)
    model = fm.compose(an)
    xn, wn = [p.name for p in fm.parameters], [p.name for p in an.parameters]
    node = qb.to_qnode(model, observable="ZZ", diff_method="backprop")
    order = node.qbridge_circuit.parameters

    def f(x, w):
        vals = dict(zip(xn, x)) | dict(zip(wn, w))
        return node(*[vals[k] for k in order])

    X, Y = make_moons(n_train + n_test, rng)
    Xtr, Ytr, Xte, Yte = X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]

    def loss(p):
        w, b = p[:-1], p[-1]
        return sum((f(x, w) + b - y) ** 2 for x, y in zip(Xtr, Ytr)) / len(Xtr)

    params = np.array(onp.r_[rng.uniform(0, 2 * onp.pi, len(wn)), 0.0], requires_grad=True)
    opt = qml.AdamOptimizer(0.1)
    t0 = time.time()
    history = []
    for s in range(steps):
        params, l = opt.step_and_cost(loss, params)
        history.append(float(l))
    train_s = time.time() - t0
    w, b = onp.asarray(params[:-1]), float(params[-1])

    bound = [model.assign_parameters({p: v for p, v in zip(fm.parameters, x)}
                                     | {p: v for p, v in zip(an.parameters, w)}) for x in Xte]
    ideal = qb.ideal_executor("ZZ")
    f_ideal = onp.array([ideal(c) for c in bound])
    out = {"train_seconds": train_s, "steps": steps, "loss_history": history,
           "bias_b": b, "n_train": n_train, "n_test": n_test,
           "ideal_accuracy": float(onp.mean(onp.sign(f_ideal + b) == Yte)), "noise": []}
    for p2 in (0.01, 0.03, 0.05, 0.08):
        noisy = qb.depolarizing_executor("ZZ", p1=p2 / 10, p2=p2)
        f_noisy = onp.array([noisy(c) for c in bound])
        res = [qb.zne(c, noisy, SCALES, "richardson", return_result=True) for c in bound]
        f_r = onp.array([r.value for r in res])
        f_l = onp.array([qb.linear_extrapolate(r.scale_factors, r.noisy_values) for r in res])
        row = {"p1": p2 / 10, "p2": p2}
        for k, v in (("noisy", f_noisy), ("zne_linear", f_l), ("zne_richardson", f_r)):
            row[k] = {"accuracy": float(onp.mean(onp.sign(v + b) == Yte)),
                      "mae_vs_ideal": float(onp.mean(onp.abs(v - f_ideal)))}
        print(row)
        out["noise"].append(row)
    return out


if __name__ == "__main__":
    res = {"meta": {"qiskit": __import__("qiskit").__version__, "pennylane": qml.__version__,
                    "qbridge": qb.__version__, "scale_factors": SCALES}}
    for name, fn in (("accuracy", bench_accuracy), ("shots", bench_shots), ("qml", bench_qml)):
        t0 = time.time()
        res[name] = fn()
        print(f"[{name}] {time.time() - t0:.1f}s", flush=True)
        (OUT / "zne.json").write_text(json.dumps(res, indent=2))
