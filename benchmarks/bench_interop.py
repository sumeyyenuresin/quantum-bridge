"""Interoperability benchmarks: Quantum-Bridge vs the PennyLane-Qiskit plugin.

Measures (B1) gate-library coverage and exactness, (B2) exactness on random
circuits, (B3) conversion/execution time vs circuit size and (B4) gradient
accuracy. Writes benchmarks/results/interop.json.

Run in an environment that has both `qbridge` and `pennylane-qiskit`:
    python benchmarks/bench_interop.py
"""
from __future__ import annotations

import inspect
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2, real_amplitudes, zz_feature_map
from qiskit.circuit.library import standard_gates as sg
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

import qbridge as qb

try:
    import pennylane_qiskit  # noqa: F401

    HAVE_PLUGIN = True
except ImportError:  # pragma: no cover
    HAVE_PLUGIN = False

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(2026)
ATOL = 1e-8


# ----------------------------------------------------------------------------- helpers
def pl_unitary(fn, n, args):
    return np.asarray(qml.matrix(lambda: fn(*args), wire_order=list(reversed(range(n))))())


def compare(ref, got):
    exact = float(np.max(np.abs(ref - got)))
    idx = np.argmax(np.abs(got))
    ph = ref.flat[idx] / got.flat[idx] if abs(got.flat[idx]) > 1e-12 else 1
    ph = ph / abs(ph) if abs(ph) > 0 else 1
    upto = float(np.max(np.abs(ref - got * ph)))
    return exact, upto


def run_tool(tool, qc, values):
    """Return (unitary in Qiskit order, conversion seconds) or raise."""
    names = [p.name for p in qc.parameters]
    args = [values[n] for n in names]
    t0 = time.perf_counter()
    if tool == "qbridge":
        fn = qb.to_pennylane(qc)
    else:
        fn = qml.from_qiskit(qc)
    dt = time.perf_counter() - t0
    return pl_unitary(fn, qc.num_qubits, args), dt


def reference(qc, values):
    bound = qc.assign_parameters({p: values[p.name] for p in qc.parameters}) if qc.parameters else qc
    return Operator(bound.remove_final_measurements(inplace=False)).data


TOOLS = ["qbridge"] + (["pennylane-qiskit"] if HAVE_PLUGIN else [])


# ----------------------------------------------------------------------------- B1
def standard_gates(symbolic):
    for name in sorted(dir(sg)):
        cls = getattr(sg, name)
        if not (inspect.isclass(cls) and name.endswith("Gate")):
            continue
        sig = list(inspect.signature(cls.__init__).parameters.values())[1:]
        n = sum(1 for p in sig if p.default is inspect._empty)
        v = (lambda i: Parameter(f"p{i}")) if symbolic else (lambda i: float(RNG.uniform(0, 2 * np.pi)))
        if name == "MCXGate":
            g = cls(3)
        elif name in ("MCPhaseGate", "MCU1Gate"):
            g = cls(v(0), 3)
        else:
            g = cls(*[v(i) for i in range(n)])
        if symbolic and not g.params:
            continue
        yield name, g


def bench_gates():
    out = {}
    for symbolic in (False, True):
        key = "symbolic" if symbolic else "numeric"
        out[key] = {t: {"ok_exact": 0, "ok_phase_only": 0, "wrong": 0, "error": 0, "fail_names": []} for t in TOOLS}
        out[key]["n_gates"] = 0
        for name, g in standard_gates(symbolic):
            out[key]["n_gates"] += 1
            qc = QuantumCircuit(max(g.num_qubits, 1))
            qc.append(g, range(g.num_qubits))
            values = {p.name: float(RNG.uniform(0, 2 * np.pi)) for p in qc.parameters}
            ref = reference(qc, values)
            for t in TOOLS:
                rec = out[key][t]
                try:
                    got, _ = run_tool(t, qc, values)
                    exact, upto = compare(ref, got)
                    if exact < ATOL:
                        rec["ok_exact"] += 1
                    elif upto < ATOL:
                        rec["ok_phase_only"] += 1
                        rec["fail_names"].append(f"{name}:phase")
                    else:
                        rec["wrong"] += 1
                        rec["fail_names"].append(f"{name}:wrong")
                except Exception as e:  # noqa: BLE001
                    rec["error"] += 1
                    rec["fail_names"].append(f"{name}:{type(e).__name__}")
    return out


# ----------------------------------------------------------------------------- B2
def bench_random(n_circuits=300):
    out = {t: {"exact": 0, "phase_only": 0, "wrong": 0, "error": 0, "times": []} for t in TOOLS}
    for s in range(n_circuits):
        n = 2 + s % 6
        qc = random_circuit(n, depth=20, max_operands=3, seed=s)
        ref = reference(qc, {})
        for t in TOOLS:
            try:
                got, dt = run_tool(t, qc, {})
                exact, upto = compare(ref, got)
                out[t]["times"].append(dt)
                key = "exact" if exact < ATOL else ("phase_only" if upto < ATOL else "wrong")
                out[t][key] += 1
            except Exception:  # noqa: BLE001
                out[t]["error"] += 1
    for t in TOOLS:
        ts = out[t].pop("times")
        out[t]["median_conversion_ms"] = 1e3 * statistics.median(ts) if ts else None
    out["n_circuits"] = n_circuits
    return out


# ----------------------------------------------------------------------------- B3
def bench_scaling():
    rows = []
    n = 8
    for depth in (10, 25, 50, 100, 200, 400):
        qc = random_circuit(n, depth=depth, max_operands=2, seed=depth)
        n_ops = sum(1 for i in qc.data if i.operation.name not in ("measure", "barrier"))
        row = {"depth": depth, "n_ops": n_ops}
        dev = qml.device("default.qubit", wires=n)
        for t in TOOLS:
            conv, execs = [], []
            for _ in range(5):
                t0 = time.perf_counter()
                fn = qb.to_pennylane(qc) if t == "qbridge" else qml.from_qiskit(qc)
                conv.append(time.perf_counter() - t0)

            @qml.qnode(dev)
            def node():
                fn()
                return qml.expval(qml.PauliZ(0))

            node()  # warm-up
            for _ in range(5):
                t0 = time.perf_counter()
                node()
                execs.append(time.perf_counter() - t0)
            row[t] = {"convert_ms": 1e3 * statistics.median(conv), "execute_ms": 1e3 * statistics.median(execs)}
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- B4
def qiskit_expval(qc, x, obs):
    return float(np.real(Statevector(qc.assign_parameters(x)).expectation_value(obs)))


def fd_grad(qc, x, obs, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (qiskit_expval(qc, x + e, obs) - qiskit_expval(qc, x - e, obs)) / (2 * h)
    return g


def bench_gradients():
    from pennylane import numpy as pnp

    circuits = {
        "RealAmplitudes(4, reps=3)": real_amplitudes(4, reps=3),
        "EfficientSU2(4, reps=2)": efficient_su2(4, reps=2),
        "ZZFeatureMap(4, reps=2)": zz_feature_map(4, reps=2),
    }
    obs = SparsePauliOp.from_list([("ZZII", 0.5), ("IXXI", -0.3), ("IIYZ", 0.2)])
    out = {}
    for label, qc in circuits.items():
        x = RNG.uniform(0, 2 * np.pi, qc.num_parameters)
        ref = fd_grad(qc, x, obs)
        out[label] = {"n_params": qc.num_parameters}
        for t in TOOLS:
            for dm in ("backprop", "parameter-shift"):
                key = f"{t}/{dm}"
                try:
                    if t == "qbridge":
                        node = qb.to_qnode(qc, observable=obs, diff_method=dm)
                    else:
                        fn = qml.from_qiskit(qc)
                        H = qb.to_pl_observable(obs, qc.num_qubits)
                        # the plugin maps a ParameterVector to ONE array argument
                        node = qml.QNode(lambda v: (fn(v), qml.expval(H))[1],
                                         qml.device("default.qubit", wires=qc.num_qubits), diff_method=dm)
                    xp = pnp.array(x, requires_grad=True)
                    g = np.asarray(qml.grad(node)(xp))
                    out[label][key] = float(np.max(np.abs(g - ref)))
                except Exception as e:  # noqa: BLE001
                    out[label][key] = f"error: {type(e).__name__}: {str(e)[:80]}"
    return out


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "pennylane": qml.__version__,
        "qbridge": qb.__version__,
        "pennylane_qiskit": getattr(sys.modules.get("pennylane_qiskit"), "__version__", None),
        "cpu_count": __import__("os").cpu_count(),
    }
    print(meta)
    res = {"meta": meta}
    only = sys.argv[1:]  # optionally run a subset, e.g. `python bench_interop.py gradients`
    prev = OUT / "interop.json"
    if only and prev.exists():
        res = json.loads(prev.read_text()) | {"meta": meta}
    for name, fn in [("gates", bench_gates), ("random", bench_random),
                     ("scaling", bench_scaling), ("gradients", bench_gradients)]:
        if only and name not in only:
            continue
        t0 = time.time()
        res[name] = fn()
        print(f"[{name}] done in {time.time() - t0:.1f}s")
        print(json.dumps(res[name], indent=1)[:2500])
    (OUT / "interop.json").write_text(json.dumps(res, indent=2))
