"""Zero-noise extrapolation: folding correctness, extrapolators, end-to-end gain."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator

import qbridge as qb


@pytest.mark.parametrize("scale", [1, 1.5, 2, 3, 4.2, 5, 7])
def test_folding_preserves_the_logical_unitary(scale):
    qc = random_circuit(3, depth=8, seed=3)
    folded = qb.fold_global(qc, scale)
    assert Operator(folded).equiv(Operator(qc))


@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_odd_scales_are_exact(scale):
    qc = random_circuit(3, depth=8, seed=4)
    assert np.isclose(qb.actual_scale(qc, qb.fold_global(qc, scale)), scale)


def test_partial_folding_is_close_to_requested_scale():
    qc = random_circuit(4, depth=20, seed=5)
    for s in (1.5, 2.0, 2.5):
        assert abs(qb.actual_scale(qc, qb.fold_global(qc, s)) - s) < 0.1


def test_measurements_survive_folding():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    folded = qb.fold_global(qc, 3)
    assert folded.count_ops()["measure"] == 2
    assert folded.data[-1].operation.name == "measure"


def test_richardson_is_exact_for_polynomials():
    f = lambda x: 0.8 - 0.1 * x + 0.02 * x**2
    xs = [1, 2, 3]
    assert np.isclose(qb.richardson_extrapolate(xs, [f(x) for x in xs]), 0.8)


def test_exponential_is_exact_for_exponentials():
    f = lambda x: 0.9 * np.exp(-0.3 * x)
    xs = [1, 3, 5]
    assert np.isclose(qb.exponential_extrapolate(xs, [f(x) for x in xs]), 0.9)


def test_rejects_unbound_parameters():
    with pytest.raises(ValueError):
        qb.zne(real_amplitudes(2), qb.ideal_executor("ZZ"))


def test_ideal_executor_is_unchanged_by_zne():
    qc = random_circuit(3, depth=6, seed=9)
    ideal = qb.ideal_executor("ZIZ")
    assert np.isclose(qb.zne(qc, ideal), ideal(qc), atol=1e-8)


@pytest.mark.parametrize("method", ["linear", "richardson", "exponential"])
def test_zne_reduces_error_under_depolarizing_noise(method):
    rng = np.random.default_rng(0)
    ansatz = real_amplitudes(3, reps=2)
    obs = "ZZI"
    ideal = qb.ideal_executor(obs)
    noisy = qb.depolarizing_executor(obs, p1=0.002, p2=0.02)

    raw_err, zne_err = [], []
    for _ in range(5):
        qc = ansatz.assign_parameters(rng.uniform(0, 2 * np.pi, ansatz.num_parameters))
        exact = ideal(qc)
        res = qb.zne(qc, noisy, (1, 3, 5), method, return_result=True)
        raw_err.append(abs(res.unmitigated - exact))
        zne_err.append(abs(res.value - exact))
    assert np.mean(zne_err) < 0.5 * np.mean(raw_err)
