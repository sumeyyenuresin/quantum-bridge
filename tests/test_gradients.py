"""Converted circuits must be differentiable, and gradients must be correct."""

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector

import qbridge as qb


def qiskit_expval(qc, values, obs):
    bound = qc.assign_parameters(values)
    return float(np.real(Statevector(bound).expectation_value(obs)))


def finite_diff(qc, x, obs, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (qiskit_expval(qc, x + e, obs) - qiskit_expval(qc, x - e, obs)) / (2 * h)
    return g


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "adjoint"])
def test_gradient_matches_qiskit_finite_difference(diff_method):
    qc = real_amplitudes(3, reps=2)
    obs = SparsePauliOp.from_list([("ZZI", 0.7), ("IXX", -0.4), ("YIY", 0.2)])
    node = qb.to_qnode(qc, observable=obs, diff_method=diff_method)

    x = pnp.array(np.random.default_rng(1).uniform(0, 2 * np.pi, qc.num_parameters), requires_grad=True)
    assert np.isclose(node(x), qiskit_expval(qc, np.asarray(x), obs))
    grad = qml.grad(node)(x)
    assert np.allclose(grad, finite_diff(qc, np.asarray(x), obs), atol=1e-5)


def test_gradient_through_nonlinear_expressions():
    """zz_feature_map uses (pi - x_i)(pi - x_j): gradients must flow through it."""
    qc = zz_feature_map(2, reps=1)
    obs = SparsePauliOp("ZZ")
    node = qb.to_qnode(qc, observable=obs, diff_method="backprop")
    x = pnp.array([0.4, 1.1], requires_grad=True)
    grad = qml.grad(node)(x)
    assert np.allclose(grad, finite_diff(qc, np.asarray(x), obs), atol=1e-5)


def test_expression_functions():
    t = Parameter("t")
    qc = QuantumCircuit(1)
    qc.ry((2 * t + 0.5) ** 2 / 3, 0)
    qc.rz(t.sin() + t.exp() * t.cos(), 0)
    assert qb.verify(qc, values={"t": 0.37}).equivalent
    node = qb.to_qnode(qc, observable="X", diff_method="backprop")
    x = pnp.array(0.37, requires_grad=True)
    fd = finite_diff(qc, np.array([0.37]), SparsePauliOp("X"))[0]
    assert np.isclose(qml.grad(node)(x), fd, atol=1e-5)


def test_torch_interface_if_available():
    torch = pytest.importorskip("torch")
    qc = real_amplitudes(2, reps=1)
    node = qb.to_qnode(qc, observable="ZZ", interface="torch", diff_method="backprop")
    x = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    out = node(x)
    out.backward()
    assert x.grad is not None
