"""Zero-noise extrapolation (ZNE) for Qiskit circuits — no extra dependencies.

ZNE runs the same circuit at several *amplified* noise levels and
extrapolates the measured expectation value back to the zero-noise limit.
Noise is amplified by **unitary folding**: ``U -> U (U^dagger U)^k``, which is
the identity on an ideal device but multiplies the gate count (and therefore
the accumulated noise) by ``1 + 2k``.

Functions here work on any executor ``f(QuantumCircuit) -> float``: a
simulator, a Qiskit primitive wrapped in a lambda, or a real backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit

Executor = Callable[[QuantumCircuit], float]


# --------------------------------------------------------------------------- #
# Noise scaling
# --------------------------------------------------------------------------- #
def _split_measurements(qc: QuantumCircuit):
    body = qc.remove_final_measurements(inplace=False)
    body_no_barrier = body.copy_empty_like()
    for inst in body.data:
        if inst.operation.name != "barrier":
            body_no_barrier.append(inst)
    meas = [inst for inst in qc.data if inst.operation.name == "measure"]
    return body_no_barrier, meas


def fold_global(qc: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    """Return a circuit whose gate count is ``scale_factor`` times larger.

    Odd integers give ``U (U^dagger U)^k`` exactly. Other values fold only the
    last gates (partial folding), as in Giurgica-Tiron et al. (2020).
    Terminal measurements are preserved; the logical action is unchanged.
    """
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1.")
    body, meas = _split_measurements(qc)
    d = len(body.data)
    k = int((scale_factor - 1) // 2)
    s = int(round(d * ((scale_factor - 1) / 2 - k)))

    out = qc.copy_empty_like()
    out.compose(body, inplace=True)
    inv = body.inverse()
    for _ in range(k):
        out.compose(inv, inplace=True)
        out.compose(body, inplace=True)
    if s:
        tail = body.copy_empty_like()
        for inst in body.data[d - s:]:
            tail.append(inst)
        tail.global_phase = 0
        out.compose(tail.inverse(), inplace=True)
        out.compose(tail, inplace=True)
    for inst in meas:
        out.append(inst)
    return out


def actual_scale(qc: QuantumCircuit, folded: QuantumCircuit) -> float:
    """Realised noise-scale factor = folded gate count / original gate count."""
    a, _ = _split_measurements(qc)
    b, _ = _split_measurements(folded)
    return len(b.data) / max(len(a.data), 1)


# --------------------------------------------------------------------------- #
# Extrapolation
# --------------------------------------------------------------------------- #
def richardson_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """Exact polynomial through all points (degree n-1), evaluated at 0."""
    x = np.asarray(scales, float)
    y = np.asarray(values, float)
    total = 0.0
    for i in range(len(x)):
        others = np.delete(x, i)
        total += y[i] * np.prod(others / (others - x[i]))
    return float(total)


def poly_extrapolate(scales, values, order: int = 1) -> float:
    """Least-squares polynomial of given order, evaluated at 0."""
    coeffs = np.polyfit(np.asarray(scales, float), np.asarray(values, float), order)
    return float(coeffs[-1])


def linear_extrapolate(scales, values) -> float:
    return poly_extrapolate(scales, values, order=1)


def exponential_extrapolate(scales, values, asymptote: float = 0.0) -> float:
    """Fit ``a * exp(-b x) + asymptote``; suited to depolarising noise.

    Falls back to linear extrapolation if values change sign.
    """
    y = np.asarray(values, float) - asymptote
    if np.any(y == 0) or not (np.all(y > 0) or np.all(y < 0)):
        return linear_extrapolate(scales, values)
    sign = np.sign(y[0])
    slope, intercept = np.polyfit(np.asarray(scales, float), np.log(np.abs(y)), 1)
    return float(sign * np.exp(intercept) + asymptote)


EXTRAPOLATORS = {
    "richardson": richardson_extrapolate,
    "linear": linear_extrapolate,
    "quadratic": lambda s, v: poly_extrapolate(s, v, 2),
    "exponential": exponential_extrapolate,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class ZNEResult:
    value: float
    unmitigated: float
    scale_factors: list = field(default_factory=list)
    noisy_values: list = field(default_factory=list)
    method: str = "richardson"

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        pts = ", ".join(f"λ={s:.2f}: {v:+.5f}" for s, v in zip(self.scale_factors, self.noisy_values))
        return (
            f"ZNE ({self.method}) -> {self.value:+.6f}  "
            f"(unmitigated {self.unmitigated:+.6f}; {pts})"
        )


def zne(
    qc: QuantumCircuit,
    executor: Executor,
    scale_factors: Sequence[float] = (1, 3, 5),
    method: str | Callable = "richardson",
    return_result: bool = False,
):
    """Zero-noise-extrapolated expectation value of ``executor(qc)``.

    >>> value = zne(qc, executor)                       # 1 line
    >>> res = zne(qc, executor, (1, 2, 3), "linear", return_result=True)
    """
    if qc.parameters:
        raise ValueError("Bind all parameters before calling zne().")
    if min(scale_factors) < 1:
        raise ValueError("Scale factors must be >= 1.")
    fn = EXTRAPOLATORS[method] if isinstance(method, str) else method

    folded = [fold_global(qc, s) for s in scale_factors]
    scales = [actual_scale(qc, f) for f in folded]
    values = [float(executor(f)) for f in folded]
    value = fn(scales, values)

    unmitigated = values[scales.index(min(scales))]
    result = ZNEResult(value, unmitigated, scales, values,
                       method if isinstance(method, str) else getattr(fn, "__name__", "custom"))
    return result if return_result else value
