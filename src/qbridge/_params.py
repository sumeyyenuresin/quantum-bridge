"""Compile Qiskit parameter expressions into interface-agnostic callables.

Qiskit gate parameters can be plain numbers, a single ``Parameter`` or an
arbitrary ``ParameterExpression`` such as ``2*x[0]`` or
``(pi - x[0])*(pi - x[1])`` (used by ``zz_feature_map``).

To keep the converted circuit differentiable under *any* PennyLane interface
(autograd, torch, jax, tensorflow) we never call ``ParameterExpression.bind``
at execution time. Instead the expression is translated once, at conversion
time, into a small tree of Python closures that only use ``+ - * / **`` and
``qml.math`` functions. Those dispatch to whatever array library the incoming
values belong to, so gradients flow through unchanged.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pennylane as qml
from qiskit.circuit import Parameter, ParameterExpression

from .exceptions import UnsupportedOperationError

Evaluator = Callable[[Mapping[str, object]], object]

# sympy function class name -> qml.math function name
_FUNCS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "Abs": "abs",
    "sign": "sign",
}


def _const(value: float) -> Evaluator:
    return lambda _values: value


def compile_param(param) -> tuple[Evaluator, bool]:
    """Return ``(evaluator, is_symbolic)`` for one Qiskit gate parameter."""
    # Plain numbers (int, float, numpy scalar)
    if not isinstance(param, ParameterExpression):
        if isinstance(param, (complex, np.complexfloating)):
            raise UnsupportedOperationError(f"Complex gate parameter {param!r} is not supported.")
        return _const(float(param)), False

    # Fully bound expression
    if not param.parameters:
        return _const(float(param)), False

    # Fast path: the parameter itself (the overwhelmingly common case)
    if isinstance(param, Parameter):
        name = param.name
        return (lambda values: values[name]), True

    # General expression: translate the sympy tree once
    return _compile_sympy(param.sympify()), True


def _compile_sympy(expr) -> Evaluator:
    import sympy  # imported lazily, only needed for non-trivial expressions

    if expr.is_Symbol:
        name = expr.name
        return lambda values: values[name]

    if expr.is_Number or expr in (sympy.pi, sympy.E):
        val = float(expr)
        return _const(val)

    if expr.is_Add:
        parts = [_compile_sympy(a) for a in expr.args]

        def add(values):
            total = parts[0](values)
            for p in parts[1:]:
                total = total + p(values)
            return total

        return add

    if expr.is_Mul:
        parts = [_compile_sympy(a) for a in expr.args]

        def mul(values):
            total = parts[0](values)
            for p in parts[1:]:
                total = total * p(values)
            return total

        return mul

    if expr.is_Pow:
        base, exponent = (_compile_sympy(a) for a in expr.args)
        return lambda values: base(values) ** exponent(values)

    fname = type(expr).__name__
    if fname in _FUNCS and len(expr.args) == 1:
        inner = _compile_sympy(expr.args[0])
        fn = getattr(qml.math, _FUNCS[fname])
        return lambda values: fn(inner(values))

    raise UnsupportedOperationError(
        f"Cannot translate parameter expression {expr} (unsupported node '{fname}')."
    )
