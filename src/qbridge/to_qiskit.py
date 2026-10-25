"""PennyLane quantum function / tape -> Qiskit ``QuantumCircuit``."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pennylane as qml
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from .exceptions import UnsupportedOperationError
from .gates import pl_to_qiskit_table
from .report import ConversionReport

_SKIP = {"Barrier", "Snapshot"}


def to_qiskit(
    qfunc_or_tape,
    *args,
    wire_order: Sequence | None = None,
    measure: bool = False,
    return_report: bool = False,
    **kwargs,
):
    """Translate a PennyLane quantum function (or tape) into a Qiskit circuit.

    Parameters
    ----------
    qfunc_or_tape:
        A quantum function, a QNode, or a ``QuantumScript``/``QuantumTape``.
    *args, **kwargs:
        Numeric arguments used to build the tape (ignored for tapes).
    wire_order:
        PennyLane wires in the order they should become Qiskit qubits
        ``0..n-1``. Defaults to sorted wires if they are all integers,
        otherwise order of first appearance.
    measure:
        Append ``measure`` instructions on the wires the PennyLane function
        measures (useful before sending to hardware).
    """
    tape = _get_tape(qfunc_or_tape, args, kwargs)

    if wire_order is None:
        wl = list(tape.wires)
        wire_order = sorted(wl) if all(isinstance(w, (int, np.integer)) for w in wl) else wl
    wire_order = list(wire_order)
    missing = [w for w in tape.wires if w not in wire_order]
    if missing:
        raise ValueError(f"wire_order is missing wires {missing}.")
    index = {w: i for i, w in enumerate(wire_order)}

    qc = QuantumCircuit(len(wire_order))
    report = ConversionReport(direction="pennylane -> qiskit")
    table = pl_to_qiskit_table()

    for op in tape.operations:
        _append(qc, op, index, table, report)

    if measure:
        m_wires = []
        for m in tape.measurements:
            mw = m.wires if len(m.wires) else tape.wires
            m_wires.extend(w for w in mw if w not in m_wires)
        qubits = [index[w] for w in m_wires]
        qc.add_register(*_creg(len(qubits)))
        qc.measure(qubits, list(range(len(qubits))))
        report.measured_qubits = qubits

    report.global_phase = float(qc.global_phase)
    return (qc, report) if return_report else qc


def _creg(n):
    from qiskit.circuit import ClassicalRegister

    return [ClassicalRegister(n, "meas")]


def _get_tape(obj, args, kwargs):
    if isinstance(obj, qml.tape.QuantumScript):
        return obj
    if isinstance(obj, qml.QNode):
        return qml.workflow.construct_tape(obj)(*args, **kwargs)
    if callable(obj):
        return qml.tape.make_qscript(obj)(*args, **kwargs)
    raise TypeError(f"Expected a quantum function, QNode or tape, got {type(obj).__name__}.")


def _num(p):
    """Strip autodiff wrappers: Qiskit needs plain floats."""
    return float(qml.math.real(qml.math.unwrap(p)))


def _append(qc, op, index, table, report, _record=True):
    name = op.name
    qubits = [index[w] for w in op.wires]
    rec = report.record if _record else (lambda *a: None)

    if name in _SKIP:
        rec(name, "skipped")
        return

    if name == "GlobalPhase":
        # PennyLane GlobalPhase(phi) = exp(-i phi)
        qc.global_phase -= _num(op.data[0])
        rec(name, "native")
        return

    factory = table.get(name)
    if factory is not None:
        qc.append(factory([_num(p) for p in op.data]), qubits)
        rec(name, "native")
        return

    # Adjoint(base) where base is mapped
    if isinstance(op, qml.ops.op_math.Adjoint) and op.base.name in table:
        gate = table[op.base.name]([_num(p) for p in op.base.data]).inverse()
        qc.append(gate, qubits)
        rec(name, "native")
        return

    # Controlled(base) where base is mapped
    if isinstance(op, qml.ops.op_math.Controlled) and op.base.name in table:
        base_gate = table[op.base.name]([_num(p) for p in op.base.data])
        ctrl_vals = [int(v) for v in op.control_values]
        ctrl_state = sum(v << i for i, v in enumerate(ctrl_vals))
        gate = base_gate.control(len(op.control_wires), ctrl_state=ctrl_state)
        ctrl_q = [index[w] for w in op.control_wires]
        tgt_q = [index[w] for w in op.target_wires]
        qc.append(gate, ctrl_q + tgt_q)
        rec(name, "controlled")
        return

    # Try PennyLane's own (exact) decomposition first
    if op.has_decomposition:
        try:
            decomp = op.decomposition()
        except Exception:  # pragma: no cover
            decomp = None
        if decomp:
            for sub in decomp:
                _append(qc, sub, index, table, report, _record=False)
            rec(name, "decomposed")
            return

    # Last resort: exact matrix
    if op.has_matrix:
        mat = qml.math.unwrap(qml.matrix(op, wire_order=op.wires))
        # PennyLane matrices are big-endian in op.wires; Qiskit is little-endian
        qc.append(UnitaryGate(np.asarray(mat), check_input=False), list(reversed(qubits)))
        rec(name, "matrix")
        return

    raise UnsupportedOperationError(f"Cannot translate PennyLane operation '{name}'.")
