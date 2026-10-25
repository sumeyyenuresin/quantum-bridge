"""Qiskit ``QuantumCircuit`` -> PennyLane quantum function.

The conversion happens once, up front: every instruction is compiled into a
small closure ("step"). Calling the resulting :class:`PennyLaneCircuit` just
replays those closures with the current parameter values, so the converted
circuit is cheap to evaluate many times (training loops, gradient
evaluations) and stays fully differentiable.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pennylane as qml
from qiskit.circuit import ControlledGate, Gate, QuantumCircuit
from qiskit.quantum_info import Operator

from ._params import compile_param
from .exceptions import ParameterBindingError, UnsupportedOperationError
from .gates import QISKIT_TO_PL, SKIPPED
from .report import ConversionReport

Step = Callable[[dict], None]

_CONTROL_FLOW = {"if_else", "for_loop", "while_loop", "switch_case", "box"}
_NON_UNITARY = {"reset", "initialize"}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
class PennyLaneCircuit:
    """A Qiskit circuit translated into a PennyLane quantum function.

    Use it exactly like a hand-written PennyLane function inside a QNode::

        circuit = qb.to_pennylane(qc)

        @qml.qnode(dev)
        def model(theta):
            circuit(theta)
            return qml.expval(qml.PauliZ(0))

    Parameters are matched by the order of ``qc.parameters`` (the same order
    Qiskit uses in ``assign_parameters``) or by name as keyword arguments.
    """

    def __init__(self, qc: QuantumCircuit, steps, wires, report, param_names):
        self.source = qc
        self._steps: list[tuple[Step, tuple]] = steps
        self.wires = list(wires)
        self.report: ConversionReport = report
        self.parameters: list[str] = param_names

    # -- properties --------------------------------------------------------
    @property
    def num_wires(self) -> int:
        return len(self.wires)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def measured_wires(self) -> list:
        """Wires that the source circuit measured at the end (or all wires)."""
        q = self.report.measured_qubits or list(range(self.num_wires))
        return [self.wires[i] for i in q]

    # -- binding -----------------------------------------------------------
    def _bind(self, args, kwargs) -> dict:
        names = self.parameters
        if kwargs and not args:
            missing = [n for n in names if n not in kwargs]
            extra = [k for k in kwargs if k not in names]
            if missing or extra:
                raise ParameterBindingError(
                    f"Expected parameters {names}; missing {missing}, unexpected {extra}."
                )
            return dict(kwargs)
        if kwargs:
            raise ParameterBindingError("Pass parameters either positionally or by name, not both.")

        # A single flat array holding all parameters is accepted as a convenience.
        if len(args) == 1 and len(names) != 1:
            shape = qml.math.shape(args[0])
            if len(shape) == 1 and shape[0] == len(names):
                args = tuple(args[0][i] for i in range(len(names)))
        if len(args) != len(names):
            raise ParameterBindingError(
                f"Circuit has {len(names)} parameter(s) {names}, got {len(args)} value(s)."
            )
        return dict(zip(names, args))

    # -- execution ---------------------------------------------------------
    def apply(self, values: dict, after_gate: Callable[[Sequence], None] | None = None) -> None:
        """Queue the circuit's operations for already-bound ``values``.

        ``after_gate(wires)`` is called after every gate; the bundled noisy
        executors use it to insert noise channels.
        """
        for step, wires in self._steps:
            step(values)
            if after_gate is not None and wires:
                after_gate(wires)

    def __call__(self, *args, **kwargs) -> None:
        self.apply(self._bind(args, kwargs))

    def __repr__(self) -> str:
        return (
            f"<PennyLaneCircuit wires={self.num_wires} params={self.num_parameters} "
            f"ops={self.report.total}>"
        )


def to_pennylane(
    qc: QuantumCircuit,
    wires: Sequence | None = None,
    *,
    keep_global_phase: bool = True,
) -> PennyLaneCircuit:
    """Translate a Qiskit circuit into a reusable PennyLane quantum function.

    Parameters
    ----------
    qc:
        The circuit. May contain unbound ``Parameter``/``ParameterExpression``
        objects; they become arguments of the returned function.
    wires:
        Optional PennyLane wire labels, one per Qiskit qubit. Defaults to
        ``range(qc.num_qubits)``.
    keep_global_phase:
        Emit ``qml.GlobalPhase`` so the translated unitary matches Qiskit's
        exactly (not just up to phase). Harmless on all PennyLane simulators;
        switch off for devices that reject ``GlobalPhase``.

    Raises
    ------
    UnsupportedOperationError
        For anything that cannot be translated exactly: classical control
        flow, mid-circuit measurement, reset.
    """
    wires = list(range(qc.num_qubits)) if wires is None else list(wires)
    if len(wires) != qc.num_qubits:
        raise ValueError(f"Got {len(wires)} wire labels for {qc.num_qubits} qubits.")

    report = ConversionReport(direction="qiskit -> pennylane")
    final_meas = _final_measurements(qc)
    report.measured_qubits = sorted({qc.find_bit(q).index for q in final_meas})

    steps = _compile_circuit(qc, wires, report, final_meas, top_level=True)

    if keep_global_phase:
        gp_step = _global_phase_step(qc.global_phase, wires)
        if gp_step is not None:
            steps.append((gp_step, ()))
            try:
                report.global_phase = float(qc.global_phase)
            except TypeError:  # symbolic phase
                report.global_phase = float("nan")

    names = [p.name for p in qc.parameters]
    return PennyLaneCircuit(qc, steps, wires, report, names)


def to_qnode(
    qc: QuantumCircuit,
    device=None,
    observable=None,
    *,
    diff_method: str = "best",
    interface: str = "auto",
    shots: int | None = None,
):
    """One-liner: Qiskit circuit -> ready-to-run (and differentiable) QNode.

    observable:
        ``None`` returns measurement probabilities over the qubits the source
        circuit measured (or all qubits), indexed in **Qiskit's bit order**, so
        the vector can be compared directly with Qiskit results.
        Otherwise a Pauli string in Qiskit order (``"ZZI"``), a
        ``SparsePauliOp``/``Pauli``, a PennyLane observable, or a list of those
        (returns one expectation value per observable).
    """
    from .observables import to_pl_observable

    circuit = to_pennylane(qc)
    if device is None:
        device = qml.device("default.qubit", wires=circuit.wires)

    if observable is None:
        prob_wires = list(reversed(circuit.measured_wires))  # Qiskit little-endian

        def measure():
            return qml.probs(wires=prob_wires)

    elif isinstance(observable, (list, tuple)):
        obs = [to_pl_observable(o, circuit.num_wires) for o in observable]

        def measure():
            return [qml.expval(o) for o in obs]

    else:
        obs1 = to_pl_observable(observable, circuit.num_wires)

        def measure():
            return qml.expval(obs1)

    def qfunc(*args, **kwargs):
        circuit(*args, **kwargs)
        return measure()

    qnode = qml.QNode(qfunc, device, diff_method=diff_method, interface=interface)
    if shots is not None:
        qnode = qml.set_shots(qnode, shots=shots)
    qnode.qbridge_circuit = circuit
    return qnode


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _final_measurements(qc: QuantumCircuit) -> set:
    """Qubits whose measurement is terminal (nothing but barriers/measures after)."""
    final = set()
    blocked = set()
    for inst in reversed(qc.data):
        name = inst.operation.name
        if name == "measure":
            q = inst.qubits[0]
            if q not in blocked:
                final.add(q)
        elif name in SKIPPED:
            continue
        else:
            blocked.update(inst.qubits)
    return final


def _global_phase_step(phase, wires):
    evaluator, symbolic = compile_param(phase)
    if not symbolic and abs(evaluator({})) < 1e-15:
        return None
    # PennyLane: GlobalPhase(phi) = exp(-i phi) I ; Qiskit: exp(+i phase)
    return lambda values: qml.GlobalPhase(-evaluator(values), wires=wires)


def _compile_circuit(qc, wires, report, final_meas, top_level=False):
    steps = []
    for inst in qc.data:
        op = inst.operation
        name = op.name
        qidx = [qc.find_bit(q).index for q in inst.qubits]
        op_wires = [wires[i] for i in qidx]

        if name in SKIPPED:
            report.record(name, "skipped")
            continue
        if name == "measure":
            if top_level and inst.qubits[0] in final_meas:
                report.record(name, "skipped")
                continue
            raise UnsupportedOperationError(
                "Mid-circuit measurement is not supported (the qubit is used again "
                "after being measured). Use PennyLane's qml.measure manually."
            )
        if name in _CONTROL_FLOW:
            raise UnsupportedOperationError(
                f"Classical control flow ('{name}') cannot be translated losslessly."
            )
        if name in _NON_UNITARY or not isinstance(op, Gate):
            raise UnsupportedOperationError(f"Non-unitary instruction '{name}' is not supported.")

        step, category = _compile_gate(op, op_wires, report)
        report.record(name, category)
        steps.append((step, tuple(op_wires)))
    return steps


def _compile_gate(op: Gate, op_wires, report):
    """Return (step, category) for a single unitary gate."""
    name = op.name
    compiled = [compile_param(p) for p in op.params]
    evaluators = [e for e, _ in compiled]
    symbolic = any(s for _, s in compiled)

    # 1) named one-to-one mapping
    builder = QISKIT_TO_PL.get(name)
    if builder is not None:
        def step(values, _b=builder, _e=evaluators, _w=op_wires):
            _b([e(values) for e in _e], _w)

        return step, "native"

    # 2) generic controlled gate whose parameters all live on the base gate
    if isinstance(op, ControlledGate) and len(op.params) == len(op.base_gate.params):
        n_ctrl = op.num_ctrl_qubits
        ctrl_w, tgt_w = op_wires[:n_ctrl], op_wires[n_ctrl:]
        ctrl_vals = [(op.ctrl_state >> i) & 1 for i in range(n_ctrl)]
        base_step, _ = _compile_gate(op.base_gate, tgt_w, report)

        def step(values, _s=base_step, _c=ctrl_w, _v=ctrl_vals):
            qml.ctrl(_s, control=_c, control_values=_v)(values)

        return step, "controlled"

    # 3) fully numeric -> exact unitary
    if not symbolic:
        try:
            mat = Operator(op).data
        except Exception:  # pragma: no cover - extremely exotic gates
            mat = None
        if mat is not None:
            # Qiskit matrices are little-endian in the gate's qubit order
            rev = list(reversed(op_wires))

            def step(values, _m=mat, _w=rev):
                qml.QubitUnitary(_m, wires=_w)

            return step, "matrix"

    # 4) symbolic and unmapped -> expand through the gate's own definition
    definition = op.definition
    if definition is None:
        raise UnsupportedOperationError(f"Gate '{name}' has no definition and no known mapping.")
    sub_steps = _compile_circuit(definition, op_wires, ConversionReport("inner"), set())
    gp = _global_phase_step(definition.global_phase, op_wires)

    def step(values, _sub=sub_steps, _gp=gp):
        for s, _ in _sub:
            s(values)
        if _gp is not None:
            _gp(values)

    return step, "decomposed"
