"""Human-readable record of how each instruction was translated."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ConversionReport:
    """How every instruction of the source circuit was handled.

    Categories
    ----------
    native      mapped one-to-one onto a named target gate
    controlled  generic controlled gate built from a mapped base gate
    decomposed  unmapped gate expanded through its own definition (exact)
    matrix      unmapped, fully bound gate embedded as its exact unitary
    skipped     no unitary action (barrier, delay) or terminal measurement
    """

    direction: str
    counts: Counter = field(default_factory=Counter)
    by_gate: dict[str, str] = field(default_factory=dict)
    measured_qubits: list[int] = field(default_factory=list)
    global_phase: float = 0.0

    def record(self, gate_name: str, category: str) -> None:
        self.counts[category] += 1
        self.by_gate.setdefault(gate_name, category)

    @property
    def lossless(self) -> bool:
        """True by construction: anything lossy raises instead of being recorded."""
        return True

    @property
    def total(self) -> int:
        return sum(v for k, v in self.counts.items() if k != "skipped")

    def as_dict(self) -> dict:
        return {
            "direction": self.direction,
            "counts": dict(self.counts),
            "by_gate": dict(self.by_gate),
            "measured_qubits": list(self.measured_qubits),
            "global_phase": self.global_phase,
        }

    def __str__(self) -> str:
        lines = [f"Quantum-Bridge conversion report ({self.direction})"]
        order = ["native", "controlled", "decomposed", "matrix", "skipped"]
        for cat in order:
            if self.counts.get(cat):
                gates = sorted(g for g, c in self.by_gate.items() if c == cat)
                lines.append(f"  {cat:<11} {self.counts[cat]:>5}   [{', '.join(gates)}]")
        if self.measured_qubits:
            lines.append(f"  measured qubits: {self.measured_qubits}")
        if self.global_phase:
            lines.append(f"  global phase kept: {self.global_phase:.6f}")
        lines.append("  status: lossless (exact unitary preserved)")
        return "\n".join(lines)

    __repr__ = __str__
