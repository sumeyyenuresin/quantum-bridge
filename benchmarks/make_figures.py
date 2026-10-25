"""Render the paper/README figures from benchmarks/results/*.json (300 dpi PNG)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).parent / "results"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

# Validated categorical palette (slots 1-4, light mode) + recessive inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=INK)


# --------------------------------------------------------------- Fig 2: exactness
inter = json.loads((R / "interop.json").read_text())
groups = ["Standard gates\n(numeric, n=56)", "Standard gates\n(symbolic, n=25)", "Random circuits\n(n=300)"]
g, rnd = inter["gates"], inter["random"]


def pct(tool):
    return [100 * g["numeric"][tool]["ok_exact"] / g["numeric"]["n_gates"],
            100 * g["symbolic"][tool]["ok_exact"] / g["symbolic"]["n_gates"],
            100 * rnd[tool]["exact"] / rnd["n_circuits"]]


fig, ax = plt.subplots(figsize=(6.2, 3.0))
x = np.arange(3)
w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, pct("qbridge"), w, color=C[0], label="Quantum-Bridge")
b2 = ax.bar(x + w / 2 + 0.01, pct("pennylane-qiskit"), w, color=C[1], label="PennyLane-Qiskit plugin")
bar_labels(ax, b1, lambda v: f"{v:.0f}%")
bar_labels(ax, b2, lambda v: f"{v:.0f}%")
ax.set_xticks(x, groups)
ax.set_ylabel("Exactly equivalent (%)")
ax.set_ylim(0, 112)
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
save(fig, "fig2_exactness.png")

# --------------------------------------------------------------- Fig 3: scaling
sc = inter["scaling"]
n_ops = [r["n_ops"] for r in sc]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
for ax, key, title in ((axs[0], "convert_ms", "Translation (ahead of time)"),
                       (axs[1], "execute_ms", "QNode execution, default.qubit")):
    for i, (tool, lab) in enumerate((("qbridge", "Quantum-Bridge"), ("pennylane-qiskit", "Plugin"))):
        y = [r[tool][key] for r in sc]
        ax.plot(n_ops, y, color=C[i], lw=2, marker="o", ms=4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gates in circuit (8 qubits)")
    ax.set_ylabel("Median time (ms)")
    ax.set_title(title, fontsize=9, color=INK)
axs[0].legend(loc="upper left")
save(fig, "fig3_scaling.png")

# --------------------------------------------------------------- Fig 4: ZNE accuracy
zne = json.loads((R / "zne.json").read_text())
rows = zne["accuracy"]
methods = ["unmitigated", "linear", "richardson", "exponential"]
labels = ["Unmitigated", "ZNE linear", "ZNE Richardson", "ZNE exponential"]
fig, ax = plt.subplots(figsize=(7.0, 3.1))
x = np.arange(len(rows))
w = 0.2
for i, m in enumerate(methods):
    vals = [r["mae"][m] for r in rows]
    ax.bar(x + (i - 1.5) * (w + 0.01), vals, w, color=C[i], label=labels[i])
short = {"RealAmplitudes": "RealAmp.", "EfficientSU2": "EffSU2"}
def tick(r):
    fam = r["family"].split("(")[0]
    return f"{short.get(fam, fam)} {r['n_qubits']}q\np₂={r['p2']}"
ax.set_xticks(x, [tick(r) for r in rows], fontsize=7.5)
ax.set_ylabel("Mean absolute error vs ideal")
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), fontsize=8)
save(fig, "fig4_zne_accuracy.png")

# --------------------------------------------------------------- Fig 5: QML
q = zne["qml"]
p2 = [r["p2"] for r in q["noise"]]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
series = (("noisy", "Unmitigated"), ("zne_linear", "ZNE linear"), ("zne_richardson", "ZNE Richardson"))
for i, (k, lab) in enumerate(series):
    ys = [r[k]["mae_vs_ideal"] for r in q["noise"]]
    ax.plot(p2, ys, color=C[i], lw=2, marker="o", ms=4, label=lab)
    ax.annotate(lab, (p2[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7, color=INK)
ax.set_xlim(0, max(p2) * 1.45)
ax.set_xlabel("Two-qubit depolarising rate p₂ (p₁ = p₂/10)")
ax.set_ylabel("Mean |f(x) − f_ideal(x)| on test set")
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig5_qml.png")
print("figures written to", FIG)


# --------------------------------------------------------------- Fig 1: architecture
def box(ax, x, y, w, h, title, body, edge):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=SURF, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.35)


def arrow(ax, a, b, text="", color=INK2, rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                                     connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, text, ha="center", fontsize=6.8, color=INK2)


fig, ax = plt.subplots(figsize=(7.2, 3.7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.9)
ax.axis("off")
box(ax, 0.1, 3.1, 2.2, 1.9, "Qiskit", "QuantumCircuit\nParameter /\nParameterExpression\nterminal measures", C[0])
box(ax, 3.4, 3.1, 3.8, 1.9, "Ahead-of-time compiler",
    "1 native gate table (40 entries)\n2 generic controlled gates\n3 exact unitary (bound, unmapped)\n"
    "4 definition expansion (symbolic)\nexpressions -> qml.math closures", C[0])
box(ax, 8.3, 3.1, 2.6, 1.9, "PennyLane", "PennyLaneCircuit\n(replayable closures)\n-> QNode: autograd /\ntorch / JAX gradients", C[0])
box(ax, 3.4, 0.2, 3.8, 1.7, "verify()  equivalence oracle",
    "Operator(qc) vs qml.matrix(...)\nendianness-corrected, incl.\nglobal phase; state vector >10 q", C[2])
box(ax, 8.3, 0.2, 2.6, 1.7, "zne()  mitigation",
    "fold_global: U(U†U)^k\nany executor(circuit)\nRichardson / linear / exp.", C[1])
box(ax, 0.1, 0.2, 2.2, 1.7, "Report", "native / controlled /\ndecomposed / matrix;\nlossless or raise", C[3])
arrow(ax, (2.35, 4.05), (3.35, 4.05))
ax.text(2.85, 4.2, "to_pennylane", ha="center", fontsize=6.5, color=INK2)
arrow(ax, (7.25, 4.05), (8.25, 4.05))
arrow(ax, (9.6, 5.05), (1.2, 5.05), rad=0.12)
ax.text(5.3, 5.65, "to_qiskit (PennyLane -> Qiskit)", ha="center", fontsize=7, color=INK2)
arrow(ax, (5.3, 3.05), (5.3, 1.95))
arrow(ax, (3.35, 1.05), (2.35, 1.05))
arrow(ax, (9.6, 3.05), (9.6, 1.95))
ax.text(9.7, 2.45, "executors", ha="left", fontsize=6.8, color=INK2)
save(fig, "fig1_architecture.png")
