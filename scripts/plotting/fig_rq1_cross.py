"""RQ1 figure: the 2x2 cross of path fixing and step denoising, as curves.

Per-step past-task (T0) accuracy across the rot-MNIST T0->T1 switch at
momentum 0 for the four cells of the RQ1 cross:

    D1  bowed path,   sampled gradient  (vanilla ER)
    D2  bowed path,   exact gradient    (full-data ER)
    C7  valley floor, sampled gradient  (shipping curriculum)
    C8  valley floor, exact gradient    (curriculum + full buffer)

Colour encodes the path (red = bowed / no gate, blue = valley floor /
curriculum), linestyle encodes the replay gradient (solid = sampled,
dashed = exact), mirroring the rows and columns of the 2x2.  Removing the
noise alone (D2) leaves the smooth arc plus the boundary overshoot; fixing
the path alone (C7) leaves the noisy overshoot; doing both (C8) holds T0
flat through the switch.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

DASH = (0, (5, 2))

# (method, ablation_key, ablation_value, color, linestyle, label)
CONDS = [
    ("er", "decomposition", "D1_vanilla",                   vw.C_VANILLA, "-",  "D1  bowed + noisy"),
    ("er", "decomposition", "D2_fulldata",                  vw.C_VANILLA, DASH, "D2  bowed, exact gradient"),
    ("er", "curriculum",    "C7_adaptive_lmin0.20",         vw.C_CURR,    "-",  "C7  valley floor + noisy"),
    ("er", "curriculum",    "C8_adaptive_lmin0.20_fullbuf", vw.C_CURR,    DASH, "C8  valley floor, exact gradient"),
]
WINDOW = (-25, 234)


def main() -> None:
    vw.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vw.mark_switch(ax)
    for method, key, value, color, ls, label in CONDS:
        vw.plot_transition(ax, method, key, value, "task_0_acc", color, label,
                           linestyle=ls, window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(0.63, 0.94)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_ylabel("past-task ($T_0$) accuracy")
    ax.set_title("Path $\\times$ step noise: only fixing both closes the gap (momentum $0$)")
    ax.annotate("task switch", xy=(0, 0.64), xytext=(10, 0.648),
                color=vw.MUTED, fontsize=9)
    leg = ax.legend(loc="lower right", ncol=1, handlelength=2.2,
                    borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "D" / "rq1_cross_t0.png")


if __name__ == "__main__":
    main()
