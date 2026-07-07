"""RQ1 figure: the deterministic arc survives every driver removal.

Per-step past-task (T0) accuracy across the T0->T1 switch on rot-MNIST at
momentum 0, for the four driver-removal conditions D1-D4.  The boundary spike
shrinks along D1 -> D3 -> D4 as the overshoot drivers are switched off, but a
smooth dip-and-recover *bend* survives every removal -- cleanest in the
exact-gradient conditions (D2, D4), whose curves carry no sampling jitter.
That surviving bend is the deterministic arc.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

KEY = "decomposition"
CONDS = [
    ("D1_vanilla",           vw.C_VANILLA, "D1  vanilla ER"),
    ("D2_fulldata",          vw.C_FULL,    "D2  full-data ER"),
    ("D3_balanced",          vw.C_EXTRA,   "D3  balanced ER"),
    ("D4_fulldata_balanced", vw.C_PER,     "D4  full-data + balanced"),
]
WINDOW = (-25, 234)


def main() -> None:
    vw.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vw.mark_switch(ax)
    for value, color, label in CONDS:
        vw.plot_transition(ax, "er", KEY, value, "task_0_acc", color, label,
                           window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(0.62, 0.94)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_ylabel("past-task ($T_0$) accuracy")
    ax.set_title("The arc survives every driver removal (momentum $0$)")
    ax.annotate("task switch", xy=(0, 0.635), xytext=(10, 0.645),
                color=vw.MUTED, fontsize=9)
    leg = ax.legend(loc="lower right", ncol=2, handlelength=1.6,
                    columnspacing=1.4, borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "D" / "drivers_t0.png")


if __name__ == "__main__":
    main()
