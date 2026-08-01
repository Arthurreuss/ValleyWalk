"""Appendix figure: the original unit-norm balanced conditions.

Per-step past-task (T0) accuracy across the rot-MNIST switch at momentum 0
for the original balanced conditions (old D3/D4), which normalise the joint
update to unit length and so fix the step length at eta, next to vanilla ER
and the revised balanced-direction condition run at vanilla's step length.

The comparison shows where the original conditions' spike removal came from:
capping the step length.  The same remix at vanilla's step length (revised
D3) deepens the drop instead.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

DASH = (0, (5, 2))

# (ablation_value, color, linestyle, label)
CONDS = [
    ("D1_vanilla",           vw.C_VANILLA, "-",  "D1  vanilla ER"),
    ("D3_balanced",          vw.C_CURR,    "-",  "old D3  remix + capped step"),
    ("D4_fulldata_balanced", vw.C_CURR,    DASH, "old D4  + exact gradient"),
    ("D3_balanced_dir",      vw.C_EXTRA,   "-",  "revised D3  remix at full step"),
]
WINDOW = (-25, 234)


def main() -> None:
    vw.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vw.mark_switch(ax)
    for value, color, ls, label in CONDS:
        vw.plot_transition(ax, "er", "decomposition", value, "task_0_acc",
                           color, label, linestyle=ls, window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(0.42, 0.97)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_ylabel("past-task ($T_0$) accuracy")
    ax.set_title("Unit-norm balancing removes the spike by capping the step (momentum $0$)")
    leg = ax.legend(loc="lower right", ncol=1, handlelength=2.2,
                    borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "D" / "old_balanced_t0.png")


if __name__ == "__main__":
    main()
