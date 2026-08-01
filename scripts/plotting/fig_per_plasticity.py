"""RQ3 plasticity evidence: new-task learning speed under the delta sweep.

Per-step new-task (T1) accuracy across the rot-MNIST switch at momentum 0
for the standard-buffer asymmetric-PER sweep (P1-P4) against vanilla ER.
The claim "no measured plasticity cost" needs the learning curve, not only
ACC/WP100.  The curve shows the honest version: the strongest filter
(delta = 0.03) is visibly slower over roughly the first fifty steps, the
selective slowdown along past-task-sharp directions, and indistinguishable
from vanilla thereafter; by the WP100 horizon no cost remains.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

# (method, ablation_key, ablation_value, color, label)
CONDS = [
    ("er",         "decomposition", "D1_vanilla", vw.C_VANILLA, "vanilla ER"),
    ("precond_er", "per_asym_sweep",     "P1_d1.0",    vw.DELTA_RAMP[1.0],  r"$\delta{=}1.0$"),
    ("precond_er", "per_asym_sweep",     "P2_d0.3",    vw.DELTA_RAMP[0.3],  r"$\delta{=}0.3$"),
    ("precond_er", "per_asym_sweep",     "P3_d0.1",    vw.DELTA_RAMP[0.1],  r"$\delta{=}0.1$"),
    ("precond_er", "per_asym_sweep",     "P4_d0.03",   vw.DELTA_RAMP[0.03], r"$\delta{=}0.03$"),
]
WINDOW = (1, 235)


def main() -> None:
    vw.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for method, key, value, color, label in CONDS:
        vw.plot_transition(ax, method, key, value, "task_1_acc", color, label,
                           window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_ylabel("new-task ($T_1$) accuracy")
    ax.set_title("A brief early slowdown, no cost thereafter (momentum $0$)")
    leg = ax.legend(loc="lower right", ncol=2, handlelength=1.6,
                    columnspacing=1.4, borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "P" / "per_plasticity.png")


if __name__ == "__main__":
    main()
