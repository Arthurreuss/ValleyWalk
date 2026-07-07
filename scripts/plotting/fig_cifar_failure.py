"""Appendix figure: the three-task CIFAR-10 GroupNorm reliability boundary.

The shipping curriculum on the three-task sequence none -> Gaussian -> shot
(ResNet-18, three seeds), GroupNorm (orange) against its BatchNorm counterpart
(blue).  Past-task T0 (a) and T1 (b) across both switches (dashed lines at
~2k and ~4k steps).  The first switch barely perturbs either norm; the second
(Gaussian -> shot, two near-identical corruptions) drives a deep transient under
GroupNorm only, while BatchNorm holds flat -- the adaptive-ratio reliability
boundary discussed in the results, localised to one boundary and one norm.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

KEY = "cifar_generalization"
GN_COLOR = vw.C_EXTRA   # orange
BN_COLOR = vw.C_CURR    # blue
BOUNDARIES = [1960, 3920]
SERIES = [("Gt1_ship_3task_gn", GN_COLOR, "GroupNorm"),
          ("Gt1_ship_3task_bn", BN_COLOR, "BatchNorm")]


def _panel(ax, task_col, title):
    for b in BOUNDARIES:
        vw.mark_switch(ax, b)
    for value, color, name in SERIES:
        steps, mean, std = vw.task_curve("er", KEY, value, task_col)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.plot(steps, mean, color=color, lw=1.8, label=name, solid_capstyle="round")
    ax.set_xlim(0, 5879)
    ax.set_ylim(0.15, 0.9)
    ax.set_xlabel("step (three tasks, two switches)")
    ax.set_title(title)


def main() -> None:
    vw.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    _panel(axL, "task_0_acc", "(a) $T_0$ (clean)")
    _panel(axR, "task_1_acc", "(b) $T_1$ (Gaussian noise)")
    axL.set_ylabel("past-task accuracy")
    axL.annotate("Gaussian$\\rightarrow$shot", xy=(3920, 0.22), xytext=(3050, 0.20),
                 color=vw.MUTED, fontsize=8)
    leg = axL.legend(loc="lower left", handlelength=1.7)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "T" / "cifar_3task_failure.png")


if __name__ == "__main__":
    main()
