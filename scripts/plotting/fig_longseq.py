"""Appendix figure: five-task rot-MNIST long sequence (L-series).

Does each gate carry past a single transition, and how does momentum act on the
long horizon?  Rotations [0, 30, 60, 90, 120] deg, four consecutive switches,
five seeds.  The L-series is the two-gate comparison: L1 vanilla ER
(red), L2 the shipping curriculum (blue), L3 asymmetric PER delta=0.1 (aqua).
Both panels show first-task T0 accuracy across the whole sequence (dashed lines
are the four switches); the two panels are the two momentum settings.

  (a) momentum 0: both gates hold T0 almost flat through every boundary, with
      asymmetric PER the smoothest of all (smallest per-boundary area); vanilla
      ER drops hard at each switch.
  (b) momentum 0.9: momentum re-inflates PER's per-step filter (exactly as on the
      two-task benchmark), so its boundary transients return and it tracks vanilla
      on cumulative retention, while the curriculum keeps the boundaries smoothest
      and holds the highest plateau.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

SERIES = [("er", "longseq", "L1_vanilla_ER", vw.C_VANILLA, "vanilla ER"),
          ("er", "longseq", "L2_adaptive_lmin0.20", vw.C_CURR, "curriculum"),
          ("precond_er", "longseq", "L3_PER_asym_d0.1", vw.C_PER, "asym PER")]
BOUNDARIES = [235, 470, 705, 940]


def _panel(ax, suffix, title):
    for b in BOUNDARIES:
        vw.mark_switch(ax, b)
    for m, k, v, color, name in SERIES:
        steps, mean, std = vw.task_curve(m, k, v + suffix, "task_0_acc")
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.13, lw=0)
        ax.plot(steps, mean, color=color, lw=1.7, label=name, solid_capstyle="round")
    ax.set_xlim(0, 1174)
    ax.set_ylim(0.6, 0.97)
    ax.set_xlabel("step (five tasks, four switches)")
    ax.set_title(title)
    for b, t in zip(BOUNDARIES, ["$T_1$", "$T_2$", "$T_3$", "$T_4$"]):
        ax.annotate(t, xy=(b, 0.955), xytext=(b + 6, 0.955), color=vw.MUTED, fontsize=8)


def main() -> None:
    vw.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    _panel(axL, "", "(a) Momentum $0$")
    _panel(axR, "_M", "(b) Momentum $0.9$")
    axL.set_ylabel("first-task ($T_0$) accuracy")
    leg = axL.legend(loc="lower left", handlelength=1.7)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "L" / "longseq.png")


if __name__ == "__main__":
    main()
