"""RQ2 figure: the scalar feedback gate (curriculum), and momentum's two roles.

Two panels, stacked so the figure fits a single text column, across the
rot-MNIST T0->T1 switch: vanilla ER (red) vs the shipping adaptive curriculum
C7 (lambda_min=0.20, blue), each at momentum 0 (dashed) and 0.9 (solid).  Both
share the step axis, so the dip in (a) lines up with the debt in (b):

  (a) past task T0 (stability): momentum collapses the curriculum's dip to a few
      points, while it barely touches vanilla's -- momentum closes the gap only
      where the gate has already tempered the push at source.
  (b) new task T1 (plasticity): the curriculum slows early T1 learning at
      momentum 0 (the plasticity debt); momentum pays it back, restoring the
      new-task learning curve to the vanilla pace.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

# (method, ablation_key, base value); "_M" appends for the momentum-0.9 leg.
VANILLA = ("er", "decomposition", "D1_vanilla")
CURR = ("er", "curriculum", "C7_adaptive_lmin0.20")
SERIES = [(VANILLA, vw.C_VANILLA, "vanilla ER"),
          (CURR, vw.C_CURR, "curriculum")]

DASH = (0, (5, 2))


def _leg(ax, **kw):
    leg = ax.legend(**kw)
    for line in leg.get_lines():
        line.set_linewidth(2.2)
    return leg


def _panel(ax, task_col, window):
    """Draw both series at both momenta on one axis (std bands on both legs)."""
    # momentum 0.9 -- solid
    for (m, k, v), color, _ in SERIES:
        vw.plot_transition(ax, m, k, v + "_M", task_col, color, None,
                           linestyle="-", window=window, alpha_band=0.16)
    # momentum 0 -- dashed, fainter band
    for (m, k, v), color, _ in SERIES:
        vw.plot_transition(ax, m, k, v, task_col, color, None,
                           linestyle=DASH, window=window, lw=1.6, alpha_band=0.10)


def main() -> None:
    vw.apply_style()
    # Stacked, not side by side: the figure sits in one text column.  The two
    # panels share the step axis so a feature of (a) sits directly above the
    # same step in (b) -- the point of the pairing is that the stability dip
    # and the plasticity debt happen at the same moment.  (b) still carries no
    # sample before the switch; it simply starts at the marker.
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(4.2, 5.2), sharex=True,
                                   constrained_layout=True)

    # -- (a) stability: T0 across the switch --------------------------------
    vw.mark_switch(axT)
    _panel(axT, "task_0_acc", (-25, 234))
    axT.set_ylim(0.66, 0.97)
    axT.set_ylabel("past-task ($T_0$) accuracy")
    axT.set_title("(a) Stability")

    # -- (b) plasticity: T1 learning curve ----------------------------------
    vw.mark_switch(axB)
    _panel(axB, "task_1_acc", (1, 235))
    axB.set_ylim(0.15, 0.99)
    axB.set_xlabel("steps into new task ($T_1$)")
    axB.set_ylabel("new-task ($T_1$) accuracy")
    axB.set_title("(b) Plasticity")
    axB.set_xlim(-25, 235)

    # -- shared legend: colour = method, style = momentum -------------------
    for color, name in [(vw.C_VANILLA, "vanilla ER"), (vw.C_CURR, "curriculum")]:
        axT.plot([], [], color=color, lw=2.2, label=name)
    axT.plot([], [], color=vw.INK_SOFT, lw=2.0, linestyle="-", label="$\\mu{=}0.9$")
    axT.plot([], [], color=vw.INK_SOFT, lw=1.6, linestyle=DASH, label="$\\mu{=}0.0$")
    _leg(axT, loc="lower right", ncol=2, handlelength=1.6, columnspacing=1.0,
         labelspacing=0.3, borderpad=0.2)

    vw.finalize(fig, vw.FIG_DIR / "curriculum" / "curriculum.png")


if __name__ == "__main__":
    main()
