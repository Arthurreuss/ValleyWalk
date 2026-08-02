"""RQ2 figures: the scalar feedback gate (curriculum), and momentum's two roles.

``curriculum.png`` (main text, one text column, panels stacked) across the
rot-MNIST T0->T1 switch: vanilla ER (red) vs the shipping adaptive curriculum
C7 (lambda_min=0.20, blue), each at momentum 0 (dashed) and 0.9 (solid).  Both
share the step axis, so the dip in (a) lines up with the debt in (b):

  (a) past task T0 (stability): momentum collapses the curriculum's dip to a few
      points, while it barely touches vanilla's -- momentum closes the gap only
      where the gate has already tempered the push at source.
  (b) new task T1 (plasticity): the curriculum slows early T1 learning at
      momentum 0 (the plasticity debt); momentum pays it back, restoring the
      new-task learning curve to the vanilla pace.

``curriculum_linear.png`` and ``curriculum_adaptive.png`` (Appendix C.2) put
every condition of the C-series behind that one headline rung, one figure per
schedule family, on a shared 2x2: rows are the momentum setting, columns are
stability (T0) and plasticity (T1), with vanilla ER carried into every panel
as the ungated reference.  Splitting the families keeps each panel to three or
four rungs of one ordinal ramp, and lets each ramp track the knob its legend
names: darker is a longer ramp in the first figure and a higher lambda_min
floor in the second.  Note that these run opposite ways as *interventions* --
a longer ramp brakes more, a higher floor brakes less -- which is exactly the
trade the two figures are there to show.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

# (method, ablation_key, base value); "_M" appends for the momentum-0.9 leg.
VANILLA = ("er", "decomposition", "D1_vanilla")
CURR = ("er", "curriculum", "C7_adaptive_lmin0.20")
SERIES = [(VANILLA, vw.C_VANILLA, "vanilla ER"),
          (CURR, vw.C_CURR, "curriculum")]

# The two C-series families, each an ordinal sweep of its own knob: the linear
# schedules lengthen the ramp, the adaptive ones raise the floor under it.
# C4 is the unfloored adaptive schedule, so it opens the second ramp.
LINEAR = [("C1_linear_N50", "$N{=}50$"),
          ("C2_linear_N100", "$N{=}100$"),
          ("C3_linear_N200", "$N{=}200$")]
ADAPTIVE = [("C4_adaptive", "no floor"),
            ("C5_adaptive_lmin0.05", "$\\lambda_{\\min}{=}0.05$"),
            ("C6_adaptive_lmin0.10", "$\\lambda_{\\min}{=}0.10$"),
            ("C7_adaptive_lmin0.20", "$\\lambda_{\\min}{=}0.20$")]

WIN_T0 = (-25, 234)
WIN_T1 = (1, 235)
DASH = (0, (5, 2))

# Momentum converges T0 several points higher before the switch, so the two
# momentum rows of a sweep figure carry the same accuracy *span* on an offset
# floor rather than one shared axis: a dip that looks deeper is deeper.
SPAN_T0 = 0.30
FLOOR_T0 = {0.0: 0.66, 0.9: 0.70}


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


def _sweep_panel(ax, conds, task_col, window, *, suffix):
    """One schedule family on one task, over the ungated reference."""
    vw.mark_switch(ax)
    vw.plot_transition(ax, *VANILLA[:2], VANILLA[2] + suffix, task_col,
                       vw.C_VANILLA, "vanilla ER", linestyle=DASH,
                       window=window, lw=1.6, alpha_band=0.08)
    # Three rungs take the light, middle and dark end of the same four-step
    # ramp, so the two families read at comparable contrast side by side.
    ramp = vw.BLUE_RAMP if len(conds) == 4 else [vw.BLUE_RAMP[i] for i in (0, 2, 3)]
    for (value, label), color in zip(conds, ramp):
        vw.plot_transition(ax, *CURR[:2], value + suffix, task_col, color,
                           label, window=window, lw=2.0, alpha_band=0.10)


def _sweep_figure(conds, name, legend_title) -> None:
    """Appendix C.2: rows are momentum, columns are stability and plasticity."""
    # Appendix C is set \onecolumn, so this is drawn at the full text width.
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True,
                             constrained_layout=True)
    rows = [(0.0, "", "(a)", "(b)"), (0.9, "_M", "(c)", "(d)")]

    for (mu, suffix, tagL, tagR), (axL, axR) in zip(rows, axes):
        _sweep_panel(axL, conds, "task_0_acc", WIN_T0, suffix=suffix)
        axL.set_ylim(FLOOR_T0[mu], FLOOR_T0[mu] + SPAN_T0)
        axL.set_ylabel("past-task ($T_0$) accuracy")
        axL.set_title(f"{tagL} Stability, momentum ${mu:g}$")

        _sweep_panel(axR, conds, "task_1_acc", WIN_T1, suffix=suffix)
        axR.set_ylim(0.15, 0.99)
        axR.set_ylabel("new-task ($T_1$) accuracy")
        axR.set_title(f"{tagR} Plasticity, momentum ${mu:g}$")

    for ax in axes[1]:
        ax.set_xlabel("steps into new task ($T_1$)")
    axes[0][0].set_xlim(-25, 235)          # shared: sharex ties all four

    # One legend for the figure: the same ramp repeats in all four panels.
    leg = axes[0][0].legend(loc="lower right", ncol=2, handlelength=1.6,
                            columnspacing=1.0, labelspacing=0.3, borderpad=0.2,
                            title=legend_title)
    leg.get_title().set_fontsize(9)
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    vw.finalize(fig, vw.FIG_DIR / "curriculum" / name)


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

    _sweep_figure(LINEAR, "curriculum_linear.png", "ramp length")
    _sweep_figure(ADAPTIVE, "curriculum_adaptive.png", "$\\lambda_{\\min}$ floor")


if __name__ == "__main__":
    main()
