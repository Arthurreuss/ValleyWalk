"""RQ3 figures: the directional feedforward gate (asymmetric PER).

Three figures come out of this module, all across the rot-MNIST T0->T1 switch:

``per_headline.png`` (main text, one text column, panels stacked) is the
counterpart of the curriculum headline and reads the same way: one operating
point of the gate, the strongest filter (delta = 0.03, aqua), against vanilla
ER (red), each at momentum 0 (dashed) and 0.9 (solid).

  (a) past task T0 (stability): at momentum 0 the filter holds T0 far above
      vanilla; at momentum 0.9 the same filter is re-inflated, because the
      accumulator rebuilds the displacement a per-step brake removes.
  (b) new task T1 (plasticity): the filter lags over roughly the first fifty
      steps and is indistinguishable from vanilla thereafter -- the selective
      slowdown along past-task-sharp directions, gone by the WP100 horizon.
      This is the scalar gate's plasticity debt inverted.

``per_sweep.png`` (Appendix C.4) is the same standard-1k-buffer sweep the
headline draws one rung of, over all four filters and both tasks: rows are the
momentum setting, columns are stability (T0) and plasticity (T1), with vanilla
ER carried into every panel as the ungated reference.  It is where the
monotonicity in delta, and its survival under momentum at a displaced level,
are read off the trajectories rather than the table.

``per_cliffs.png`` (Appendix C.4) is the exact-gradient diagnostic: the full
60k buffer at momentum 0, where the variance driver is off and the residual
excursion is the arc plus the step's overshoot of it.  The excursion shrinks
with the filter down to delta = 0.1 and then breaks monotonicity at delta =
0.03, where 3 of 5 seeds hold flat for ~200 steps and slide off a late cliff
(shown per seed, critical red).  This is a diagnostic probe, not an operating
point: no seed on the practical 1k buffer shows one.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

KEY = "per_asym_sweep"
VANILLA = ("er", "decomposition", "D1_vanilla")
HEADLINE = ("precond_er", KEY, "P4_d0.03")     # the strongest filter
STD = [("P1_d1.0", 1.0), ("P2_d0.3", 0.3), ("P3_d0.1", 0.1), ("P4_d0.03", 0.03)]
# No delta = 1.0 cell was run at momentum 0.9; the momentum leg covers the
# three strongest filters, which are the ones the narrative turns on.
STD_M = [("P2_d0.3_M", 0.3), ("P3_d0.1_M", 0.1), ("P4_d0.03_M", 0.03)]
FULL = [("P1_d1.0_fullbuf", 1.0), ("P2_d0.3_fullbuf", 0.3), ("P3_d0.1_fullbuf", 0.1)]
CLIFF = "P4_d0.03_fullbuf"
CRITICAL = "#d03b3b"

WIN_T0 = (-25, 234)
WIN_T1 = (1, 235)
DASH = (0, (5, 2))

# Momentum converges T0 several points higher before the switch, so the two
# momentum rows of the sweep figure carry the same accuracy *span* on an
# offset floor rather than one shared axis: a dip that looks deeper is deeper.
SPAN_T0 = 0.30
FLOOR_T0 = {0.0: 0.66, 0.9: 0.70}


# ---------------------------------------------------------------------------
# Main text: one filter against the baseline, at both momenta
# ---------------------------------------------------------------------------

def _headline_panel(ax, task_col, window):
    """Vanilla ER and the delta = 0.03 filter, momentum 0.9 solid / 0 dashed."""
    series = [(VANILLA, vw.C_VANILLA), (HEADLINE, vw.C_PER)]
    for (m, k, v), color in series:                      # momentum 0.9 -- solid
        vw.plot_transition(ax, m, k, v + "_M", task_col, color, None,
                           linestyle="-", window=window, alpha_band=0.16)
    for (m, k, v), color in series:                      # momentum 0 -- dashed
        vw.plot_transition(ax, m, k, v, task_col, color, None,
                           linestyle=DASH, window=window, lw=1.6, alpha_band=0.10)


def _headline_figure() -> None:
    """Stacked stability/plasticity pair for a single text column."""
    # Same geometry and reading order as the curriculum headline, so the two
    # gates can be compared panel for panel: (a) sits directly above (b) on a
    # shared step axis, and colour = method, linestyle = momentum throughout.
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(4.2, 5.2), sharex=True,
                                   constrained_layout=True)

    # -- (a) stability: T0 across the switch --------------------------------
    vw.mark_switch(axT)
    _headline_panel(axT, "task_0_acc", WIN_T0)
    axT.set_ylim(0.66, 0.97)
    axT.set_ylabel("past-task ($T_0$) accuracy")
    axT.set_title("(a) Stability")

    # -- (b) plasticity: T1 learning curve ----------------------------------
    vw.mark_switch(axB)
    _headline_panel(axB, "task_1_acc", WIN_T1)
    axB.set_ylim(0.15, 0.99)
    axB.set_xlabel("steps into new task ($T_1$)")
    axB.set_ylabel("new-task ($T_1$) accuracy")
    axB.set_title("(b) Plasticity")
    axB.set_xlim(-25, 235)

    # -- shared legend: colour = method, style = momentum -------------------
    axT.plot([], [], color=vw.C_VANILLA, lw=2.2, label="vanilla ER")
    axT.plot([], [], color=vw.C_PER, lw=2.2, label="asym. PER $\\delta{=}0.03$")
    axT.plot([], [], color=vw.INK_SOFT, lw=2.0, linestyle="-", label="$\\mu{=}0.9$")
    axT.plot([], [], color=vw.INK_SOFT, lw=1.6, linestyle=DASH, label="$\\mu{=}0.0$")
    leg = axT.legend(loc="lower right", ncol=2, handlelength=1.6,
                     columnspacing=1.0, labelspacing=0.3, borderpad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2.2)

    vw.finalize(fig, vw.FIG_DIR / "asymmetric_per" / "per_headline.png")


# ---------------------------------------------------------------------------
# Appendix: the full sweep, both tasks, both momenta
# ---------------------------------------------------------------------------

def _sweep_panel(ax, series, task_col, window, *, vanilla_value):
    """One delta sweep on one task, over the ungated reference."""
    vw.mark_switch(ax)
    vw.plot_transition(ax, *VANILLA[:2], vanilla_value, task_col, vw.C_VANILLA,
                       "vanilla ER", linestyle=DASH, window=window, lw=1.6,
                       alpha_band=0.08)
    for value, delta in series:
        vw.plot_transition(ax, "precond_er", KEY, value, task_col,
                           vw.DELTA_RAMP[delta], f"$\\delta{{=}}{delta}$",
                           window=window, lw=2.0, alpha_band=0.10)


def _sweep_figure() -> None:
    """Appendix C.4: rows are momentum, columns are stability and plasticity."""
    # Appendix C is set \onecolumn, so this is drawn at the full text width.
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True,
                             constrained_layout=True)
    rows = [(0.0, STD, "", "(a)", "(b)"), (0.9, STD_M, "_M", "(c)", "(d)")]

    for (mu, series, suffix, tagL, tagR), (axL, axR) in zip(rows, axes):
        _sweep_panel(axL, series, "task_0_acc", WIN_T0,
                     vanilla_value="D1_vanilla" + suffix)
        axL.set_ylim(FLOOR_T0[mu], FLOOR_T0[mu] + SPAN_T0)
        axL.set_ylabel("past-task ($T_0$) accuracy")
        axL.set_title(f"{tagL} Stability, momentum ${mu:g}$")

        _sweep_panel(axR, series, "task_1_acc", WIN_T1,
                     vanilla_value="D1_vanilla" + suffix)
        axR.set_ylim(0.15, 0.99)
        axR.set_ylabel("new-task ($T_1$) accuracy")
        axR.set_title(f"{tagR} Plasticity, momentum ${mu:g}$")

    for ax in axes[1]:
        ax.set_xlabel("steps into new task ($T_1$)")
    axes[0][0].set_xlim(-25, 235)          # shared: sharex ties all four

    # One legend for the figure: the ramp repeats in all four panels, and the
    # momentum-0 row is the only one carrying the delta = 1.0 rung.
    leg = axes[0][0].legend(loc="lower right", ncol=2, handlelength=1.6,
                            columnspacing=1.0, labelspacing=0.3, borderpad=0.2,
                            title="filter strength")
    leg.get_title().set_fontsize(9)
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    vw.finalize(fig, vw.FIG_DIR / "asymmetric_per" / "per_sweep.png")


# ---------------------------------------------------------------------------
# Appendix: the exact-gradient diagnostic
# ---------------------------------------------------------------------------

def _seed_curves(ax, value, color, label):
    """Overlay each seed's raw T0 curve (thin), for the cliff condition."""
    sw = vw.switch_step("precond_er", KEY, value)
    first = True
    for d in vw.run_dirs("precond_er", KEY, value):
        f = d / "results" / "accuracy_curves.csv"
        if not f.exists():
            continue
        c = vw.read_curve_csv(f)[["step", "task_0_acc"]].dropna()
        # +1 matches vw.plot_transition: the first evaluated post-switch step
        # already contains one new-task update, so it sits at x = 1 and x = 0
        # is the pre-update state marked by vw.mark_switch.
        x = c["step"].to_numpy(float) - sw + 1
        y = c["task_0_acc"].to_numpy(float)
        # Same hold-last anchor as vw.plot_transition, applied before the mask.
        x, y = vw.anchor_boundary(x, y, at=0.0)
        m = (x >= WIN_T0[0]) & (x <= WIN_T0[1])
        ax.plot(x[m], y[m], color=color, lw=1.1,
                alpha=0.85, label=label if first else None, zorder=3)
        first = False


def _cliffs_figure() -> None:
    """Appendix C.4: the exact-gradient diagnostic and its late cliffs."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    vw.mark_switch(ax)
    for value, delta in FULL:
        vw.plot_transition(ax, "precond_er", KEY, value, "task_0_acc",
                           vw.DELTA_RAMP[delta], f"$\\delta{{=}}{delta}$",
                           window=WIN_T0, lw=2.0, alpha_band=0.10)
    ax.set_xlim(*WIN_T0)
    ax.set_ylabel("past-task ($T_0$) accuracy")
    _seed_curves(ax, CLIFF, CRITICAL, "$\\delta{=}0.03$ (per seed)")
    ax.set_ylim(0.66, 0.93)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_title("Full $60$k buffer (exact replay gradient)")
    # Lower *left*: the cliffs themselves fall through the lower right.
    leg = ax.legend(loc="lower left", handlelength=1.6, labelspacing=0.3,
                    borderpad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    vw.finalize(fig, vw.FIG_DIR / "asymmetric_per" / "per_cliffs.png")


def main() -> None:
    vw.apply_style()
    _headline_figure()
    _sweep_figure()
    _cliffs_figure()


if __name__ == "__main__":
    main()
