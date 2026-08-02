"""RQ3 figures: the directional feedforward gate (asymmetric PER), delta sweep.

Two figures come out of this module, both on past-task (T0) accuracy across the
rot-MNIST switch and both sweeping the damping delta (weaker filter light ->
stronger filter dark, a single-hue ordinal ramp):

``per_sweep.png`` (main text, one text column, panels stacked) stays on the
standard 1k buffer, the operating point of every other experiment, and turns
the momentum leg into the second panel:

  (a) momentum 0: the boundary dip shrinks monotonically as delta -> 0, and
      accuracy rises with it -- protection at no plasticity cost.
  (b) momentum 0.9: the same three filters, re-inflated.  The sweep still
      orders monotonically, but every rung sits deeper than its momentum-0
      counterpart: the accumulator rebuilds the very directions the per-step
      filter brakes.  The two panels carry the same accuracy span (0.27), so a
      dip that looks deeper is deeper; the pre-switch level is higher in (b)
      only because momentum converges T0 further before the switch.

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
STD = [("P1_d1.0", 1.0), ("P2_d0.3", 0.3), ("P3_d0.1", 0.1), ("P4_d0.03", 0.03)]
# No delta = 1.0 cell was run at momentum 0.9; the momentum leg covers the
# three strongest filters, which are the ones the narrative turns on.
STD_M = [("P2_d0.3_M", 0.3), ("P3_d0.1_M", 0.1), ("P4_d0.03_M", 0.03)]
FULL = [("P1_d1.0_fullbuf", 1.0), ("P2_d0.3_fullbuf", 0.3), ("P3_d0.1_fullbuf", 0.1)]
CLIFF = "P4_d0.03_fullbuf"
CRITICAL = "#d03b3b"
WINDOW = (-25, 234)

# One accuracy span for both panels of the main figure, so the two momentum
# legs are read against the same scale.  Momentum 0.9 converges T0 about six
# points higher before the switch, hence the offset floor rather than a
# shared axis.
SPAN = 0.27
YLIM = {0.0: (0.66, 0.66 + SPAN), 0.9: (0.72, 0.72 + SPAN)}


def _sweep(ax, series, *, legend_title=None):
    """Draw one delta sweep (mean +- std band per rung) on one axis."""
    vw.mark_switch(ax)
    for value, delta in series:
        vw.plot_transition(ax, "precond_er", KEY, value, "task_0_acc",
                           vw.DELTA_RAMP[delta], f"$\\delta{{=}}{delta:g}$",
                           window=WINDOW, lw=2.0, alpha_band=0.10)
    ax.set_xlim(*WINDOW)
    ax.set_ylabel("past-task ($T_0$) accuracy")
    if legend_title is not None:
        leg = ax.legend(loc="lower right", handlelength=1.6, title=legend_title,
                        labelspacing=0.3, borderpad=0.2)
        leg.get_title().set_fontsize(9)
        for line in leg.get_lines():
            line.set_linewidth(2.4)


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
        m = (x >= WINDOW[0]) & (x <= WINDOW[1])
        ax.plot(x[m], y[m], color=color, lw=1.1,
                alpha=0.85, label=label if first else None, zorder=3)
        first = False


def _main_figure() -> None:
    """Standard 1k buffer at both momenta, stacked for a single text column."""
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(4.2, 5.4), sharex=True,
                                   constrained_layout=True)

    # -- (a) momentum 0: monotone shrink ------------------------------------
    # The legend lives here alone: (b) reuses the same ramp, and repeating it
    # would cost a quarter of the lower panel.
    _sweep(axT, STD, legend_title="filter strength")
    axT.set_ylim(*YLIM[0.0])
    axT.set_title("(a) Momentum $0$")

    # -- (b) momentum 0.9: the same filters, re-inflated ---------------------
    _sweep(axB, STD_M)
    axB.set_ylim(*YLIM[0.9])
    axB.set_xlabel("steps into new task ($T_1$)")
    axB.set_title("(b) Momentum $0.9$")

    vw.finalize(fig, vw.FIG_DIR / "asymmetric_per" / "per_sweep.png")


def _cliffs_figure() -> None:
    """Appendix C.4: the exact-gradient diagnostic and its late cliffs."""
    # Appendix C is set \onecolumn, so this one is drawn at the full text
    # width like its neighbour fig_per_plasticity -- not at the column width
    # the main-text figure uses.
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    _sweep(ax, FULL)
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
    _main_figure()
    _cliffs_figure()


if __name__ == "__main__":
    main()
