"""RQ4b figure: both gates carry to CIFAR-10 / ResNet-18.

Per-step past-task (T0) accuracy across the clean -> Gaussian-noise switch
(ResNet-18, momentum 0.9, buffer 5000), one panel per normalisation, each
overlaying vanilla ER (red), the shipping curriculum (blue), and asymmetric PER
(aqua):

The panels are stacked, one above the other, so the figure fits a single text
column and the two normalisations are read against a shared step axis:

  (a) BatchNorm: vanilla dips and re-adapts slowly; both gates hold T0 near its
      plateau through the switch.
  (b) GroupNorm: vanilla ER crashes T0 almost to chance and crawls back over the
      whole recovery window; both gates hold it flat, cutting the deep transient
      and lifting worst-case retention from ~0.31 to ~0.67.

``cifar_plasticity.png`` (Appendix C.5) is the same figure on the new task: same
conditions, same axis, T1 instead of T0.  It is what prices the protection above.
Under batch norm the gates track vanilla ER with a small early lag that closes;
under group norm they are ahead of it throughout, because the baseline's crash
costs it the shared features the new task starts from.  The window stops at 400
steps like the headline figure (T1 is not evaluated before the switch, so the
empty pre-switch margin is dropped), and the caption carries the end-of-training
values.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

WINDOW = (-120, 400)
BN = [("er", "cifar_headline", "G1_vanilla_BN_buf5k", vw.C_VANILLA, "vanilla ER"),
      ("er", "cifar_headline", "G3_adaptive_lmin0.20_BN_buf5k", vw.C_CURR, "curriculum"),
      ("precond_er", "cifar_per_asym", "G7_PER_asym_d0.1_BN_buf5k", vw.C_PER, "asym PER")]
GN = [("er", "cifar_headline", "G4_vanilla_GN_buf5k", vw.C_VANILLA, "vanilla ER"),
      ("er", "cifar_headline", "G6_adaptive_lmin0.20_GN_buf5k", vw.C_CURR, "curriculum"),
      ("precond_er", "cifar_per_asym", "G8_PER_asym_d0.1_GN_buf5k", vw.C_PER, "asym PER")]


def _panel(ax, series, title, *, task_col="task_0_acc", window=WINDOW,
           xlim=WINDOW, ylim=(0.24, 0.85),
           ylabel="past-task ($T_0$) accuracy"):
    vw.mark_switch(ax)
    for m, k, v, color, name in series:
        vw.plot_transition(ax, m, k, v, task_col, color, name, window=window)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plasticity_figure() -> None:
    """Appendix C.5: the same panels on T1 -- what the protection costs."""
    # T1 has no sample before the switch, so the curve starts at the marker;
    # the x limits still come from WINDOW so this figure and the headline can
    # be read against each other step for step.
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(4.2, 5.2), sharex=True,
                                   sharey=True, constrained_layout=True)
    # The pre-switch margin the headline figure carries is empty here -- T1 is
    # not evaluated before the switch -- so the axis starts just left of the
    # marker and keeps the same post-switch extent.
    kw = dict(task_col="task_1_acc", window=(1, WINDOW[1]),
              xlim=(-25, WINDOW[1]), ylim=(0.17, 0.71),
              ylabel="new-task ($T_1$) accuracy")
    _panel(axT, BN, "(a) BatchNorm", **kw)
    _panel(axB, GN, "(b) GroupNorm", **kw)
    axB.set_xlabel("steps into new task ($T_1$)")
    leg = axT.legend(loc="lower right", handlelength=1.7, labelspacing=0.3,
                     borderpad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "cifar" / "cifar_plasticity.png")


def main() -> None:
    vw.apply_style()
    # Stacked for a single text column; shared x and y, so the two
    # normalisations are directly comparable and only the bottom panel needs
    # the step axis.  Both panels carry the same three methods, so the legend
    # is drawn once, in (a).
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(4.2, 5.2), sharex=True,
                                   sharey=True, constrained_layout=True)
    _panel(axT, BN, "(a) BatchNorm")
    _panel(axB, GN, "(b) GroupNorm")
    axB.set_xlabel("steps into new task ($T_1$)")
    leg = axT.legend(loc="lower right", handlelength=1.7, labelspacing=0.3,
                     borderpad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "cifar" / "cifar_headline.png")

    _plasticity_figure()


if __name__ == "__main__":
    main()
