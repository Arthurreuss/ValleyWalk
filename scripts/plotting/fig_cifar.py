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


def _panel(ax, series, title):
    vw.mark_switch(ax)
    for m, k, v, color, name in series:
        vw.plot_transition(ax, m, k, v, "task_0_acc", color, name, window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(0.24, 0.85)
    ax.set_ylabel("past-task ($T_0$) accuracy")
    ax.set_title(title)


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


if __name__ == "__main__":
    main()
