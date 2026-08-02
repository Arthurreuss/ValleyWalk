"""RQ4b figure: both gates carry to CIFAR-10 / ResNet-18.

Per-step past-task (T0) accuracy across the clean -> Gaussian-noise switch
(ResNet-18, momentum 0.9, buffer 5000), one panel per normalisation, each
overlaying vanilla ER (red), the shipping curriculum (blue), and asymmetric PER
(aqua):

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
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_title(title)


def main() -> None:
    vw.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    _panel(axL, BN, "(a) BatchNorm")
    _panel(axR, GN, "(b) GroupNorm")
    axL.set_ylabel("past-task ($T_0$) accuracy")
    leg = axL.legend(loc="lower right", handlelength=1.7)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "cifar" / "cifar_headline.png")


if __name__ == "__main__":
    main()
