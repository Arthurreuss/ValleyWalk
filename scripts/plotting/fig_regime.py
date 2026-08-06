"""Momentum-synthesis figure: the gates' ordering swaps with momentum.

Two panels of per-step past-task (T0) accuracy across the rot-MNIST switch,
all three methods at the matched 1k buffer -- vanilla ER (red), the shipping
curriculum C7 (blue), asymmetric PER at its strongest standard-buffer filter
delta=0.03 (aqua):

  (a) momentum 0:   the directional gate holds T0 highest; the curriculum
      opens a deep noisy dip (its plasticity-debt regime).
  (b) momentum 0.9: the ordering swaps -- the curriculum holds T0 near its
      plateau while the directional gate's per-step filter is re-inflated
      by accumulated velocity.

Colour follows the method across both panels; the panels differ only in the
momentum setting, so the swap is read directly.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

# (method, ablation_key, momentum-0 value, momentum-0.9 value, color, label)
SERIES = [
    ("er", "decomposition", "D1_vanilla", "D1_vanilla_M",
     vw.C_VANILLA, "vanilla ER"),
    ("er", "curriculum", "C7_adaptive_lmin0.20", "C7_adaptive_lmin0.20_M",
     vw.C_CURR, "curriculum"),
    ("precond_er", "per_asym_sweep", "P4_d0.03", "P4_d0.03_M",
     vw.C_PER, "asym. PER $\\delta{=}0.03$"),
]
WINDOW = (-25, 234)


def _panel(ax, leg_idx: int, title: str) -> None:
    vw.mark_switch(ax)
    for method, key, v0, v9, color, label in SERIES:
        value = (v0, v9)[leg_idx]
        vw.plot_transition(ax, method, key, value, "task_0_acc", color, label,
                           window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(0.63, 0.97)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_title(title)


def main() -> None:
    vw.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    _panel(axL, 0, "(a) momentum $0$: directional gate ahead")
    _panel(axR, 1, "(b) momentum $0.9$: scalar gate ahead")
    axL.set_ylabel("past-task ($T_0$) accuracy")
    leg = axL.legend(loc="lower right", handlelength=1.8, borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "momentum" / "regime_flip.png")


if __name__ == "__main__":
    main()
