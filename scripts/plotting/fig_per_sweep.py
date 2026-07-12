"""RQ3 figure: the directional feedforward gate (asymmetric PER), delta sweep.

Two panels of past-task (T0) accuracy across the rot-MNIST switch at momentum 0,
sweeping the damping delta in {1.0, 0.3, 0.1, 0.03} (weaker filter light ->
stronger filter dark, a single-hue ordinal ramp):

  (a) standard 1k buffer: the boundary dip shrinks monotonically as delta -> 0,
      and accuracy rises with it -- protection at no plasticity cost.
  (b) full 60k buffer (variance driver off, so the residual excursion IS the
      deterministic arc): the arc shrinks with the filter for delta >= 0.1, but
      at the strongest setting delta=0.03 the feedforward gate hits its
      curvature blind spot -- 3 of 5 seeds hold flat for ~200 steps and then
      slide off a late cliff (shown per seed, critical red).
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

KEY = "per_asym_sweep"
STD = [("P1_d1.0", 1.0), ("P2_d0.3", 0.3), ("P3_d0.1", 0.1), ("P4_d0.03", 0.03)]
FULL = [("P1_d1.0_fullbuf", 1.0), ("P2_d0.3_fullbuf", 0.3), ("P3_d0.1_fullbuf", 0.1)]
CLIFF = "P4_d0.03_fullbuf"
CRITICAL = "#d03b3b"
WINDOW = (-25, 234)


def _seed_curves(ax, value, color, label):
    """Overlay each seed's raw T0 curve (thin), for the cliff condition."""
    sw = vw.switch_step("precond_er", KEY, value)
    first = True
    for d in vw.run_dirs("precond_er", KEY, value):
        f = d / "results" / "accuracy_curves.csv"
        if not f.exists():
            continue
        c = vw.read_curve_csv(f)[["step", "task_0_acc"]].dropna()
        x = c["step"].to_numpy(float) - sw
        m = (x >= WINDOW[0]) & (x <= WINDOW[1])
        ax.plot(x[m], c["task_0_acc"].to_numpy(float)[m], color=color, lw=1.1,
                alpha=0.85, label=label if first else None, zorder=3)
        first = False


def main() -> None:
    vw.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)

    # -- (a) standard buffer: monotone shrink -------------------------------
    vw.mark_switch(axL)
    for value, delta in STD:
        vw.plot_transition(axL, "precond_er", KEY, value, "task_0_acc",
                           vw.DELTA_RAMP[delta], f"$\\delta{{=}}{delta:g}$",
                           window=WINDOW, lw=2.0, alpha_band=0.10)
    axL.set_xlim(*WINDOW)
    axL.set_ylim(0.66, 0.93)
    axL.set_xlabel("steps into new task ($T_1$)")
    axL.set_ylabel("past-task ($T_0$) accuracy")
    axL.set_title("(a) Standard 1k buffer")
    leg = axL.legend(loc="lower right", handlelength=1.6, title="filter strength")
    leg.get_title().set_fontsize(9)
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    # -- (b) full buffer: arc shrinks, then the cliff -----------------------
    vw.mark_switch(axR)
    for value, delta in FULL:
        vw.plot_transition(axR, "precond_er", KEY, value, "task_0_acc",
                           vw.DELTA_RAMP[delta], f"$\\delta{{=}}{delta:g}$",
                           window=WINDOW, lw=2.0, alpha_band=0.10)
    _seed_curves(axR, CLIFF, CRITICAL, "$\\delta{=}0.03$ (per seed)")
    axR.set_xlim(*WINDOW)
    axR.set_xlabel("steps into new task ($T_1$)")
    axR.set_title("(b) Full 60k buffer")
    leg = axR.legend(loc="lower right", handlelength=1.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    vw.finalize(fig, vw.FIG_DIR / "P" / "per_sweep.png")


if __name__ == "__main__":
    main()
