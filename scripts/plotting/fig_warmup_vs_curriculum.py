"""RQ2a falsification figure: curriculum vs symmetric LR warm-up at matched N.

Gap depth against ramp length N in {50, 100, 200} for the linear lambda-
curriculum (C1-C3) and the symmetric learning-rate warm-up (M1-M3), at
momentum 0 (left) and momentum 0.9 (right).  Thin lines join per-seed pairs;
thick lines join condition means.

At momentum 0 the two ramps are statistically tied: with no accumulator,
"small steps early" is the whole effect and it does not matter whether the
attenuation is one-sided.  Under momentum the curriculum wins at every N,
by the most at the shortest ramp: the warm-up scales only the applied step,
so the velocity buffer still accumulates full-magnitude boundary gradients
during the ramp, while the curriculum shrinks the push at source before it
enters the accumulator.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt
import numpy as np

NS = [50, 100, 200]
PAIRS = {
    0.0: [("C%d_linear_N%d" % (i + 1, n), "M%d_lrwarmup_N%d" % (i + 1, n))
          for i, n in enumerate(NS)],
    0.9: [("C%d_linear_N%d_M" % (i + 1, n), "M%d_lrwarmup_N%d_M" % (i + 1, n))
          for i, n in enumerate(NS)],
}


def depth_by_seed(value: str) -> np.ndarray:
    df = vw.master()
    sel = df[(df["ablation_value"] == value) & (df["status"] == "completed")]
    return 100.0 * sel.sort_values("seed")["stab_gap_depth"].to_numpy()


def main() -> None:
    vw.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (mom, pairs) in zip(axes, PAIRS.items()):
        c_means, m_means = [], []
        for x, (c_val, m_val) in zip(NS, pairs):
            c, m = depth_by_seed(c_val), depth_by_seed(m_val)
            jitter = np.linspace(-4, 4, len(c))
            ax.plot(x + jitter, c, "o", color=vw.C_CURR, ms=3.5, alpha=0.45, zorder=2)
            ax.plot(x + jitter, m, "o", color=vw.C_WARM, ms=3.5, alpha=0.45, zorder=2)
            c_means.append(c.mean())
            m_means.append(m.mean())
        ax.plot(NS, c_means, "-o", color=vw.C_CURR, lw=2.2, ms=5,
                label="curriculum (one-sided)", zorder=3)
        ax.plot(NS, m_means, "-o", color=vw.C_WARM, lw=2.2, ms=5,
                label="LR warm-up (symmetric)", zorder=3)
        ax.set_xticks(NS)
        ax.set_xlabel("ramp length $N$ (steps)")
        ax.set_title(f"momentum ${mom}$")
    axes[0].set_ylabel("gap depth (pp)")
    axes[0].legend(loc="upper right")
    fig.suptitle("One-sided vs symmetric ramp: the gap opens under momentum",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    vw.finalize(fig, vw.FIG_DIR / "M" / "warmup_vs_curriculum.png")


if __name__ == "__main__":
    main()
