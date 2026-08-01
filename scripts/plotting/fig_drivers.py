"""RQ1 figure: the step is the acute lever, the field remix is not.

Per-step past-task (T0) accuracy across the T0->T1 switch on rot-MNIST at
momentum 0, for the intervention conditions of the revised driver block:

    D1  vanilla ER              -- reference (all levers untouched)
    D2  full-data ER            -- step: estimator noise off
    M3  LR warm-up N=200        -- step: first steps shortened
    D3  balanced-direction ER   -- field: imbalance out of the direction,
                                   step length untouched

The warm-up (a pure step-size intervention) shrinks the boundary drop; the
balanced-direction remix (a pure field intervention at matched step length)
*deepens* it, because at the boundary the replay gradient is near zero and
its normalised direction carries no information.  The smooth dip-and-recover
bend that survives the exact-gradient condition (D2) is the deterministic
arc.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt

# (ablation_key, ablation_value, color, label)
CONDS = [
    ("decomposition", "D1_vanilla",         vw.C_VANILLA, "D1  vanilla ER"),
    ("decomposition", "D2_fulldata",        vw.C_FULL,    "D2  full-data ER (step)"),
    ("lr_warmup",     "M3_lrwarmup_N200",   vw.C_WARM,    "M3  LR warm-up $N{=}200$ (step)"),
    ("decomposition", "D3_balanced_dir",    vw.C_EXTRA,   "D3  balanced-direction (field)"),
]
WINDOW = (-25, 234)


def main() -> None:
    vw.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vw.mark_switch(ax)
    for key, value, color, label in CONDS:
        vw.plot_transition(ax, "er", key, value, "task_0_acc", color, label,
                           window=WINDOW)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(0.42, 0.95)
    ax.set_xlabel("steps into new task ($T_1$)")
    ax.set_ylabel("past-task ($T_0$) accuracy")
    ax.set_title("Step interventions cut the drop; the field remix deepens it (momentum $0$)")
    ax.annotate("task switch", xy=(0, 0.435), xytext=(10, 0.445),
                color=vw.MUTED, fontsize=9)
    leg = ax.legend(loc="lower right", ncol=2, handlelength=1.6,
                    columnspacing=1.4, borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.4)
    vw.finalize(fig, vw.FIG_DIR / "D" / "drivers_t0.png")


if __name__ == "__main__":
    main()
