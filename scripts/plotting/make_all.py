"""Regenerate every results/appendix figure from outputs/.

    python -m scripts.plotting.make_all

Each figure module is self-contained and reads the aggregated run data via
scripts.plotting.vw_style (master_index.csv + per-run accuracy_curves.csv).
Run scripts/aggregate_results.py first if outputs/ has changed.
"""

from __future__ import annotations

from scripts.plotting import (
    fig_lr_ladder,
    fig_curriculum, fig_per_sweep, fig_per_plasticity, fig_regime,
    fig_cifar, fig_longseq, fig_cifar_failure,
)

FIGURES = [
    ("RQ1  step-size ladder",      fig_lr_ladder),
    ("RQ2  curriculum",            fig_curriculum),
    ("RQ3  asym-PER delta sweep",  fig_per_sweep),
    ("RQ3  asym-PER plasticity",   fig_per_plasticity),
    ("syn  momentum regime",       fig_regime),
    ("RQ4  CIFAR headline",        fig_cifar),
    ("app  long sequence",         fig_longseq),
    ("app  CIFAR 3-task outlier",  fig_cifar_failure),
]


def main() -> None:
    for name, mod in FIGURES:
        print(f"[{name}]")
        mod.main()
    print("all figures regenerated.")


if __name__ == "__main__":
    main()
