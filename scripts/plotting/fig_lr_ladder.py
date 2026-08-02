"""RQ1 figure: the step-size ladder, past-task accuracy against flow time.

A uniform rescaling of eta is a time reparametrisation of one and the same
gradient flow, so at matched flow time ``tau = sum of eta over steps`` the three
rungs are three discretisations of the same intended trajectory.  Plotted
against tau they therefore have to be read as a limit, not as three conditions:
what shrinks down the ladder is the discretisation, what the curves converge to
is the trajectory.

The two panels are stacked, one above the other, so the figure fits a single
text column of the thesis:

  (a) exact replay gradient (60k full buffer): the estimator noise is off, so
      whatever residual excursion survives eta -> 0 is deterministic — the arc.
  (b) sampled 1k reservoir: the operating point of every other experiment.

The ladder is exactly paired — one shared theta*_0 and a bit-identical replay
buffer per (arm, seed), only eta differs — so the rung-to-rung difference at a
given tau is attributable to the step size alone.  Momentum 0 in both panels:
momentum is not a reparametrisation of the flow, and the mu = 0.9 leg is
flow-matched only within itself.

    python -m scripts.plotting.fig_lr_ladder
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python scripts/plotting/fig_lr_ladder.py` as well as the -m form:
    # run as a script, sys.path[0] is this directory, not the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.patheffects as pe  # noqa: E402  (after the path fix)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import scripts.plotting.vw_style as vw  # noqa: E402
from scripts.analyze_ladder import (  # noqa: E402
    RUNG_ORDER, condition_band, discover_runs, load_curve, tau_grid,
)

# The whole of T1 in flow time: 235 steps x eta = 23.5 at every rung.  The
# spike lives in the first few tenths and the arc spans the rest, so the full
# range is the honest window — cropping to the spike would hide the tail that
# is the point of the comparison.
WINDOW = (-0.35, 23.5)
PANELS = [("exact", "(a) Exact replay gradient (60k buffer)"),
          ("sampled", "(b) Sampled 1k reservoir")]

# The bare rung tags, for the reader-facing legend labels.  The arm and the
# momentum leg are already named by the panel title and the caption, so the
# legend carries only what varies inside a panel: the step size.
RUNG_TAG = {"S1": "S1_eta0.1", "S2": "S2_eta0.033", "S3": "S3_eta0.01",
            "S4": "S4_eta0.001"}

# The three finest rungs converge to within 0.007 in accuracy, which is the
# finding — and which draws them as a single stroke.  Colour alone therefore
# cannot show that there are three of them, so the fine rungs are separated by
# dash pattern as well; the coarse rung, which is nowhere near the others,
# keeps the solid line.
# The finest rung (S4) is drawn thick and solid and the rungs above it ride on
# top as thinner dashed lines with shortening dashes, so the group reads as
# several curves that coincide rather than as one stroke.  (style, lw, zorder)
RUNG_STROKE = {"S1": ("solid", 2.0, 3),
               "S2": ((0, (1.4, 1.8)), 1.5, 6),
               "S3": ((0, (5.0, 2.6)), 1.5, 5),
               "S4": ("solid", 3.0, 4)}


def _rung_curves(runs):
    """Group completed momentum-0 runs by (arm, rung) -> (eta, [Curve, ...])."""
    out: dict[tuple[str, str], tuple[float, list]] = {}
    for run in runs:
        if run.momentum != 0.0 or run.status != "completed":
            continue
        curve = load_curve(run)
        if curve is None:
            continue
        eta, curves = out.setdefault((run.arm, run.rung), (run.eta, []))
        curves.append(curve)
    return out


def _panel(ax, by_rung, arm, title, legend=True):
    """Draw one buffer arm: one mean+-std band per rung, light -> dark as eta falls."""
    vw.mark_switch(ax)
    drawn = 0
    for rung in RUNG_ORDER:
        entry = by_rung.get((arm, rung))
        if entry is None:
            print(f"warning: {arm} arm has no completed seeds for rung {rung}; "
                  "omitted from the figure", file=sys.stderr)
            continue
        eta, curves = entry

        # One grid per rung, extended by a single cell at tau = -eta to carry
        # the pre-switch record.  That segment is a genuine one-step descent,
        # not the sparse->dense cadence artifact vw.anchor_boundary exists to
        # repair: these runs warm-start at the boundary and log every dense
        # step from it, so there is nothing to hold flat (anchor_boundary is a
        # no-op here — a sample already sits at tau = 0).
        grid = np.concatenate(([-eta], tau_grid(curves)))
        mean, std, n = condition_band(curves, grid)
        # condition_band interpolates Curve.acc, which holds the post-switch
        # records only, so the cell just added comes back NaN and would be
        # filtered out below — leaving every rung to start at the state *after*
        # its first new-task update, with the drop itself off the left edge.
        # The warm start is why the level cannot be recovered from the curve:
        # these runs skip task-0 training, so step 234 is the single pre-switch
        # record there is.  load_curve keeps it as Curve.pre (and rejects a run
        # that lacks it), so the band at tau = -eta is written in from there.
        pre = np.array([c.pre for c in curves], dtype=float)
        mean[0], std[0], n[0] = pre.mean(), pre.std(), pre.size
        keep = (~np.isnan(mean)) & (grid >= WINDOW[0]) & (grid <= WINDOW[1])
        x, mean, std = grid[keep], mean[keep], std[keep]
        x, mean, std = vw.anchor_boundary(x, mean, std, at=0.0)

        color = vw.ladder_color(eta)
        label = f"{vw.display_name(RUNG_TAG[rung])}  ({int(n[keep].max())} seeds)"
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14,
                        linewidth=0)
        style, lw, z = RUNG_STROKE[rung]
        # A dashed rung sits on top of a rung it coincides with, so its dashes
        # are outlined in the background colour: without the outline the two
        # same-hue curves fuse into a single stroke and the convergence, which
        # is the finding, reads as a missing line.
        effects = None if style == "solid" else [
            pe.withStroke(linewidth=lw + 2.2, foreground="white")]
        ax.plot(x, mean, color=color, lw=lw, label=label, linestyle=style,
                zorder=z, solid_capstyle="round", path_effects=effects)
        drawn += 1

    ax.set_xlim(*WINDOW)
    ax.set_title(title)
    if drawn and legend:
        leg = ax.legend(loc="lower right", handlelength=1.6, title="step size",
                        labelspacing=0.3, borderpad=0.2)
        leg.get_title().set_fontsize(9)
        for line in leg.get_lines():
            line.set_linewidth(2.4)
    if not drawn:
        ax.text(0.5, 0.5, "no completed runs yet", transform=ax.transAxes,
                ha="center", va="center", color=vw.MUTED, fontsize=10)
    return drawn


def main() -> None:
    vw.apply_style()
    runs = discover_runs(vw.OUTPUTS)
    by_rung = _rung_curves(runs)
    if not by_rung:
        print("warning: no completed lr_ladder runs found; drawing an empty figure",
              file=sys.stderr)

    # Stacked, not side by side: the figure sits in one text column, so the
    # panels share the x-axis vertically and the width is spent on flow time.
    # The rungs are the same in both panels, so one legend (in (a)) serves both.
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, (ax, (arm, title)) in enumerate(zip(axes, PANELS)):
        _panel(ax, by_rung, arm, title, legend=(i == 0))
    axes[-1].set_xlabel("flow time $\\tau = \\eta \\cdot$ steps into $T_1$")
    fig.supylabel("past-task ($T_0$) accuracy", fontsize=11)
    axes[0].set_ylim(0.55, 0.93)
    axes[0].annotate("task switch", xy=(0, 0.565), xytext=(0.6, 0.565),
                     color=vw.MUTED, fontsize=8)
    fig.suptitle("Driving the step to zero: what vanishes is\n"
                 "discretisation, what remains is the arc",
                 fontsize=11, fontweight="bold")
    vw.finalize(fig, vw.FIG_DIR / "ladder" / "step_size_ladder.png")


if __name__ == "__main__":
    main()
