"""Shared style and data-loading helpers for the thesis results figures.

All figures in ``scripts/plotting/`` import from here so they share one visual
system (colours, fonts, spines, transition markers) and one data path
(``outputs/master_index.csv`` + per-run ``results/accuracy_curves.csv``).

Colour assignment follows the entity, never its rank, and is stable across
every figure:

    vanilla ER (no gate)            -> red      (the gap-heavy baseline)
    curriculum / scalar gate        -> blue     (feedback gate)
    asymmetric PER / directional    -> aqua      (feedforward gate)

The damping (delta) sweep is an *ordinal* magnitude, so it uses a single-hue
blue ramp (weaker filter light -> stronger filter dark), not the categorical
slots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"
MASTER_INDEX = OUTPUTS / "master_index.csv"
FIG_DIR = PROJECT_ROOT / "thesis_latex" / "img" / "results"

# ---------------------------------------------------------------------------
# Colours (validated categorical palette, light surface)
# ---------------------------------------------------------------------------

C_VANILLA = "#e34948"   # red    -- vanilla ER, the no-gate baseline
C_CURR    = "#2a78d6"   # blue   -- curriculum / scalar feedback gate
C_PER     = "#1baf7a"   # aqua   -- asymmetric PER / directional feedforward gate
C_FULL    = "#4a3aa7"   # violet -- full-data / exact-gradient variant (driver block)
C_EXTRA   = "#eb6834"   # orange -- fourth driver-block slot
C_WARM    = "#b8477d"   # magenta -- learning-rate warm-up (symmetric step control)

INK        = "#0b0b0b"
INK_SOFT   = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
AXIS       = "#c3c2b7"

# Ordinal blue ramp for the delta sweep (light = weak filter, dark = strong).
DELTA_RAMP = {
    1.0:  "#86b6ef",
    0.3:  "#5598e7",
    0.1:  "#2a78d6",
    0.03: "#184f95",
}

# Ordinal ramp for the S-series step-size ladder.  Every rung *is* vanilla ER —
# only eta differs — so the ramp stays on the vanilla hue and darkens as the
# step shrinks: eta = 0.1 is the standard operating point and keeps exactly
# C_VANILLA, the smaller steps sit below it on the same hue.
LADDER_RAMP = {
    0.1:     C_VANILLA,
    0.03333: "#a5302f",
    0.01:    "#66191a",
}


def ladder_color(eta: float) -> str:
    """Colour for a ladder rung, matched to the nearest tabulated eta.

    The exact eta of the middle rung is 0.1/3 = 0.03333..., not the 0.033 of
    its label, so a dict lookup on the raw value would miss.
    """
    key = min(LADDER_RAMP, key=lambda e: abs(np.log(e) - np.log(float(eta))))
    return LADDER_RAMP[key]

# ---------------------------------------------------------------------------
# Reader-facing condition names
# ---------------------------------------------------------------------------
#
# ``ablation_value`` is an internal key: it encodes the block a condition was
# launched in (D1, C7, P3, S2) rather than what the reader is being shown.
# Figures and tables should carry the second, so the same condition reads the
# same way everywhere; this dict is the single place that mapping lives.
#
# Keys are looked up whole first, so a variant with its own established name
# (C8, the curriculum + full-buffer cell) wins over the suffix rules below.

DISPLAY_NAMES: dict[str, str] = {
    # Driver block (D-series) — plain ER is the reference every figure returns to.
    "D1_vanilla": "Experience replay",
    # Step-size ladder (S-series): the rung *is* the step size.
    # The label rounds; the true value is 0.1/c (see run_S.sh) and is always
    # read from the run config, never parsed back out of the tag.
    "S1_eta0.1":   "η = 0.1",
    "S2_eta0.033": "η = 0.033",
    "S3_eta0.01":  "η = 0.01",
    # Curriculum (C-series) — only the two conditions the narrative keeps.
    "C7_adaptive_lmin0.20": "loss curriculum",
    "C8_adaptive_lmin0.20_fullbuf": "curriculum + exact replay",
    # Asymmetric PER (P-series): an ordinal damping sweep.
    "P1_d1.0":  "curvature filter (δ = 1.0)",
    "P2_d0.3":  "curvature filter (δ = 0.3)",
    "P3_d0.1":  "curvature filter (δ = 0.1)",
    "P4_d0.03": "curvature filter (δ = 0.03)",
}

# Suffixes that modify a base condition rather than naming a new one.  Both
# ``_exact`` (S-series) and ``_fullbuf`` (C/P-series) mean the same thing —
# replay over the whole past-task set, i.e. the exact past-task gradient — so
# they get the same qualifier; the two spellings are historical.
_NAME_SUFFIXES: list[tuple[str, str]] = [
    ("_exact", ", exact replay gradient"),
    ("_fullbuf", ", exact replay gradient"),
    ("_M", " (momentum 0.9)"),
]


def display_name(ablation_value: str) -> str:
    """Reader-facing name for an ``ablation_value``.

    Whole-key matches win; otherwise the recognised variant suffixes are
    stripped (in any order and combination), the base is looked up, and the
    qualifiers are appended in the fixed order of :data:`_NAME_SUFFIXES` so
    that ``X_exact_M`` and ``X_M_exact`` read identically.  An unknown base is
    returned unchanged rather than raising: a missing entry should show up as
    an ugly label on a draft figure, not as a crashed figure run.
    """
    if ablation_value in DISPLAY_NAMES:
        return DISPLAY_NAMES[ablation_value]
    base = ablation_value
    qualifiers: list[str] = []
    changed = True
    while changed:                       # peel suffixes until none match
        changed = False
        for suffix, qualifier in _NAME_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                if qualifier not in qualifiers:
                    qualifiers.append(qualifier)
                changed = True
        if base in DISPLAY_NAMES:        # a named variant, not a bare base
            break
    if base not in DISPLAY_NAMES:
        return ablation_value
    order = [q for _, q in _NAME_SUFFIXES]
    qualifiers.sort(key=order.index)
    return DISPLAY_NAMES[base] + "".join(qualifiers)


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Set global rcParams for a clean, publication-ready look."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_MASTER: pd.DataFrame | None = None


def master() -> pd.DataFrame:
    """Load (and cache) the master index."""
    global _MASTER
    if _MASTER is None:
        _MASTER = pd.read_csv(MASTER_INDEX)
    return _MASTER


def run_dirs(method: str, ablation_key: str, ablation_value: str) -> list[Path]:
    """Return the run directories for a condition, one per seed (sorted)."""
    df = master()
    sel = df[(df["method"] == method)
             & (df["ablation_key"] == ablation_key)
             & (df["ablation_value"] == ablation_value)
             & (df["status"] == "completed")]
    return [Path(p) for p in sorted(sel["run_dir"].tolist())]


def read_curve_csv(path) -> pd.DataFrame:
    """Read an accuracy_curves.csv, tolerating the occasional malformed line.

    A few runs have a logging-flush artifact where newlines were dropped and
    several rows were concatenated on one physical line; those lines are skipped
    (a handful of steps out of ~1000, negligible for a plot).
    """
    return pd.read_csv(path, on_bad_lines="skip", engine="python")


def _switch_step(curve: pd.DataFrame, new_task_col: str) -> int:
    """First step at which ``new_task_col`` is evaluated (the task switch)."""
    valid = curve[curve[new_task_col].notna()]
    if valid.empty:
        return int(curve["step"].min())
    return int(valid["step"].min())


def task_curve(method: str, ablation_key: str, ablation_value: str,
               task_col: str, *, seeds_axis: bool = True):
    """Load a per-step accuracy curve for one task, averaged over seeds.

    Returns ``(steps, mean, std)`` where ``steps`` is the common step grid,
    ``mean``/``std`` are over seeds.  Curves are interpolated onto the union
    step grid so seeds with slightly different logging cadence still align.
    """
    dirs = run_dirs(method, ablation_key, ablation_value)
    series = []
    for d in dirs:
        f = d / "results" / "accuracy_curves.csv"
        if not f.exists():
            continue
        c = read_curve_csv(f)
        if task_col not in c.columns:
            continue
        sub = c[["step", task_col]].dropna()
        if sub.empty:
            continue
        series.append((sub["step"].to_numpy(dtype=float),
                       sub[task_col].to_numpy(dtype=float)))
    if not series:
        raise ValueError(f"no curve data for {method}/{ablation_key}/{ablation_value}/{task_col}")
    # Drop seeds whose step range is a gross outlier: a handful of runs have a
    # logging-flush artifact that mangles the step column (concatenated rows with
    # bogus step values), which would otherwise blow up the seed band. Their
    # aggregate metrics (from metrics_summary.json) are unaffected.
    max_steps = np.array([s.max() for s, _ in series])
    ref = float(np.median(max_steps))
    series = [sv for sv, mx in zip(series, max_steps) if mx <= 1.5 * ref]
    all_steps: set[float] = set()
    for s, _ in series:
        all_steps.update(s.tolist())
    grid = np.array(sorted(all_steps))
    mat = np.vstack([np.interp(grid, s, v, left=np.nan, right=np.nan) for s, v in series])
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return grid, mean, std


def switch_step(method: str, ablation_key: str, ablation_value: str,
                new_task_col: str = "task_1_acc") -> int:
    """Median switch step across the condition's seeds."""
    dirs = run_dirs(method, ablation_key, ablation_value)
    steps = []
    for d in dirs:
        f = d / "results" / "accuracy_curves.csv"
        if not f.exists():
            continue
        c = read_curve_csv(f)
        if new_task_col in c.columns:
            steps.append(_switch_step(c, new_task_col))
    return int(np.median(steps)) if steps else 0


# ---------------------------------------------------------------------------
# Aggregated scalar metrics (matches scripts/aggregate_results.py grouping)
# ---------------------------------------------------------------------------

_METRIC_SCALE = {"stab_gap_depth": 100.0, "stab_gap_max_drop": 100.0,
                 "min_ACC": 100.0}  # report depth / min-ACC in pp where useful


def agg(method: str, ablation_key: str, ablation_value: str) -> dict:
    """Mean/std over seeds for one condition. Depth returned in pp.

    Only rows with ``status == "completed"`` are aggregated, matching
    :func:`run_dirs` — a run that crashed, is still in flight, or whose
    manifest failed to parse still has a row in the master index, and its
    partial or absent metrics must not enter a published mean.  Every row in
    the archive is currently ``completed``, so this changes no reported number;
    it is a guard for the sweeps that are still running.
    """
    df = master()
    sel = df[(df["method"] == method)
             & (df["ablation_key"] == ablation_key)
             & (df["ablation_value"] == ablation_value)
             & (df["status"] == "completed")]
    out: dict = {"n": len(sel)}
    for col in ["ACC", "FORG", "min_ACC", "WF10", "WF100", "WP10", "WP100",
                "WC_ACC", "stab_gap_depth", "stab_gap_max_drop",
                "stab_gap_area", "stab_gap_area_end", "stab_gap_recovery_steps"]:
        vals = pd.to_numeric(sel[col], errors="coerce").dropna()
        if len(vals) == 0:
            out[col + "_mean"], out[col + "_std"] = float("nan"), float("nan")
        else:
            out[col + "_mean"] = float(vals.mean())
            out[col + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    # convenience: depth in pp
    out["depth_pp"] = out["stab_gap_depth_mean"] * 100.0
    out["depth_pp_std"] = out["stab_gap_depth_std"] * 100.0
    out["area_end"] = out["stab_gap_area_end_mean"]
    out["area_end_std"] = out["stab_gap_area_end_std"]
    return out


# ---------------------------------------------------------------------------
# Shared plotting primitives
# ---------------------------------------------------------------------------


def anchor_boundary(x, *series, at=0.0):
    """Insert a hold-last sample at ``at`` so a sparse->dense cadence change
    does not linearly interpolate the drop across the boundary.

    Per-step accuracy is logged every ten steps *before* a task switch and every
    step *after* it, so the last pre-switch sample can sit several steps short of
    the boundary (rot-MNIST: step 230 then step 235, with the boundary at 234).
    Matplotlib joins those two samples with one straight segment, which reads as
    a drop that began before the switch -- an artifact of the cadence, not of the
    run: steps 231-234 are still past-task training, so the level is flat there.
    Repeating the last pre-boundary value at ``at`` restores that plateau and
    puts the whole descent after the marker.

    ``x`` must be sorted ascending.  Returns ``(x, *series)`` with one sample
    inserted in each, or the inputs unchanged when there is nothing to anchor:
    a sample already sits at ``at``, or none lies on one side of it (the
    new-task curves legitimately start at the boundary).
    """
    x = np.asarray(x, dtype=float)
    before = np.flatnonzero(x < at)
    if before.size == 0 or not np.any(x > at) or np.any(np.isclose(x, at)):
        return (x, *series)
    i = int(before[-1])          # last sample strictly before the boundary
    out = [np.insert(x, i + 1, at)]
    for s in series:
        s = np.asarray(s, dtype=float)
        out.append(np.insert(s, i + 1, s[i]))
    return tuple(out)


def plot_transition(ax, method, key, value, task_col, color, label, *,
                    linestyle="-", zero_at_switch=True, band=True,
                    window=None, lw=2.0, alpha_band=0.16):
    """Plot one per-step transition curve (mean + std band) on ``ax``.

    With ``zero_at_switch`` the x axis counts completed new-task updates:
    the first evaluated step after the switch already includes one update,
    so it is placed at x = 1, and x = 0 is the state at the switch, before
    any new-task training.
    """
    steps, mean, std = task_curve(method, key, value, task_col)
    sw = switch_step(method, key, value)
    x = steps - sw + 1 if zero_at_switch else steps
    if zero_at_switch:
        # Hold the last pre-switch level up to x = 0 before the window is cut,
        # so the anchor cannot be clipped away by the mask.
        x, mean, std = anchor_boundary(x, mean, std, at=0.0)
    if window is not None:
        lo, hi = window
        m = (x >= lo) & (x <= hi)
        x, mean, std = x[m], mean[m], std[m]
    if band:
        ax.fill_between(x, mean - std, mean + std, color=color,
                        alpha=alpha_band, linewidth=0)
    ax.plot(x, mean, color=color, lw=lw, linestyle=linestyle, label=label,
            solid_capstyle="round")
    return x, mean, std


def seed_metric(runs, fn) -> dict[int, float]:
    """Apply ``fn(curve)`` to each run's ``accuracy_curves.csv``, keyed by seed.

    ``runs`` maps ``seed -> run directory`` (or is any iterable of such pairs).
    A run whose curve is missing or unreadable, or for which ``fn`` returns
    ``None``, is skipped rather than contributing a NaN: the callers of this
    helper are paired analyses, where a NaN seed silently unpairs a comparison.
    Seeds come from the caller because the run directory name does not carry
    one — the master index and ``.hydra/overrides.yaml`` do.
    """
    items = runs.items() if hasattr(runs, "items") else runs
    out: dict[int, float] = {}
    for seed, run_dir in items:
        f = Path(run_dir) / "results" / "accuracy_curves.csv"
        if not f.exists():
            continue
        try:
            curve = read_curve_csv(f)
        except Exception:                # a truncated / half-flushed CSV
            continue
        value = fn(curve)
        if value is not None:
            out[int(seed)] = float(value)
    return out


def dot_ladder(ax, xs, means, stds, color, label=None, *, marker="o",
               ms=6.0, lw=1.4, capsize=3.0, zorder=3):
    """Plot a mean±std dot ladder: one marker per rung, joined by a guide line.

    The companion to :func:`plot_transition` for the summary view of an
    ordinal sweep (the delta sweep, the step-size ladder): the trajectory
    panels show *what happens*, this shows the scalar it collapses to as the
    swept parameter moves.  ``xs`` is normally set on a log axis by the caller.
    """
    xs = np.asarray(xs, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    order = np.argsort(xs)
    xs, means, stds = xs[order], means[order], stds[order]
    ax.plot(xs, means, color=color, lw=lw, alpha=0.55, zorder=zorder - 1)
    ax.errorbar(xs, means, yerr=stds, color=color, fmt=marker, ms=ms,
                lw=0, elinewidth=lw, capsize=capsize, capthick=lw,
                label=label, zorder=zorder)
    return xs, means, stds


def mark_switch(ax, x=0.0):
    """Draw the task-switch reference line."""
    ax.axvline(x, color=MUTED, lw=0.9, linestyle=(0, (4, 3)), zorder=0)


def finalize(fig, path: Path | str):
    """Save a figure to the thesis image tree and report the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
