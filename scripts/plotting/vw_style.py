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
slots.  Palette validated with the data-viz skill's ``validate_palette.js``.
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
    """Mean/std over seeds for one condition. Depth returned in pp."""
    df = master()
    sel = df[(df["method"] == method)
             & (df["ablation_key"] == ablation_key)
             & (df["ablation_value"] == ablation_value)]
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


def plot_transition(ax, method, key, value, task_col, color, label, *,
                    linestyle="-", zero_at_switch=True, band=True,
                    window=None, lw=2.0, alpha_band=0.16):
    """Plot one per-step transition curve (mean + std band) on ``ax``."""
    steps, mean, std = task_curve(method, key, value, task_col)
    sw = switch_step(method, key, value)
    x = steps - sw if zero_at_switch else steps
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
