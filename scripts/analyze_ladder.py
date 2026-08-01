#!/usr/bin/env python3
"""Analysis of the S-series learning-rate ladder (RQ1).

    .venv/bin/python scripts/analyze_ladder.py

What the ladder is
------------------
A uniform rescaling of the step size is a *time reparametrisation* of the same
gradient flow: forward Euler at step eta integrates ``dtheta/dtau = -g(theta)``
on a finer grid as eta shrinks, so at matched flow time
``tau = sum of eta over steps`` every rung is following the same intended
trajectory, only more accurately.  Whatever vanishes as eta -> 0 is
discretisation; whatever survives belongs to the trajectory itself — the arc.
``scripts/run_S.sh`` builds the grid and its header documents the four design
constraints that make it read as a limit (matched flow time, frozen theta*_0,
matched record grid, cadence counted from the boundary).

What this script computes
-------------------------
For every ladder run found on disk:

  * ``depth`` — the largest drop of task-0 accuracy below its pre-switch value.
    Dimensionless, and eta-invariant by construction, so it is directly
    comparable across rungs.
  * ``area_tau`` — the shortfall below the *recovered* level, integrated over
    flow time.  The reference level mirrors ``gap_area(reference="end")`` in
    ``src/eval/stability_gap.py``: ``b = min(pre-switch, trailing-window mean)``,
    which scores permanent forgetting and continued learning at ~0 and so
    isolates the dip-and-recover transient.  This is the primary area number:
    it is the only one whose units (accuracy x flow time) are the same in every
    rung.
  * ``area_steps`` — the same integral over *training steps*, which is what
    ``stab_gap_area_end`` in ``master_index.csv`` reports.  Kept for
    cross-checking against the archive, and because it exposes a wrinkle in the
    run_S.sh header: ``gap_area`` is not a sum over records but a trapezoid
    weighted by step *differences*, so it scales with the stretch factor
    c = 0.1/eta and is **not** comparable across rungs as it stands.  Exactly:
    ``area_tau = eta * area_steps``.

and then, per (arm, momentum, seed):

  * the O(eta) extrapolation ``A(eta) = A_0 + k*eta`` — forward Euler's global
    error is first order in the step, so the eta -> 0 limit comes with an error
    bar instead of being asserted to be the smallest rung.  The fit residual is
    reported alongside: a departure from linearity at eta = 0.1 is expected,
    and is itself the signature of the discrete instability that produces the
    first-step spike.  A two-point fit over the two small rungs is reported as
    well, with the eta = 0.1 rung's departure from it quantified.

  * paired Wilcoxon contrasts between adjacent rungs and between S1 and S3,
    via ``scripts/paired_stats.py`` — the rungs are exactly paired (identical
    theta*_0 and bit-identical replay buffer at a given arm/momentum/seed;
    only eta differs), so the seed-paired test is the right one.

  * a distributional cross-check of the eta = 0.1 rung against the archived
    ``D1_vanilla`` rows.  This is **not** a reproduction: the ladder warm-starts
    from a shared checkpoint and so skips task-0 training, which moves the RNG
    stream and gives a different (though identically distributed) replay
    buffer.  Only means +- std and their overlap are reported.

Partial sweeps
--------------
The grid takes hours, so this script is written to run against it while it is
still filling in: a missing cell, a missing seed, a run without a manifest, a
truncated CSV are all skipped with a warning on stderr.  Only runs whose
manifest says ``completed`` are analysed — a half-written curve has no
recovered level, so its area reference would be meaningless.

Produces
--------
``outputs/summary_tables/lr_ladder.csv`` — one row per (arm, momentum, rung)
with n, eta and mean +- std of each metric, plus the ``kind="limit"`` rows
holding the O(eta) extrapolation to eta = 0.  Everything else prints to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    # Allow `python scripts/analyze_ladder.py` as well as
    # `python -m scripts.analyze_ladder`: run as a script, sys.path[0] is
    # scripts/, so the `scripts` package itself is not importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paired_stats import compare  # noqa: E402  (after the path fix)

# ---------------------------------------------------------------------------
# Constants of the ladder design (see scripts/run_S.sh)
# ---------------------------------------------------------------------------

ABLATION_KEY = "lr_ladder"

# Task 0 of two-task rot-MNIST is one epoch of 235 batches, so task 1 begins at
# global step 235 in every run — including the ladder's, which skip task-0
# training but keep the step accounting.  The single pre-switch record sits one
# step earlier; the record *at* the boundary step is the state after the first
# new-task update.
BOUNDARY_STEP = 235
PRE_SWITCH_STEP = BOUNDARY_STEP - 1

# eta * c, constant across the ladder by construction: the dense eval cadence
# is c steps and eta = 0.1/c, so consecutive dense records are 0.1 of flow time
# apart in every rung.  This is the spacing of the common tau grid.
TAU_SPACING = 0.1

# Archived reference block for the eta = 0.1 consistency check.
D1_METHOD, D1_KEY, D1_VALUE = "er", "decomposition", "D1_vanilla"

RUNG_ORDER = ["S1", "S2", "S3"]


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderRun:
    """One ladder cell-seed, as found on disk."""

    run_dir: Path
    ablation_value: str          # e.g. S2_eta0.01_exact_M
    rung: str                    # S1 | S2 | S3
    arm: str                     # "exact" (60k full buffer) | "sampled" (1k)
    eta: float                   # read from the run's own overrides, not the label
    momentum: float
    epochs: int                  # = c = the stretch factor = the dense cadence
    seed: int
    status: str

    @property
    def stretch(self) -> int:
        """c = 0.1/eta, the flow-time stretch factor."""
        return self.epochs


def _read_overrides(path: Path) -> Dict[str, str]:
    """Parse a Hydra ``overrides.yaml`` (a list of ``key=value`` strings)."""
    with open(path, encoding="utf-8") as fh:
        entries = yaml.safe_load(fh) or []
    out: Dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, str) and "=" in entry:
            key, value = entry.split("=", 1)
            out[key] = value
    return out


def _manifest_status(run_dir: Path) -> str:
    """Status from the run manifest; ``"no_manifest"`` while a run is in flight.

    The manifest is written when the run finishes, so its absence is the
    cleanest available signal that a directory belongs to a job that is still
    running (or died before it could report).
    """
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return "no_manifest"
    try:
        with open(path, encoding="utf-8") as fh:
            return str(json.load(fh).get("status", "unknown"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"unreadable manifest {path}: {exc}")
        return "parse_error"


def discover_runs(outputs: Path) -> List[LadderRun]:
    """Find every ``lr_ladder`` run under ``outputs/er/rot_mnist/``.

    The filesystem is the source of truth here rather than
    ``master_index.csv``: the sweep is aggregated only once it finishes, and
    the point of this script is to read it while it fills in.  eta, momentum
    and the epoch count come from the run's own ``.hydra/overrides.yaml`` —
    the ablation_value carries a rounded label (``S2_eta0.033`` for
    0.1/3 = 0.03333...) which must never be used as a number.
    """
    runs: List[LadderRun] = []
    root = outputs / "er" / "rot_mnist"
    if not root.is_dir():
        warn(f"no such directory: {root}")
        return runs

    for overrides_path in sorted(root.glob("*/.hydra/overrides.yaml")):
        try:
            ov = _read_overrides(overrides_path)
        except (OSError, yaml.YAMLError) as exc:
            warn(f"unreadable overrides {overrides_path}: {exc}")
            continue
        if ov.get("+ablation_key") != ABLATION_KEY:
            continue

        run_dir = overrides_path.parent.parent
        value = ov.get("+ablation_value", "")
        rung = value.split("_", 1)[0]
        if rung not in RUNG_ORDER:
            warn(f"unrecognised rung in ablation_value {value!r} ({run_dir}); skipped")
            continue
        try:
            eta = float(ov["training.lr"])
            momentum = float(ov.get("training.momentum", 0.0))
            epochs = int(ov["training.epochs_per_task"])
            seed = int(ov.get("seed", -1))
        except (KeyError, ValueError) as exc:
            warn(f"incomplete overrides in {run_dir}: {exc}; skipped")
            continue

        # Flow time is matched only if eta * epochs is the same in every rung;
        # if it is not, the cell is not on the ladder and comparing it would be
        # comparing an under-trained trajectory to a finished one.  The target
        # is TAU_SPACING because the dense cadence is also c steps: eta * c is
        # at once the flow time per epoch and the flow time between records.
        if not math.isclose(eta * epochs, TAU_SPACING, rel_tol=1e-6):
            warn(f"{value} seed {seed}: eta*epochs = {eta * epochs:.6g}, "
                 f"expected {TAU_SPACING:g} — flow time is not matched; skipped")
            continue

        runs.append(LadderRun(
            run_dir=run_dir,
            ablation_value=value,
            rung=rung,
            arm="exact" if "_exact" in value else "sampled",
            eta=eta,
            momentum=momentum,
            epochs=epochs,
            seed=seed,
            status=_manifest_status(run_dir),
        ))
    return runs


# ---------------------------------------------------------------------------
# Curves and per-run metrics
# ---------------------------------------------------------------------------


@dataclass
class Curve:
    """A run's past-task trajectory on the dense stability-gap record grid."""

    pre: float          # task-0 accuracy at the pre-switch record (step 234)
    step: np.ndarray    # global step, post-switch records only
    tau: np.ndarray     # eta * (step - 235): flow time since the boundary
    acc: np.ndarray     # task-0 accuracy


def load_curve(run: LadderRun) -> Optional[Curve]:
    """Load one run's task-0 curve, restricted to the dense record grid.

    Two cadences write into ``accuracy_curves.csv``: the stability-gap tracker
    (every ``c`` steps through the whole of task 1, which is the grid the
    ladder was designed to match across rungs) and the ordinary eval loop
    (every ``10c`` steps, on multiples of the *global* step).  The latter adds
    ~23 samples at flow times that are not on the matched grid.  Keeping only
    records at ``(step - 235) % c == 0`` both restores the exact rung-to-rung
    correspondence and makes the integral below reproduce the tracker's own
    ``gap_area``, which sees only its own records.

    Returns ``None`` — with a warning — for anything unusable: no file, a
    truncated or malformed CSV, a missing pre-switch record, or fewer than two
    post-switch records.
    """
    path = run.run_dir / "results" / "accuracy_curves.csv"
    if not path.is_file():
        warn(f"{run.ablation_value} seed {run.seed}: no accuracy_curves.csv; skipped")
        return None
    try:
        # on_bad_lines="skip" mirrors vw_style.read_curve_csv: a run whose log
        # was flushed mid-line leaves one concatenated row behind.
        curve = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except (OSError, pd.errors.ParserError) as exc:
        warn(f"{run.ablation_value} seed {run.seed}: unreadable curve ({exc}); skipped")
        return None
    if not {"step", "task_0_acc"}.issubset(curve.columns):
        warn(f"{run.ablation_value} seed {run.seed}: curve lacks step/task_0_acc; skipped")
        return None

    try:
        curve = (curve[["step", "task_0_acc"]]
                 .apply(pd.to_numeric, errors="coerce").dropna()
                 .astype({"step": int, "task_0_acc": float})
                 .drop_duplicates(subset="step")
                 .sort_values("step"))
    except (TypeError, ValueError) as exc:
        warn(f"{run.ablation_value} seed {run.seed}: malformed curve ({exc}); skipped")
        return None

    pre_rows = curve[curve["step"] == PRE_SWITCH_STEP]
    if pre_rows.empty:
        warn(f"{run.ablation_value} seed {run.seed}: no pre-switch record at step "
             f"{PRE_SWITCH_STEP}; skipped")
        return None
    pre = float(pre_rows["task_0_acc"].iloc[0])

    post = curve[curve["step"] >= BOUNDARY_STEP]
    on_grid = (post["step"] - BOUNDARY_STEP) % run.stretch == 0
    dropped = int((~on_grid).sum())
    post = post[on_grid]
    if len(post) < 2:
        warn(f"{run.ablation_value} seed {run.seed}: only {len(post)} post-switch "
             f"record(s); skipped")
        return None
    if dropped and run.stretch == 1:
        # At c = 1 every step is on the dense grid, so nothing should be cut.
        warn(f"{run.ablation_value} seed {run.seed}: dropped {dropped} off-grid "
             "record(s) at c=1 — unexpected")

    step = post["step"].to_numpy(dtype=float)
    tau = run.eta * (step - BOUNDARY_STEP)

    # Task 1 is c epochs of BOUNDARY_STEP batches, so its last dense record
    # sits at step 235 + 234c and every rung should reach the same flow time,
    # 234 * eta * c = 23.4.  A curve that stops appreciably short is truncated
    # — a job killed mid-run, or a CSV caught mid-flush — and must not be
    # scored: the area references the level the task *settles* at, which a
    # missing tail does not contain, so the number would be quietly wrong
    # rather than obviously absent.
    expected_tau = TAU_SPACING * (BOUNDARY_STEP - 1)
    if tau[-1] < 0.99 * expected_tau:
        warn(f"{run.ablation_value} seed {run.seed}: curve ends at tau = "
             f"{tau[-1]:.3g} of an expected {expected_tau:.3g} (truncated); skipped")
        return None

    return Curve(pre=pre, step=step, tau=tau,
                 acc=post["task_0_acc"].to_numpy(dtype=float))


def _end_reference(acc: np.ndarray, pre: float) -> float:
    """The level the task settles at, as ``gap_area(reference="end")`` defines it.

    ``min(pre-switch level, trailing-window mean)`` over the last
    ``max(5, ceil(0.1 n))`` samples.  Referencing the recovered level rather
    than the pre-switch one is what makes the area a *transient* measure:
    permanent forgetting and continued learning both score ~0.
    """
    w = max(5, math.ceil(0.1 * len(acc)))
    return min(pre, float(np.mean(acc[-w:])))


def run_metrics(run: LadderRun, curve: Curve) -> Dict[str, float]:
    """Depth and area for one run, in flow time and (for the archive) in steps."""
    depth = max(0.0, curve.pre - float(np.min(curve.acc)))
    b = _end_reference(curve.acc, curve.pre)
    shortfall = np.maximum(0.0, b - curve.acc)
    area_tau = float(np.trapezoid(shortfall, curve.tau))
    area_steps = float(np.trapezoid(shortfall, curve.step))
    return {
        "depth": depth,
        "area_tau": area_tau,
        "area_steps": area_steps,
        "pre": curve.pre,
        "end_ref": b,
        "tau_max": float(curve.tau[-1]),
        "n_records": float(len(curve.acc)),
    }


def collect(runs: Sequence[LadderRun]) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], Curve]]:
    """Metrics table (one row per completed run) plus the curves behind it."""
    rows: List[dict] = []
    curves: Dict[Tuple[str, int], Curve] = {}
    for run in runs:
        if run.status != "completed":
            warn(f"{run.ablation_value} seed {run.seed}: status={run.status}; skipped")
            continue
        curve = load_curve(run)
        if curve is None:
            continue
        curves[(run.ablation_value, run.seed)] = curve
        rows.append({
            "condition": run.ablation_value,
            "rung": run.rung,
            "arm": run.arm,
            "momentum": run.momentum,
            "eta": run.eta,
            # Momentum multiplies the asymptotic step by 1/(1-mu), so this is
            # the step the trajectory actually takes per unit of "wall" step;
            # tau_eff = eta*(step-235)/(1-mu) puts the two momentum legs on one
            # axis.  It is a matched-effective-step read, not a
            # reparametrisation: momentum is not a rescaling of the flow.
            "eta_eff": run.eta / (1.0 - run.momentum),
            "seed": run.seed,
            "run_dir": str(run.run_dir),
            **run_metrics(run, curve),
        })
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------------------
# Common flow-time grid
# ---------------------------------------------------------------------------


def resample(curves: Sequence[Curve], grid: np.ndarray) -> np.ndarray:
    """Interpolate curves onto a shared tau grid; NaN outside each one's range.

    Rungs record at the same flow-time spacing but not, in general, at exactly
    the same tau (the third rung's eta is 0.1/33, so its grid carries rounding),
    and a partial run stops early.  Interpolating with NaN outside the observed
    range lets ``nanmean`` average whatever seeds exist at each tau without
    inventing a tail.
    """
    return np.vstack([
        np.interp(grid, c.tau, c.acc, left=np.nan, right=np.nan) for c in curves
    ]) if curves else np.empty((0, grid.size))


def tau_grid(curves: Sequence[Curve], spacing: float = TAU_SPACING) -> np.ndarray:
    """Common flow-time grid spanning the curves, at the matched record spacing."""
    if not curves:
        return np.zeros(0)
    tau_max = max(float(c.tau[-1]) for c in curves)
    return np.arange(0.0, tau_max + 0.5 * spacing, spacing)


def condition_band(curves: Sequence[Curve], grid: np.ndarray):
    """Seed mean/std of a condition on the common grid, plus the seed count."""
    mat = resample(curves, grid)
    if mat.size == 0:
        return np.full(grid.shape, np.nan), np.full(grid.shape, np.nan), np.zeros(grid.shape, int)
    # The grid runs to the longest curve's end, so its last cell(s) can be
    # empty for every seed once floating-point rounding of tau is taken into
    # account.  An all-NaN column is the intended "no data here" answer, not a
    # problem to report, so nanmean's empty-slice warning is silenced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
    return mean, std, np.sum(~np.isnan(mat), axis=0)


# ---------------------------------------------------------------------------
# O(eta) extrapolation
# ---------------------------------------------------------------------------


def fit_o_eta(etas: Sequence[float], values: Sequence[float]) -> Optional[dict]:
    """Least-squares fit of ``A(eta) = A_0 + k*eta``; ``None`` if under-determined.

    Forward Euler's global error is first order in the step, so this is the
    model the ladder is built to test.  ``resid_max`` is the largest absolute
    departure of a rung from the fitted line: with three rungs a visible
    residual means the leading-order model has broken down somewhere on the
    ladder, which at eta = 0.1 is the expected signature of the discrete
    instability rather than a failure of the fit.
    """
    etas = np.asarray(etas, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(etas) < 2 or len(np.unique(etas)) < 2:
        return None
    slope, intercept = np.polyfit(etas, values, 1)
    resid = values - (intercept + slope * etas)
    return {
        "A0": float(intercept),
        "k": float(slope),
        "resid_max": float(np.max(np.abs(resid))),
        "resid_rms": float(np.sqrt(np.mean(resid ** 2))),
        "n_rungs": int(len(etas)),
    }


def extrapolate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (arm, momentum) eta -> 0 limit of *metric*, fitted per seed.

    The fit is done *within* a seed, because the rungs are exactly paired: the
    same theta*_0 and the same replay buffer, so the seed-to-seed spread is
    shared across the ladder and cancels out of the slope.  The spread reported
    on A_0 is therefore over independent fits, not over pooled points.

    Two fits are reported per (arm, momentum):
      * ``A0`` over every available rung;
      * ``A0_small`` over the rungs with eta <= 0.01 only, together with
        ``S1_departure`` = (measured at eta=0.1) - (that fit's prediction
        there).  A large positive departure is the discrete instability: the
        coarse rung sits above the line the fine rungs lie on.
    """
    out: List[dict] = []
    for (arm, mu), leg in df.groupby(["arm", "momentum"]):
        per_seed: List[dict] = []
        short: List[int] = []
        for seed, grp in leg.groupby("seed"):
            grp = grp.sort_values("eta")
            fit = fit_o_eta(grp["eta"], grp[metric])
            if fit is None:
                short.append(int(seed))    # reported once per leg below
                continue
            row = {"seed": seed, **fit}

            small = grp[grp["eta"] <= 0.011]
            coarse = grp[grp["eta"] > 0.011]
            small_fit = fit_o_eta(small["eta"], small[metric])
            if small_fit is not None:
                row["A0_small"] = small_fit["A0"]
                if not coarse.empty:
                    predicted = small_fit["A0"] + small_fit["k"] * coarse["eta"].to_numpy()
                    row["S1_departure"] = float(
                        (coarse[metric].to_numpy() - predicted).max()
                    )
            per_seed.append(row)
        if short:
            warn(f"{arm}/mu={mu:g}: {metric} not extrapolated for seed(s) "
                 f"{','.join(map(str, short))} — fewer than two rungs completed")
        if not per_seed:
            continue
        fits = pd.DataFrame(per_seed)
        rec = {"arm": arm, "momentum": mu, "metric": metric, "n_seeds": len(fits),
               "n_rungs": int(fits["n_rungs"].max())}
        for col in ["A0", "k", "resid_max", "A0_small", "S1_departure"]:
            vals = fits[col].dropna() if col in fits else pd.Series(dtype=float)
            rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else (
                0.0 if len(vals) == 1 else float("nan"))
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_SUMMARY_METRICS = ["depth", "area_tau", "area_steps"]


def summary_table(df: pd.DataFrame, limits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Condition-level table: one row per rung, plus the eta -> 0 limit rows."""
    rows: List[dict] = []
    for (arm, mu, rung), grp in df.groupby(["arm", "momentum", "rung"]):
        row = {"kind": "rung", "arm": arm, "momentum": mu, "rung": rung,
               "condition": grp["condition"].iloc[0],
               "eta": float(grp["eta"].iloc[0]),
               "eta_eff": float(grp["eta_eff"].iloc[0]),
               "n_seeds": len(grp)}
        for metric in _SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_std"] = float(grp[metric].std(ddof=1)) if len(grp) > 1 else 0.0
        rows.append(row)

    for metric, lim in limits.items():
        for _, r in lim.iterrows():
            rows.append({
                "kind": "limit", "arm": r["arm"], "momentum": r["momentum"],
                "rung": f"eta->0 ({int(r['n_rungs'])} rungs)",
                "condition": "", "eta": 0.0, "eta_eff": 0.0,
                "n_seeds": int(r["n_seeds"]),
                f"{metric}_mean": r["A0_mean"], f"{metric}_std": r["A0_std"],
                f"{metric}_slope": r["k_mean"], f"{metric}_resid_max": r["resid_max_mean"],
                f"{metric}_A0_small": r.get("A0_small_mean"),
                f"{metric}_S1_departure": r.get("S1_departure_mean"),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Collapse the per-metric limit rows (one per metric) onto one row per leg.
    keys = ["kind", "arm", "momentum", "rung", "condition", "eta", "eta_eff", "n_seeds"]
    table = table.groupby(keys, dropna=False, as_index=False).first()
    return table.sort_values(["arm", "momentum", "eta"], ascending=[True, True, False])


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _fmt(v, prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{prec}g}"


def print_coverage(runs: Sequence[LadderRun]) -> None:
    print("\n" + "=" * 78)
    print("  Coverage — cells found on disk")
    print("=" * 78)
    if not runs:
        print("  (no lr_ladder runs found)")
        return
    rows = []
    for (arm, mu, rung), grp in pd.DataFrame([{
        "arm": r.arm, "momentum": r.momentum, "rung": r.rung,
        "status": r.status, "seed": r.seed} for r in runs
    ]).groupby(["arm", "momentum", "rung"]):
        done = sorted(grp.loc[grp["status"] == "completed", "seed"])
        other = sorted(grp.loc[grp["status"] != "completed", "seed"])
        rows.append({"arm": arm, "momentum": mu, "rung": rung,
                     "completed": f"{len(done)}/5", "seeds": ",".join(map(str, done)) or "-",
                     "in flight": ",".join(map(str, other)) or "-"})
    full = pd.DataFrame(rows).sort_values(["arm", "momentum", "rung"])
    print(full.to_string(index=False))
    n_done = sum(r.status == "completed" for r in runs)
    print(f"\n  {n_done} completed of {len(runs)} run dirs found; "
          f"the full grid is 12 cells x 5 seeds = 60.")


def print_rungs(table: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  Per-rung metrics (mean +- std over seeds)")
    print("=" * 78)
    rung_rows = table[table["kind"] == "rung"]
    if rung_rows.empty:
        print("  (nothing to report)")
        return
    for (arm, mu), leg in rung_rows.groupby(["arm", "momentum"]):
        print(f"\n  arm={arm}  momentum={mu:g}")
        print(f"    {'rung':>5} {'eta':>10} {'n':>2}  {'depth':>17} "
              f"{'area (flow time)':>19} {'area (steps)':>19}")
        for _, r in leg.sort_values("eta", ascending=False).iterrows():
            print(f"    {r['rung']:>5} {r['eta']:>10.6g} {int(r['n_seeds']):>2}  "
                  f"{_fmt(r['depth_mean'], 3):>8} +- {_fmt(r['depth_std'], 2):<6} "
                  f"{_fmt(r['area_tau_mean'], 4):>10} +- {_fmt(r['area_tau_std'], 2):<6} "
                  f"{_fmt(r['area_steps_mean'], 4):>10} +- {_fmt(r['area_steps_std'], 2):<6}")
    print("\n  depth is dimensionless and eta-invariant; area (flow time) has units "
          "\n  accuracy x tau and is the comparable one.  area (steps) is what "
          "\n  stab_gap_area_end reports: it scales with c = 0.1/eta, so it grows "
          "\n  down the ladder even when the trajectory is unchanged.")


def print_limits(limits: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 78)
    print("  O(eta) extrapolation:  A(eta) = A_0 + k*eta, fitted per seed")
    print("=" * 78)
    any_rows = False
    for metric, lim in limits.items():
        if lim.empty:
            continue
        any_rows = True
        print(f"\n  {metric}")
        print(f"    {'arm':>8} {'mu':>4} {'seeds':>5} {'rungs':>5}  {'A_0':>16} "
              f"{'k':>13} {'resid_max':>10} {'A_0 (eta<=.01)':>15} {'S1 departure':>13}")
        for _, r in lim.iterrows():
            print(f"    {r['arm']:>8} {r['momentum']:>4g} {int(r['n_seeds']):>5} "
                  f"{int(r['n_rungs']):>5}  "
                  f"{_fmt(r['A0_mean'], 3):>7} +- {_fmt(r['A0_std'], 2):<6} "
                  f"{_fmt(r['k_mean'], 3):>13} {_fmt(r['resid_max_mean'], 3):>10} "
                  f"{_fmt(r.get('A0_small_mean'), 3):>15} "
                  f"{_fmt(r.get('S1_departure_mean'), 3):>13}")
    if not any_rows:
        print("  (no leg has two or more completed rungs yet)")
        return
    print("\n  A_0 is the eta -> 0 limit: what survives the step going to zero, i.e. "
          "\n  the arc.  Read resid_max with care — with three rungs spread over a "
          "\n  30x lever arm the least-squares line is pinned by the coarse rung, so "
          "\n  a real breakdown of the O(eta) model leaves only a small residual "
          "\n  there.  'S1 departure', how far the eta = 0.1 rung sits above the line "
          "\n  fitted through the eta <= 0.01 rungs alone, is the sharper diagnostic: "
          "\n  it is the non-linear excess, i.e. the discrete instability itself.  "
          "\n  Where the two limits disagree, A_0 (eta <= .01) is the trustworthy one.")


def print_separation(curves_by_condition: Dict[str, List[Curve]]) -> None:
    """How far apart the rungs' mean trajectories run, on the common tau grid."""
    print("\n" + "=" * 78)
    print("  Trajectory separation on the common flow-time grid")
    print("=" * 78)
    by_leg: Dict[Tuple[str, float], Dict[str, List[Curve]]] = {}
    for cond, curves in curves_by_condition.items():
        arm = "exact" if "_exact" in cond else "sampled"
        mu = 0.9 if cond.endswith("_M") else 0.0
        by_leg.setdefault((arm, mu), {})[cond] = curves

    for (arm, mu), conds in sorted(by_leg.items()):
        # Difference every rung against the finest one available: it is the
        # closest thing on hand to the flow the whole ladder is approximating.
        finest = max(conds, key=lambda c: RUNG_ORDER.index(c.split("_", 1)[0]))
        grid = tau_grid([c for cs in conds.values() for c in cs])
        ref_mean, _, ref_n = condition_band(conds[finest], grid)
        print(f"\n  arm={arm}  momentum={mu:g}   reference rung: {finest}")
        for cond in sorted(conds, key=lambda c: RUNG_ORDER.index(c.split("_", 1)[0])):
            mean, _, n = condition_band(conds[cond], grid)
            both = (~np.isnan(mean)) & (~np.isnan(ref_mean))
            if not both.any():
                continue
            diff = mean[both] - ref_mean[both]
            i = int(np.argmax(np.abs(diff)))
            tau_at = grid[both][i]
            print(f"    {cond:<24} seeds={int(np.nanmax(n)):d}  "
                  f"max |T0 - reference| = {abs(diff[i]):.4f} at tau = {tau_at:.2f}  "
                  f"(overlap: tau <= {grid[both].max():.2f})")


def print_paired(df: pd.DataFrame) -> None:
    """Seed-paired Wilcoxon contrasts, adjacent rungs and S1 vs S3."""
    print("\n" + "=" * 78)
    print("  Paired contrasts between rungs (Wilcoxon signed-rank, two-sided)")
    print("=" * 78)
    rows: List[dict] = []
    for (arm, mu), leg in df.groupby(["arm", "momentum"]):
        present = {r: c for r, c in zip(leg["rung"], leg["condition"])}
        for a, b in [("S1", "S2"), ("S2", "S3"), ("S1", "S3")]:
            if a not in present or b not in present:
                continue
            for metric in ["depth", "area_tau"]:
                row = compare(leg, present[a], present[b], metric)
                row["arm"], row["momentum"] = arm, mu
                rows.append(row)
    if not rows:
        print("  (no leg has two comparable rungs yet)")
        return
    out = pd.DataFrame(rows)
    cols = ["arm", "momentum", "metric", "cond_a", "cond_b", "n", "mean_a", "mean_b",
            "mean_delta", "signs_agree", "wilcoxon_p", "ci95_lo", "ci95_hi", "note"]
    fmt = out[cols].copy()
    for c in ["mean_a", "mean_b", "mean_delta", "ci95_lo", "ci95_hi"]:
        fmt[c] = fmt[c].map(lambda v: _fmt(v, 4))
    fmt["wilcoxon_p"] = fmt["wilcoxon_p"].map(lambda v: _fmt(v, 3))
    print(fmt.to_string(index=False))
    print("\n  With five paired seeds the smallest attainable two-sided p is "
          "2/2^5 = 0.0625,\n  reached exactly when all five seeds agree in sign "
          "(scripts/paired_stats.py).")


def print_d1_check(df: pd.DataFrame, master_path: Path) -> None:
    """Distributional cross-check of the eta = 0.1 rung against archived D1."""
    print("\n" + "=" * 78)
    print("  S1 (eta = 0.1) vs archived D1_vanilla — distributional check only")
    print("=" * 78)
    if not master_path.is_file():
        warn(f"no master index at {master_path}; D1 check skipped")
        return
    master = pd.read_csv(master_path)
    d1 = master[(master["method"] == D1_METHOD)
                & (master["ablation_key"] == D1_KEY)
                & (master["ablation_value"] == D1_VALUE)
                & (master["status"] == "completed")]
    if d1.empty:
        warn(f"no completed {D1_VALUE} rows in {master_path}; D1 check skipped")
        return

    s1 = df[(df["rung"] == "S1") & (df["arm"] == "sampled") & (df["momentum"] == 0.0)]
    if s1.empty:
        print("  (the eta = 0.1, 1k-buffer, momentum-0 rung has no completed seeds yet)")
        return

    print("\n  D1 numbers come from the archive's manifests (its own tracker "
          "baseline);\n  S1 numbers are recomputed here from the curve.  The two "
          "are NOT paired: the\n  ladder warm-starts from a shared theta*_0 and "
          "skips task-0 training, so the RNG\n  stream and hence the replay buffer "
          "differ seed for seed.  D1's last pre-switch\n  record is also at step 230, "
          "not 234, so pre-switch levels are not compared.\n")
    for label, s1_col, d1_col in [("depth", "depth", "stab_gap_depth"),
                                  ("area (steps)", "area_steps", "stab_gap_area_end")]:
        a = s1[s1_col].astype(float)
        b = pd.to_numeric(d1[d1_col], errors="coerce").dropna()
        if b.empty:
            continue
        am, asd = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
        bm, bsd = float(b.mean()), float(b.std(ddof=1)) if len(b) > 1 else 0.0
        overlap = (am - asd) <= (bm + bsd) and (bm - bsd) <= (am + asd)
        print(f"    {label:<14} S1: {am:.4f} +- {asd:.4f} (n={len(a)})   "
              f"D1: {bm:.4f} +- {bsd:.4f} (n={len(b)})   "
              f"1-sd intervals {'overlap' if overlap else 'DO NOT overlap'}")


def print_area_consistency(df: pd.DataFrame) -> None:
    """Recomputed area vs the archived ``stab_gap_area_end``, run by run."""
    print("\n" + "=" * 78)
    print("  Cross-check: recomputed area (steps) vs the run's own manifest")
    print("=" * 78)
    diffs: List[Tuple[str, int, float, float]] = []
    for _, r in df.iterrows():
        manifest = Path(r["run_dir"]) / "run_manifest.json"
        if not manifest.is_file():
            continue
        try:
            with open(manifest, encoding="utf-8") as fh:
                reported = (json.load(fh).get("final_metrics") or {}).get(
                    "stability_gap_area_end")
        except (OSError, json.JSONDecodeError):
            continue
        if reported is None:
            continue
        diffs.append((r["condition"], int(r["seed"]), float(reported),
                      float(r["area_steps"])))
    if not diffs:
        print("  (no manifest area to compare against)")
        return
    rel = np.array([abs(a - b) / max(abs(a), 1e-12) for _, _, a, b in diffs])
    worst = int(np.argmax(rel))
    cond, seed, reported, recomputed = diffs[worst]
    print(f"  {len(diffs)} run(s) compared; max relative difference "
          f"{rel.max():.2e} ({cond} seed {seed}: "
          f"manifest {reported:.6g} vs recomputed {recomputed:.6g}).")
    print("  A difference of order 1e-15 confirms the recomputation reproduces "
          "\n  StabilityGapTracker.gap_area(reference='end') on the dense grid; "
          "\n  anything larger means the two are not seeing the same records.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--outputs", type=Path, default=root / "outputs",
                   help="Run-output root to scan (default: outputs/).")
    p.add_argument("--master", type=Path, default=None,
                   help="master_index.csv for the D1 check "
                        "(default: <outputs>/master_index.csv).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Summary table destination "
                        "(default: <outputs>/summary_tables/lr_ladder.csv).")
    args = p.parse_args(argv)
    master_path = args.master or args.outputs / "master_index.csv"
    csv_path = args.csv or args.outputs / "summary_tables" / "lr_ladder.csv"

    runs = discover_runs(args.outputs)
    print_coverage(runs)

    df, curves = collect(runs)
    if df.empty:
        print("\nNo completed ladder runs to analyse yet.")
        return 0

    limits = {m: extrapolate(df, m) for m in ["depth", "area_tau"]}
    table = summary_table(df, limits)

    print_rungs(table)
    print_limits(limits)

    by_condition: Dict[str, List[Curve]] = {}
    for (cond, _seed), curve in curves.items():
        by_condition.setdefault(cond, []).append(curve)
    print_separation(by_condition)

    print_paired(df)
    print_d1_check(df, master_path)
    print_area_consistency(df)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(table)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
