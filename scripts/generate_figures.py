#!/usr/bin/env python3
"""Publication figure generation (T7.3).

Reads ``master_index.csv`` (produced by aggregate_results.py) to identify
runs, then reads raw CSVs from each run's ``results/`` directory.  Produces
8 publication-ready matplotlib figures, each saved as both PDF and PNG.

Figures produced
----------------
fig_accuracy_matrix      — R[i,j] heatmap, one subplot per method.
fig_cone_sweep           — ACC & BWT vs cone angle α (ablation A1).
fig_ablation_summary     — ACC for every ablation + baseline on both datasets.
fig_trust_radius         — Trust-radius evolution over all training steps.
fig_eigenvalue_evolution — Top-5 Hessian eigenvalues over training steps.
fig_stability_gap        — Per-step accuracy curves at a task boundary.
fig_cost                 — Wall-clock seconds per task (methods + K variants).
fig_memory_budget        — ACC vs memory budget (ablation A10 + baselines).

All figures have axis labels and legends.  One-column figures are 3.25 in
wide; full-width figures are 6.75 in wide (two-column paper format).

Every figure function prints the ``run_id`` values it consumed, and the
combined provenance is written to ``<outdir>/figure_provenance.json`` so
every paper number can be traced to an exact run.

Usage
-----
    python scripts/generate_figures.py \\
        --index outputs/master_index.csv \\
        --outdir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")   # headless backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Publication style constants
# ---------------------------------------------------------------------------

ONE_COL  = 3.25   # single-column figure width, inches
FULL_COL = 6.75   # full-page figure width, inches
FIG_H    = 2.4    # default figure height, inches

# IBM-inspired colorblind-friendly palette (8 colors)
PALETTE = [
    "#0f62fe",   # blue      — CACL
    "#fa4d56",   # red       — ER
    "#08bdba",   # teal      — GEM-joint
    "#ff7eb6",   # magenta   — GEM-lasttask
    "#be95ff",   # purple    — NCL
    "#6fdc8c",   # green
    "#ffd6e8",   # pink
    "#d4bbff",   # lavender
]

# Display labels for method names
METHOD_LABELS: Dict[str, str] = {
    "cacl":          "CACL",
    "er":            "ER",
    "gem":           "GEM",
    "agem":          "A-GEM",
    "ncl":           "NCL",
}

# Apply global rcParams for consistent, publication-quality appearance.
plt.rcParams.update({
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "pdf.fonttype":      42,    # embed fonts in PDF (required for ACL/NeurIPS)
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_run_csv(run_dir: str, relative_path: str) -> Optional[pd.DataFrame]:
    """Load ``{run_dir}/{relative_path}`` as a DataFrame.

    Returns ``None`` (with a warning to stderr) when the file is absent or
    cannot be parsed.  Callers treat ``None`` as "no data for this run".
    """
    path = Path(run_dir) / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def infer_task_boundaries(run_dir: str) -> List[int]:
    """Return the global step at the start of each task.

    Reads ``results/task_curves/task_{:02d}_train.csv`` files and extracts the
    minimum step per task, sorted by task id.  Returns ``[]`` if the directory
    is absent.

    Used to draw vertical boundary lines in time-series plots.
    """
    curves_dir = Path(run_dir) / "results" / "task_curves"
    if not curves_dir.is_dir():
        return []

    boundaries: List[int] = []
    for csv_path in sorted(curves_dir.glob("task_*_train.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=["step"])
            if not df.empty:
                boundaries.append(int(df["step"].min()))
        except Exception:
            pass
    return sorted(set(boundaries))


def pick_median_run(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the row whose ACC is closest to the group's median ACC.

    Skips rows with NaN ACC.  Returns ``None`` when no valid row exists.
    """
    valid = df.dropna(subset=["ACC"])
    if valid.empty:
        return None
    median_acc = valid["ACC"].median()
    idx = (valid["ACC"] - median_acc).abs().idxmin()
    return valid.loc[idx]


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    """Save *fig* as PDF + PNG in *outdir*, then close it."""
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _add_task_boundaries(ax: plt.Axes, boundaries: List[int]) -> None:
    """Draw vertical dotted gray lines at each task boundary (except task 0)."""
    for step in boundaries[1:]:  # skip task-0 start at step 0
        ax.axvline(step, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)


# ---------------------------------------------------------------------------
# Figure 1: Accuracy matrix heatmap
# ---------------------------------------------------------------------------

def fig_accuracy_matrix(index: pd.DataFrame, outdir: Path) -> List[str]:
    """R[i,j] heatmap for the median-accuracy seed of each method.

    For each unique method in the index, selects the run whose ACC is closest
    to the per-method median, then loads ``results/accuracy_matrix.npy``.
    Methods are arranged as side-by-side subplots in a full-width (6.75 in) figure.

    Returns list of run_ids consumed (one per method subplot).
    """
    used_run_ids: List[str] = []

    df = index[index["status"].eq("completed")].copy()
    if df.empty:
        print("WARNING: fig_accuracy_matrix — no completed runs. Skipping.")
        return used_run_ids

    methods = sorted(df["method"].dropna().unique())
    # Collect (method, row, R) triples for methods that have a .npy file.
    matrices: List[Tuple[str, np.ndarray]] = []
    for method in methods:
        row = pick_median_run(df[df["method"] == method])
        if row is None:
            continue
        mat_path = Path(row["run_dir"]) / "results" / "accuracy_matrix.npy"
        if not mat_path.exists():
            continue
        try:
            R = np.load(mat_path)
        except Exception as exc:
            warnings.warn(f"Could not load {mat_path}: {exc}")
            continue
        matrices.append((method, R))
        used_run_ids.append(str(row["run_id"]))

    if not matrices:
        print("WARNING: fig_accuracy_matrix — no accuracy_matrix.npy files found. Skipping.")
        return used_run_ids

    n = len(matrices)
    # Shared vmin/vmax so heatmaps are colour-comparable across methods.
    vmin = min(float(R.min()) for _, R in matrices)
    vmax = max(float(R.max()) for _, R in matrices)

    fig, axes = plt.subplots(
        1, n,
        figsize=(min(FULL_COL, n * 1.6), FIG_H + 0.4),
        squeeze=False,
    )

    for ax, (method, R) in zip(axes[0], matrices):
        im = ax.imshow(
            R,
            vmin=vmin, vmax=vmax,
            cmap="Blues",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Task j (trained up to)")
        ax.set_ylabel("Task i (evaluated)")
        num_tasks = R.shape[0]
        ticks = list(range(num_tasks))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=5)
        ax.set_yticklabels(ticks, fontsize=5)

    # One shared colorbar attached to the last subplot.
    fig.colorbar(im, ax=axes[0][-1], fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_accuracy_matrix")
    print(f"Wrote fig_accuracy_matrix  [run_ids: {used_run_ids}]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 2: Cone angle sweep
# ---------------------------------------------------------------------------

def fig_cone_sweep(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Grouped bar chart: ACC & FORG vs cone angle α (ablation A1_cone_sweep).

    One subplot per dataset.  x-axis: α in degrees; grouped bars: ACC (blue)
    and FORG (red) with error bars from all seeds.
    Single-column width (3.25 in) per dataset panel.

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[
        index["ablation_key"].eq("A1_cone_sweep") &
        index["status"].eq("completed")
    ].copy()

    if df.empty:
        print("WARNING: fig_cone_sweep — no A1_cone_sweep runs found. Skipping.")
        return used_run_ids

    # Parse numeric alpha from "alpha_45" → 45
    def _parse_alpha(v: Any) -> Optional[float]:
        try:
            return float(str(v).replace("alpha_", ""))
        except (ValueError, AttributeError):
            return None

    df["alpha_deg"] = df["ablation_value"].apply(_parse_alpha)
    df = df.dropna(subset=["alpha_deg"])
    df["alpha_deg"] = df["alpha_deg"].astype(int)
    used_run_ids = df["run_id"].dropna().tolist()

    datasets = sorted(df["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(ONE_COL * max(len(datasets), 1), FIG_H),
        squeeze=False,
    )

    for ax, ds in zip(axes[0], datasets):
        ds_df = df[df["dataset"] == ds]
        alphas = sorted(ds_df["alpha_deg"].unique())
        x = np.arange(len(alphas))
        width = 0.35

        acc_means, acc_stds, forg_means, forg_stds = [], [], [], []
        for alpha in alphas:
            subset = ds_df[ds_df["alpha_deg"] == alpha]
            acc_vals  = pd.to_numeric(subset["ACC"],  errors="coerce").dropna()
            forg_vals = pd.to_numeric(subset["FORG"], errors="coerce").dropna()
            acc_means.append(float(acc_vals.mean()) if len(acc_vals) else float("nan"))
            acc_stds.append(float(acc_vals.std(ddof=1)) if len(acc_vals) > 1 else 0.0)
            forg_means.append(float(forg_vals.mean()) if len(forg_vals) else float("nan"))
            forg_stds.append(float(forg_vals.std(ddof=1)) if len(forg_vals) > 1 else 0.0)

        ax.bar(
            x - width / 2, acc_means, width,
            yerr=acc_stds, capsize=2,
            label="ACC", color=PALETTE[0], alpha=0.85,
        )
        ax.bar(
            x + width / 2, forg_means, width,
            yerr=forg_stds, capsize=2,
            label="FORG", color=PALETTE[1], alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}°" for a in alphas])
        ax.set_xlabel("Cone angle α")
        ax.set_ylabel("Metric value")
        ax.set_title(ds)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_cone_sweep")
    n = len(used_run_ids)
    print(f"Wrote fig_cone_sweep  [{n} run(s), e.g. {used_run_ids[:2]}]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 3: Ablation summary
# ---------------------------------------------------------------------------

def fig_ablation_summary(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Grouped bar chart: ACC for every ablation (A1–A10) and baseline on both datasets.

    Full-width figure (6.75 in).  x-axis: ablation group; grouped bars by dataset.
    Baselines are placed first (left) for easy comparison.

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[index["status"].eq("completed")].copy()
    if df.empty:
        print("WARNING: fig_ablation_summary — no completed runs. Skipping.")
        return used_run_ids

    used_run_ids = df["run_id"].dropna().tolist()

    # Build a concise group label for each (method, ablation_key, ablation_value).
    def _group_label(row: pd.Series) -> str:
        ak  = row.get("ablation_key")  or ""
        av  = row.get("ablation_value") or ""
        m   = row.get("method")         or ""
        if ak == "baseline":
            return METHOD_LABELS.get(str(m), str(m))
        if ak:
            return f"{ak}\n{av}" if av else str(ak)
        return METHOD_LABELS.get(str(m), str(m))

    df["group_label"] = df.apply(_group_label, axis=1)

    # Mean ACC per (group_label, dataset)
    agg = (
        df.groupby(["group_label", "dataset"])["ACC"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    # Ordering: baseline labels first, then ablations alphabetically
    baseline_labels = set(
        df[df["ablation_key"] == "baseline"]["group_label"].unique()
    )
    all_labels = sorted(agg["group_label"].unique())
    ordered_labels = sorted(baseline_labels) + [
        lbl for lbl in all_labels if lbl not in baseline_labels
    ]

    datasets = sorted(df["dataset"].dropna().unique())
    x = np.arange(len(ordered_labels))
    width = 0.8 / max(len(datasets), 1)

    fig, ax = plt.subplots(figsize=(FULL_COL, FIG_H + 0.8))

    for i, ds in enumerate(datasets):
        ds_agg = agg[agg["dataset"] == ds].set_index("group_label")
        means = [
            float(ds_agg.loc[lbl, "mean"]) if lbl in ds_agg.index else float("nan")
            for lbl in ordered_labels
        ]
        stds = [
            float(ds_agg.loc[lbl, "std"]) if lbl in ds_agg.index else 0.0
            for lbl in ordered_labels
        ]
        offset = (i - (len(datasets) - 1) / 2) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=2,
            label=ds, color=PALETTE[i % len(PALETTE)], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("ACC")
    ax.set_title("Ablation summary (all ablations and baselines)")
    ax.legend(title="Dataset")
    fig.tight_layout()

    # Source run_ids: used_run_ids (all completed runs — see figure_provenance.json)
    save_figure(fig, outdir, "fig_ablation_summary")
    print(f"Wrote fig_ablation_summary  [{len(used_run_ids)} run(s)]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 4: Trust radius evolution
# ---------------------------------------------------------------------------

def fig_trust_radius(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Trust-radius evolution over all training steps for CACL variants.

    Compares:
    - A2 variants (trust_region.enabled = true vs false)
    - A4 variants (hessian target = replay vs joint)

    When trust is disabled (tr_false), there is no radius file, so that
    variant is omitted from the line plot.
    Vertical dotted lines mark task boundaries.
    Full-width figure (6.75 in).

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[
        index["method"].eq("cacl") &
        index["status"].eq("completed")
    ].copy()

    if df.empty:
        print("WARNING: fig_trust_radius — no completed CACL runs. Skipping.")
        return used_run_ids

    # Variant groups of interest (ablation_key → list of ablation_values to plot)
    variant_groups = [
        ("A2_trust_region",   ["tr_true", "tr_false"]),
        ("A4_hessian_target", ["hessian_replay", "hessian_joint"]),
    ]

    datasets = sorted(df["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(FULL_COL, FIG_H),
        squeeze=False,
    )

    for ax, ds in zip(axes[0], datasets):
        ds_df = df[df["dataset"] == ds]
        color_idx = 0
        plotted_any = False
        last_boundary_run_dir: Optional[str] = None

        for ablation_key, ablation_values in variant_groups:
            ak_df = ds_df[ds_df["ablation_key"].eq(ablation_key)]
            if ak_df.empty:
                continue

            for av in ablation_values:
                av_df = ak_df[ak_df["ablation_value"].eq(av)]
                if av_df.empty:
                    continue
                row = pick_median_run(av_df)
                if row is None:
                    continue

                trust_df = load_run_csv(
                    row["run_dir"],
                    "results/diagnostics/trust_radius_history.csv",
                )
                if trust_df is None or trust_df.empty or "radius" not in trust_df.columns:
                    color_idx += 1
                    continue

                # Sub-sample to ≤ 2 000 points for legibility
                if len(trust_df) > 2000:
                    step = max(1, len(trust_df) // 2000)
                    trust_df = trust_df.iloc[::step]

                used_run_ids.append(str(row["run_id"]))
                last_boundary_run_dir = row["run_dir"]
                color = PALETTE[color_idx % len(PALETTE)]
                color_idx += 1

                ax.plot(
                    trust_df["step"],
                    trust_df["radius"],
                    label=f"{ablation_key}: {av}",
                    color=color,
                    linewidth=0.8,
                    alpha=0.85,
                )
                plotted_any = True

        # Vertical task boundary lines from the last run that had trust data
        if last_boundary_run_dir is not None:
            _add_task_boundaries(ax, infer_task_boundaries(last_boundary_run_dir))

        ax.set_xlabel("Global step")
        ax.set_ylabel("Trust radius")
        ax.set_title(ds)
        if plotted_any:
            ax.legend(fontsize=6)
        else:
            ax.text(
                0.5, 0.5, "No trust-radius data",
                transform=ax.transAxes, ha="center", va="center",
            )

    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_trust_radius")
    print(f"Wrote fig_trust_radius  [run_ids: {used_run_ids}]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 5: Eigenvalue spectrum evolution
# ---------------------------------------------------------------------------

def fig_eigenvalue_evolution(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Top-5 Hessian eigenvalues over all training steps (per dataset).

    For each dataset, selects the CACL run with median ACC and loads
    ``results/diagnostics/eigenvalue_spectrum.csv``.  Only rows recorded on
    Lanczos-recompute steps are present; the plot is automatically sparse.
    Vertical dotted lines mark task boundaries.
    Full-width figure (6.75 in).

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[
        index["method"].eq("cacl") &
        index["status"].eq("completed")
    ].copy()

    if df.empty:
        print("WARNING: fig_eigenvalue_evolution — no completed CACL runs. Skipping.")
        return used_run_ids

    datasets = sorted(df["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(FULL_COL, FIG_H),
        squeeze=False,
    )

    for ax, ds in zip(axes[0], datasets):
        ds_df = df[df["dataset"] == ds]
        row = pick_median_run(ds_df)
        if row is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(ds)
            continue

        eig_df = load_run_csv(
            row["run_dir"],
            "results/diagnostics/eigenvalue_spectrum.csv",
        )
        if eig_df is None or eig_df.empty:
            ax.text(
                0.5, 0.5, "No eigenvalue data",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_title(ds)
            continue

        used_run_ids.append(str(row["run_id"]))

        # Eigenvalue columns are named lambda_1, lambda_2, ... (arbitrary count)
        eig_cols = sorted(
            [c for c in eig_df.columns if c.startswith("lambda_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        top_k = eig_cols[:5]   # at most top-5 eigenvalues

        for i, col in enumerate(top_k):
            vals = pd.to_numeric(eig_df[col], errors="coerce")
            ax.plot(
                eig_df["step"],
                vals,
                label=f"λ{i + 1}",
                color=PALETTE[i % len(PALETTE)],
                linewidth=0.8,
                marker=".",
                markersize=2,
                alpha=0.85,
            )

        _add_task_boundaries(ax, infer_task_boundaries(row["run_dir"]))

        ax.set_xlabel("Global step")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(ds)
        ax.legend(ncol=2, fontsize=6)

    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_eigenvalue_evolution")
    print(f"Wrote fig_eigenvalue_evolution  [run_ids: {used_run_ids}]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 6: Stability gap comparison
# ---------------------------------------------------------------------------

def fig_stability_gap(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Per-step accuracy curves at a task boundary for all methods.

    Uses the transition into task 4 (0-indexed; i.e. the 5th task) as the
    primary target.  Falls back to task 9 → task 3 → task 2 → task 1 →
    any available gap CSV, in that order.

    For each method, plots the *mean* accuracy over all previously seen tasks
    (averaged column-wise across the stability gap CSV).  One line per method.
    Single-column width per dataset panel.

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[index["status"].eq("completed")].copy()
    if df.empty:
        print("WARNING: fig_stability_gap — no completed runs. Skipping.")
        return used_run_ids

    datasets = sorted(df["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(ONE_COL * max(len(datasets), 1), FIG_H),
        squeeze=False,
    )

    # Candidate task_ids in preference order
    CANDIDATE_TASK_IDS = [4, 9, 3, 2, 1]

    for ax, ds in zip(axes[0], datasets):
        ds_df = df[df["dataset"] == ds]
        methods = sorted(ds_df["method"].dropna().unique())

        for i, method in enumerate(methods):
            row = pick_median_run(ds_df[ds_df["method"] == method])
            if row is None:
                continue

            gap_dir = Path(row["run_dir"]) / "results" / "stability_gap"
            if not gap_dir.is_dir():
                continue

            # Try preferred task_ids first, then any available file
            gap_df: Optional[pd.DataFrame] = None
            for tid in CANDIDATE_TASK_IDS:
                p = gap_dir / f"task_{tid:02d}_gap.csv"
                if p.exists():
                    try:
                        gap_df = pd.read_csv(p)
                        if not gap_df.empty:
                            break
                    except Exception:
                        gap_df = None

            if gap_df is None:
                for p in sorted(gap_dir.glob("task_*_gap.csv")):
                    try:
                        gap_df = pd.read_csv(p)
                        if not gap_df.empty:
                            break
                    except Exception:
                        gap_df = None

            if gap_df is None or gap_df.empty:
                continue

            # Identify per-task accuracy columns (task_0_acc, task_1_acc, ...)
            acc_cols = [c for c in gap_df.columns if c.endswith("_acc")]
            if not acc_cols:
                continue

            used_run_ids.append(str(row["run_id"]))
            step_col = (
                gap_df["step_within_task"]
                if "step_within_task" in gap_df.columns
                else pd.Series(range(len(gap_df)), name="step")
            )
            mean_acc = gap_df[acc_cols].mean(axis=1)

            ax.plot(
                step_col,
                mean_acc,
                label=METHOD_LABELS.get(method, method),
                color=PALETTE[i % len(PALETTE)],
                linewidth=1.0,
                alpha=0.85,
            )

        ax.set_xlabel("Step within task")
        ax.set_ylabel("Mean acc. (prev. tasks)")
        ax.set_title(ds)
        ax.set_ylim(bottom=0)
        ax.legend()

    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_stability_gap")
    print(f"Wrote fig_stability_gap  [run_ids: {used_run_ids}]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 7: Computational cost
# ---------------------------------------------------------------------------

def fig_cost(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Bar chart: mean wall-clock seconds per task for each method.

    Includes:
    - Baseline methods (ER, GEM-joint, GEM-lasttask, NCL) from ablation_key=baseline.
    - CACL amortization variants (K=1..50) from ablation_key=A7_amortize_K.

    Bar heights are mean total_seconds / task across seeds.
    Full-width figure (6.75 in).

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[index["status"].eq("completed")].copy()
    if df.empty:
        print("WARNING: fig_cost — no completed runs. Skipping.")
        return used_run_ids

    datasets = sorted(df["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(FULL_COL, FIG_H),
        squeeze=False,
    )

    for ax, ds in zip(axes[0], datasets):
        ds_df = df[df["dataset"] == ds]
        # Collect (label, seconds_per_task) pairs across seeds
        entries: List[Dict[str, Any]] = []

        # --- Baseline methods ---
        baselines = ds_df[ds_df["ablation_key"].eq("baseline")]
        for method in sorted(baselines["method"].dropna().unique()):
            for _, row in baselines[baselines["method"] == method].iterrows():
                wc = load_run_csv(row["run_dir"], "results/timing/wall_clock.csv")
                if wc is not None and not wc.empty and "total_seconds" in wc.columns:
                    used_run_ids.append(str(row["run_id"]))
                    entries.append({
                        "label": METHOD_LABELS.get(method, method),
                        "spt":   float(wc["total_seconds"].mean()),
                    })

        # --- CACL amortization variants (A7) ---
        a7 = ds_df[ds_df["ablation_key"].eq("A7_amortize_K")]
        for av in sorted(a7["ablation_value"].dropna().unique()):
            for _, row in a7[a7["ablation_value"] == av].iterrows():
                wc = load_run_csv(row["run_dir"], "results/timing/wall_clock.csv")
                if wc is not None and not wc.empty and "total_seconds" in wc.columns:
                    used_run_ids.append(str(row["run_id"]))
                    entries.append({
                        "label": f"CACL {av}",
                        "spt":   float(wc["total_seconds"].mean()),
                    })

        if not entries:
            ax.text(
                0.5, 0.5, "No timing data",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_title(ds)
            continue

        # Aggregate mean ± std per label (over seeds)
        entries_df = pd.DataFrame(entries)
        agg = (
            entries_df.groupby("label")["spt"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        agg["std"] = agg["std"].fillna(0.0)

        labels = agg["label"].tolist()
        means  = agg["mean"].tolist()
        stds   = agg["std"].tolist()
        x = np.arange(len(labels))

        ax.bar(
            x, means,
            yerr=stds, capsize=2,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Seconds / task (mean)")
        ax.set_title(ds)

    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_cost")
    print(f"Wrote fig_cost  [{len(used_run_ids)} run(s)]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 8: Memory budget sensitivity
# ---------------------------------------------------------------------------

def fig_memory_budget(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Line plot: ACC vs memory budget for CACL (A10) and baselines.

    CACL data comes from A10_memory_budget ablation runs (budgets 100–2000).
    ER and GEM-joint baselines are shown as horizontal dashed reference lines
    at their default budget of 500 samples.
    Single-column width per dataset panel.

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[index["status"].eq("completed")].copy()
    if df.empty:
        print("WARNING: fig_memory_budget — no completed runs. Skipping.")
        return used_run_ids

    # Parse budget from "budget_500" → 500.0
    def _parse_budget(v: Any) -> Optional[float]:
        try:
            return float(str(v).replace("budget_", ""))
        except (ValueError, AttributeError):
            return None

    a10 = df[df["ablation_key"].eq("A10_memory_budget")].copy()
    a10["memory_budget"] = a10["ablation_value"].apply(_parse_budget)
    a10 = a10.dropna(subset=["memory_budget"])

    # Baseline methods for reference lines (default budget = 500)
    baselines = df[
        df["ablation_key"].eq("baseline") &
        df["method"].isin(["er", "agem"])
    ].copy()
    baselines["memory_budget"] = 500.0

    datasets = sorted(
        set(a10["dataset"].dropna().tolist()) |
        set(baselines["dataset"].dropna().tolist())
    )

    if not datasets:
        print("WARNING: fig_memory_budget — no A10 or baseline runs found. Skipping.")
        return used_run_ids

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(ONE_COL * max(len(datasets), 1), FIG_H),
        squeeze=False,
    )

    for ax, ds in zip(axes[0], datasets):
        color_idx = 0

        # CACL sweep line
        cacl_ds = a10[a10["dataset"] == ds]
        if not cacl_ds.empty:
            used_run_ids.extend(cacl_ds["run_id"].dropna().tolist())
            agg = (
                cacl_ds.groupby("memory_budget")["ACC"]
                .agg(mean="mean", std="std")
                .reset_index()
                .sort_values("memory_budget")
            )
            ax.errorbar(
                agg["memory_budget"],
                agg["mean"],
                yerr=agg["std"].fillna(0),
                label="CACL",
                color=PALETTE[color_idx],
                marker="o",
                markersize=3,
                linewidth=1.0,
                capsize=2,
            )
            color_idx += 1

        # Baseline reference lines (single horizontal line at default budget)
        for method in ["er", "agem"]:
            b_ds = baselines[
                (baselines["dataset"] == ds) & (baselines["method"] == method)
            ]
            if b_ds.empty:
                continue
            used_run_ids.extend(b_ds["run_id"].dropna().tolist())
            mean_acc = float(b_ds["ACC"].dropna().mean())
            ax.axhline(
                mean_acc,
                label=f"{METHOD_LABELS.get(method, method)} (budget=500)",
                color=PALETTE[color_idx],
                linewidth=1.0,
                linestyle="--",
            )
            color_idx += 1

        ax.set_xlabel("Memory budget (samples)")
        ax.set_ylabel("ACC")
        ax.set_title(ds)
        ax.legend(fontsize=6)

    fig.tight_layout()

    # Source run_ids: used_run_ids (see figure_provenance.json)
    save_figure(fig, outdir, "fig_memory_budget")
    n = len(used_run_ids)
    print(f"Wrote fig_memory_budget  [{n} run(s)]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Figure 9: Continuous accuracy curves
# ---------------------------------------------------------------------------

def fig_accuracy_curves(index: pd.DataFrame, outdir: Path) -> List[str]:
    """Continuous per-task accuracy over all training steps, one subplot per method.

    Reads ``results/accuracy_curves.csv`` (written by train.py at every
    ``eval_every_n_steps`` step, evaluating all tasks seen so far).  This gives
    a single unbroken curve for each task spanning the full training run —
    including the stability gap period at task boundaries where old-task accuracy
    temporarily drops before recovering.

    Vertical dotted lines mark task boundaries.
    Full-width figure (6.75 in).

    Returns list of run_ids consumed.
    """
    used_run_ids: List[str] = []

    df = index[index["status"].eq("completed")].copy()
    if df.empty:
        print("WARNING: fig_accuracy_curves — no completed runs. Skipping.")
        return used_run_ids

    methods = sorted(df["method"].dropna().unique())
    if not methods:
        return used_run_ids

    n = len(methods)
    fig, axes = plt.subplots(
        1, n,
        figsize=(ONE_COL * n, FIG_H),
        squeeze=False,
        sharey=True,
    )

    for ax, method in zip(axes[0], methods):
        row = pick_median_run(df[df["method"] == method])
        if row is None:
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        acc_df = load_run_csv(row["run_dir"], "results/accuracy_curves.csv")
        if acc_df is None or acc_df.empty or "step" not in acc_df.columns:
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.text(
                0.5, 0.5, "No accuracy_curves.csv",
                transform=ax.transAxes, ha="center", va="center",
            )
            continue

        used_run_ids.append(str(row["run_id"]))

        acc_cols = sorted(
            [c for c in acc_df.columns if c.endswith("_acc")],
            key=lambda c: int(c.split("_")[1]),
        )
        for i, col in enumerate(acc_cols):
            task_label = f"Task {col.split('_')[1]}"
            vals = pd.to_numeric(acc_df[col], errors="coerce")
            ax.plot(
                acc_df["step"],
                vals,
                label=task_label,
                color=PALETTE[i % len(PALETTE)],
                linewidth=0.9,
                alpha=0.85,
            )

        _add_task_boundaries(ax, infer_task_boundaries(row["run_dir"]))
        ax.set_xlabel("Global step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.legend(fontsize=6)

    fig.tight_layout()
    save_figure(fig, outdir, "fig_accuracy_curves")
    print(f"Wrote fig_accuracy_curves  [run_ids: {used_run_ids}]")
    return used_run_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from master_index.csv."
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path to master_index.csv produced by aggregate_results.py.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for PDF/PNG figures and figure_provenance.json.",
    )
    args = parser.parse_args()

    index_path: Path = args.index.resolve()
    outdir: Path     = args.outdir.resolve()

    if not index_path.exists():
        print(f"ERROR: --index '{index_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    index = pd.read_csv(index_path)
    if index.empty:
        print("WARNING: master_index.csv is empty. No figures will be generated.")
        sys.exit(0)

    print(f"Loaded {len(index)} run(s) from {index_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Run all 8 figure generators.  Each is fully independent: missing data
    # yields a warning and an empty run_id list, not a crash.
    provenance: Dict[str, List[str]] = {}

    provenance["fig_accuracy_matrix"]      = fig_accuracy_matrix(index, outdir)
    provenance["fig_accuracy_curves"]      = fig_accuracy_curves(index, outdir)
    provenance["fig_cone_sweep"]           = fig_cone_sweep(index, outdir)
    provenance["fig_ablation_summary"]     = fig_ablation_summary(index, outdir)
    provenance["fig_trust_radius"]         = fig_trust_radius(index, outdir)
    provenance["fig_eigenvalue_evolution"] = fig_eigenvalue_evolution(index, outdir)
    provenance["fig_stability_gap"]        = fig_stability_gap(index, outdir)
    provenance["fig_cost"]                 = fig_cost(index, outdir)
    provenance["fig_memory_budget"]        = fig_memory_budget(index, outdir)

    # Write figure→run_id mapping (traceability anchor for paper numbers).
    prov_path = outdir / "figure_provenance.json"
    with open(prov_path, "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"\nWrote figure provenance: {prov_path}")

    generated = [name for name, ids in provenance.items() if ids]
    skipped   = [name for name, ids in provenance.items() if not ids]

    print(f"Generated {len(generated)} figure(s): {generated}")
    if skipped:
        print(f"Skipped   {len(skipped)} (no data): {skipped}")


if __name__ == "__main__":
    main()
