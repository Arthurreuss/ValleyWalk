"""Aggregate run_manifest.json files into a master index and per-dataset tables.

Usage
-----
    python scripts/aggregate_results.py \\
        --run-dir outputs/ \\
        [--outdir outputs/] \\
        [--seeds 1,2,3,4,5]

Produces
--------
<outdir>/master_index.csv
    One row per run.  Columns: run_id, method, dataset, ablation_key,
    ablation_value, seed, ACC, FORG, min_ACC, WF10, WF100, WP10, WP100,
    WC_ACC, stab_gap_max_drop, stab_gap_depth, stab_gap_area,
    stab_gap_area_end, stab_gap_recovery_steps, true_grad_cosine_mean,
    true_grad_cosine_min,
    true_grad_mag_ratio_mean, wall_clock_total, run_dir, wandb_run_url,
    git_commit, status.  This is the single source of truth for all
    downstream analysis and figure generation.

<outdir>/summary_tables/<dataset>_summary.csv
<outdir>/summary_tables/<dataset>_summary.tex
    For each dataset found in master_index.csv: a table whose rows are
    (method, ablation_key, ablation_value) groups and whose columns are
    mean±std of each metric over seeds.  The .tex file is a self-contained
    LaTeX table environment (requires ``\\usepackage{booktabs}`` in the
    parent document).

<outdir>/missing_runs.csv
    Any (method, dataset, ablation_key, ablation_value, seed) combination
    that is missing or has status != "completed".  Written only when
    problems are found; a warning is also printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

MASTER_COLUMNS: List[str] = [
    "run_id",
    "method",
    "dataset",
    "ablation_key",
    "ablation_value",
    "seed",
    "ACC",
    "FORG",
    "min_ACC",
    "WF10",
    "WF100",
    "WP10",
    "WP100",
    "WC_ACC",
    "stab_gap_max_drop",
    "stab_gap_depth",
    "stab_gap_area",
    "stab_gap_area_end",
    "stab_gap_recovery_steps",
    "true_grad_cosine_mean",
    "true_grad_cosine_min",
    "true_grad_mag_ratio_mean",
    "wall_clock_total",
    "run_dir",
    "wandb_run_url",
    "git_commit",
    "status",
]

METRIC_COLUMNS: List[str] = [
    "ACC",
    "FORG",
    "min_ACC",
    "WF10",
    "WF100",
    "WP10",
    "WP100",
    "WC_ACC",
    "stab_gap_max_drop",
    "stab_gap_depth",
    "stab_gap_area",
    "stab_gap_area_end",
    "stab_gap_recovery_steps",
    "true_grad_cosine_mean",
    "true_grad_cosine_min",
    "true_grad_mag_ratio_mean",
]

GROUP_COLUMNS: List[str] = ["method", "ablation_key", "ablation_value"]

# Per-ablation_key overrides of the expected seed set.  Most blocks run the
# full default seed set, but a few are deliberately run at fewer seeds — the
# completeness check uses these overrides so those groups are not flagged as
# missing the seeds they were never meant to have.
#
#   cifar_generalization — run at 3 seeds (SEEDS_GEN="1,2,3" in
#                          scripts/run_G.sh), not the full headline set.
EXPECTED_SEEDS_BY_KEY: Dict[str, List[int]] = {
    "cifar_generalization": [1, 2, 3],
}

_METRIC_LABELS: Dict[str, str] = {
    "ACC": "ACC",
    "FORG": "FORG",
    "min_ACC": "min-ACC",
    "WF10": "WF$_{10}$",
    "WF100": "WF$_{100}$",
    "WP10": "WP$_{10}$",
    "WP100": "WP$_{100}$",
    "WC_ACC": "WC-ACC",
    "stab_gap_max_drop": "$\\Delta_{\\max}$",
    "stab_gap_depth": "$G_{\\text{depth}}$",
    "stab_gap_area": "$G_{\\text{area}}$",
    "stab_gap_area_end": "$G_{\\text{area}}^{\\text{end}}$",
    "stab_gap_recovery_steps": "Recov.",
    "true_grad_cosine_mean": "$\\bar{\\cos}$",
    "true_grad_cosine_min": "$\\cos_{\\min}$",
    "true_grad_mag_ratio_mean": "$\\bar{r}$",
}


# Directories under --run-dir that hold runs which are *not* experiments and so
# must not become rows in the master index.
#
#   S_anchors — the S-series' shared task-0 checkpoints (scripts/run_S.sh,
#               phase 0).  One single-task run per (momentum, seed), whose only
#               product is theta*_0 for the ladder cells to warm-start from.
#               They measure no stability gap (num_tasks=1) and carry no
#               ablation_key, so they would add ten all-"---" rows to the
#               rot_mnist summary table under an empty ablation group.
EXCLUDED_DIR_NAMES = {"S_anchors"}


def find_manifests(run_dir: Path) -> List[Path]:
    """Recursively find all ``run_manifest.json`` files under *run_dir*.

    Paths passing through a directory in :data:`EXCLUDED_DIR_NAMES` are
    skipped.  Results are sorted for deterministic ordering.
    """
    return sorted(
        p for p in run_dir.rglob("run_manifest.json")
        if not EXCLUDED_DIR_NAMES.intersection(p.relative_to(run_dir).parts)
    )


def parse_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Parse a single ``run_manifest.json`` into a flat dict matching MASTER_COLUMNS.

    Missing fields produce ``None`` rather than raising — so a partial or
    malformed manifest contributes a row with NaN metrics rather than
    crashing the aggregation.
    """
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            "WARNING: could not parse " + str(manifest_path) + ": " + str(exc),
            file=sys.stderr,
        )
        row = {col: None for col in MASTER_COLUMNS}
        row["status"] = "parse_error"
        row["run_dir"] = str(manifest_path.parent)
        return row

    final = m.get("final_metrics") or {}
    return {
        "run_id": m.get("run_id"),
        "method": m.get("method"),
        "dataset": m.get("dataset"),
        "ablation_key": m.get("ablation_key"),
        "ablation_value": m.get("ablation_value"),
        "seed": m.get("seed"),
        "ACC": final.get("ACC"),
        "FORG": final.get("FORG"),
        "min_ACC": final.get("min_ACC"),
        "WF10": final.get("WF10"),
        "WF100": final.get("WF100"),
        "WP10": final.get("WP10"),
        "WP100": final.get("WP100"),
        "WC_ACC": final.get("WC_ACC"),
        "stab_gap_max_drop": final.get("stability_gap_max_drop"),
        "stab_gap_depth": final.get("stability_gap_depth"),
        "stab_gap_area": final.get("stability_gap_area"),
        "stab_gap_area_end": final.get("stability_gap_area_end"),
        "stab_gap_recovery_steps": final.get("stability_gap_recovery_steps"),
        "true_grad_cosine_mean": final.get("true_grad_cosine_mean"),
        "true_grad_cosine_min": final.get("true_grad_cosine_min"),
        "true_grad_mag_ratio_mean": final.get("true_grad_mag_ratio_mean"),
        "wall_clock_total": m.get("wall_clock_total_seconds"),
        "wandb_run_url": m.get("wandb_run_url"),
        "git_commit": m.get("git_commit"),
        "status": m.get("status", "unknown"),
        "run_dir": str(manifest_path.parent),
    }


def compute_area_end_from_curve(curve_path: Path) -> Optional[float]:
    """Recompute ``stability_gap_area_end`` from a run's ``accuracy_curves.csv``.

    Older runs predate the ``stability_gap_area_end`` metric and so leave it
    absent from their manifest.  The value reported at training time is
    ``StabilityGapTracker.gap_area(reference="end")`` for the *final* task
    transition (``scripts/train.py`` re-instantiates the tracker per task and
    reports the last one).  Because ``gap_area`` depends only on step
    *differences*, it can be reproduced from the logged accuracy curve:

      * the per-task pre-switch baseline ``b_j`` is the last recorded accuracy
        of task ``j`` before the final task begins — the same model state the
        live tracker evaluates at construction;
      * the per-step drop ``max(0, min(b_j, end_mean_j) - acc)`` is integrated
        over ``step`` by the trapezoidal rule and summed across prior tasks,
        where ``end_mean_j`` is the trailing-window mean (window
        ``max(5, ceil(0.1 n))``) — matching :meth:`StabilityGapTracker.gap_area`
        with ``reference="end"``.

    Returns ``None`` (leaving the value absent rather than filled with a bogus
    number) when the curve is missing, has only a single task (no previous-task
    gap), lacks a pre-switch baseline, or has non-monotonic steps — the latter
    a tell-tale of a corrupted/interleaved CSV.
    """
    if not curve_path.is_file():
        return None
    try:
        with open(curve_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None

    task_ids = sorted(
        int(c.split("_")[1])
        for c in rows[0]
        if c.startswith("task_") and c.endswith("_acc")
    )
    if not task_ids or task_ids[-1] == 0:
        return None  # single task -> no previous-task stability gap
    last = task_ids[-1]
    last_col = f"task_{last}_acc"

    # The final task starts at the first row where its accuracy is populated;
    # the row before that holds the pre-switch baseline.
    start = next(
        (i for i, r in enumerate(rows) if r.get(last_col) not in (None, "")),
        None,
    )
    if not start:  # None, or 0 (no preceding row to read a baseline from)
        return None

    try:
        steps = [int(r["step"]) for r in rows[start:]]
    except (KeyError, ValueError):
        return None
    if any(b < a for a, b in zip(steps, steps[1:])):
        return None  # corrupted CSV (interleaved writes)

    base_row = rows[start - 1]
    pre: Dict[int, float] = {}
    for j in range(last):
        v = base_row.get(f"task_{j}_acc")
        if v not in (None, ""):
            pre[j] = float(v)

    total = 0.0
    for j, b_pre in pre.items():
        series = [
            (int(r["step"]), float(r[f"task_{j}_acc"]))
            for r in rows[start:]
            if r.get(f"task_{j}_acc") not in (None, "")
        ]
        if len(series) < 2:
            continue
        w = max(5, math.ceil(0.1 * len(series)))
        tail = series[-w:]
        end_mean = sum(a for _, a in tail) / len(tail)
        b_j = min(b_pre, end_mean)

        prev_step: Optional[int] = None
        prev_drop = 0.0
        for step, acc in series:
            drop = max(0.0, b_j - acc)
            if prev_step is not None:
                total += (step - prev_step) * 0.5 * (drop + prev_drop)
            prev_step = step
            prev_drop = drop
    return float(total)


def backfill_area_end(df: pd.DataFrame) -> int:
    """Fill missing ``stab_gap_area_end`` values in *df* from accuracy curves.

    For every row whose ``stab_gap_area_end`` is absent, recompute it from
    ``<run_dir>/results/accuracy_curves.csv`` via
    :func:`compute_area_end_from_curve`.  Modifies *df* in place and returns the
    number of rows backfilled.
    """
    if "stab_gap_area_end" not in df.columns:
        return 0
    backfilled = 0
    for idx in df.index[df["stab_gap_area_end"].isna()]:
        run_dir_str = df.at[idx, "run_dir"]
        if not run_dir_str or pd.isna(run_dir_str):
            continue
        curve = Path(run_dir_str) / "results" / "accuracy_curves.csv"
        val = compute_area_end_from_curve(curve)
        if val is not None:
            df.at[idx, "stab_gap_area_end"] = val
            backfilled += 1
    return backfilled


def _fmt_cell(mean: float, std: float) -> str:
    """Format a numeric cell as 'mean ± std' (3 decimal places).

    Returns '---' when *mean* is NaN.
    """
    if math.isnan(mean):
        return "---"
    return f"{mean:.3f} ± {std:.3f}"


def build_summary_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Build a mean±std summary DataFrame for *dataset*.

    Returns a DataFrame indexed by (method, ablation_key, ablation_value)
    whose columns are METRIC_COLUMNS with string 'mean ± std' cells.
    Returns an empty DataFrame if *dataset* is not present.
    """
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["ablation_key"] = subset["ablation_key"].fillna("")
    subset["ablation_value"] = subset["ablation_value"].fillna("")

    for col in METRIC_COLUMNS:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    rows = []
    keys = []
    for key, grp in subset.groupby(GROUP_COLUMNS):
        method, ablation_key, ablation_value = key
        row = {}
        for col in METRIC_COLUMNS:
            vals = grp[col].dropna()
            if len(vals) == 0:
                row[col] = "---"
            elif len(vals) == 1:
                row[col] = f"{float(vals.iloc[0]):.3f}"
            else:
                row[col] = _fmt_cell(float(vals.mean()), float(vals.std()))
        rows.append(row)
        keys.append(key)

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(keys, names=GROUP_COLUMNS),
        columns=METRIC_COLUMNS,
    )


def _tex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    for old, new in [("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")]:
        text = text.replace(old, new)
    return text


def write_latex_table(summary: pd.DataFrame, dataset: str, out_path: Path) -> None:
    """Write a compilable LaTeX ``table`` environment to *out_path*.

    The file is designed to be ``\\input{}``-ed into a paper that has::

        \\usepackage{booktabs}

    in its preamble.
    """
    header_cols = [_METRIC_LABELS.get(c, c) for c in METRIC_COLUMNS]
    n_metric_cols = len(METRIC_COLUMNS)

    lines = [
        "% Requires \\usepackage{booktabs} in document preamble.",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{Results on \\textsc{{{dataset}}} (mean$\\pm$std over seeds).}}",
        f"  \\label{{tab:{dataset.lower().replace('_', '-').replace(' ', '-')}}}",
        "  \\begin{tabular}{lll" + "r" * n_metric_cols + "}",
        "    \\toprule",
        "    Method & Ablation & Value & " + " & ".join(header_cols) + " \\\\",
        "    \\midrule",
    ]

    for (method, ablation_key, ablation_value), row in summary.iterrows():
        cells = [
            _tex_escape(str(method)),
            _tex_escape(str(ablation_key)),
            _tex_escape(str(ablation_value)),
        ] + [str(row[c]) for c in METRIC_COLUMNS]
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_completeness(
    df: pd.DataFrame,
    expected_seeds: List[int],
    expected_seeds_by_key: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """Return a DataFrame of missing or non-completed runs.

    Checks two kinds of problems:
    1. Runs that exist but have ``status != "completed"`` (e.g. "running",
       "failed", "unknown", "parse_error").
    2. Expected seeds that are absent for a given (method, dataset,
       ablation_key, ablation_value) combination.

    Parameters
    ----------
    df:
        The master index DataFrame.
    expected_seeds:
        Seeds that every (method, dataset, ablation) combo is expected to have,
        unless overridden for its ``ablation_key`` via *expected_seeds_by_key*.
    expected_seeds_by_key:
        Optional per-``ablation_key`` override of the expected seed set, for
        blocks deliberately run at fewer seeds (e.g. ``cifar_generalization``).
        Defaults to :data:`EXPECTED_SEEDS_BY_KEY`.

    Returns
    -------
    pd.DataFrame with columns: method, dataset, ablation_key, ablation_value,
    seed, issue.  Empty when everything is in order.
    """
    if expected_seeds_by_key is None:
        expected_seeds_by_key = EXPECTED_SEEDS_BY_KEY
    issues = []
    group_cols = ["method", "dataset", "ablation_key", "ablation_value"]

    for _, row in df.iterrows():
        if row["status"] != "completed":
            issues.append(
                {
                    "method": row["method"],
                    "dataset": row["dataset"],
                    "ablation_key": row["ablation_key"],
                    "ablation_value": row["ablation_value"],
                    "seed": row["seed"],
                    "issue": f"status={row['status']}",
                }
            )

    for key, grp in df.groupby(group_cols, dropna=False):
        method, dataset, ablation_key, ablation_value = key
        present_seeds = set(grp["seed"].dropna().apply(lambda x: int(x)).tolist())
        group_expected = expected_seeds_by_key.get(ablation_key, expected_seeds)
        for seed in group_expected:
            if seed not in present_seeds:
                issues.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "ablation_key": ablation_key,
                        "ablation_value": ablation_value,
                        "seed": seed,
                        "issue": "missing",
                    }
                )

    return pd.DataFrame(issues)


def main() -> None:
    """Aggregate run_manifest.json files into master_index.csv and per-dataset LaTeX/CSV summary tables."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--run-dir",
        metavar="DIR",
        type=Path,
        required=True,
        help="Root directory to search recursively for run_manifest.json files.",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        type=Path,
        default=None,
        help="Output directory for master_index.csv and summary_tables/. Defaults to --run-dir.",
    )
    parser.add_argument(
        "--seeds",
        metavar="N,N,...",
        default="1,2,3,4,5",
        help="Comma-separated expected seeds for the completeness check. Default: 1,2,3,4,5. "
        "Per-block overrides (e.g. cifar_generalization at 3 seeds) are applied automatically.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    outdir: Path = (args.outdir or run_dir).resolve()
    expected_seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if not run_dir.is_dir():
        print(f"ERROR: --run-dir '{run_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    manifest_paths = find_manifests(run_dir)
    if not manifest_paths:
        print(f"WARNING: no run_manifest.json files found under {run_dir}.")

    outdir.mkdir(parents=True, exist_ok=True)

    rows = [parse_manifest(p) for p in sorted(manifest_paths)]
    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)

    # Backfill stab_gap_area_end for older runs that predate the metric,
    # recomputing it from each run's accuracy_curves.csv.
    n_backfilled = backfill_area_end(df)
    if n_backfilled:
        print(
            f"Backfilled stab_gap_area_end for {n_backfilled} run(s) "
            "from accuracy_curves.csv"
        )

    master_csv = outdir / "master_index.csv"
    df.to_csv(master_csv, index=False)
    print(f"Found {len(manifest_paths)} run manifest(s) under {run_dir}")
    print(f"Wrote {master_csv}  ({len(df)} row(s))")

    summary_dir = outdir / "summary_tables"
    datasets = df["dataset"].dropna().unique()
    if len(datasets) == 0:
        print("WARNING: no dataset column found in any manifest.")
    for dataset in sorted(datasets):
        summary = build_summary_df(df, str(dataset))
        if summary.empty:
            continue
        summary_dir.mkdir(parents=True, exist_ok=True)
        csv_path = summary_dir / f"{dataset}_summary.csv"
        tex_path = summary_dir / f"{dataset}_summary.tex"
        summary.to_csv(csv_path)
        write_latex_table(summary, str(dataset), tex_path)

    missing = check_completeness(df, expected_seeds)
    if missing.empty:
        print(
            f"\nAll expected seeds {expected_seeds} present and completed for every "
            "(method, dataset, ablation) group. No missing runs."
        )
    else:
        missing_csv = outdir / "missing_runs.csv"
        missing.to_csv(missing_csv, index=False)
        n_missing = len(missing)
        print(f"\nWARNING: {n_missing} missing or incomplete run(s). See {missing_csv}")
        for issue_type, grp in missing.groupby("issue"):
            print(f"  [{issue_type}] {len(grp)} run(s):")
            for _, row in grp.iterrows():
                ak = row["ablation_key"] if row["ablation_key"] else ""
                av = row["ablation_value"] if row["ablation_value"] else ""
                ablation_str = f"{ak}/{av}" if ak else "(baseline)"
                print(
                    f"    method={row['method']}  dataset={row['dataset']}"
                    f"  ablation={ablation_str}  seed={row['seed']}"
                )
        sys.exit(2)


if __name__ == "__main__":
    main()
