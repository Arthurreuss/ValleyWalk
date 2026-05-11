#!/usr/bin/env python3
"""Aggregate run_manifest.json files into a master index and per-dataset tables.

T7.2 — Results aggregation into master index.

Usage
-----
    python scripts/aggregate_results.py \\
        --run-dir outputs/ \\
        [--outdir outputs/] \\
        [--seeds 42,123,456,789,1337]

Produces
--------
<outdir>/master_index.csv
    One row per run.  Columns: run_id, method, dataset, ablation_key,
    ablation_value, seed, ACC, BWT, FWT, forgetting, LA, stab_gap_max_drop,
    stab_gap_recovery_steps, wall_clock_total, run_dir, wandb_run_url,
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
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column definitions (spec-mandated order)
# ---------------------------------------------------------------------------

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
    "stab_gap_recovery_steps",
    "wall_clock_total",
    "run_dir",
    "wandb_run_url",
    "git_commit",
    "status",
]

# Metrics that are averaged and reported in summary tables.
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
    "stab_gap_recovery_steps",
]

# Columns that identify a unique (experiment, condition) group.
GROUP_COLUMNS: List[str] = ["method", "ablation_key", "ablation_value"]

# Human-readable metric labels for LaTeX column headers.
_METRIC_LABELS: Dict[str, str] = {
    "ACC":                      "ACC",
    "FORG":                     "FORG",
    "min_ACC":                  "min-ACC",
    "WF10":                     r"WF$_{10}$",
    "WF100":                    r"WF$_{100}$",
    "WP10":                     r"WP$_{10}$",
    "WP100":                    r"WP$_{100}$",
    "WC_ACC":                   "WC-ACC",
    "stab_gap_max_drop":        r"$\Delta_{\max}$",
    "stab_gap_recovery_steps":  "Recov.",
}


# ---------------------------------------------------------------------------
# 1. Walk and parse run_manifest.json files
# ---------------------------------------------------------------------------

def find_manifests(run_dir: Path) -> List[Path]:
    """Recursively find all ``run_manifest.json`` files under *run_dir*.

    Results are sorted for deterministic ordering.
    """
    return sorted(run_dir.rglob("run_manifest.json"))


def parse_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Parse a single ``run_manifest.json`` into a flat dict matching MASTER_COLUMNS.

    Missing fields produce ``None`` rather than raising — so a partial or
    malformed manifest contributes a row with NaN metrics rather than
    crashing the aggregation.
    """
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            m = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"WARNING: could not parse {manifest_path}: {exc}",
            file=sys.stderr,
        )
        # Return a skeleton row so the caller can still report the path.
        return {col: None for col in MASTER_COLUMNS} | {
            "run_dir": str(manifest_path.parent),
            "status":  "parse_error",
        }

    final: Dict[str, Any] = m.get("final_metrics") or {}

    return {
        "run_id":                   m.get("run_id"),
        "method":                   m.get("method"),
        "dataset":                  m.get("dataset"),
        "ablation_key":             m.get("ablation_key"),
        "ablation_value":           m.get("ablation_value"),
        "seed":                     m.get("seed"),
        # Metrics live inside final_metrics; rename stability gap keys to match spec.
        "ACC":                      final.get("ACC"),
        "FORG":                     final.get("FORG"),
        "min_ACC":                  final.get("min_ACC"),
        "WF10":                     final.get("WF10"),
        "WF100":                    final.get("WF100"),
        "WP10":                     final.get("WP10"),
        "WP100":                    final.get("WP100"),
        "WC_ACC":                   final.get("WC_ACC"),
        "stab_gap_max_drop":        final.get("stability_gap_max_drop"),
        "stab_gap_recovery_steps":  final.get("stability_gap_recovery_steps"),
        "wall_clock_total":         m.get("wall_clock_total_seconds"),
        "run_dir":                  str(manifest_path.parent),
        "wandb_run_url":            m.get("wandb_run_url"),
        "git_commit":               m.get("git_commit"),
        "status":                   m.get("status", "unknown"),
    }


# ---------------------------------------------------------------------------
# 2. Summary tables (mean ± std per group per dataset)
# ---------------------------------------------------------------------------

def _fmt_cell(mean: float, std: float) -> str:
    """Format a numeric cell as 'mean ± std' (3 decimal places).

    Returns '---' when *mean* is NaN.
    """
    if math.isnan(mean):
        return "---"
    if math.isnan(std):
        # Only one observation — report the value without uncertainty.
        return f"{mean:.3f}"
    return f"{mean:.3f} \u00b1 {std:.3f}"  # ± in UTF-8


def build_summary_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Build a mean±std summary DataFrame for *dataset*.

    Returns a DataFrame indexed by (method, ablation_key, ablation_value)
    whose columns are METRIC_COLUMNS with string 'mean ± std' cells.
    Returns an empty DataFrame if *dataset* is not present.
    """
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for key, grp in subset.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        method, ablation_key, ablation_value = key
        row: Dict[str, Any] = {
            "method":         method,
            "ablation_key":   ablation_key if pd.notna(ablation_key) else "",
            "ablation_value": ablation_value if pd.notna(ablation_value) else "",
        }
        for col in METRIC_COLUMNS:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) == 0:
                row[col] = "---"
            elif len(vals) == 1:
                row[col] = f"{float(vals.iloc[0]):.3f}"
            else:
                row[col] = _fmt_cell(float(vals.mean()), float(vals.std(ddof=1)))
        rows.append(row)

    result = pd.DataFrame(rows).set_index(["method", "ablation_key", "ablation_value"])
    return result


def _tex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    # Order matters: backslash must come first.
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
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
        r"\begin{table}[t]",
        r"  \centering",
        (
            f"  \\caption{{Results on \\textsc{{{_tex_escape(dataset)}}}"
            r" (mean$\pm$std over seeds).}"
        ),
        f"  \\label{{tab:{dataset.lower().replace('-', '_').replace(' ', '_')}}}",
        "  \\begin{tabular}{lll" + "r" * n_metric_cols + "}",
        r"    \toprule",
        "    Method & Ablation & Value & "
        + " & ".join(header_cols)
        + r" \\",
        r"    \midrule",
    ]

    for (method, ablation_key, ablation_value), row in summary.iterrows():
        cells = [str(row.get(c, "---")) for c in METRIC_COLUMNS]
        lines.append(
            f"    {_tex_escape(str(method))} & "
            f"{_tex_escape(str(ablation_key))} & "
            f"{_tex_escape(str(ablation_value))} & "
            + " & ".join(cells)
            + r" \\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Completeness check
# ---------------------------------------------------------------------------

def check_completeness(
    df: pd.DataFrame,
    expected_seeds: List[int],
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
        Seeds that every (method, dataset, ablation) combo is expected to have.

    Returns
    -------
    pd.DataFrame with columns: method, dataset, ablation_key, ablation_value,
    seed, issue.  Empty when everything is in order.
    """
    issues: List[Dict[str, Any]] = []

    # --- Problem 1: runs with non-completed status ---
    bad_status = df[df["status"] != "completed"]
    for _, row in bad_status.iterrows():
        issues.append({
            "method":         row["method"],
            "dataset":        row["dataset"],
            "ablation_key":   row["ablation_key"],
            "ablation_value": row["ablation_value"],
            "seed":           row["seed"],
            "issue":          f"status={row['status']}",
        })

    # --- Problem 2: missing seeds per experiment group ---
    group_cols = ["method", "dataset", "ablation_key", "ablation_value"]
    for key, grp in df.groupby(group_cols, dropna=False, sort=True):
        method, dataset, ablation_key, ablation_value = key
        present_seeds = set(
            grp["seed"].dropna().apply(lambda x: int(x)).tolist()
        )
        for seed in expected_seeds:
            if seed not in present_seeds:
                issues.append({
                    "method":         method,
                    "dataset":        dataset,
                    "ablation_key":   ablation_key,
                    "ablation_value": ablation_value,
                    "seed":           seed,
                    "issue":          "missing",
                })

    return pd.DataFrame(
        issues,
        columns=["method", "dataset", "ablation_key", "ablation_value", "seed", "issue"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate run_manifest.json files into master_index.csv "
            "and per-dataset LaTeX/CSV summary tables."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root directory to search recursively for run_manifest.json files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Output directory for master_index.csv and summary_tables/. "
            "Defaults to --run-dir."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,1337",
        metavar="N,N,...",
        help=(
            "Comma-separated expected seeds for the completeness check. "
            "Default: 42,123,456,789,1337."
        ),
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    outdir: Path = (args.outdir or args.run_dir).resolve()
    expected_seeds: List[int] = [int(s.strip()) for s in args.seeds.split(",")]

    if not run_dir.is_dir():
        print(f"ERROR: --run-dir '{run_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # ── 1. Collect all manifests ───────────────────────────────────────────
    manifest_paths = find_manifests(run_dir)
    if not manifest_paths:
        print(
            f"WARNING: no run_manifest.json files found under {run_dir}.",
            file=sys.stderr,
        )
        # Write an empty master_index.csv so downstream scripts don't crash.
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=MASTER_COLUMNS).to_csv(
            outdir / "master_index.csv", index=False
        )
        sys.exit(0)

    print(f"Found {len(manifest_paths)} run manifest(s) under {run_dir}")
    rows = [parse_manifest(p) for p in manifest_paths]
    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)

    # ── 2. Write master_index.csv ──────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    master_csv = outdir / "master_index.csv"
    df.to_csv(master_csv, index=False)
    print(f"Wrote {master_csv}  ({len(df)} row(s))")

    # ── 3. Per-dataset summary tables ─────────────────────────────────────
    summary_dir = outdir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].dropna().unique())
    if not datasets:
        print("WARNING: no dataset column found in any manifest.", file=sys.stderr)
    for dataset in datasets:
        summary = build_summary_df(df, dataset)
        if summary.empty:
            continue

        # CSV (machine- and human-readable)
        csv_path = summary_dir / f"{dataset}_summary.csv"
        summary.to_csv(csv_path)
        print(f"Wrote {csv_path}")

        # LaTeX table
        tex_path = summary_dir / f"{dataset}_summary.tex"
        write_latex_table(summary, dataset, tex_path)
        print(f"Wrote {tex_path}")

    # ── 4. Completeness check ──────────────────────────────────────────────
    missing = check_completeness(df, expected_seeds)
    if missing.empty:
        print(
            f"\nAll expected seeds {expected_seeds} present and completed "
            "for every (method, dataset, ablation) group. No missing runs."
        )
    else:
        missing_csv = outdir / "missing_runs.csv"
        missing.to_csv(missing_csv, index=False)
        n_missing = len(missing)
        print(
            f"\nWARNING: {n_missing} missing or incomplete run(s). "
            f"See {missing_csv}",
            file=sys.stderr,
        )
        # Print a concise grouped summary to stderr.
        for issue_type, grp in missing.groupby("issue", sort=False):
            print(f"  [{issue_type}] {len(grp)} run(s):", file=sys.stderr)
            for _, row in grp.iterrows():
                ak = row["ablation_key"] if pd.notna(row["ablation_key"]) else ""
                av = row["ablation_value"] if pd.notna(row["ablation_value"]) else ""
                ablation_str = f"{ak}/{av}" if ak else "(baseline)"
                print(
                    f"    method={row['method']}  dataset={row['dataset']}"
                    f"  ablation={ablation_str}  seed={row['seed']}",
                    file=sys.stderr,
                )
        # Non-zero exit so CI/sweep scripts can detect incompleteness.
        sys.exit(2)


if __name__ == "__main__":
    main()
