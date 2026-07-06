#!/usr/bin/env python3
"""Paired statistical comparisons between conditions from the master index.

Computes the seed-paired statistics reported in the results section: the paired
Wilcoxon signed-rank test (the default contrast) and the bootstrap 95 %
confidence interval of the mean paired difference (used for equivalence-style
claims).  Both operate on per-seed deltas ``delta_s = m(A, s) - m(B, s)``, with
seeds matched by their ``seed`` value across the two conditions.

A *condition* is a ``run_id`` with its trailing ``_s<seed>`` suffix removed, so
``er_rot_mnist_decomposition_D1_vanilla`` names the five-seed group
``..._s1 ... _s5``.  Pass two such keys to ``--pair`` to compare them.

Note on five-seed power: with five paired seeds the smallest attainable
two-sided Wilcoxon p-value is ``2 / 2^5 = 0.0625``, reached exactly when all
five deltas share a sign.  A reported ``p = 0.0625`` therefore means "all five
seeds agree in direction"; read effect sizes and the bootstrap CI alongside it.

Usage
-----
    # One explicit comparison on gap depth (the momentum-alone null, p ~ 0.44):
    python scripts/paired_stats.py \
        --pair er_rot_mnist_decomposition_D1_vanilla \
               er_rot_mnist_decomposition_D1_vanilla_M \
        --metric stab_gap_depth

    # Several metrics at once, and write a CSV:
    python scripts/paired_stats.py --pair A B --metric stab_gap_depth,ACC --csv out.csv

    # Every momentum on/off contrast (base vs base_M) on gap depth:
    python scripts/paired_stats.py --auto-momentum --metric stab_gap_depth

    # No pair given: list the available condition keys and exit.
    python scripts/paired_stats.py

Produces
--------
A table on stdout (one row per pair x metric) with: n paired seeds, the two
condition means, the mean paired difference, how many seeds agree in sign, the
Wilcoxon statistic and p-value, and the bootstrap 95 % CI of the mean
difference.  With ``--csv`` the same rows are also written to disk.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Metric columns in master_index.csv that a paired contrast may target.
_DEFAULT_METRIC = "stab_gap_depth"
_SEED_SUFFIX = re.compile(r"_s\d+$")


def condition_key(run_id: str) -> str:
    """Strip the trailing ``_s<seed>`` suffix to get the condition group key."""
    return _SEED_SUFFIX.sub("", run_id)


def load_master(master_path: Path) -> pd.DataFrame:
    """Load the master index and attach a ``condition`` key column."""
    df = pd.read_csv(master_path)
    if "run_id" not in df.columns or "seed" not in df.columns:
        sys.exit(f"{master_path} is missing required 'run_id'/'seed' columns.")
    df = df[df.get("status", "completed").fillna("completed") != "missing"].copy()
    df["condition"] = df["run_id"].map(condition_key)
    return df


def _paired_series(
    df: pd.DataFrame, cond_a: str, cond_b: str, metric: str
) -> Tuple[pd.Series, pd.Series]:
    """Return seed-aligned metric series for two conditions (inner-joined on seed)."""
    a = df[df["condition"] == cond_a].set_index("seed")[metric].dropna()
    b = df[df["condition"] == cond_b].set_index("seed")[metric].dropna()
    shared = a.index.intersection(b.index)
    return a.loc[shared].sort_index(), b.loc[shared].sort_index()


def bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of the paired deltas."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(lo), float(hi)


def compare(
    df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    metric: str,
    alternative: str = "two-sided",
    n_boot: int = 10_000,
    boot_seed: int = 0,
) -> dict:
    """Run the paired Wilcoxon test and bootstrap CI for one pair and metric."""
    a, b = _paired_series(df, cond_a, cond_b, metric)
    n = len(a)
    row = {
        "metric": metric,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n": n,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "mean_delta": np.nan,
        "signs_agree": "",
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
        "ci95_lo": np.nan,
        "ci95_hi": np.nan,
        "note": "",
    }
    if n < 1:
        row["note"] = "no shared seeds"
        return row

    deltas = (a.values - b.values).astype(float)
    row["mean_a"] = float(a.mean())
    row["mean_b"] = float(b.mean())
    row["mean_delta"] = float(deltas.mean())
    nonzero = deltas[deltas != 0]
    n_pos = int((nonzero > 0).sum())
    row["signs_agree"] = f"{max(n_pos, len(nonzero) - n_pos)}/{len(nonzero)}"

    if len(nonzero) < 1:
        row["note"] = "all deltas zero; Wilcoxon undefined"
    else:
        try:
            res = stats.wilcoxon(a.values, b.values, alternative=alternative)
            row["wilcoxon_stat"] = float(res.statistic)
            row["wilcoxon_p"] = float(res.pvalue)
        except ValueError as exc:  # e.g. all-zero after dropping ties
            row["note"] = f"wilcoxon: {exc}"

    row["ci95_lo"], row["ci95_hi"] = bootstrap_ci(
        deltas, n_boot=n_boot, seed=boot_seed
    )
    return row


def find_momentum_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """All (base, base_M) condition pairs present in the index (momentum off/on)."""
    conds = set(df["condition"].unique())
    pairs = [(c[:-2], c) for c in conds if c.endswith("_M") and c[:-2] in conds]
    return sorted(pairs)


def _format_table(rows: List[dict]) -> str:
    out = pd.DataFrame(rows)
    fmt = out.copy()
    for c in ["mean_a", "mean_b", "mean_delta", "ci95_lo", "ci95_hi"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.4g}" if pd.notna(v) else "--")
    fmt["wilcoxon_p"] = fmt["wilcoxon_p"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "--"
    )
    fmt["wilcoxon_stat"] = fmt["wilcoxon_stat"].map(
        lambda v: f"{v:g}" if pd.notna(v) else "--"
    )
    fmt["ci95"] = "[" + fmt["ci95_lo"] + ", " + fmt["ci95_hi"] + "]"
    cols = [
        "metric", "cond_a", "cond_b", "n", "mean_a", "mean_b", "mean_delta",
        "signs_agree", "wilcoxon_stat", "wilcoxon_p", "ci95", "note",
    ]
    return fmt[cols].to_string(index=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", type=Path, default=Path("outputs/master_index.csv"),
                   help="Path to master_index.csv (default: outputs/master_index.csv).")
    p.add_argument("--pair", nargs=2, metavar=("COND_A", "COND_B"), action="append",
                   default=[], help="A condition pair to compare; repeatable.")
    p.add_argument("--auto-momentum", action="store_true",
                   help="Compare every (base, base_M) momentum on/off pair found.")
    p.add_argument("--metric", default=_DEFAULT_METRIC,
                   help="Comma-separated metric column(s) (default: stab_gap_depth).")
    p.add_argument("--alternative", default="two-sided",
                   choices=["two-sided", "less", "greater"],
                   help="Wilcoxon alternative hypothesis (default: two-sided).")
    p.add_argument("--n-boot", type=int, default=10_000,
                   help="Bootstrap resamples for the CI (default: 10000).")
    p.add_argument("--boot-seed", type=int, default=0,
                   help="RNG seed for the bootstrap, for reproducibility (default: 0).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Also write the result rows to this CSV path.")
    args = p.parse_args(argv)

    if not args.master.exists():
        sys.exit(f"Master index not found: {args.master}")
    df = load_master(args.master)
    metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        sys.exit(f"Unknown metric column(s): {missing}\nAvailable: {list(df.columns)}")

    pairs: List[Tuple[str, str]] = [tuple(pr) for pr in args.pair]
    if args.auto_momentum:
        pairs.extend(find_momentum_pairs(df))

    if not pairs:
        print("No --pair given. Available condition keys:\n")
        for ds, sub in df.groupby("dataset"):
            print(f"  [{ds}]")
            for c in sorted(sub["condition"].unique()):
                n = int((sub["condition"] == c).sum())
                print(f"    {c}  (n={n})")
        print("\nPass two keys to --pair, or use --auto-momentum.")
        return 0

    known = set(df["condition"].unique())
    rows: List[dict] = []
    for cond_a, cond_b in pairs:
        for c in (cond_a, cond_b):
            if c not in known:
                print(f"warning: condition not found, skipping pair: {c}", file=sys.stderr)
        if cond_a not in known or cond_b not in known:
            continue
        for metric in metrics:
            rows.append(compare(
                df, cond_a, cond_b, metric,
                alternative=args.alternative,
                n_boot=args.n_boot, boot_seed=args.boot_seed,
            ))

    if not rows:
        sys.exit("No valid pairs to compare.")

    print(_format_table(rows))
    if args.csv is not None:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nWrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
