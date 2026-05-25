"""One-shot patcher for the run_cifar.sh sweep diagonal bug (TODO 3.2).

The buggy ``ContinualMetrics._acc_at_boundary`` returned the first record
*after* the boundary eval at the same global_step (i.e. the post-one-step
dip), corrupting the diagonal ``R[i, i]`` for every task ``i < N - 1``.
Consequently ``FORG`` in ``metrics_summary.json`` (and the mirror in
``run_manifest.json::final_metrics``) was wrong.

This script:
  1. Finds every run with ``ablation_key in {cifar_headline,
     cifar_generalization}`` under ``outputs/``.
  2. For each task ``NN < N - 1`` reads
     ``results/task_curves/task_NN_eval.csv`` and uses its last
     ``task_NN_acc`` value as the corrected diagonal (TODO 3.2 recipe —
     a 10-step approximation of the true boundary eval value, which
     isn't persisted to disk).
  3. Patches ``accuracy_matrix.npy`` in place (only ``R[i, i]`` for
     ``i < N - 1``; off-diagonals and the last row are correct).
  4. Recomputes only ``FORG`` from the patched matrix; every other
     metric is left untouched (ACC / WC_ACC / min_ACC / WF* / WP* /
     stability_gap_* are not affected by the bug).
  5. Writes back ``metrics_summary.json`` and
     ``run_manifest.json::final_metrics``.

Idempotent: running twice is a no-op.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SWEEP_KEYS = {"cifar_headline", "cifar_generalization"}


def _patch_run(run_dir: Path) -> dict | None:
    """Return a summary dict if patched (or already-correct), or ``None`` if skipped."""
    manifest_path = run_dir / "run_manifest.json"
    matrix_path = run_dir / "results" / "accuracy_matrix.npy"
    summary_path = run_dir / "results" / "metrics_summary.json"
    curves_dir = run_dir / "results" / "task_curves"

    if not (manifest_path.exists() and matrix_path.exists() and summary_path.exists()):
        return None

    with manifest_path.open() as fh:
        manifest = json.load(fh)

    R = np.load(matrix_path)
    N = R.shape[0]

    # Recover the corrected diagonal from the last row of each task_NN_eval.csv.
    old_diag = [float(R[i, i]) for i in range(N)]
    new_diag = list(old_diag)
    for i in range(N - 1):
        csv_path = curves_dir / f"task_{i:02d}_eval.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        col = f"task_{i}_acc"
        new_diag[i] = float(df[col].iloc[-1])
        R[i, i] = new_diag[i]

    # R[N-1, N-1] is correct in both buggy and fixed runs — leave it.
    # Recompute FORG from the patched diagonal.
    if N < 2:
        forg_new = 0.0
    else:
        forg_new = float(
            sum(R[i, i] - R[N - 1, i] for i in range(N - 1)) / (N - 1)
        )

    # Persist patched matrix.
    np.save(matrix_path, R)

    # Update metrics_summary.json — preserve every other field.
    with summary_path.open() as fh:
        summary = json.load(fh)
    forg_old = summary.get("FORG")
    summary["FORG"] = forg_new
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Update run_manifest.json::final_metrics — same surgical edit.
    if isinstance(manifest.get("final_metrics"), dict):
        manifest["final_metrics"]["FORG"] = forg_new
        with manifest_path.open("w") as fh:
            json.dump(manifest, fh, indent=2)

    return {
        "run_dir": str(run_dir),
        "ablation_value": manifest.get("ablation_value"),
        "seed": manifest.get("seed"),
        "N": N,
        "old_diag": old_diag,
        "new_diag": new_diag,
        "forg_old": forg_old,
        "forg_new": forg_new,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    outputs = project_root / "outputs"
    if not outputs.exists():
        print(f"ERROR: no outputs directory at {outputs}", file=sys.stderr)
        return 1

    candidates: list[Path] = []
    for method_dir in outputs.iterdir():
        if not method_dir.is_dir():
            continue
        for dataset_dir in method_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for run_dir in dataset_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                manifest = run_dir / "run_manifest.json"
                if not manifest.exists():
                    continue
                with manifest.open() as fh:
                    m = json.load(fh)
                if m.get("ablation_key") in SWEEP_KEYS:
                    candidates.append(run_dir)

    candidates.sort()
    print(f"Found {len(candidates)} run_cifar.sh sweep runs.")

    patched, skipped = 0, 0
    for run_dir in candidates:
        summary = _patch_run(run_dir)
        if summary is None:
            skipped += 1
            print(f"  SKIP {run_dir.name} (missing files)")
            continue
        patched += 1
        delta_forg = summary["forg_new"] - (summary["forg_old"] or 0.0)
        print(
            f"  {summary['ablation_value']:35s} seed={summary['seed']} "
            f"N={summary['N']}  "
            f"diag {summary['old_diag']} → {summary['new_diag']}  "
            f"FORG {summary['forg_old']:+.4f} → {summary['forg_new']:+.4f} "
            f"(Δ={delta_forg:+.4f})"
        )

    print()
    print(f"Patched: {patched}    Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
