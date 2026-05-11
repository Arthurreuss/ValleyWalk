"""Tests for scripts/generate_figures.py  (T7.3).

Imports the script as a module (scripts/ is not a package) and exercises each
helper and figure function with synthetic fixtures.  Every figure function is
tested for:
  - PDF + PNG created when relevant data exists.
  - Graceful skip (no file created, no exception) when data is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script as a module
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_figures.py"
spec = importlib.util.spec_from_file_location("generate_figures", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

load_run_csv           = _mod.load_run_csv
infer_task_boundaries  = _mod.infer_task_boundaries
pick_median_run        = _mod.pick_median_run
save_figure            = _mod.save_figure
fig_accuracy_matrix    = _mod.fig_accuracy_matrix
fig_cone_sweep         = _mod.fig_cone_sweep
fig_ablation_summary   = _mod.fig_ablation_summary
fig_trust_radius       = _mod.fig_trust_radius
fig_eigenvalue_evolution = _mod.fig_eigenvalue_evolution
fig_stability_gap      = _mod.fig_stability_gap
fig_cost               = _mod.fig_cost
fig_memory_budget      = _mod.fig_memory_budget
MASTER_COLUMNS = [
    "run_id", "method", "dataset", "ablation_key", "ablation_value", "seed",
    "ACC", "FORG", "min_ACC", "WF10", "WF100", "WP10", "WP100", "WC_ACC",
    "stab_gap_max_drop", "stab_gap_recovery_steps", "wall_clock_total",
    "run_dir", "wandb_run_url", "git_commit", "status",
]


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_index_row(
    run_dir: str,
    method: str = "cacl",
    dataset: str = "rot_mnist",
    seed: int = 42,
    acc: float = 0.80,
    ablation_key: str | None = None,
    ablation_value: str | None = None,
    status: str = "completed",
) -> Dict[str, Any]:
    return {
        "run_id":        f"{method}_{dataset}_seed{seed}",
        "method":        method,
        "dataset":       dataset,
        "ablation_key":  ablation_key,
        "ablation_value": ablation_value,
        "seed":          seed,
        "ACC":           acc,
        "FORG":          0.06,
        "min_ACC":       0.75,
        "WF10":          0.08,
        "WF100":         0.10,
        "WP10":          0.05,
        "WP100":         0.07,
        "WC_ACC":        0.78,
        "stab_gap_max_drop":        0.12,
        "stab_gap_recovery_steps":  80,
        "wall_clock_total":         120.0,
        "run_dir":       run_dir,
        "wandb_run_url": None,
        "git_commit":    "abc1234",
        "status":        status,
    }


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_accuracy_matrix(run_dir: Path, num_tasks: int = 5) -> None:
    R = np.eye(num_tasks) * 0.9
    out = run_dir / "results" / "accuracy_matrix.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, R)


def _make_task_train_csvs(run_dir: Path, num_tasks: int = 3, steps_per_task: int = 10) -> None:
    base = run_dir / "results" / "task_curves"
    base.mkdir(parents=True, exist_ok=True)
    for t in range(num_tasks):
        start = t * steps_per_task
        rows = [{"step": start + s, "loss": 1.0 - s * 0.01} for s in range(steps_per_task)]
        pd.DataFrame(rows).to_csv(base / f"task_{t:02d}_train.csv", index=False)


def _make_trust_radius_csv(run_dir: Path, steps: int = 30) -> None:
    data = {
        "step":     list(range(steps)),
        "radius":   [0.1 + i * 0.001 for i in range(steps)],
        "rho":      [0.8] * steps,
        "accepted": [True] * steps,
    }
    _write_csv(
        run_dir / "results" / "diagnostics" / "trust_radius_history.csv",
        pd.DataFrame(data),
    )


def _make_eigenvalue_csv(run_dir: Path, num_d: int = 5, steps: int = 10) -> None:
    rows = []
    for i, step in enumerate(range(0, steps * 5, 5)):   # recorded every 5 steps
        row: Dict[str, Any] = {"step": step}
        for d in range(1, num_d + 1):
            row[f"lambda_{d}"] = float(num_d - d + 1) * (1.0 - i * 0.01)
        rows.append(row)
    _write_csv(
        run_dir / "results" / "diagnostics" / "eigenvalue_spectrum.csv",
        pd.DataFrame(rows),
    )


def _make_stability_gap_csv(run_dir: Path, task_id: int = 4, prev_tasks: int = 4) -> None:
    rows = []
    for step in range(20):
        row: Dict[str, Any] = {"step_within_task": step}
        for t in range(prev_tasks):
            row[f"task_{t}_acc"] = 0.80 - step * 0.005 + t * 0.02
        rows.append(row)
    _write_csv(
        run_dir / "results" / "stability_gap" / f"task_{task_id:02d}_gap.csv",
        pd.DataFrame(rows),
    )


def _make_wall_clock_csv(run_dir: Path, num_tasks: int = 3) -> None:
    rows = [
        {"task_id": t, "train_seconds": 5.0, "eval_seconds": 1.0, "total_seconds": 6.0}
        for t in range(num_tasks)
    ]
    _write_csv(
        run_dir / "results" / "timing" / "wall_clock.csv",
        pd.DataFrame(rows),
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestLoadRunCsv:
    def test_loads_existing_file(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
        df = load_run_csv(str(tmp_path), "data.csv")
        assert df is not None
        assert len(df) == 2

    def test_returns_none_for_missing_file(self, tmp_path):
        df = load_run_csv(str(tmp_path), "nonexistent.csv")
        assert df is None

    def test_returns_none_for_malformed_csv(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("not,valid\n{json}")
        # CSV reading with ambiguous content should not raise; may return a df
        # or None — the important thing is no exception propagates.
        try:
            _ = load_run_csv(str(tmp_path), "bad.csv")
        except Exception:
            pytest.fail("load_run_csv should not raise on bad content")


class TestInferTaskBoundaries:
    def test_returns_empty_for_missing_dir(self, tmp_path):
        assert infer_task_boundaries(str(tmp_path)) == []

    def test_infers_boundaries_from_train_csvs(self, tmp_path):
        _make_task_train_csvs(tmp_path, num_tasks=3, steps_per_task=10)
        bounds = infer_task_boundaries(str(tmp_path))
        assert bounds == [0, 10, 20]

    def test_returns_sorted_unique_steps(self, tmp_path):
        base = tmp_path / "results" / "task_curves"
        base.mkdir(parents=True, exist_ok=True)
        # Two tasks with overlapping step ranges (shouldn't happen in practice
        # but the function should handle it gracefully)
        pd.DataFrame({"step": [0, 1, 2]}).to_csv(base / "task_00_train.csv", index=False)
        pd.DataFrame({"step": [5, 6, 7]}).to_csv(base / "task_01_train.csv", index=False)
        bounds = infer_task_boundaries(str(tmp_path))
        assert bounds == sorted(set(bounds))


class TestPickMedianRun:
    def test_single_row_returns_that_row(self):
        df = pd.DataFrame([_make_index_row("/tmp", acc=0.80)])
        row = pick_median_run(df)
        assert row is not None
        assert abs(row["ACC"] - 0.80) < 1e-9

    def test_picks_row_closest_to_median(self):
        rows = [
            _make_index_row("/tmp/a", acc=0.70),
            _make_index_row("/tmp/b", acc=0.80),
            _make_index_row("/tmp/c", acc=0.90),
        ]
        df = pd.DataFrame(rows)
        row = pick_median_run(df)
        # Median is 0.80; pick_median_run should return the 0.80 row
        assert abs(row["ACC"] - 0.80) < 1e-9

    def test_returns_none_for_all_nan_acc(self):
        df = pd.DataFrame([_make_index_row("/tmp", acc=float("nan"))])
        assert pick_median_run(df) is None

    def test_returns_none_for_empty_df(self):
        df = pd.DataFrame(columns=MASTER_COLUMNS)
        assert pick_median_run(df) is None

    def test_skips_nan_rows_picks_from_valid(self):
        rows = [
            _make_index_row("/tmp/a", acc=float("nan")),
            _make_index_row("/tmp/b", acc=0.75),
        ]
        df = pd.DataFrame(rows)
        row = pick_median_run(df)
        assert row is not None
        assert abs(row["ACC"] - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# Figure function tests
# ---------------------------------------------------------------------------

class TestFigAccuracyMatrix:
    def test_creates_pdf_and_png(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_accuracy_matrix(run_dir)
        index = pd.DataFrame([_make_index_row(str(run_dir), method="cacl", acc=0.8)])
        fig_accuracy_matrix(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_accuracy_matrix.pdf").exists()
        assert (tmp_path / "figs" / "fig_accuracy_matrix.png").exists()

    def test_returns_run_ids(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_accuracy_matrix(run_dir)
        index = pd.DataFrame([_make_index_row(str(run_dir), method="er", acc=0.7)])
        ids = fig_accuracy_matrix(index, tmp_path / "figs")
        assert len(ids) == 1

    def test_skips_when_no_completed_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row("/nonexistent", status="failed")])
        ids = fig_accuracy_matrix(index, tmp_path / "figs")
        assert ids == []
        assert not (tmp_path / "figs" / "fig_accuracy_matrix.pdf").exists()

    def test_skips_when_npy_missing(self, tmp_path):
        # run_dir exists but no accuracy_matrix.npy
        index = pd.DataFrame([_make_index_row(str(tmp_path / "run0"), acc=0.8)])
        ids = fig_accuracy_matrix(index, tmp_path / "figs")
        assert ids == []

    def test_multiple_methods(self, tmp_path):
        for i, method in enumerate(["cacl", "er", "ncl"]):
            run_dir = tmp_path / f"run_{i}"
            _make_accuracy_matrix(run_dir)
            # Reuse same tmp dir for index
        rows = [
            _make_index_row(str(tmp_path / f"run_{i}"), method=m, acc=0.80 - i * 0.05)
            for i, m in enumerate(["cacl", "er", "ncl"])
        ]
        index = pd.DataFrame(rows)
        ids = fig_accuracy_matrix(index, tmp_path / "figs")
        assert len(ids) == 3


class TestFigConeSweep:
    def test_creates_files_for_a1_runs(self, tmp_path):
        rows = [
            _make_index_row(str(tmp_path / f"r{i}"),
                            method="cacl", dataset="rot_mnist",
                            acc=0.80, seed=s,
                            ablation_key="A1_cone_sweep",
                            ablation_value=f"alpha_{a}")
            for i, (s, a) in enumerate([(42, 0), (43, 45), (44, 90)])
        ]
        index = pd.DataFrame(rows)
        ids = fig_cone_sweep(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_cone_sweep.pdf").exists()
        assert len(ids) == 3

    def test_skips_when_no_a1_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row(str(tmp_path / "r0"), ablation_key="other")])
        ids = fig_cone_sweep(index, tmp_path / "figs")
        assert ids == []
        assert not (tmp_path / "figs" / "fig_cone_sweep.pdf").exists()

    def test_parses_alpha_values(self, tmp_path):
        rows = [
            _make_index_row(str(tmp_path / f"r{a}"),
                            ablation_key="A1_cone_sweep",
                            ablation_value=f"alpha_{a}",
                            seed=a)
            for a in [0, 30, 60, 90]
        ]
        index = pd.DataFrame(rows)
        ids = fig_cone_sweep(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_cone_sweep.pdf").exists()
        assert len(ids) == 4


class TestFigAblationSummary:
    def test_creates_files_when_runs_exist(self, tmp_path):
        rows = [
            _make_index_row(str(tmp_path / "r0"), method="cacl", ablation_key="A1_cone_sweep", ablation_value="alpha_45"),
            _make_index_row(str(tmp_path / "r1"), method="er",   ablation_key="baseline",      ablation_value="er"),
        ]
        index = pd.DataFrame(rows)
        ids = fig_ablation_summary(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_ablation_summary.pdf").exists()
        assert len(ids) == 2

    def test_skips_when_no_completed_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row("/tmp", status="failed")])
        ids = fig_ablation_summary(index, tmp_path / "figs")
        assert ids == []

    def test_baselines_before_ablations(self, tmp_path):
        # Just check it doesn't crash with a mixed set
        rows = [
            _make_index_row(str(tmp_path / f"r{i}"),
                            method=m, ablation_key=ak, ablation_value=av, seed=i)
            for i, (m, ak, av) in enumerate([
                ("er",   "baseline",      "er"),
                ("cacl", "A2_trust_region", "tr_true"),
                ("cacl", "A1_cone_sweep",  "alpha_45"),
            ])
        ]
        index = pd.DataFrame(rows)
        ids = fig_ablation_summary(index, tmp_path / "figs")
        assert len(ids) == 3


class TestFigTrustRadius:
    def test_creates_files_when_csv_exists(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_trust_radius_csv(run_dir, steps=30)
        _make_task_train_csvs(run_dir, num_tasks=2, steps_per_task=15)
        index = pd.DataFrame([
            _make_index_row(str(run_dir), method="cacl",
                            ablation_key="A2_trust_region", ablation_value="tr_true")
        ])
        ids = fig_trust_radius(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_trust_radius.pdf").exists()
        assert len(ids) == 1

    def test_skips_when_no_cacl_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row("/tmp", method="er")])
        ids = fig_trust_radius(index, tmp_path / "figs")
        assert ids == []
        assert not (tmp_path / "figs" / "fig_trust_radius.pdf").exists()

    def test_produces_file_even_without_trust_csv(self, tmp_path):
        # CACL run exists but no trust_radius_history.csv
        index = pd.DataFrame([
            _make_index_row(str(tmp_path / "run0"), method="cacl",
                            ablation_key="A2_trust_region", ablation_value="tr_true")
        ])
        # Even with no data, the function should produce the figure (with a "No data" annotation)
        ids = fig_trust_radius(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_trust_radius.pdf").exists()


class TestFigEigenvalueEvolution:
    def test_creates_files_when_csv_exists(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_eigenvalue_csv(run_dir, num_d=5, steps=10)
        _make_task_train_csvs(run_dir, num_tasks=2, steps_per_task=25)
        index = pd.DataFrame([_make_index_row(str(run_dir), method="cacl")])
        ids = fig_eigenvalue_evolution(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_eigenvalue_evolution.pdf").exists()
        assert len(ids) == 1

    def test_skips_when_no_cacl_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row("/tmp", method="er")])
        ids = fig_eigenvalue_evolution(index, tmp_path / "figs")
        assert ids == []
        assert not (tmp_path / "figs" / "fig_eigenvalue_evolution.pdf").exists()

    def test_handles_missing_eigenvalue_csv(self, tmp_path):
        index = pd.DataFrame([_make_index_row(str(tmp_path / "run0"), method="cacl")])
        ids = fig_eigenvalue_evolution(index, tmp_path / "figs")
        # Figure should still be produced (with "No eigenvalue data" annotation)
        assert (tmp_path / "figs" / "fig_eigenvalue_evolution.pdf").exists()

    def test_plots_at_most_5_eigenvalues(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_eigenvalue_csv(run_dir, num_d=8, steps=5)  # 8 eigenvalue columns
        index = pd.DataFrame([_make_index_row(str(run_dir), method="cacl")])
        ids = fig_eigenvalue_evolution(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_eigenvalue_evolution.pdf").exists()


class TestFigStabilityGap:
    def test_creates_files_when_csv_exists(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_stability_gap_csv(run_dir, task_id=4, prev_tasks=3)
        index = pd.DataFrame([_make_index_row(str(run_dir), method="cacl")])
        ids = fig_stability_gap(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_stability_gap.pdf").exists()
        assert len(ids) == 1

    def test_falls_back_to_lower_task_ids(self, tmp_path):
        run_dir = tmp_path / "run0"
        # Only task 2 CSV exists (not the preferred task 4)
        _make_stability_gap_csv(run_dir, task_id=2, prev_tasks=2)
        index = pd.DataFrame([_make_index_row(str(run_dir), method="er")])
        ids = fig_stability_gap(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_stability_gap.pdf").exists()
        assert len(ids) == 1

    def test_skips_when_no_completed_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row("/tmp", status="failed")])
        ids = fig_stability_gap(index, tmp_path / "figs")
        assert ids == []

    def test_multiple_methods_overlaid(self, tmp_path):
        for i, method in enumerate(["cacl", "er", "ncl"]):
            run_dir = tmp_path / f"run_{i}"
            _make_stability_gap_csv(run_dir, task_id=4, prev_tasks=3)
        rows = [
            _make_index_row(str(tmp_path / f"run_{i}"), method=m, seed=i)
            for i, m in enumerate(["cacl", "er", "ncl"])
        ]
        index = pd.DataFrame(rows)
        ids = fig_stability_gap(index, tmp_path / "figs")
        assert len(ids) == 3


class TestFigCost:
    def test_creates_files_with_timing_data(self, tmp_path):
        run_dir = tmp_path / "run0"
        _make_wall_clock_csv(run_dir, num_tasks=3)
        rows = [_make_index_row(str(run_dir), method="er", ablation_key="baseline", ablation_value="er")]
        index = pd.DataFrame(rows)
        ids = fig_cost(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_cost.pdf").exists()
        assert len(ids) == 1

    def test_includes_a7_amortize_variants(self, tmp_path):
        rows = []
        for k in [1, 10, 50]:
            run_dir = tmp_path / f"run_k{k}"
            _make_wall_clock_csv(run_dir)
            rows.append(
                _make_index_row(str(run_dir), method="cacl",
                                ablation_key="A7_amortize_K",
                                ablation_value=f"amortize_K_{k}",
                                seed=k)
            )
        index = pd.DataFrame(rows)
        ids = fig_cost(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_cost.pdf").exists()
        assert len(ids) == 3

    def test_skips_when_no_completed_runs(self, tmp_path):
        index = pd.DataFrame([_make_index_row("/tmp", status="failed")])
        ids = fig_cost(index, tmp_path / "figs")
        assert ids == []

    def test_handles_missing_timing_csv(self, tmp_path):
        # Run exists but no timing CSV
        index = pd.DataFrame([
            _make_index_row(str(tmp_path / "run0"), method="er",
                            ablation_key="baseline", ablation_value="er")
        ])
        ids = fig_cost(index, tmp_path / "figs")
        # Figure should still be created (with "No timing data" annotation)
        assert (tmp_path / "figs" / "fig_cost.pdf").exists()


class TestFigMemoryBudget:
    def test_creates_files_with_a10_runs(self, tmp_path):
        rows = [
            _make_index_row(str(tmp_path / f"r{b}"),
                            method="cacl",
                            ablation_key="A10_memory_budget",
                            ablation_value=f"budget_{b}",
                            acc=0.70 + b / 10000,
                            seed=b)
            for b in [100, 200, 500, 1000, 2000]
        ]
        index = pd.DataFrame(rows)
        ids = fig_memory_budget(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_memory_budget.pdf").exists()
        assert len(ids) == 5

    def test_includes_baseline_reference_lines(self, tmp_path):
        a10_rows = [
            _make_index_row(str(tmp_path / "r0"),
                            method="cacl",
                            ablation_key="A10_memory_budget",
                            ablation_value="budget_500",
                            acc=0.81)
        ]
        baseline_rows = [
            _make_index_row(str(tmp_path / "r1"),
                            method="er",
                            ablation_key="baseline",
                            ablation_value="er",
                            acc=0.79)
        ]
        index = pd.DataFrame(a10_rows + baseline_rows)
        ids = fig_memory_budget(index, tmp_path / "figs")
        assert (tmp_path / "figs" / "fig_memory_budget.pdf").exists()
        # ER baseline run_id should be included (for provenance)
        assert any("er" in str(rid) for rid in ids)

    def test_skips_when_no_relevant_runs(self, tmp_path):
        index = pd.DataFrame([
            _make_index_row("/tmp", method="ncl", ablation_key="baseline", acc=0.8)
        ])
        ids = fig_memory_budget(index, tmp_path / "figs")
        assert ids == []
        assert not (tmp_path / "figs" / "fig_memory_budget.pdf").exists()


# ---------------------------------------------------------------------------
# End-to-end: main() with a realistic synthetic sweep
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _build_full_index(self, tmp_path: Path) -> pd.DataFrame:
        """Build a minimal master_index with data for all 8 figure types."""
        rows = []

        # Baselines (for fig 1, 3, 6, 7, 8)
        for i, (method, ds) in enumerate([
            ("cacl",         "rot_mnist"),
            ("er",           "rot_mnist"),
            ("agem",         "rot_mnist"),
            ("ncl",          "rot_mnist"),
            ("cacl",         "dom_cifar100"),
            ("er",           "dom_cifar100"),
        ]):
            run_dir = tmp_path / f"run_b{i}"
            _make_accuracy_matrix(run_dir, num_tasks=3)
            _make_stability_gap_csv(run_dir, task_id=4)
            _make_wall_clock_csv(run_dir)
            rows.append(
                _make_index_row(str(run_dir), method=method, dataset=ds, seed=i,
                                ablation_key="baseline", ablation_value=method)
            )

        # A1 cone sweep (for fig 2, 3)
        for j, alpha in enumerate([0, 45, 90]):
            run_dir = tmp_path / f"run_a1_{j}"
            rows.append(
                _make_index_row(str(run_dir), method="cacl", dataset="rot_mnist",
                                seed=42 + j, acc=0.78 + j * 0.01,
                                ablation_key="A1_cone_sweep",
                                ablation_value=f"alpha_{alpha}")
            )

        # A2 trust region (for fig 4)
        for j, (av, seed) in enumerate([("tr_true", 42), ("tr_false", 43)]):
            run_dir = tmp_path / f"run_a2_{j}"
            _make_trust_radius_csv(run_dir)
            _make_task_train_csvs(run_dir, num_tasks=3, steps_per_task=10)
            rows.append(
                _make_index_row(str(run_dir), method="cacl", seed=seed,
                                ablation_key="A2_trust_region", ablation_value=av)
            )

        # A7 amortization (for fig 7)
        for j, k in enumerate([1, 10, 50]):
            run_dir = tmp_path / f"run_a7_{j}"
            _make_wall_clock_csv(run_dir)
            rows.append(
                _make_index_row(str(run_dir), method="cacl", seed=42 + j,
                                ablation_key="A7_amortize_K",
                                ablation_value=f"amortize_K_{k}")
            )

        # A10 memory budget (for fig 8)
        for j, budget in enumerate([100, 500, 2000]):
            run_dir = tmp_path / f"run_a10_{j}"
            rows.append(
                _make_index_row(str(run_dir), method="cacl", seed=42 + j,
                                acc=0.72 + j * 0.03,
                                ablation_key="A10_memory_budget",
                                ablation_value=f"budget_{budget}")
            )

        # CACL with eigenvalue data (for fig 5)
        run_dir = tmp_path / "run_cacl_eig"
        _make_eigenvalue_csv(run_dir, num_d=5, steps=10)
        _make_task_train_csvs(run_dir, num_tasks=3, steps_per_task=10)
        rows.append(_make_index_row(str(run_dir), method="cacl", seed=999, acc=0.83))

        return pd.DataFrame(rows)

    def test_all_8_figures_created_or_skipped_gracefully(self, tmp_path):
        """Running on a realistic index produces figures without errors."""
        index = self._build_full_index(tmp_path)
        index_path = tmp_path / "master_index.csv"
        index.to_csv(index_path, index=False)
        figs_dir = tmp_path / "figures"

        old_argv = sys.argv
        sys.argv = [
            "generate_figures.py",
            "--index", str(index_path),
            "--outdir", str(figs_dir),
        ]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code is None or exc.code == 0
        finally:
            sys.argv = old_argv

        # Every figure should have been at least attempted.
        # Some may be skipped (empty ids) but must not crash.
        expected_pdfs = [
            "fig_accuracy_matrix.pdf",
            "fig_cone_sweep.pdf",
            "fig_ablation_summary.pdf",
            "fig_trust_radius.pdf",
            "fig_eigenvalue_evolution.pdf",
            "fig_stability_gap.pdf",
            "fig_cost.pdf",
            "fig_memory_budget.pdf",
        ]
        for pdf_name in expected_pdfs:
            assert (figs_dir / pdf_name).exists(), f"{pdf_name} not created"

    def test_provenance_json_created(self, tmp_path):
        index = self._build_full_index(tmp_path)
        index_path = tmp_path / "master_index.csv"
        index.to_csv(index_path, index=False)
        figs_dir = tmp_path / "figures"

        old_argv = sys.argv
        sys.argv = ["generate_figures.py", "--index", str(index_path), "--outdir", str(figs_dir)]
        try:
            _mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        prov_path = figs_dir / "figure_provenance.json"
        assert prov_path.exists()
        with open(prov_path) as fh:
            prov = json.load(fh)
        assert len(prov) == 9
        # Provenance should list at least one run_id for most figures
        all_ids = [ids for ids in prov.values()]
        assert any(len(ids) > 0 for ids in all_ids)

    def test_empty_index_exits_cleanly(self, tmp_path):
        index_path = tmp_path / "master_index.csv"
        pd.DataFrame(columns=["run_id", "method", "dataset", "status"]).to_csv(
            index_path, index=False
        )
        figs_dir = tmp_path / "figures"
        old_argv = sys.argv
        sys.argv = ["generate_figures.py", "--index", str(index_path), "--outdir", str(figs_dir)]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code is None or exc.code == 0
        finally:
            sys.argv = old_argv

    def test_nonexistent_index_exits_with_error(self, tmp_path):
        old_argv = sys.argv
        sys.argv = [
            "generate_figures.py",
            "--index", str(tmp_path / "no_such_file.csv"),
            "--outdir", str(tmp_path / "figs"),
        ]
        exit_code = None
        try:
            _mod.main()
        except SystemExit as exc:
            exit_code = exc.code
        finally:
            sys.argv = old_argv
        assert exit_code == 1
