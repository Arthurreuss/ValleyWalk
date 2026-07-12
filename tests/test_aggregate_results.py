"""Tests for scripts/aggregate_results.py.

We import the script as a module by adding the scripts/ directory to sys.path,
then exercising each function in isolation and via a full end-to-end run on
synthetic run_manifest.json fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (scripts/ is not a package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "aggregate_results.py"

spec = importlib.util.spec_from_file_location("aggregate_results", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

find_manifests = _mod.find_manifests
parse_manifest = _mod.parse_manifest
build_summary_df = _mod.build_summary_df
write_latex_table = _mod.write_latex_table
check_completeness = _mod.check_completeness
compute_area_end_from_curve = _mod.compute_area_end_from_curve
backfill_area_end = _mod.backfill_area_end
MASTER_COLUMNS = _mod.MASTER_COLUMNS
METRIC_COLUMNS = _mod.METRIC_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    tmp_path: Path,
    subdir: str = "run_00",
    method: str = "cacl",
    dataset: str = "rot_mnist",
    seed: int = 42,
    acc: float = 0.82,
    status: str = "completed",
    ablation_key: str | None = None,
    ablation_value: str | None = None,
) -> Path:
    """Write a synthetic run_manifest.json and return its path."""
    run_dir = tmp_path / subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "run_id": f"{method}_{dataset}_seed{seed}",
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "ablation_key": ablation_key,
        "ablation_value": ablation_value,
        "status": status,
        "git_commit": "abc1234",
        "git_dirty": False,
        "wandb_run_id": None,
        "wandb_run_url": None,
        "wall_clock_total_seconds": 120.5,
        "final_metrics": {
            "ACC": acc,
            "FORG": 0.06,
            "min_ACC": 0.75,
            "WF10": 0.08,
            "WF100": 0.10,
            "WP10": 0.05,
            "WP100": 0.07,
            "WC_ACC": 0.78,
            "stability_gap_max_drop": 0.12,
            "stability_gap_depth": 0.10,
            "stability_gap_area": 5.2,
            "stability_gap_recovery_steps": 80,
            "true_grad_cosine_mean": 0.71,
            "true_grad_cosine_min": 0.31,
            "true_grad_mag_ratio_mean": 0.95,
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# Tests: find_manifests
# ---------------------------------------------------------------------------


class TestFindManifests:
    def test_finds_single_manifest(self, tmp_path):
        p = _make_manifest(tmp_path, "r0")
        found = find_manifests(tmp_path)
        assert p in found

    def test_finds_nested_manifests(self, tmp_path):
        p1 = _make_manifest(tmp_path, "a/r0")
        p2 = _make_manifest(tmp_path, "b/c/r1", seed=123)
        found = find_manifests(tmp_path)
        assert p1 in found and p2 in found

    def test_returns_empty_when_none(self, tmp_path):
        assert find_manifests(tmp_path) == []

    def test_returns_sorted(self, tmp_path):
        _make_manifest(tmp_path, "z/run")
        _make_manifest(tmp_path, "a/run", seed=99)
        found = find_manifests(tmp_path)
        assert found == sorted(found)


# ---------------------------------------------------------------------------
# Tests: parse_manifest
# ---------------------------------------------------------------------------


class TestParseManifest:
    def test_all_master_columns_present(self, tmp_path):
        p = _make_manifest(tmp_path)
        row = parse_manifest(p)
        for col in MASTER_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_metric_values_extracted(self, tmp_path):
        p = _make_manifest(tmp_path, acc=0.77)
        row = parse_manifest(p)
        assert abs(row["ACC"] - 0.77) < 1e-9
        assert abs(row["FORG"] - 0.06) < 1e-9
        assert abs(row["min_ACC"] - 0.75) < 1e-9

    def test_stability_gap_keys_remapped(self, tmp_path):
        p = _make_manifest(tmp_path)
        row = parse_manifest(p)
        # Spec wants stab_gap_max_drop (not stability_gap_max_drop)
        assert "stab_gap_max_drop" in row
        assert "stab_gap_recovery_steps" in row
        assert abs(row["stab_gap_max_drop"] - 0.12) < 1e-9
        assert row["stab_gap_recovery_steps"] == 80

    def test_wall_clock_from_total_seconds(self, tmp_path):
        p = _make_manifest(tmp_path)
        row = parse_manifest(p)
        assert abs(row["wall_clock_total"] - 120.5) < 1e-9

    def test_run_dir_is_parent_of_manifest(self, tmp_path):
        p = _make_manifest(tmp_path, "deep/nested/run")
        row = parse_manifest(p)
        assert row["run_dir"] == str(p.parent)

    def test_ablation_fields_present(self, tmp_path):
        p = _make_manifest(tmp_path, ablation_key="A1_cone", ablation_value="alpha_30")
        row = parse_manifest(p)
        assert row["ablation_key"] == "A1_cone"
        assert row["ablation_value"] == "alpha_30"

    def test_malformed_json_returns_skeleton(self, tmp_path):
        bad = tmp_path / "bad" / "run_manifest.json"
        bad.parent.mkdir()
        bad.write_text("{not valid json")
        row = parse_manifest(bad)
        assert row["status"] == "parse_error"
        assert row["run_dir"] == str(bad.parent)

    def test_missing_fields_return_none(self, tmp_path):
        # Minimal manifest — many fields absent
        p = tmp_path / "min" / "run_manifest.json"
        p.parent.mkdir()
        p.write_text(json.dumps({"method": "er", "status": "completed"}))
        row = parse_manifest(p)
        assert row["dataset"] is None
        assert row["ACC"] is None


# ---------------------------------------------------------------------------
# Tests: compute_area_end_from_curve / backfill_area_end
# ---------------------------------------------------------------------------


def _write_curve(path: Path, header: list[str], rows: list[list]) -> None:
    """Write an accuracy_curves.csv-style file ('' for absent cells)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines += [",".join("" if c is None else str(c) for c in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


class TestComputeAreaEndFromCurve:
    def test_matches_tracker_reference_end(self, tmp_path):
        """Recompute must equal StabilityGapTracker.gap_area(reference='end')."""
        from src.eval.stability_gap import StabilityGapTracker

        # task_0 trains alone, then task_1 starts: task_0 dips and recovers.
        # Pre-switch baseline = last task_0 acc before task_1 appears (0.90).
        header = ["step", "task_0_acc", "task_1_acc"]
        pre_rows = [[0, 0.50, None], [10, 0.80, None], [20, 0.90, None]]
        task1_series = [
            (30, 0.90),
            (40, 0.60),
            (50, 0.55),
            (60, 0.70),
            (70, 0.85),
            (80, 0.88),
            (90, 0.88),
        ]
        rows = pre_rows + [[s, a, 0.30] for s, a in task1_series]
        curve = tmp_path / "results" / "accuracy_curves.csv"
        _write_curve(curve, header, rows)

        tracker = StabilityGapTracker.__new__(StabilityGapTracker)
        tracker._pre_task_acc = {0: 0.90}
        tracker._records = [(s, {0: a}) for s, a in task1_series]
        expected = tracker.gap_area(reference="end")

        got = compute_area_end_from_curve(curve)
        assert got == pytest.approx(expected)
        assert got > 0.0

    def test_missing_file_returns_none(self, tmp_path):
        assert compute_area_end_from_curve(tmp_path / "nope.csv") is None

    def test_single_task_returns_none(self, tmp_path):
        curve = tmp_path / "accuracy_curves.csv"
        _write_curve(curve, ["step", "task_0_acc"], [[0, 0.5], [10, 0.8]])
        assert compute_area_end_from_curve(curve) is None

    def test_nonmonotonic_steps_returns_none(self, tmp_path):
        """Corrupted (interleaved) CSV -> None, never a bogus number."""
        header = ["step", "task_0_acc", "task_1_acc"]
        rows = [[0, 0.9, None], [10, 0.9, 0.3], [44441, 0.4, 0.4], [50, 0.8, 0.5]]
        curve = tmp_path / "accuracy_curves.csv"
        _write_curve(curve, header, rows)
        assert compute_area_end_from_curve(curve) is None

    def test_backfill_fills_only_missing(self, tmp_path):
        header = ["step", "task_0_acc", "task_1_acc"]
        rows = [[0, 0.9, None]] + [
            [s, a, 0.3]
            for s, a in [(10, 0.9), (20, 0.5), (30, 0.7), (40, 0.85), (50, 0.88)]
        ]
        # Row 0 already has a value -> must be preserved untouched.
        d0, d1 = tmp_path / "r0", tmp_path / "r1"
        _write_curve(d0 / "results" / "accuracy_curves.csv", header, rows)
        _write_curve(d1 / "results" / "accuracy_curves.csv", header, rows)
        df = pd.DataFrame(
            [
                {"stab_gap_area_end": 99.0, "run_dir": str(d0)},
                {"stab_gap_area_end": None, "run_dir": str(d1)},
            ]
        )
        n = backfill_area_end(df)
        assert n == 1
        assert df.loc[0, "stab_gap_area_end"] == 99.0  # preserved
        assert df.loc[1, "stab_gap_area_end"] > 0.0  # filled


# ---------------------------------------------------------------------------
# Tests: build_summary_df
# ---------------------------------------------------------------------------


class TestBuildSummaryDf:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=MASTER_COLUMNS)

    def _base_row(
        self,
        seed,
        acc,
        method="cacl",
        dataset="rot_mnist",
        ablation_key=None,
        ablation_value=None,
    ):
        return {
            "run_id": f"r{seed}",
            "method": method,
            "dataset": dataset,
            "ablation_key": ablation_key,
            "ablation_value": ablation_value,
            "seed": seed,
            "ACC": acc,
            "FORG": 0.06,
            "min_ACC": 0.75,
            "WF10": 0.08,
            "WF100": 0.10,
            "WP10": 0.05,
            "WP100": 0.07,
            "WC_ACC": 0.78,
            "stab_gap_max_drop": 0.10,
            "stab_gap_depth": 0.08,
            "stab_gap_area": 4.5,
            "stab_gap_area_w250": 9.0,
            "stab_gap_area_end": 3.2,
            "stab_gap_recovery_steps": 50,
            "true_grad_cosine_mean": 0.71,
            "true_grad_cosine_min": 0.31,
            "true_grad_mag_ratio_mean": 0.95,
            "wall_clock_total": 120.0,
            "run_dir": "/tmp",
            "wandb_run_url": None,
            "git_commit": "abc",
            "status": "completed",
        }

    def test_returns_empty_for_unknown_dataset(self, tmp_path):
        df = self._make_df([self._base_row(42, 0.8)])
        result = build_summary_df(df, "dom_cifar100")
        assert result.empty

    def test_single_seed_no_std(self):
        df = pd.DataFrame([self._base_row(42, 0.80)])
        result = build_summary_df(df, "rot_mnist")
        assert "0.800" in result.loc[("cacl", "", ""), "ACC"]
        assert "±" not in result.loc[("cacl", "", ""), "ACC"]

    def test_multiple_seeds_mean_std(self):
        rows = [
            self._base_row(s, a) for s, a in [(42, 0.80), (123, 0.82), (456, 0.84)]
        ]  # noqa: E501
        df = pd.DataFrame(rows)
        result = build_summary_df(df, "rot_mnist")
        cell = result.loc[("cacl", "", ""), "ACC"]
        assert "±" in cell, f"Expected ±, got: {cell}"
        # Mean should be 0.82
        mean_str = cell.split("±")[0].strip()
        assert abs(float(mean_str) - 0.82) < 1e-3

    def test_multiple_methods_separate_rows(self):
        rows = [
            self._base_row(42, 0.80, method="er"),
            self._base_row(42, 0.75, method="cacl"),
        ]
        df = pd.DataFrame(rows)
        result = build_summary_df(df, "rot_mnist")
        assert ("er", "", "") in result.index
        assert ("cacl", "", "") in result.index

    def test_nan_values_produce_dashes(self):
        row = self._base_row(42, acc=None)
        df = pd.DataFrame([row])
        result = build_summary_df(df, "rot_mnist")
        assert result.loc[("cacl", "", ""), "ACC"] == "---"

    def test_ablation_groups_separately(self):
        rows = [
            self._base_row(42, 0.80, ablation_key="A1", ablation_value="v0"),
            self._base_row(42, 0.75, ablation_key="A1", ablation_value="v1"),
        ]
        df = pd.DataFrame(rows)
        result = build_summary_df(df, "rot_mnist")
        assert ("cacl", "A1", "v0") in result.index
        assert ("cacl", "A1", "v1") in result.index


# ---------------------------------------------------------------------------
# Tests: write_latex_table
# ---------------------------------------------------------------------------


class TestWriteLatexTable:
    def _make_summary(self):
        data = {m: ["0.82 ± 0.01"] for m in METRIC_COLUMNS}
        idx = pd.MultiIndex.from_tuples(
            [("cacl", "", ""), ("er", "", "")],
            names=["method", "ablation_key", "ablation_value"],
        )
        return pd.DataFrame(data, index=idx)

    def test_creates_file(self, tmp_path):
        summary = self._make_summary()
        out = tmp_path / "tables" / "test.tex"
        write_latex_table(summary, "rot_mnist", out)
        assert out.exists()

    def test_contains_required_latex_environments(self, tmp_path):
        summary = self._make_summary()
        out = tmp_path / "test.tex"
        write_latex_table(summary, "rot_mnist", out)
        text = out.read_text()
        assert r"\begin{table}" in text
        assert r"\end{table}" in text
        assert r"\begin{tabular}" in text
        assert r"\end{tabular}" in text
        assert r"\toprule" in text
        assert r"\midrule" in text
        assert r"\bottomrule" in text

    def test_contains_caption_and_label(self, tmp_path):
        summary = self._make_summary()
        out = tmp_path / "test.tex"
        write_latex_table(summary, "rot_mnist", out)
        text = out.read_text()
        assert r"\caption" in text
        assert r"\label" in text
        assert "rot_mnist" in text

    def test_booktabs_comment_present(self, tmp_path):
        summary = self._make_summary()
        out = tmp_path / "test.tex"
        write_latex_table(summary, "rot_mnist", out)
        text = out.read_text()
        assert "booktabs" in text

    def test_tex_escaping(self, tmp_path):
        # Underscore in method name must be escaped
        data = {m: ["0.80"] for m in METRIC_COLUMNS}
        idx = pd.MultiIndex.from_tuples(
            [("cacl_v2", "A_1", "val_x")],
            names=["method", "ablation_key", "ablation_value"],
        )
        summary = pd.DataFrame(data, index=idx)
        out = tmp_path / "test.tex"
        write_latex_table(summary, "rot_mnist", out)
        text = out.read_text()
        # Raw underscores in column data must be escaped
        assert r"cacl\_v2" in text
        assert r"A\_1" in text

    def test_each_method_row_present(self, tmp_path):
        summary = self._make_summary()
        out = tmp_path / "test.tex"
        write_latex_table(summary, "rot_mnist", out)
        text = out.read_text()
        assert "cacl" in text
        assert "er" in text


# ---------------------------------------------------------------------------
# Tests: check_completeness
# ---------------------------------------------------------------------------


class TestCheckCompleteness:
    def _make_df(self, records):
        rows = []
        for method, dataset, ablation_key, ablation_value, seed, status in records:
            rows.append(
                {
                    "run_id": f"r{seed}",
                    "method": method,
                    "dataset": dataset,
                    "ablation_key": ablation_key,
                    "ablation_value": ablation_value,
                    "seed": seed,
                    "ACC": 0.8,
                    "FORG": 0.06,
                    "min_ACC": 0.75,
                    "WF10": 0.08,
                    "WF100": 0.10,
                    "WP10": 0.05,
                    "WP100": 0.07,
                    "WC_ACC": 0.78,
                    "stab_gap_max_drop": 0.1,
                    "stab_gap_depth": 0.08,
                    "stab_gap_area": 4.5,
                    "stab_gap_recovery_steps": 50,
                    "true_grad_cosine_mean": 0.71,
                    "true_grad_cosine_min": 0.31,
                    "true_grad_mag_ratio_mean": 0.95,
                    "wall_clock_total": 100.0,
                    "run_dir": "/tmp",
                    "wandb_run_url": None,
                    "git_commit": "abc",
                    "status": status,
                }
            )
        return pd.DataFrame(rows, columns=MASTER_COLUMNS)

    def test_no_issues_when_all_seeds_completed(self):
        records = [
            ("er", "rot_mnist", None, None, 42, "completed"),
            ("er", "rot_mnist", None, None, 123, "completed"),
        ]
        df = self._make_df(records)
        result = check_completeness(df, [42, 123])
        assert result.empty

    def test_flags_failed_status(self):
        records = [
            ("er", "rot_mnist", None, None, 42, "failed"),
            ("er", "rot_mnist", None, None, 123, "completed"),
        ]
        df = self._make_df(records)
        result = check_completeness(df, [42, 123])
        failed_rows = result[result["issue"].str.startswith("status=")]
        assert len(failed_rows) == 1
        assert failed_rows.iloc[0]["seed"] == 42

    def test_flags_running_status(self):
        records = [("er", "rot_mnist", None, None, 42, "running")]
        df = self._make_df(records)
        result = check_completeness(df, [42])
        assert any(result["issue"].str.startswith("status="))

    def test_flags_missing_seed(self):
        # Only seed 42 present; seed 123 missing
        records = [("er", "rot_mnist", None, None, 42, "completed")]
        df = self._make_df(records)
        result = check_completeness(df, [42, 123])
        missing_rows = result[result["issue"] == "missing"]
        assert len(missing_rows) == 1
        assert missing_rows.iloc[0]["seed"] == 123

    def test_flags_multiple_missing_seeds(self):
        records = [("cacl", "rot_mnist", "A1", "v0", 42, "completed")]
        df = self._make_df(records)
        expected = [42, 123, 456]
        result = check_completeness(df, expected)
        missing = result[result["issue"] == "missing"]
        assert set(missing["seed"].tolist()) == {123, 456}

    def test_separate_groups_checked_independently(self):
        # Group 1 has all seeds; group 2 is missing seed 456
        records = [
            ("er", "rot_mnist", None, None, 42, "completed"),
            ("er", "rot_mnist", None, None, 123, "completed"),
            ("cacl", "rot_mnist", None, None, 42, "completed"),
            ("cacl", "rot_mnist", None, None, 123, "completed"),
        ]
        df = self._make_df(records)
        result = check_completeness(df, [42, 123, 456])
        missing = result[result["issue"] == "missing"]
        # Both groups are missing seed 456
        assert len(missing) == 2
        assert all(missing["seed"] == 456)


# ---------------------------------------------------------------------------
# Tests: end-to-end (master_index.csv + tables + completeness)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_creates_master_index_csv(self, tmp_path):
        """Five completed seeds → master_index.csv with 5 rows."""
        outdir = tmp_path / "out"
        seeds = [42, 123, 456, 789, 1337]
        for i, seed in enumerate(seeds):
            _make_manifest(tmp_path, f"run_{i:02d}", seed=seed, acc=0.80 + i * 0.01)

        # Run main via subprocess-style: patch sys.argv then call main
        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--run-dir",
            str(tmp_path),
            "--outdir",
            str(outdir),
            "--seeds",
            ",".join(str(s) for s in seeds),
        ]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code == 0 or exc.code is None
        finally:
            sys.argv = old_argv

        master = outdir / "master_index.csv"
        assert master.exists(), "master_index.csv not created"
        df = pd.read_csv(master)
        assert len(df) == 5
        for col in MASTER_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_creates_summary_tables(self, tmp_path):
        """5 seeds → both .csv and .tex summary files for rot_mnist."""
        outdir = tmp_path / "out"
        seeds = [42, 123, 456, 789, 1337]
        for i, seed in enumerate(seeds):
            _make_manifest(tmp_path, f"run_{i:02d}", seed=seed)

        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--run-dir",
            str(tmp_path),
            "--outdir",
            str(outdir),
            "--seeds",
            ",".join(str(s) for s in seeds),
        ]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code == 0 or exc.code is None
        finally:
            sys.argv = old_argv

        assert (outdir / "summary_tables" / "rot_mnist_summary.csv").exists()
        assert (outdir / "summary_tables" / "rot_mnist_summary.tex").exists()

    def test_latex_table_contains_mean_std(self, tmp_path):
        """5 seeds → LaTeX table cell has ± symbol."""
        outdir = tmp_path / "out"
        seeds = [42, 123, 456, 789, 1337]
        for i, seed in enumerate(seeds):
            _make_manifest(tmp_path, f"run_{i:02d}", seed=seed, acc=0.78 + i * 0.01)

        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--run-dir",
            str(tmp_path),
            "--outdir",
            str(outdir),
            "--seeds",
            ",".join(str(s) for s in seeds),
        ]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code == 0 or exc.code is None
        finally:
            sys.argv = old_argv

        tex = (outdir / "summary_tables" / "rot_mnist_summary.tex").read_text()
        assert "±" in tex or r"\pm" in tex or "±" in tex

    def test_missing_seeds_exits_nonzero(self, tmp_path):
        """3 of 5 seeds present → exit code 2 and missing_runs.csv created."""
        outdir = tmp_path / "out"
        for i, seed in enumerate([42, 123, 456]):
            _make_manifest(tmp_path, f"run_{i:02d}", seed=seed)

        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--run-dir",
            str(tmp_path),
            "--outdir",
            str(outdir),
            "--seeds",
            "42,123,456,789,1337",
        ]
        exit_code = None
        try:
            _mod.main()
        except SystemExit as exc:
            exit_code = exc.code
        finally:
            sys.argv = old_argv

        assert exit_code == 2, f"Expected exit code 2, got {exit_code}"
        assert (outdir / "missing_runs.csv").exists()
        df = pd.read_csv(outdir / "missing_runs.csv")
        assert set(df["seed"].tolist()) == {789, 1337}

    def test_empty_run_dir_exits_cleanly(self, tmp_path):
        """No manifests → exit 0, empty master_index.csv."""
        outdir = tmp_path / "out"
        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--run-dir",
            str(tmp_path),
            "--outdir",
            str(outdir),
        ]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code == 0 or exc.code is None
        finally:
            sys.argv = old_argv

        master = outdir / "master_index.csv"
        assert master.exists()
        df = pd.read_csv(master)
        assert len(df) == 0

    def test_multiple_datasets_separate_tables(self, tmp_path):
        """Runs from two datasets → two sets of summary files."""
        outdir = tmp_path / "out"
        for i, (seed, ds) in enumerate([(42, "rot_mnist"), (42, "dom_cifar100")]):
            _make_manifest(tmp_path, f"run_{i:02d}", seed=seed, dataset=ds)

        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--run-dir",
            str(tmp_path),
            "--outdir",
            str(outdir),
            "--seeds",
            "42",
        ]
        try:
            _mod.main()
        except SystemExit as exc:
            assert exc.code == 0 or exc.code is None
        finally:
            sys.argv = old_argv

        assert (outdir / "summary_tables" / "rot_mnist_summary.tex").exists()
        assert (outdir / "summary_tables" / "dom_cifar100_summary.tex").exists()
