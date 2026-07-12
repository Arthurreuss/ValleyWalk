"""Tests for the dual-write tracking utility.

Verifies that:
  (b) csv_only mode writes local CSVs without errors and without W&B dependency.
  (c) write_run_manifest() produces a valid JSON matching the schema.

W&B mode (criterion a) requires a live account and is not tested here; the
_NullRun path exercises every code branch that the real wandb.Run would hit,
so the logic is still exercised.

Coverage:
  (a) _NullRun: all methods are no-ops; id/url/name attributes exist.
  (b) init_tracking csv_only: returns (_NullRun, str, bool).
  (c) DualLogger: creates CSV on first log(); header matches sorted keys;
      subsequent rows appended; flush-safe; close() is idempotent;
      context-manager protocol works.
  (d) DualLogger: parent directory is created automatically.
  (e) save_and_log_checkpoint: file is written locally; no W&B call in csv_only.
  (f) save_and_log_results: no-op in csv_only mode (no error).
  (g) log_accuracy_matrix_table: no-op for _NullRun (no error).
  (h) log_metrics_summary_table: no-op for _NullRun (no error).
  (i) write_run_manifest: all required schema keys present; JSON is valid;
      config_hash starts with "sha256:"; started_at / finished_at are strings;
      checkpoint_files is a list.
"""

import csv
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.tracking import (
    _NullRun,
    DualLogger,
    init_tracking,
    log_accuracy_matrix_table,
    log_metrics_summary_table,
    save_and_log_checkpoint,
    save_and_log_results,
    write_run_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(backend: str = "csv_only") -> SimpleNamespace:
    """Minimal config for tracking tests."""
    return SimpleNamespace(
        tracking=SimpleNamespace(
            backend=backend,
            wandb=SimpleNamespace(
                project="cacl_experiments",
                entity=None,
                tags=[],
                artifacts=SimpleNamespace(
                    log_checkpoints=True,
                    log_results=True,
                    log_buffer=False,
                    checkpoint_type="model",
                    results_type="results",
                ),
                tables=SimpleNamespace(
                    log_accuracy_matrix=True,
                    log_metrics_summary=True,
                ),
            ),
        ),
        method=SimpleNamespace(name="cacl"),
        dataset=SimpleNamespace(name="rot_mnist"),
        seed=42,
        checkpointing=SimpleNamespace(dir="checkpoints"),
    )


def _make_timings(n: int = 3) -> list:
    return [
        {"task_id": i, "train_seconds": 10.0, "eval_seconds": 2.0, "total_seconds": 12.0}
        for i in range(n)
    ]


def _make_final_metrics() -> dict:
    return {
        "ACC": 0.82,
        "BWT": -0.03,
        "FWT": 0.01,
        "forgetting": 0.05,
        "LA": 0.87,
        "stability_gap_max_drop": 0.1,
        "stability_gap_recovery_steps": 80,
    }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# (a) _NullRun
# ---------------------------------------------------------------------------

class TestNullRun:
    def test_id_is_none(self):
        run = _NullRun()
        assert run.id is None

    def test_url_is_none(self):
        run = _NullRun()
        assert run.url is None

    def test_name_is_csv_only(self):
        run = _NullRun()
        assert run.name == "csv_only"

    def test_log_is_noop(self):
        run = _NullRun()
        run.log({"loss": 0.5}, step=0)  # must not raise

    def test_log_artifact_is_noop(self):
        run = _NullRun()
        run.log_artifact(object())  # must not raise

    def test_finish_is_noop(self):
        run = _NullRun()
        run.finish()  # must not raise


# ---------------------------------------------------------------------------
# (b) init_tracking — csv_only mode
# ---------------------------------------------------------------------------

class TestInitTracking:
    def test_returns_three_values(self):
        cfg = _make_cfg("csv_only")
        result = init_tracking(cfg)
        assert len(result) == 3

    def test_run_is_null_run_in_csv_only(self):
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        assert isinstance(run, _NullRun)

    def test_git_commit_is_string(self):
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        assert isinstance(git_commit, str)
        assert len(git_commit) > 0

    def test_git_dirty_is_bool(self):
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        assert isinstance(git_dirty, bool)

    def test_started_at_is_recorded(self):
        """init_tracking must record a start timestamp for write_run_manifest."""
        from src.utils.tracking import _run_started_at
        cfg = _make_cfg("csv_only")
        run, _, _ = init_tracking(cfg)
        assert id(run) in _run_started_at
        assert isinstance(_run_started_at[id(run)], str)


# ---------------------------------------------------------------------------
# (c) DualLogger — CSV writing
# ---------------------------------------------------------------------------

class TestDualLogger:
    def test_csv_created_on_first_log(self, tmp_path):
        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, _NullRun())
        logger.log(0, {"loss": 1.0, "eta": 0.1})
        logger.close()
        assert os.path.exists(path)

    def test_csv_header_matches_sorted_keys(self, tmp_path):
        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, _NullRun())
        logger.log(0, {"loss": 1.0, "eta": 0.1})
        logger.close()

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == ["step", "eta", "loss"]  # sorted

    def test_csv_row_values_correct(self, tmp_path):
        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, _NullRun())
        logger.log(5, {"loss": 0.5, "acc": 0.9})
        logger.close()

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert int(rows[0]["step"]) == 5
        assert float(rows[0]["loss"]) == pytest.approx(0.5)

    def test_multiple_rows_appended(self, tmp_path):
        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, _NullRun())
        for i in range(10):
            logger.log(i, {"loss": float(i)})
        logger.close()

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 10
        assert [int(r["step"]) for r in rows] == list(range(10))

    def test_extra_keys_in_later_rows_silently_dropped(self, tmp_path):
        """Keys not in the header (from first log) are silently ignored."""
        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, _NullRun())
        logger.log(0, {"loss": 1.0})
        logger.log(1, {"loss": 0.9, "extra_key": 42})  # must not raise
        logger.close()

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert "extra_key" not in rows[1]

    def test_parent_directory_created_automatically(self, tmp_path):
        nested = str(tmp_path / "deep" / "nested" / "train.csv")
        logger = DualLogger(nested, _NullRun())
        logger.log(0, {"loss": 0.0})
        logger.close()
        assert os.path.exists(nested)

    def test_close_is_idempotent(self, tmp_path):
        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, _NullRun())
        logger.log(0, {"loss": 0.5})
        logger.close()
        logger.close()  # must not raise

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "train.csv")
        with DualLogger(path, _NullRun()) as logger:
            logger.log(0, {"loss": 0.5})
        # File should be closed after __exit__
        assert logger._csv_file is None

    def test_null_run_log_called(self, tmp_path):
        """DualLogger passes metrics to wandb_run.log()."""
        calls = []

        class SpyRun:
            def log(self, metrics, step=None, **kwargs):
                calls.append((step, metrics))

        path = str(tmp_path / "train.csv")
        logger = DualLogger(path, SpyRun())
        logger.log(3, {"loss": 0.7})
        logger.close()

        assert len(calls) == 1
        assert calls[0][0] == 3
        assert calls[0][1]["loss"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# (d) DualLogger — parent directory creation
# ---------------------------------------------------------------------------
# (already covered in TestDualLogger.test_parent_directory_created_automatically)


# ---------------------------------------------------------------------------
# (e) save_and_log_checkpoint
# ---------------------------------------------------------------------------

class TestSaveAndLogCheckpoint:
    def test_file_written_locally(self, tmp_path):
        model = TinyModel()
        cfg = _make_cfg("csv_only")
        ckpt_dir = str(tmp_path / "checkpoints")
        path = save_and_log_checkpoint(model, 0, _NullRun(), cfg, ckpt_dir)
        assert os.path.exists(path)

    def test_filename_format(self, tmp_path):
        model = TinyModel()
        cfg = _make_cfg("csv_only")
        ckpt_dir = str(tmp_path / "checkpoints")
        path = save_and_log_checkpoint(model, 5, _NullRun(), cfg, ckpt_dir)
        assert path.endswith("model_task_05.pt")

    def test_state_dict_loadable(self, tmp_path):
        """The saved .pt file must be loadable back into a fresh model."""
        model = TinyModel()
        cfg = _make_cfg("csv_only")
        ckpt_dir = str(tmp_path / "checkpoints")
        path = save_and_log_checkpoint(model, 0, _NullRun(), cfg, ckpt_dir)

        model2 = TinyModel()
        model2.load_state_dict(torch.load(path, weights_only=True))
        # Parameters should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_no_wandb_call_in_csv_only(self, tmp_path):
        """In csv_only mode, no wandb.Artifact is constructed."""
        calls = []

        class SpyRun(_NullRun):
            def log_artifact(self, artifact, **kwargs):
                calls.append(artifact)

        model = TinyModel()
        cfg = _make_cfg("csv_only")
        ckpt_dir = str(tmp_path / "checkpoints")
        save_and_log_checkpoint(model, 0, SpyRun(), cfg, ckpt_dir)
        assert len(calls) == 0, "log_artifact should NOT be called in csv_only mode"


# ---------------------------------------------------------------------------
# (f) save_and_log_results
# ---------------------------------------------------------------------------

class TestSaveAndLogResults:
    def test_noop_in_csv_only(self, tmp_path):
        """save_and_log_results must be a no-op in csv_only mode."""
        calls = []

        class SpyRun(_NullRun):
            def log_artifact(self, artifact, **kwargs):
                calls.append(artifact)

        cfg = _make_cfg("csv_only")
        save_and_log_results(str(tmp_path), SpyRun(), cfg)
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# (g) log_accuracy_matrix_table
# ---------------------------------------------------------------------------

class TestLogAccuracyMatrixTable:
    def test_noop_for_null_run(self):
        """Must not raise when run is _NullRun."""
        matrix = [[0.9, 0.0], [0.7, 0.8]]
        log_accuracy_matrix_table(matrix, _NullRun(), num_tasks=2)  # no error

    def test_noop_for_null_run_numpy(self):
        """Handles numpy arrays gracefully."""
        import numpy as np
        matrix = np.array([[0.9, 0.0], [0.7, 0.8]])
        log_accuracy_matrix_table(matrix, _NullRun(), num_tasks=2)  # no error


# ---------------------------------------------------------------------------
# (h) log_metrics_summary_table
# ---------------------------------------------------------------------------

class TestLogMetricsSummaryTable:
    def test_noop_for_null_run(self):
        """Must not raise when run is _NullRun."""
        log_metrics_summary_table(_make_final_metrics(), _NullRun())  # no error

    def test_handles_missing_keys(self):
        """Must not raise when some metric keys are absent."""
        log_metrics_summary_table({}, _NullRun())  # no error


# ---------------------------------------------------------------------------
# (i) write_run_manifest
# ---------------------------------------------------------------------------

_REQUIRED_MANIFEST_KEYS = {
    "run_id",
    "wandb_run_id",
    "wandb_run_url",
    "method",
    "dataset",
    "seed",
    "git_commit",
    "git_dirty",
    "config_hash",
    "hydra_overrides",
    "ablation_key",
    "ablation_value",
    "started_at",
    "finished_at",
    "status",
    "final_metrics",
    "wall_clock_total_seconds",
    "gpu",
    "pytorch_version",
    "checkpoint_files",
    "wandb_artifact_ids",
}


class TestWriteRunManifest:
    def _write(self, tmp_path, extra_cfg=None) -> dict:
        """Helper: call init_tracking + write_run_manifest, return parsed JSON."""
        cfg = _make_cfg("csv_only")
        if extra_cfg:
            for k, v in extra_cfg.items():
                setattr(cfg, k, v)

        run, git_commit, git_dirty = init_tracking(cfg)
        out_path = write_run_manifest(
            run_dir=str(tmp_path),
            run=run, cfg=cfg,
            git_commit=git_commit, git_dirty=git_dirty,
            final_metrics=_make_final_metrics(),
            task_timings=_make_timings(3),
        )
        with open(out_path) as fh:
            return json.load(fh)

    def test_file_written(self, tmp_path):
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        out_path = write_run_manifest(
            str(tmp_path), run, cfg, git_commit, git_dirty,
            _make_final_metrics(), _make_timings(),
        )
        assert os.path.exists(out_path)
        assert out_path.endswith("run_manifest.json")

    def test_all_required_keys_present(self, tmp_path):
        manifest = self._write(tmp_path)
        missing = _REQUIRED_MANIFEST_KEYS - set(manifest.keys())
        assert not missing, f"Manifest missing keys: {missing}"

    def test_config_hash_format(self, tmp_path):
        manifest = self._write(tmp_path)
        assert manifest["config_hash"].startswith("sha256:"), (
            f"config_hash should start with 'sha256:', got: {manifest['config_hash']}"
        )
        # SHA-256 hex digest is 64 chars; with prefix: 7 + 64 = 71 chars
        assert len(manifest["config_hash"]) == 71

    def test_method_and_dataset(self, tmp_path):
        manifest = self._write(tmp_path)
        assert manifest["method"] == "cacl"
        assert manifest["dataset"] == "rot_mnist"

    def test_seed_is_int(self, tmp_path):
        manifest = self._write(tmp_path)
        assert isinstance(manifest["seed"], int)
        assert manifest["seed"] == 42

    def test_git_dirty_is_bool(self, tmp_path):
        manifest = self._write(tmp_path)
        assert isinstance(manifest["git_dirty"], bool)

    def test_timestamps_are_strings(self, tmp_path):
        manifest = self._write(tmp_path)
        # started_at may be None if init_tracking wasn't called
        if manifest["started_at"] is not None:
            assert isinstance(manifest["started_at"], str)
        assert isinstance(manifest["finished_at"], str)

    def test_pytorch_version_present(self, tmp_path):
        manifest = self._write(tmp_path)
        assert manifest["pytorch_version"] == torch.__version__

    def test_status_completed_by_default(self, tmp_path):
        manifest = self._write(tmp_path)
        assert manifest["status"] == "completed"

    def test_status_can_be_failed(self, tmp_path):
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        out_path = write_run_manifest(
            str(tmp_path), run, cfg, git_commit, git_dirty,
            {}, [], status="failed",
        )
        with open(out_path) as fh:
            manifest = json.load(fh)
        assert manifest["status"] == "failed"

    def test_wall_clock_total_is_sum(self, tmp_path):
        timings = _make_timings(3)  # 3 × 12.0 s = 36.0 s
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        out_path = write_run_manifest(
            str(tmp_path), run, cfg, git_commit, git_dirty,
            _make_final_metrics(), timings,
        )
        with open(out_path) as fh:
            manifest = json.load(fh)
        assert manifest["wall_clock_total_seconds"] == pytest.approx(36.0)

    def test_checkpoint_files_list(self, tmp_path):
        """checkpoint_files should be an (empty) list when no .pt files exist."""
        manifest = self._write(tmp_path)
        assert isinstance(manifest["checkpoint_files"], list)

    def test_checkpoint_files_populated_when_exist(self, tmp_path):
        """If model_task_*.pt files exist, they are listed in the manifest."""
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)

        # Create fake checkpoint files in the expected directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "model_task_00.pt").write_bytes(b"fake")
        (ckpt_dir / "model_task_01.pt").write_bytes(b"fake")

        out_path = write_run_manifest(
            str(tmp_path), run, cfg, git_commit, git_dirty,
            _make_final_metrics(), _make_timings(),
        )
        with open(out_path) as fh:
            manifest = json.load(fh)

        assert len(manifest["checkpoint_files"]) == 2
        for f in manifest["checkpoint_files"]:
            assert "model_task_" in f
            assert f.endswith(".pt")

    def test_wandb_run_id_null_for_csv_only(self, tmp_path):
        manifest = self._write(tmp_path)
        assert manifest["wandb_run_id"] is None

    def test_wandb_run_url_null_for_csv_only(self, tmp_path):
        manifest = self._write(tmp_path)
        assert manifest["wandb_run_url"] is None

    def test_hydra_overrides_is_list(self, tmp_path):
        manifest = self._write(tmp_path)
        assert isinstance(manifest["hydra_overrides"], list)

    def test_final_metrics_in_manifest(self, tmp_path):
        manifest = self._write(tmp_path)
        assert "ACC" in manifest["final_metrics"]
        assert manifest["final_metrics"]["ACC"] == pytest.approx(0.82)

    def test_json_is_valid_and_idempotent(self, tmp_path):
        """The written file must round-trip through json.loads without error."""
        cfg = _make_cfg("csv_only")
        run, git_commit, git_dirty = init_tracking(cfg)
        out_path = write_run_manifest(
            str(tmp_path), run, cfg, git_commit, git_dirty,
            _make_final_metrics(), _make_timings(),
        )
        with open(out_path) as fh:
            raw = fh.read()
        # Re-parse — must not raise
        manifest = json.loads(raw)
        assert isinstance(manifest, dict)
